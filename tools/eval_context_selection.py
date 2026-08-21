"""Run deterministic context-selection evaluation cases."""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from rag.context_evaluation import ContextSelectionEvaluator
from rag.context_manager import ContextManager


def load_cases(path: Path) -> Iterable[Dict[str, Any]]:
    """Load context evaluation cases from JSONL."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def check_source_coverage(
    selected_report: Dict[str, Any], expected_sources: List[str]
) -> Dict[str, Any]:
    """Check whether selected context covers expected sources."""
    if not expected_sources:
        return {"coverage_rate": 1.0, "missing_sources": []}
    selected_sources = {
        item.get("source") for item in selected_report.get("selected", [])
        if item.get("source")
    }
    covered = sum(1 for src in expected_sources if src in selected_sources)
    missing = [src for src in expected_sources if src not in selected_sources]
    return {
        "coverage_rate": covered / len(expected_sources),
        "missing_sources": missing,
    }


def check_answer_hint_coverage(
    selected_docs: List[Any], expected_hints: List[str]
) -> Dict[str, Any]:
    """Check whether selected content contains expected answer keywords."""
    if not expected_hints:
        return {"hint_coverage_rate": 1.0, "missing_hints": []}
    selected_text = " ".join(
        str(doc.get("content", "")).lower() if isinstance(doc, dict)
        else str(getattr(doc, "content", "")).lower()
        for doc in selected_docs
    )
    covered = sum(1 for hint in expected_hints if hint.lower() in selected_text)
    missing = [hint for hint in expected_hints if hint.lower() not in selected_text]
    return {
        "hint_coverage_rate": covered / len(expected_hints),
        "missing_hints": missing,
    }


def check_answer_faithfulness(
    selected_docs: List[Any], expected_answer: str
) -> Dict[str, Any]:
    """Measure the fraction of answer terms supported by selected context."""
    answer_terms = set(re.findall(r"[a-z0-9]+", expected_answer.lower()))
    if not answer_terms:
        return {"faithfulness_score": 1.0, "unsupported_terms": []}

    selected_text = " ".join(
        str(doc.get("content", "")).lower() if isinstance(doc, dict)
        else str(getattr(doc, "content", "")).lower()
        for doc in selected_docs
    )
    context_terms = set(re.findall(r"[a-z0-9]+", selected_text))
    unsupported = sorted(answer_terms - context_terms)
    return {
        "faithfulness_score": (
            len(answer_terms - set(unsupported)) / len(answer_terms)
        ),
        "unsupported_terms": unsupported,
    }


def evaluate_cases(cases: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate context selection cases and return per-case and aggregate metrics."""
    results: List[Dict[str, Any]] = []
    for case in cases:
        manager = ContextManager(
            max_tokens=case.get("max_tokens", 120000),
            system_prompt_reserve=case.get("system_prompt_reserve", 1000),
            max_docs_per_source=case.get("max_docs_per_source", 2),
        )
        selected_docs = manager.truncate_context(
            case.get("documents", []),
            query=case.get("query"),
            history=case.get("history"),
        )
        metrics = ContextSelectionEvaluator.evaluate(manager.last_selection_report)
        source_quality = check_source_coverage(
            manager.last_selection_report,
            case.get("expected_sources", []),
        )
        hint_quality = check_answer_hint_coverage(
            selected_docs, case.get("expected_answer_hints", [])
        )
        faithfulness = check_answer_faithfulness(
            selected_docs, case.get("expected_answer", "")
        )
        quality_signals = [
            source_quality["coverage_rate"],
            hint_quality["hint_coverage_rate"],
        ]
        if case.get("expected_answer"):
            quality_signals.append(faithfulness["faithfulness_score"])
        answer_quality = {
            "source_coverage_rate": source_quality["coverage_rate"],
            "hint_coverage_rate": hint_quality["hint_coverage_rate"],
            "score": sum(quality_signals) / len(quality_signals),
        }
        if case.get("expected_answer"):
            answer_quality["faithfulness_score"] = faithfulness["faithfulness_score"]
        results.append({
            "name": case.get("name", "unnamed"),
            "metrics": metrics.__dict__,
            "source_quality": source_quality,
            "hint_quality": hint_quality,
            "faithfulness": faithfulness,
            "answer_quality": answer_quality,
            "selection_report": manager.last_selection_report,
        })

    metric_names = (
        "selected_count",
        "dropped_count",
        "duplicate_drop_rate",
        "source_diversity",
        "budget_utilization",
    )
    aggregate = {
        name: sum(result["metrics"][name] for result in results) / len(results)
        if results else 0.0
        for name in metric_names
    }
    aggregate["source_coverage_rate"] = (
        sum(result["source_quality"]["coverage_rate"] for result in results)
        / len(results)
        if results
        else 0.0
    )
    aggregate["hint_coverage_rate"] = (
        sum(result["hint_quality"]["hint_coverage_rate"] for result in results)
        / len(results)
        if results
        else 0.0
    )
    aggregate["answer_quality_score"] = (
        sum(result["answer_quality"]["score"] for result in results)
        / len(results)
        if results
        else 0.0
    )
    aggregate["faithfulness_score"] = (
        sum(result["faithfulness"]["faithfulness_score"] for result in results)
        / len(results)
        if results
        else 0.0
    )
    return {"cases": results, "aggregate": aggregate}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).with_name("context_eval_cases.jsonl"),
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    report = evaluate_cases(load_cases(args.cases))
    print(json.dumps(report["aggregate"], indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
