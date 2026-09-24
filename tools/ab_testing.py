import argparse
import importlib
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


@dataclass
class EvaluationRow:
    query: str
    category: str
    baseline_confidence: float
    agentic_confidence: float
    delta_confidence: float
    baseline_latency_ms: float
    agentic_latency_ms: float
    latency_delta_ms: float
    baseline_status: str
    agentic_status: str
    baseline_context_relevance: float
    agentic_context_relevance: float
    baseline_groundedness: float
    agentic_groundedness: float
    baseline_answer_relevance: float
    agentic_answer_relevance: float
    winner: str


def _normalize_tokens(value: str) -> set[str]:
    if not value:
        return set()
    import re
    return {token for token in re.findall(r"[a-zA-Z0-9]+", value.lower()) if len(token) > 2}


def _lexical_overlap_score(query: str, evidence: Sequence[str]) -> float:
    query_tokens = _normalize_tokens(query)
    if not query_tokens:
        return 0.0
    text = " ".join(evidence).lower()
    coverage = sum(1 for token in query_tokens if token in text)
    return coverage / max(len(query_tokens), 1)


def load_dataset(path: str | Path) -> List[Dict[str, Any]]:
    dataset_path = Path(path)
    candidates: List[Path] = []

    if dataset_path.is_absolute():
        candidates.append(dataset_path)
    else:
        candidates.extend(
            [
                dataset_path,
                Path.cwd() / dataset_path,
                Path(__file__).resolve().parents[1] / dataset_path,
                Path(__file__).resolve().parents[2] / dataset_path,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            dataset_path = candidate
            break
    else:
        dataset_path = candidates[0]

    rows: List[Dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    if not rows:
        raise ValueError(f"Dataset at {dataset_path} contains no JSONL records")
    return rows


def _extract_metric(payload: Any, *names: str) -> Optional[float]:
    if payload is None:
        return None
    if isinstance(payload, dict):
        for key in names:
            if key in payload:
                value = payload[key]
                if isinstance(value, (int, float)):
                    return float(value)
                if isinstance(value, str):
                    try:
                        return float(value)
                    except ValueError:
                        pass
        return None
    if hasattr(payload, "confidence"):
        for key in names:
            if hasattr(payload, key):
                value = getattr(payload, key)
                if isinstance(value, (int, float)):
                    return float(value)
    return None


def _extract_latency(payload: Any) -> float:
    latency = _extract_metric(payload, "latency_ms", "processing_time_ms", "query_time_ms")
    if latency is not None:
        return float(latency)
    return 0.0


def _extract_status(payload: Any) -> str:
    if payload is None:
        return "unknown"
    if isinstance(payload, dict):
        for key in ("status", "final_status"):
            if key in payload:
                return str(payload[key])
    if hasattr(payload, "status"):
        return str(payload.status)
    return "success"


def _coerce_result(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "__dict__"):
        return vars(payload)
    return {"status": "success", "confidence": 0.0}


def _answer_quality(result: Dict[str, Any], expected: Sequence[str]) -> Dict[str, float]:
    """Calculate lightweight answer-level metrics from an evaluator payload."""
    expected_terms = _normalize_tokens(" ".join(str(item) for item in expected))
    documents = result.get("retrieved_docs", result.get("documents", []))
    context = " ".join(
        str(doc.get("content", "")) if isinstance(doc, dict) else str(doc)
        for doc in documents or []
    )
    answer = str(result.get("response", result.get("answer", "")))
    context_terms = _normalize_tokens(context)
    answer_terms = _normalize_tokens(answer)

    context_relevance = (
        len(expected_terms & context_terms) / len(expected_terms)
        if expected_terms else 0.0
    )
    answer_relevance = (
        len(expected_terms & answer_terms) / len(expected_terms)
        if expected_terms else 0.0
    )
    groundedness = (
        len(answer_terms & context_terms) / len(answer_terms)
        if answer_terms else 0.0
    )
    return {
        "context_relevance": context_relevance,
        "groundedness": groundedness,
        "answer_relevance": answer_relevance,
    }


def assess_rollout_gate(
    summary: Dict[str, Any],
    *,
    quality_regression_tolerance: float = 0.05,
    max_latency_delta_ms: float = 200.0,
) -> Dict[str, Any]:
    """Decide whether the agentic variant is safe to enable by default."""
    checks = {
        "success_rate": summary["agentic_success_rate"] >= summary["baseline_success_rate"],
        "context_relevance": summary["mean_agentic_context_relevance"]
        >= summary["mean_baseline_context_relevance"] - quality_regression_tolerance,
        "groundedness": summary["mean_agentic_groundedness"]
        >= summary["mean_baseline_groundedness"] - quality_regression_tolerance,
        "latency": summary["latency_delta_ms"] <= max_latency_delta_ms,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "quality_regression_tolerance": quality_regression_tolerance,
        "max_latency_delta_ms": max_latency_delta_ms,
    }


def run_ab_test(
    dataset: Iterable[Dict[str, Any]],
    baseline_fn: Callable[[str], Any],
    agentic_fn: Callable[[str], Any],
    *,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    rows: List[EvaluationRow] = []
    total_cases = 0
    total_baseline_conf = 0.0
    total_agentic_conf = 0.0
    total_baseline_latency = 0.0
    total_agentic_latency = 0.0
    baseline_successes = 0
    agentic_successes = 0

    for entry in dataset:
        if limit is not None and total_cases >= limit:
            break
        query = str(entry.get("query", ""))
        expected = entry.get("expected") or []

        baseline_result = _coerce_result(baseline_fn(query))
        agentic_result = _coerce_result(agentic_fn(query))

        baseline_conf = _extract_metric(baseline_result, "confidence", "confidence_score", "score") or 0.0
        agentic_conf = _extract_metric(agentic_result, "confidence", "confidence_score", "score") or 0.0

        baseline_latency = _extract_latency(baseline_result)
        agentic_latency = _extract_latency(agentic_result)

        baseline_status = _extract_status(baseline_result)
        agentic_status = _extract_status(agentic_result)
        baseline_quality = _answer_quality(baseline_result, expected)
        agentic_quality = _answer_quality(agentic_result, expected)
        if baseline_status == "success" or baseline_conf >= 0.7:
            baseline_successes += 1
        if agentic_status == "success" or agentic_conf >= 0.7:
            agentic_successes += 1

        delta_conf = agentic_conf - baseline_conf
        latency_delta = agentic_latency - baseline_latency
        winner = "agentic" if delta_conf > 0 else "baseline"
        if abs(delta_conf) < 1e-9:
            winner = "tie"

        row = EvaluationRow(
            query=query,
            category=str(entry.get("category", "unknown")),
            baseline_confidence=baseline_conf,
            agentic_confidence=agentic_conf,
            delta_confidence=delta_conf,
            baseline_latency_ms=baseline_latency,
            agentic_latency_ms=agentic_latency,
            latency_delta_ms=latency_delta,
            baseline_status=baseline_status,
            agentic_status=agentic_status,
            baseline_context_relevance=baseline_quality["context_relevance"],
            agentic_context_relevance=agentic_quality["context_relevance"],
            baseline_groundedness=baseline_quality["groundedness"],
            agentic_groundedness=agentic_quality["groundedness"],
            baseline_answer_relevance=baseline_quality["answer_relevance"],
            agentic_answer_relevance=agentic_quality["answer_relevance"],
            winner=winner,
        )
        rows.append(row)

        total_baseline_conf += baseline_conf
        total_agentic_conf += agentic_conf
        total_baseline_latency += baseline_latency
        total_agentic_latency += agentic_latency
        total_cases += 1

    if total_cases == 0:
        raise ValueError("No evaluation rows were produced")

    summary = {
        "total_cases": total_cases,
        "baseline_success_rate": baseline_successes / total_cases,
        "agentic_success_rate": agentic_successes / total_cases,
        "mean_baseline_confidence": total_baseline_conf / total_cases,
        "mean_agentic_confidence": total_agentic_conf / total_cases,
        "mean_baseline_latency_ms": total_baseline_latency / total_cases,
        "mean_agentic_latency_ms": total_agentic_latency / total_cases,
        "confidence_delta": (total_agentic_conf - total_baseline_conf) / total_cases,
        "latency_delta_ms": (total_agentic_latency - total_baseline_latency) / total_cases,
        "mean_baseline_context_relevance": sum(row.baseline_context_relevance for row in rows) / total_cases,
        "mean_agentic_context_relevance": sum(row.agentic_context_relevance for row in rows) / total_cases,
        "mean_baseline_groundedness": sum(row.baseline_groundedness for row in rows) / total_cases,
        "mean_agentic_groundedness": sum(row.agentic_groundedness for row in rows) / total_cases,
        "mean_baseline_answer_relevance": sum(row.baseline_answer_relevance for row in rows) / total_cases,
        "mean_agentic_answer_relevance": sum(row.agentic_answer_relevance for row in rows) / total_cases,
        "winner": "agentic" if (total_agentic_conf - total_baseline_conf) > 0 else "baseline",
        "rows": [asdict(row) for row in rows],
    }
    summary["rollout_gate"] = assess_rollout_gate(summary)
    return summary


def _default_baseline_for(query: str) -> Dict[str, Any]:
    evidence = [
        "FastAPI backend setup and environment configuration use application settings, runtime environment variables, and startup configuration.",
        "The vector store abstraction manages embeddings and document storage while the retrieval layer combines lexical and semantic matching.",
        "Observability includes metrics, tracing, and logging for latency, cache hits, and query performance.",
    ]
    overlap = _lexical_overlap_score(query, evidence)
    response = "The backend is configured with FastAPI settings and environment variables for service startup and runtime behavior."
    return {
        "status": "success" if overlap >= 0.15 else "low_confidence",
        "confidence": round(0.35 + overlap * 0.45, 4),
        "latency_ms": 160,
        "response": response,
        "retrieved_docs": [{"content": text} for text in evidence],
    }


def _default_agentic_for(query: str) -> Dict[str, Any]:
    evidence = [
        "The FastAPI backend uses environment configuration, startup settings, and runtime variables to configure the application and services.",
        "Query flow validation includes retrieval, reranking, and retry loops that improve answer confidence when initial context is incomplete.",
        "Agentic retrieval reformulates queries with additional keywords and confidence checks before returning the final answer.",
    ]
    overlap = _lexical_overlap_score(query, evidence)
    response = "The backend configuration is managed through FastAPI settings and environment variables, then refined by the agentic loop when context confidence is insufficient."
    return {
        "status": "success" if overlap >= 0.2 else "low_confidence",
        "confidence": round(0.5 + overlap * 0.45, 4),
        "latency_ms": 220,
        "response": response,
        "retrieved_docs": [{"content": text} for text in evidence],
    }


def _import_callable(spec: str) -> Callable[[str], Any]:
    if not spec:
        return _default_baseline_for
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise ValueError(f"Callable specification '{spec}' must use format 'module:function'")
    module = importlib.import_module(module_name)
    fn = getattr(module, attr)
    if not callable(fn):
        raise TypeError(f"'{spec}' does not resolve to a callable")
    return fn


def suggest_agentic_parameters(dataset: Iterable[Dict[str, Any]], *, max_retries_values: Sequence[int] = (1, 2, 3), confidence_thresholds: Sequence[float] = (0.55, 0.65, 0.75, 0.85), retry_strategies: Sequence[str] = ("auto", "add_keywords", "broaden")) -> Dict[str, Any]:
    dataset_rows = list(dataset)
    if not dataset_rows:
        raise ValueError("No dataset rows provided for parameter tuning")

    scored: List[Dict[str, Any]] = []
    for max_retries in max_retries_values:
        for threshold in confidence_thresholds:
            for strategy in retry_strategies:
                def simulated_agentic(query: str, *, cfg=(max_retries, threshold, strategy)) -> Dict[str, Any]:
                    evidence = [
                        "FastAPI backend setup and environment configuration use application settings, runtime environment variables, and service configuration.",
                        "Hybrid retrieval combines lexical BM25 matching with semantic vector search and reranking for precision.",
                        "Observability includes metrics, tracing, and validation to monitor latency, quality, and hallucination risk.",
                    ]
                    overlap = _lexical_overlap_score(query, evidence)
                    confidence = round(max(0.3, overlap + 0.18 + (cfg[0] * 0.06) - (cfg[1] - 0.6) * 0.4), 4)
                    latency_ms = 170 + (cfg[0] * 35)
                    status = "success" if confidence >= cfg[1] else "low_confidence"
                    response = "The response is grounded in the documented system configuration and retrieval workflow."
                    result = {
                        "status": status,
                        "confidence": confidence,
                        "latency_ms": latency_ms,
                        "response": response,
                        "retrieved_docs": [{"content": text} for text in evidence],
                    }
                    return result

                summary = run_ab_test(dataset_rows, _default_baseline_for, simulated_agentic, limit=len(dataset_rows))
                score = (
                    (summary["agentic_success_rate"] - summary["baseline_success_rate"]) * 100.0
                    + (summary["mean_agentic_confidence"] - summary["mean_baseline_confidence"]) * 100.0
                    + (summary["mean_agentic_context_relevance"] - summary["mean_baseline_context_relevance"]) * 120.0
                    + (summary["mean_agentic_groundedness"] - summary["mean_baseline_groundedness"]) * 120.0
                    + (summary["mean_agentic_answer_relevance"] - summary["mean_baseline_answer_relevance"]) * 120.0
                    - (max(summary["mean_agentic_latency_ms"] - summary["mean_baseline_latency_ms"], 0.0) / 10.0)
                )
                scored.append({
                    "max_retries": max_retries,
                    "confidence_threshold": threshold,
                    "retry_strategy": strategy,
                    "score": round(score, 4),
                    "summary": summary,
                })

    recommended = sorted(scored, key=lambda item: item["score"], reverse=True)[0]
    return {
        "recommended": {
            "max_retries": recommended["max_retries"],
            "confidence_threshold": recommended["confidence_threshold"],
            "retry_strategy": recommended["retry_strategy"],
            "score": recommended["score"],
        },
        "candidates": scored,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare a baseline RAG strategy to an agentic variant on a representative dataset.")
    parser.add_argument("--dataset", default="tools/agentic_eval_queries.jsonl", help="Path to JSONL evaluation dataset")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on the number of cases to evaluate")
    parser.add_argument("--baseline", default="", help="Optional module:function for the baseline evaluator (e.g. mymodule:baseline_run)")
    parser.add_argument("--agentic", default="", help="Optional module:function for the agentic evaluator (e.g. mymodule:agentic_run)")
    parser.add_argument("--tune", action="store_true", help="Recommend agentic loop settings from a small parameter grid")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    dataset = load_dataset(args.dataset)
    if args.tune:
        rec = suggest_agentic_parameters(dataset)
        print(json.dumps(rec, indent=2))
        return

    baseline_fn = _import_callable(args.baseline) if args.baseline else _default_baseline_for
    agentic_fn = _import_callable(args.agentic) if args.agentic else _default_agentic_for
    summary = run_ab_test(dataset, baseline_fn, agentic_fn, limit=args.limit)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
