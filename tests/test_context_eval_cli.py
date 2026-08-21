import json
from pathlib import Path

from tools.eval_context_selection import (
    benchmark_context_variants,
    evaluate_cases,
    load_cases,
)


def test_context_evaluation_fixture_produces_aggregate_metrics():
    cases_path = Path(__file__).parents[1] / "tools" / "context_eval_cases.jsonl"

    report = evaluate_cases(load_cases(cases_path))

    assert len(report["cases"]) == 2
    assert report["aggregate"]["selected_count"] > 0
    assert report["aggregate"]["budget_utilization"] > 0
    assert report["cases"][0]["metrics"]["duplicate_drop_rate"] > 0
    assert report["aggregate"]["source_coverage_rate"] >= 0.0
    assert report["aggregate"]["hint_coverage_rate"] >= 0.0
    for variant in ("full", "truncated", "compressed", "cached"):
        assert report["aggregate"][f"{variant}_average_ms"] >= 0.0


def test_context_evaluation_report_is_json_serializable():
    report = evaluate_cases([{
        "name": "serializable",
        "query": "test",
        "max_tokens": 30,
        "system_prompt_reserve": 0,
        "expected_sources": ["test.txt"],
        "expected_answer_hints": ["content"],
        "documents": [{"doc_id": "doc-1", "source": "test.txt", "content": "test content"}],
    }])

    json.dumps(report)
    assert report["aggregate"]["source_coverage_rate"] == 1.0
    assert report["aggregate"]["hint_coverage_rate"] == 1.0


def test_context_evaluation_reports_answer_quality_gaps():
    report = evaluate_cases([{
        "name": "missing-evidence",
        "query": "test",
        "max_tokens": 30,
        "system_prompt_reserve": 0,
        "expected_sources": ["required.txt"],
        "expected_answer_hints": ["missing term"],
        "documents": [{
            "doc_id": "doc-1",
            "source": "other.txt",
            "content": "unrelated content",
        }],
    }])

    quality = report["cases"][0]["answer_quality"]
    assert quality == {
        "source_coverage_rate": 0.0,
        "hint_coverage_rate": 0.0,
        "score": 0.0,
    }
    assert report["aggregate"]["answer_quality_score"] == 0.0


def test_context_evaluation_reports_answer_faithfulness():
    report = evaluate_cases([{
        "name": "grounded-answer",
        "query": "engine performance",
        "max_tokens": 30,
        "system_prompt_reserve": 0,
        "expected_answer": "Efficient cooling improves engine performance.",
        "documents": [{
            "doc_id": "doc-1",
            "source": "engine.txt",
            "content": "Efficient cooling improves engine performance.",
        }],
    }])

    quality = report["cases"][0]["answer_quality"]
    assert quality["faithfulness_score"] == 1.0
    assert report["aggregate"]["faithfulness_score"] == 1.0


def test_context_latency_benchmark_reports_all_variants():
    report = benchmark_context_variants({
        "full": lambda: "full context",
        "truncated": lambda: "short context",
        "compressed": lambda: "compressed",
        "cached": lambda: "cached context",
    }, repeats=2)

    assert set(report) == {"full", "truncated", "compressed", "cached"}
    for result in report.values():
        assert result["repeats"] == 2
        assert result["elapsed_ms"] >= 0
        assert result["token_count"] > 0
