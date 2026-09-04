import json

from tools.ab_testing import load_dataset, run_ab_test, suggest_agentic_parameters


def test_load_dataset_reads_jsonl_records():
    dataset = load_dataset("tools/agentic_eval_queries.jsonl")

    assert len(dataset) >= 10
    assert dataset[0]["query"].startswith("What is the FastAPI backend")


def test_run_ab_test_compares_baseline_and_agentic_outcomes():
    def baseline_fn(query):
        return {
            "status": "success",
            "confidence": 0.62,
            "latency_ms": 150,
            "response": "The answer mentions config.",
            "retrieved_docs": [{"content": "config details"}],
        }

    def agentic_fn(query):
        return {
            "status": "success",
            "confidence": 0.81,
            "latency_ms": 220,
            "response": "config details",
            "retrieved_docs": [{"content": "config details"}],
        }

    rows = [
        {"query": "How does the backend configure the environment?", "category": "setup", "expected": ["config"]},
        {"query": "What is the retrieval architecture?", "category": "architecture", "expected": ["retrieval"]},
    ]

    summary = run_ab_test(rows, baseline_fn, agentic_fn)

    assert summary["total_cases"] == 2
    assert summary["mean_agentic_confidence"] > summary["mean_baseline_confidence"]
    assert summary["winner"] == "agentic"
    assert summary["rows"][0]["winner"] == "agentic"
    assert summary["rows"][0]["agentic_groundedness"] == 1.0
    assert summary["rows"][0]["agentic_answer_relevance"] == 1.0


def test_suggest_agentic_parameters_returns_recommended_settings():
    dataset = [
        {"query": "How does the backend configure the environment?", "category": "setup", "expected": ["config"]},
        {"query": "What is the retrieval architecture?", "category": "architecture", "expected": ["retrieval"]},
        {"query": "Explain the API and query flow.", "category": "flow", "expected": ["api", "query"]},
    ]

    recommendation = suggest_agentic_parameters(dataset)

    assert recommendation["recommended"]["max_retries"] in {1, 2, 3}
    assert recommendation["recommended"]["confidence_threshold"] >= 0.55
    assert recommendation["recommended"]["retry_strategy"] in {"auto", "add_keywords", "broaden"}
    assert recommendation["recommended"]["score"]
