import json

from tools.ab_testing import load_dataset, run_ab_test


def test_load_dataset_reads_jsonl_records():
    dataset = load_dataset("tools/agentic_eval_queries.jsonl")

    assert len(dataset) >= 10
    assert dataset[0]["query"].startswith("What is the FastAPI backend")


def test_run_ab_test_compares_baseline_and_agentic_outcomes():
    def baseline_fn(query):
        return {"status": "success", "confidence": 0.62, "latency_ms": 150}

    def agentic_fn(query):
        return {"status": "success", "confidence": 0.81, "latency_ms": 220}

    rows = [
        {"query": "How does the backend configure the environment?", "category": "setup", "expected": ["config"]},
        {"query": "What is the retrieval architecture?", "category": "architecture", "expected": ["retrieval"]},
    ]

    summary = run_ab_test(rows, baseline_fn, agentic_fn)

    assert summary["total_cases"] == 2
    assert summary["mean_agentic_confidence"] > summary["mean_baseline_confidence"]
    assert summary["winner"] == "agentic"
    assert summary["rows"][0]["winner"] == "agentic"
