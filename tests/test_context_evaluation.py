from rag.context_evaluation import ContextSelectionEvaluator


def test_context_selection_metrics_summarize_report():
    report = {
        "selected": [
            {"id": "one", "source": "manual.txt", "truncated": False},
            {"id": "two", "source": "engine.txt", "truncated": True},
        ],
        "dropped": [
            {"id": "duplicate", "reason": "duplicate"},
            {"id": "limited", "reason": "source_limit"},
        ],
        "available_tokens": 100,
        "used_tokens": 75,
        "ordering": "score_descending",
    }

    metrics = ContextSelectionEvaluator.evaluate(report)

    assert metrics.selected_count == 2
    assert metrics.dropped_count == 2
    assert metrics.duplicate_drop_rate == 0.5
    assert metrics.source_diversity == 2
    assert metrics.budget_utilization == 0.75
    assert metrics.ordering == "score_descending"


def test_context_selection_metrics_handle_empty_report():
    metrics = ContextSelectionEvaluator.evaluate({})

    assert metrics.selected_count == 0
    assert metrics.dropped_count == 0
    assert metrics.duplicate_drop_rate == 0.0
    assert metrics.source_diversity == 0
    assert metrics.budget_utilization == 0.0
    assert metrics.ordering == "retrieval_order"
