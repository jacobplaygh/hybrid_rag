from rag.analytics import QueryAnalytics


def test_analyze_latency_correlates_pipeline_stages_and_cache_status(tmp_path):
    analytics = QueryAnalytics(str(tmp_path / "queries.jsonl"))
    analytics.log_query({
        "cache_status": "MISS",
        "query_time_ms": 120,
        "response_time_ms": 120,
        "retrieval_time_ms": 30,
        "first_token_time_ms": 80,
    })
    analytics.log_query({
        "cache_status": "HIT",
        "query_time_ms": 10,
        "response_time_ms": 10,
        "retrieval_time_ms": 0,
        "first_token_time_ms": 5,
    })

    report = analytics.analyze_latency()

    assert report["total_queries"] == 2
    assert report["average_query_time_ms"] == 65.0
    assert report["average_retrieval_time_ms"] == 15.0
    assert report["cache_breakdown"]["MISS"]["queries"] == 1
    assert report["cache_breakdown"]["MISS"]["average_first_token_time_ms"] == 80.0
    assert report["cache_breakdown"]["HIT"]["average_retrieval_time_ms"] == 0.0