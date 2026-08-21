from fastapi.testclient import TestClient

from api.main import app
from observability import metrics


def test_request_logging_records_basic_metadata():
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert len(app.state.request_log) >= 1
    assert any(entry["path"] == "/health" and entry["method"] == "GET" for entry in app.state.request_log)
    entry = next(entry for entry in reversed(app.state.request_log) if entry["path"] == "/health")
    assert "status_code" in entry


def test_stream_latency_metrics_are_registered():
    assert metrics.stream_first_chunk_latency is not None
    assert metrics.stream_first_chunk_latency._name == "rag_stream_first_chunk_duration_seconds"
    assert metrics.stream_duration is not None
    assert metrics.stream_duration._name == "rag_stream_duration_seconds"
