from fastapi.testclient import TestClient

from api.main import app


def test_request_logging_records_basic_metadata():
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert len(app.state.request_log) >= 1
    assert any(entry["path"] == "/health" and entry["method"] == "GET" for entry in app.state.request_log)
    entry = next(entry for entry in reversed(app.state.request_log) if entry["path"] == "/health")
    assert "status_code" in entry
