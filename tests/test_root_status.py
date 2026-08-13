from fastapi.testclient import TestClient
from api.main import app, _request_history

def test_root_endpoint_reports_api_status():
    from api.config import get_settings
    settings = get_settings()
    original_rate_limit = settings.RATE_LIMIT_ENABLED
    settings.RATE_LIMIT_ENABLED = False
    try:
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        body = response.json()
        assert body["message"] == "Hybrid RAG API"
        assert body["status"] == "running"
        assert "version" in body
        assert "docs" in body
        assert "ui" in body
    finally:
        settings.RATE_LIMIT_ENABLED = original_rate_limit


def test_ui_endpoint_returns_html():
    from api.config import get_settings
    settings = get_settings()
    original_rate_limit = settings.RATE_LIMIT_ENABLED
    settings.RATE_LIMIT_ENABLED = False
    try:
        client = TestClient(app)
        response = client.get("/ui")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert "Hybrid RAG Web UI" in response.text
    finally:
        settings.RATE_LIMIT_ENABLED = original_rate_limit
