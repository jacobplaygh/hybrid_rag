from fastapi.testclient import TestClient

from api.main import app


def test_health_endpoint_reports_dependency_status():
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert "dependencies" in body
    assert "vector_store" in body["dependencies"]
    assert "nvidia_api" in body["dependencies"]
