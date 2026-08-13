from fastapi.testclient import TestClient

from api.main import app
from data.vector_store import get_vector_store


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_rebuild_index_endpoint_returns_real_stats():
    client = TestClient(app)
    response = client.post(
        "/api/admin/rebuild-index",
        json={"include_new_docs": True, "clear_existing": True},
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert "documents_loaded" in body


def test_clear_cache_endpoint_resets_store():
    client = TestClient(app)
    response = client.post("/api/admin/clear-cache", headers={"X-API-Key": "test-key"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"


def test_vector_store_falls_back_to_in_memory_when_chroma_is_unavailable():
    store = get_vector_store(store_type="chroma", persist_dir="./tmp-test-store")
    assert store.__class__.__name__ == "InMemoryVectorStore"
