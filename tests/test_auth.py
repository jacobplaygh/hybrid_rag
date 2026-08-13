import importlib
import os

from fastapi.testclient import TestClient


def test_protected_routes_require_api_key(monkeypatch):
    monkeypatch.setenv("AUTH_REQUIRED", "True")
    monkeypatch.setenv("API_KEY", "test-key")

    import api.config as config_module
    import api.main as main_module

    config_module.get_settings.cache_clear()
    main_module = importlib.reload(main_module)

    client = TestClient(main_module.app)

    unauthenticated = client.get("/api/documents/list")
    assert unauthenticated.status_code == 401

    authenticated = client.get(
        "/api/documents/list",
        headers={"X-API-Key": "test-key"},
    )
    assert authenticated.status_code == 200
