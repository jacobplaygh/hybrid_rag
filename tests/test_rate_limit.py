from fastapi.testclient import TestClient

from api.main import app


def test_rate_limit_returns_429_after_threshold(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "True")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "2")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "60")

    import importlib
    import api.config as config_module
    import api.main as main_module

    config_module.get_settings.cache_clear()
    main_module = importlib.reload(main_module)

    client = TestClient(main_module.app)

    first = client.get("/health")
    second = client.get("/health")
    third = client.get("/health")

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
