from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from refaat.api import app as app_module
from refaat.api.app import create_app
from refaat.api.middleware import RedisRateLimiter
from refaat.config import Settings


class TrackedRuntime:
    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


class TrackedLimiter:
    requests = 10
    window_seconds = 60

    def __init__(self) -> None:
        self.closed = 0

    def consume(self, key: str) -> bool:
        del key
        return True

    def close(self) -> None:
        self.closed += 1


def test_app_lifespan_closes_limiter_but_not_supplied_runtime(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = TrackedRuntime()
    limiter = TrackedLimiter()
    monkeypatch.setattr(app_module, "_create_limiter", lambda value: limiter)

    with TestClient(create_app(settings, runtime)) as client:  # type: ignore[arg-type]
        assert client.get("/health/live").status_code == 200

    assert limiter.closed == 1
    assert runtime.closed == 0


def test_app_construction_failure_closes_owned_runtime_and_limiter(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = TrackedRuntime()
    limiter = TrackedLimiter()

    class Builder:
        def __init__(self, value: Settings) -> None:
            assert value is settings

        def build(self) -> TrackedRuntime:
            return runtime

    monkeypatch.setattr(app_module, "RuntimeBuilder", Builder)
    monkeypatch.setattr(app_module, "_create_limiter", lambda value: limiter)
    monkeypatch.setattr(
        app_module,
        "configure_telemetry",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("telemetry failed")),
    )

    with pytest.raises(RuntimeError, match="telemetry failed"):
        create_app(settings)

    assert limiter.closed == 1
    assert runtime.closed == 1


def test_openapi_declares_api_key_security(settings: Settings) -> None:
    app = create_app(settings)
    with TestClient(app):
        schema = app.openapi()

    schemes = schema["components"]["securitySchemes"]
    assert schemes["bearerAuth"] == {"type": "http", "scheme": "bearer"}
    assert schemes["apiKeyAuth"] == {"type": "apiKey", "in": "header", "name": "X-API-Key"}
    assert {"bearerAuth": []} in schema["security"]
    assert {"apiKeyAuth": []} in schema["security"]

    protected = schema["paths"]["/v1/corpora/{corpus_id}/query"]["post"]
    assert "security" not in protected

    assert schema["paths"]["/v1/bootstrap/workspaces"]["post"]["security"] == []
    assert schema["paths"]["/health/live"]["get"]["security"] == []


def test_multi_worker_selects_the_shared_limiter(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    monkeypatch.setattr(app_module, "RedisRateLimiter", lambda *args, **kwargs: sentinel)
    multi = settings.model_copy(update={"api_workers": 2, "redis_url": "redis://example"})

    assert app_module._create_limiter(multi) is sentinel, "more than one worker selects the shared limiter"


def test_multi_worker_without_redis_is_rejected(settings: Settings) -> None:
    from refaat.errors import ConfigurationError

    multi = settings.model_copy(update={"api_workers": 2, "redis_url": None})

    with pytest.raises(ConfigurationError):
        app_module._create_limiter(multi)


def test_redis_rate_limiter_closes_owned_client(monkeypatch: pytest.MonkeyPatch) -> None:
    class Pipeline:
        def __enter__(self) -> Pipeline:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def incr(self, key: str) -> None:
            assert key.startswith("refaat:rate:")

        def expire(self, key: str, seconds: int) -> None:
            assert key.startswith("refaat:rate:")
            assert seconds == 62

        def execute(self) -> tuple[int, bool]:
            return 1, True

    class Client:
        def __init__(self) -> None:
            self.closed = False

        def pipeline(self) -> Pipeline:
            return Pipeline()

        def close(self) -> None:
            self.closed = True

    client = Client()

    class RedisFactory:
        @staticmethod
        def from_url(url: str, *, decode_responses: bool) -> Client:
            assert url == "redis://example"
            assert decode_responses
            return client

    fake_redis = SimpleNamespace(Redis=RedisFactory)
    monkeypatch.setitem(sys.modules, "redis", fake_redis)

    limiter = RedisRateLimiter("redis://example", requests=10, window_seconds=60)
    assert limiter.consume("principal")
    limiter.close()
    assert client.closed
