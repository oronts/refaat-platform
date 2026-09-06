from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Protocol

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import ORJSONResponse

from refaat.api.middleware import (
    InMemoryRateLimiter,
    MaxBodySizeMiddleware,
    RateLimitMiddleware,
    RedisRateLimiter,
    RequestContextMiddleware,
)
from refaat.api.routes import router
from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.config import Settings, get_settings
from refaat.errors import ConfigurationError, RefaatError, ToolTimeoutError
from refaat.observability import configure_telemetry, current_trace_id, get_logger
from refaat.version import __version__

logger = get_logger(__name__)


class RateLimiter(Protocol):
    def consume(self, key: str) -> bool: ...

    def close(self) -> None: ...


_OPEN_PATHS = frozenset({"/health/live", "/health/ready", "/v1/bootstrap/workspaces"})


def _install_security_scheme(app: FastAPI) -> None:
    """Advertise the API-key auth that every non-open route enforces.

    The auth dependency accepts the raw key either as an ``Authorization: Bearer``
    credential or an ``X-API-Key`` header, so both schemes are declared and the
    default requirement admits either. The unauthenticated health and bootstrap
    routes carry an empty requirement so the document does not claim they need a key.
    """

    def custom_openapi() -> dict[str, object]:
        if app.openapi_schema:
            return app.openapi_schema
        document = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        components = document.setdefault("components", {})
        assert isinstance(components, dict)
        components["securitySchemes"] = {
            "bearerAuth": {"type": "http", "scheme": "bearer"},
            "apiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-API-Key"},
        }
        document["security"] = [{"bearerAuth": []}, {"apiKeyAuth": []}]
        paths = document.get("paths", {})
        assert isinstance(paths, dict)
        for path in _OPEN_PATHS:
            operations = paths.get(path)
            if not isinstance(operations, dict):
                continue
            for operation in operations.values():
                if isinstance(operation, dict):
                    operation["security"] = []
        app.openapi_schema = document
        return document

    app.openapi = custom_openapi  # type: ignore[method-assign]


def _create_limiter(settings: Settings) -> RateLimiter:
    # A multi-worker deployment must use the shared backend: the in-memory bucket is
    # per process, so N workers would each grant the full allowance. When more than
    # one worker is declared the shared limiter is required, not merely preferred.
    if settings.api_workers > 1 and not settings.redis_url:
        raise ConfigurationError(
            "REFAAT_API_WORKERS greater than 1 requires REFAAT_REDIS_URL: the in-memory rate limiter is per "
            "process and cannot enforce a shared limit across workers"
        )
    if settings.redis_url:
        return RedisRateLimiter(
            settings.redis_url,
            settings.rate_limit_requests,
            settings.rate_limit_window_seconds,
        )
    return InMemoryRateLimiter(
        settings.rate_limit_requests,
        settings.rate_limit_window_seconds,
        max_buckets=settings.rate_limit_max_buckets,
    )


def create_app(
    settings: Settings | None = None,
    runtime: Refaat | None = None,
    *,
    legal_hold_management: bool = False,
) -> FastAPI:
    actual_settings = settings or get_settings()
    owned_runtime = runtime is None
    if runtime is not None:
        # An injected runtime already carries its own management wiring.
        actual_runtime = runtime
    else:
        builder = RuntimeBuilder(actual_settings)
        if legal_hold_management:
            builder = builder.with_legal_hold_registry()
        actual_runtime = builder.build()
    limiter: RateLimiter | None = None

    try:
        active_limiter = _create_limiter(actual_settings)
        limiter = active_limiter

        @asynccontextmanager
        async def lifespan(_: FastAPI) -> AsyncIterator[None]:
            try:
                yield
            finally:
                try:
                    active_limiter.close()
                finally:
                    if owned_runtime:
                        actual_runtime.close()

        app = FastAPI(
            title="Refaat",
            version=__version__,
            description=("Adaptive evidence and context runtime for RAG, CAG, KAG, GraphRAG, and agents."),
            default_response_class=ORJSONResponse,
            lifespan=lifespan,
            docs_url="/docs" if actual_settings.environment != "production" else None,
            redoc_url=None,
        )
        app.state.runtime = actual_runtime
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=actual_settings.trusted_hosts,
        )
        if actual_settings.cors_origins:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=actual_settings.cors_origins,
                allow_credentials=False,
                allow_methods=["GET", "POST", "DELETE"],
                allow_headers=[
                    "Authorization",
                    "Content-Type",
                    "X-API-Key",
                    "X-Request-ID",
                ],
            )
        app.add_middleware(
            RateLimitMiddleware,
            limiter=active_limiter,
            trusted_proxies=actual_settings.rate_limit_trusted_proxies,
        )
        app.add_middleware(
            MaxBodySizeMiddleware,
            max_body_bytes=actual_settings.max_body_bytes,
        )
        app.add_middleware(RequestContextMiddleware)
        app.include_router(router)
        _install_security_scheme(app)

        @app.exception_handler(ToolTimeoutError)
        async def handle_tool_timeout(
            request: Request,
            exc: ToolTimeoutError,
        ) -> ORJSONResponse:
            del request
            # First-class fields so a client can never mistake the 409 for a safe gateway-timeout retry:
            # an abandoned write-capable tool was not terminated, so its outcome is unknown, not nothing.
            return ORJSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {
                        "code": exc.code,
                        "message": str(exc),
                        "outcome": exc.outcome,
                        "side_effect": exc.side_effect,
                        "execution_terminated": exc.execution_terminated,
                        "retryable": False,
                        "trace_id": current_trace_id(),
                    }
                },
            )

        @app.exception_handler(RefaatError)
        async def handle_refaat_error(
            request: Request,
            exc: RefaatError,
        ) -> ORJSONResponse:
            del request
            return ORJSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {
                        "code": exc.code,
                        "message": str(exc),
                        "trace_id": current_trace_id(),
                    }
                },
            )

        @app.exception_handler(RequestValidationError)
        async def handle_validation_error(
            request: Request,
            exc: RequestValidationError,
        ) -> ORJSONResponse:
            del request
            return ORJSONResponse(
                status_code=422,
                content={
                    "error": {
                        "code": "request_validation_error",
                        "message": "Request validation failed",
                        "details": exc.errors(),
                        "trace_id": current_trace_id(),
                    }
                },
            )

        @app.exception_handler(Exception)
        async def handle_unexpected_error(
            request: Request,
            exc: Exception,
        ) -> ORJSONResponse:
            route = getattr(request.scope.get("route"), "path", None)
            logger.exception(
                "unhandled_request_error",
                route=route or "unmatched",
                exc_info=exc,
            )
            return ORJSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": "internal_error",
                        "message": "An internal error occurred",
                        "trace_id": current_trace_id(),
                    }
                },
            )

        configure_telemetry(actual_settings, app=app)
        return app
    except Exception:
        if limiter is not None:
            limiter.close()
        if owned_runtime:
            actual_runtime.close()
        raise


def run() -> None:
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=8000, log_level="info")
