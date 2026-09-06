"""The showcase FastAPI application factory.

``create_app`` builds the showcase runtime (offline by default, OpenAI when a key is
present), ingests the Oronts knowledge base once, and wires the REST surface and the
single-page UI onto the showcase's own FastAPI instance. This is not the refaat API
app: it wraps the showcase retrieval, program, and agent layers as HTTP endpoints and
serves a self-contained page with no auth and no external host.

The runtime is built and the corpus ingested when the app is created, so an in-process
TestClient exercises the whole app offline, and ``uvicorn ... --factory`` runs it as a
single self-contained image with SQLite and in-process vector search. The runtime is
closed on shutdown through the lifespan.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api import ShowcaseService, api_router
from .config import ShowcaseConfig, load_config
from .ingest import ingest_knowledge, open_project
from .runtime import build_runtime
from .ui import static_files, ui_router


def create_app(config: ShowcaseConfig | None = None) -> FastAPI:
    resolved = config or load_config()
    runtime = build_runtime(resolved)
    project = open_project(runtime)
    ingest_knowledge(project)
    service = ShowcaseService(
        project,
        mode=resolved.mode,
        online=resolved.online,
        generation_model=resolved.generation_model,
        embedding_model=resolved.embedding_model,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            runtime.close()

    app = FastAPI(title="Oronts Knowledge Terminal", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.service = service
    app.include_router(ui_router())
    app.include_router(api_router(service))
    app.mount("/static", static_files(), name="static")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "mode": resolved.mode}

    return app


__all__ = ["create_app"]
