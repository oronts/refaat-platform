"""Serve the self-contained single-page UI.

The page and its assets are local files under ``examples/showcase/static``: no login,
no external CDN, no external font or script host, so it works fully offline and inside
one image. The index route returns the page and the static mount serves its CSS and JS.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

STATIC_ROOT = Path(__file__).resolve().parents[1] / "static"
INDEX_PATH = STATIC_ROOT / "index.html"


def ui_router() -> APIRouter:
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(INDEX_PATH.read_text(encoding="utf-8"))

    return router


def static_files() -> StaticFiles:
    return StaticFiles(directory=str(STATIC_ROOT))


__all__ = ["INDEX_PATH", "STATIC_ROOT", "static_files", "ui_router"]
