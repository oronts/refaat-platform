"""Ingest the Oronts knowledge base into a corpus.

Ingestion runs through the configured runtime, so the graph extractor, HyPE, RAPTOR,
and contextual retrieval are all applied according to the active settings. The
operation is idempotent: a corpus already holding the full knowledge base is left
untouched, and re-ingesting a single document replaces it by source id.
"""

from __future__ import annotations

import struct
import zlib

from refaat import Refaat, RefaatProject
from refaat.domain.models import DocumentInput, IngestResult

from .knowledge import KnowledgeDocument, load_documents

WORKSPACE_SLUG = "oronts"
WORKSPACE_NAME = "Oronts GmbH"
CORPUS_NAME = "oronts-knowledge"

CAPTION = "A crimson badge on a white background marking an Oronts knowledge asset"
IMAGE_QUERY = "crimson badge Oronts knowledge asset"
IMAGE_SOURCE_ID = "asset-badge.png"
IMAGE_TITLE = "Oronts asset badge"


def _document_input(document: KnowledgeDocument) -> DocumentInput:
    return DocumentInput(
        source_id=document.source_id,
        title=document.title,
        media_type="text/markdown",
        content=document.text.encode("utf-8"),
        metadata=document.as_metadata(),
    )


def ingest_knowledge(project: RefaatProject, *, force: bool = False) -> int:
    documents = load_documents()
    if not force and project.stats().get("documents", 0) >= len(documents):
        return 0
    ingested = 0
    for document in documents:
        project.runtime.ingest(project.workspace_id, project.corpus_id, _document_input(document))
        ingested += 1
    return ingested


def open_project(runtime: Refaat) -> RefaatProject:
    return runtime.project(
        workspace=WORKSPACE_SLUG,
        workspace_name=WORKSPACE_NAME,
        corpus=CORPUS_NAME,
    )


def _png_bytes(red: int, green: int, blue: int) -> bytes:
    """A minimal valid 1x1 PNG generated in-process, deterministic per color.

    No external file and no image library, so the deterministic descriptor path is
    stable across runs.
    """

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    raw = b"\x00" + bytes((red, green, blue))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def ingest_image_asset(
    project: RefaatProject,
    *,
    source_id: str = IMAGE_SOURCE_ID,
    title: str = IMAGE_TITLE,
    caption: str = CAPTION,
) -> IngestResult:
    """Ingest a deterministic in-process image as one citable descriptor chunk.

    The raw bytes are stored as the chunk's image reference, never as retrievable
    text: only the descriptor built from the caption and title is embedded and
    citable, so a citation can never surface the raw image bytes.
    """
    return project.ingest_image(
        content=_png_bytes(220, 20, 60),
        media_type="image/png",
        source_id=source_id,
        title=title,
        caption=caption,
    )


__all__ = [
    "CAPTION",
    "CORPUS_NAME",
    "IMAGE_QUERY",
    "IMAGE_SOURCE_ID",
    "IMAGE_TITLE",
    "WORKSPACE_NAME",
    "WORKSPACE_SLUG",
    "_png_bytes",
    "ingest_image_asset",
    "ingest_knowledge",
    "open_project",
]
