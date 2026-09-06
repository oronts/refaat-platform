from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from _shared import build_demo_runtime, emit
from showcase.app.ingest import _png_bytes

from refaat.domain.models import DocumentInput

CAPTION = "A red circle on a white background system architecture diagram"
IMAGE_QUERY = "red circle architecture diagram"
TEXT_QUERY = "What does a production release require"


def main() -> None:
    report: dict[str, object] = {"mode": "offline-deterministic"}
    image_bytes = _png_bytes(220, 20, 60)
    with (
        TemporaryDirectory(prefix="refaat-multimodal-") as directory,
        build_demo_runtime(Path(directory)) as runtime,
    ):
        project = runtime.project(workspace="oronts", corpus="assets")

        ingest = project.ingest_image(
            content=image_bytes,
            media_type="image/png",
            source_id="architecture.png",
            title="System architecture",
            caption=CAPTION,
        )
        report["image_document_ingested"] = ingest.changed and ingest.chunks == 1

        stored = runtime.store.list_chunks(project.workspace_id, project.corpus_id)
        image_chunk = stored[0]
        report["descriptor_text"] = image_chunk.text
        report["image_ref_present"] = bool(image_chunk.image_ref) and image_chunk.media_type == "image/png"

        answer = project.ask(IMAGE_QUERY)
        cited_ids = tuple(citation.chunk_id for citation in answer.citations)
        resolved = {chunk.id for chunk in runtime.store.get_chunks(project.workspace_id, project.corpus_id, cited_ids)}
        report["image_chunk_cited"] = bool(answer.citations) and all(cid == image_chunk.id for cid in cited_ids)
        report["citation_resolves_to_stored_chunk"] = bool(cited_ids) and set(cited_ids) <= resolved
        report["raw_image_not_cited"] = all(
            image_bytes.decode("latin-1") not in citation.quote for citation in answer.citations
        )

        # A second image ingested with no caption gets an honest deterministic
        # descriptor built from the bytes digest and the title, never a fabricated
        # description of pixels.
        no_caption = runtime.project(workspace="oronts", corpus="raw")
        no_caption.ingest_image(
            content=_png_bytes(30, 144, 255),
            media_type="image/png",
            source_id="capture.png",
            title="Raw capture",
        )
        raw_chunk = runtime.store.list_chunks(no_caption.workspace_id, no_caption.corpus_id)[0]
        report["deterministic_descriptor_ok"] = raw_chunk.text.startswith("image asset ") and (
            "Raw capture" in raw_chunk.text
        )

        # A text-only control corpus is unchanged: the image columns are NULL and a
        # repeated query returns the exact answer-cache hit.
        control = runtime.project(workspace="oronts", corpus="handbook")
        runtime.ingest(
            control.workspace_id,
            control.corpus_id,
            DocumentInput(
                source_id="handbook.txt",
                title="Handbook",
                media_type="text/plain",
                content=b"A production release requires a signed checklist and an on-call commander.",
            ),
        )
        control_chunks = runtime.store.list_chunks(control.workspace_id, control.corpus_id)
        report["text_only_columns_null"] = all(
            chunk.image_ref is None and chunk.media_type is None for chunk in control_chunks
        )
        first = control.ask(TEXT_QUERY)
        second = control.ask(TEXT_QUERY)
        report["text_only_control_grounded"] = bool(first.citations) and not first.cached
        report["text_only_cache_hit"] = second.cached and second.routes[0] == "cag:exact"

        report["features_exercised"] = [
            "image_ingest",
            "descriptor_chunk",
            "citation_contract",
            "deterministic_descriptor",
            "text_only_byte_identity",
        ]
    checks = [
        "image_document_ingested",
        "image_ref_present",
        "image_chunk_cited",
        "citation_resolves_to_stored_chunk",
        "raw_image_not_cited",
        "deterministic_descriptor_ok",
        "text_only_columns_null",
        "text_only_control_grounded",
        "text_only_cache_hit",
    ]
    if not all(report[name] for name in checks):
        raise SystemExit(f"Multimodal example assertion failed: {report}")
    emit(report)


if __name__ == "__main__":
    main()
