from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import select

from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.config import Settings
from refaat.domain.models import DocumentInput, GenerationRequest, GenerationResult
from refaat.errors import ValidationError
from refaat.infrastructure.database.schema import chunks as chunks_table

pytestmark = pytest.mark.integration

_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc``\x00\x00\x00"
    b"\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
)
_CAPTION = "A red circle on a white background architecture diagram"


class _CapturingVisionGenerator:
    """A stub vision generator: records the images it saw and cites the retrieved chunks.

    Stands in for a real multimodal model so the gate exercises the image plumbing
    without a network call. It grounds on the retrieved image-descriptor chunks by
    citing their real stored ids, so the citation contract is preserved.
    """

    model_id = "stub-vision-v1"

    def __init__(self) -> None:
        self.seen: list[tuple[str, ...]] = []
        self.extra_citation: str | None = None

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.seen.append(tuple(image.chunk_id for image in request.images))
        cited = [hit.chunk.id for hit in request.context.hits]
        if self.extra_citation is not None:
            cited = [self.extra_citation, *cited]
        answer = " ".join(f"looking at the image [chunk:{chunk_id}]" for chunk_id in cited)
        return GenerationResult(answer=answer, cited_chunk_ids=tuple(cited), usage={"provider": "stub-vision"})

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        yield self.generate(request).answer


def _multimodal_settings(settings: Settings) -> Settings:
    return settings.model_copy(update={"query_top_k": 6})


def _project(runtime: Refaat) -> object:
    return runtime.project(workspace="acme", corpus="assets")


def _stored_chunks(runtime: Refaat, workspace_id: str, corpus_id: str) -> list[object]:
    return runtime.store.list_chunks(workspace_id, corpus_id)


def test_image_ingests_as_citable_descriptor_chunk(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings).build()
    try:
        project = _project(runtime)
        result = project.ingest_image(  # type: ignore[attr-defined]
            content=_PNG,
            media_type="image/png",
            source_id="diagram.png",
            title="System diagram",
            caption=_CAPTION,
        )
        assert result.changed
        stored = _stored_chunks(runtime, project.workspace_id, project.corpus_id)  # type: ignore[attr-defined]
        assert len(stored) == 1
        chunk = stored[0]
        assert chunk.image_ref is not None
        assert chunk.media_type == "image/png"
        assert chunk.text == _CAPTION

        response = project.ask("red circle architecture diagram")  # type: ignore[attr-defined]
        assert response.citations
        resolved_ids = {
            resolved.id
            for resolved in runtime.store.get_chunks(
                project.workspace_id,  # type: ignore[attr-defined]
                project.corpus_id,  # type: ignore[attr-defined]
                tuple(citation.chunk_id for citation in response.citations),
            )
        }
        for citation in response.citations:
            assert citation.chunk_id == chunk.id
            assert citation.chunk_id in resolved_ids
            assert citation.quote in chunk.text
    finally:
        runtime.close()


def test_deterministic_descriptor_when_no_caption(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings).build()
    try:
        project = _project(runtime)
        project.ingest_image(  # type: ignore[attr-defined]
            content=_PNG,
            media_type="image/png",
            source_id="raw.png",
            title="Raw capture",
        )
        stored = _stored_chunks(runtime, project.workspace_id, project.corpus_id)  # type: ignore[attr-defined]
        assert len(stored) == 1
        assert stored[0].text.startswith("image asset ")
        assert "Raw capture" in stored[0].text
    finally:
        runtime.close()


def test_reingestion_is_idempotent(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings).build()
    try:
        project = _project(runtime)
        first = project.ingest_image(  # type: ignore[attr-defined]
            content=_PNG, media_type="image/png", source_id="d.png", title="D", caption=_CAPTION
        )
        again = project.ingest_image(  # type: ignore[attr-defined]
            content=_PNG, media_type="image/png", source_id="d.png", title="D", caption=_CAPTION
        )
        assert first.document_id == again.document_id
        assert again.changed is False
    finally:
        runtime.close()


def test_validation_rejects_bad_type_and_oversize(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings.model_copy(update={"image_max_bytes": 16})).build()
    try:
        project = _project(runtime)
        with pytest.raises(ValidationError):
            project.ingest_image(  # type: ignore[attr-defined]
                content=_PNG, media_type="application/pdf", source_id="x", title="x"
            )
        with pytest.raises(ValidationError):
            project.ingest_image(  # type: ignore[attr-defined]
                content=_PNG, media_type="image/png", source_id="x", title="x"
            )
    finally:
        runtime.close()


def test_vision_generator_receives_image_only_for_image_chunks(settings: Settings) -> None:
    generator = _CapturingVisionGenerator()
    runtime = RuntimeBuilder(_multimodal_settings(settings)).with_multimodal_generator(generator).build()
    try:
        project = _project(runtime)
        project.ingest_image(  # type: ignore[attr-defined]
            content=_PNG, media_type="image/png", source_id="diagram.png", title="Diagram", caption=_CAPTION
        )
        runtime.ingest(
            project.workspace_id,  # type: ignore[attr-defined]
            project.corpus_id,  # type: ignore[attr-defined]
            DocumentInput(
                source_id="notes.txt",
                title="Notes",
                media_type="text/plain",
                content=b"The architecture notes describe a red circle diagram in plain text form.",
            ),
        )
        all_chunks = _stored_chunks(runtime, project.workspace_id, project.corpus_id)  # type: ignore[attr-defined]
        stored = {chunk.text: chunk for chunk in all_chunks}
        image_chunk = stored[_CAPTION]
        response = project.ask("red circle architecture diagram")  # type: ignore[attr-defined]

        assert generator.seen, "the vision generator was not exercised"
        attached = generator.seen[-1]
        assert image_chunk.id in attached
        text_ids = {chunk.id for text, chunk in stored.items() if text != _CAPTION}
        assert not (set(attached) & text_ids)
        assert response.citations
        for citation in response.citations:
            assert citation.chunk_id in {chunk.id for chunk in stored.values()}
    finally:
        runtime.close()


def test_raw_image_and_fabricated_id_are_never_cited(settings: Settings) -> None:
    generator = _CapturingVisionGenerator()
    generator.extra_citation = "chunk_fabricated_not_stored"
    runtime = RuntimeBuilder(_multimodal_settings(settings)).with_multimodal_generator(generator).build()
    try:
        project = _project(runtime)
        project.ingest_image(  # type: ignore[attr-defined]
            content=_PNG, media_type="image/png", source_id="diagram.png", title="Diagram", caption=_CAPTION
        )
        response = project.ask("red circle architecture diagram")  # type: ignore[attr-defined]
        all_chunks = _stored_chunks(runtime, project.workspace_id, project.corpus_id)  # type: ignore[attr-defined]
        stored_ids = {chunk.id for chunk in all_chunks}
        assert response.citations
        for citation in response.citations:
            assert citation.chunk_id in stored_ids
            assert citation.chunk_id != "chunk_fabricated_not_stored"
            assert _PNG.decode("latin-1") not in citation.quote
    finally:
        runtime.close()


def test_text_only_corpus_is_byte_identical_with_multimodal_code_present(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings).build()
    try:
        project = _project(runtime)
        runtime.ingest(
            project.workspace_id,  # type: ignore[attr-defined]
            project.corpus_id,  # type: ignore[attr-defined]
            DocumentInput(
                source_id="handbook.txt",
                title="Handbook",
                media_type="text/plain",
                content=b"A production release requires a signed checklist and an on-call commander.",
            ),
        )
        with runtime.store.engine.connect() as conn:
            rows = conn.execute(select(chunks_table.c.image_ref, chunks_table.c.media_type)).all()
        assert rows
        assert all(row[0] is None and row[1] is None for row in rows)

        first = project.ask("What does a production release require?")  # type: ignore[attr-defined]
        second = project.ask("What does a production release require?")  # type: ignore[attr-defined]
        assert first.citations
        assert not first.cached
        assert second.cached
        assert second.routes[0] == "cag:exact"
    finally:
        runtime.close()
