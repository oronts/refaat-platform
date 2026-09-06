from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import func, select

from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.config import Settings
from refaat.domain.models import DocumentInput, QueryRequest
from refaat.infrastructure.database.schema import chunk_hypotheticals
from refaat.infrastructure.hype import DeterministicHypotheticalGenerator, ModelHypotheticalGenerator

_CONTENT = (
    b"Photosynthesis converts sunlight into chemical energy inside plant cells. "
    b"Chlorophyll absorbs light in the leaves and drives the reaction."
)
_QUERY = "What is photosynthesis and how does it work"


def _hype_settings(settings: Settings) -> Settings:
    return settings.model_copy(update={"hype_enabled": True})


def _count_hypotheticals(runtime: Refaat) -> int:
    with runtime.store.engine.connect() as conn:
        return int(conn.execute(select(func.count()).select_from(chunk_hypotheticals)).scalar_one())


def _hypothetical_rows(runtime: Refaat) -> list[dict[str, object]]:
    with runtime.store.engine.connect() as conn:
        return [dict(row) for row in conn.execute(select(chunk_hypotheticals)).mappings()]


def _ingest(runtime: Refaat, content: bytes = _CONTENT) -> tuple[str, str]:
    project = runtime.project(workspace="acme", corpus="handbook")
    runtime.ingest(
        project.workspace_id,
        project.corpus_id,
        DocumentInput(source_id="bio.txt", title="Biology", media_type="text/plain", content=content),
    )
    return project.workspace_id, project.corpus_id


def test_deterministic_generator_is_stable_and_content_derived() -> None:
    generator = DeterministicHypotheticalGenerator()
    first = generator.questions("Photosynthesis converts sunlight into energy")
    second = generator.questions("Photosynthesis converts sunlight into energy")
    assert first == second
    assert first, "expected at least one templated question"
    assert all("photosynthesis" in question.casefold() for question in first)
    assert generator.questions("   ") == []


def test_disabled_path_writes_no_rows_and_keeps_identity(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings).build()
    try:
        workspace_id, corpus_id = _ingest(runtime)
        assert _count_hypotheticals(runtime) == 0
        assert runtime.store.list_hypotheticals(workspace_id, corpus_id) == []
        assert runtime.queries.retriever.hype_enabled is False
        assert runtime.queries.retriever.hype_cache_id is None
        answer = runtime.project(workspace="acme", corpus="handbook")  # existing project
        response = answer.ask(_QUERY)
        assert "hype" not in response.routes
    finally:
        runtime.close()


def test_enabled_ingestion_writes_deterministic_embedded_rows(settings: Settings) -> None:
    runtime = RuntimeBuilder(_hype_settings(settings)).build()
    try:
        _ingest(runtime)
        rows = _hypothetical_rows(runtime)
        assert rows, "expected hypothetical rows when hype is enabled"
        for row in rows:
            expected = list(runtime.ingestion.embedder.embed([str(row["question"])])[0])
            stored = [float(value) for value in row["embedding"]]  # type: ignore[union-attr]
            assert stored == pytest.approx(expected)
            assert str(row["embedding_model"]) == runtime.ingestion.embedder.model_id
    finally:
        runtime.close()


def test_question_match_maps_back_to_real_chunk(settings: Settings) -> None:
    runtime = RuntimeBuilder(_hype_settings(settings)).build()
    try:
        workspace_id, corpus_id = _ingest(runtime)
        stored_ids = {chunk.id for chunk in runtime.store.list_chunks(workspace_id, corpus_id)}
        assert stored_ids
        retriever = runtime.queries.retriever
        request = QueryRequest(workspace_id=workspace_id, corpus_id=corpus_id, query=_QUERY)
        embedding = retriever.embedder.embed([_QUERY])[0]  # type: ignore[union-attr]
        policy = retriever.resolve_policy(request, {})
        hits = retriever._hypothetical_hits(request, embedding, policy)
        assert hits, "expected a hypothetical match for a question-shaped query"
        for hit in hits:
            assert hit.chunk.id in stored_ids
            assert hit.reasons == ("hypothetical",)
    finally:
        runtime.close()


def test_citation_never_comes_from_hypothetical_text(settings: Settings) -> None:
    runtime = RuntimeBuilder(_hype_settings(settings)).build()
    try:
        workspace_id, corpus_id = _ingest(runtime)
        stored = {chunk.id: chunk for chunk in runtime.store.list_chunks(workspace_id, corpus_id)}
        questions = {str(row["question"]) for row in _hypothetical_rows(runtime)}
        project = runtime.project(workspace="acme", corpus="handbook")
        response = project.ask(_QUERY)
        assert "hype" in response.routes
        assert response.citations
        for citation in response.citations:
            assert citation.chunk_id in stored
            assert citation.quote in stored[citation.chunk_id].text
            assert citation.quote not in questions
    finally:
        runtime.close()


def test_reingestion_is_idempotent(settings: Settings) -> None:
    runtime = RuntimeBuilder(_hype_settings(settings)).build()
    try:
        _ingest(runtime)
        first = {str(row["id"]) for row in _hypothetical_rows(runtime)}
        first_count = _count_hypotheticals(runtime)
        _ingest(runtime, content=_CONTENT + b" ")  # change checksum to force re-index
        _ingest(runtime)  # back to the original content
        again = {str(row["id"]) for row in _hypothetical_rows(runtime)}
        assert _count_hypotheticals(runtime) == first_count
        assert again == first
    finally:
        runtime.close()


_DIFF_QUERY = "how does photosynthesis work"


def _diff_corpus() -> list[DocumentInput]:
    return [
        DocumentInput(
            source_id="target.txt",
            title="Target",
            media_type="text/plain",
            content=b"Photosynthesis sustains plant life across the biosphere.",
        ),
        DocumentInput(
            source_id="d1.txt",
            title="D1",
            media_type="text/plain",
            content=b"How does a gearbox work while machines perform repetitive tasks.",
        ),
        DocumentInput(
            source_id="d2.txt",
            title="D2",
            media_type="text/plain",
            content=b"Engineers describe how turbines work when systems perform under load.",
        ),
        DocumentInput(
            source_id="d3.txt",
            title="D3",
            media_type="text/plain",
            content=b"Photographers show how cameras work as light performs on film.",
        ),
    ]


def _diff_settings(settings: Settings, tmp_path: Path, name: str, *, hype: bool) -> Settings:
    return settings.model_copy(
        update={
            "database_url": f"sqlite:///{tmp_path / f'{name}.db'}",
            "blob_root": tmp_path / f"{name}-blobs",
            "connector_file_root": tmp_path / f"{name}-connectors",
            "skill_root": tmp_path / f"{name}-skills",
            "hype_enabled": hype,
        }
    )


def _ingest_diff(runtime: Refaat) -> tuple[str, str, str]:
    project = runtime.project(workspace="acme", corpus="handbook")
    for doc in _diff_corpus():
        runtime.ingest(project.workspace_id, project.corpus_id, doc)
    target_id = next(
        chunk.id
        for chunk in runtime.store.list_chunks(project.workspace_id, project.corpus_id)
        if "sustains plant life" in chunk.text
    )
    return project.workspace_id, project.corpus_id, target_id


def test_hype_rescues_a_chunk_plain_retrieval_ranks_out_of_top_k(settings: Settings, tmp_path: Path) -> None:
    """Same corpus and query, hype off vs on: the target chunk that a question-shaped

    query barely matches on the semantic and lexical path is absent from top_k
    without HyPE and rescued into it with HyPE, because its precomputed
    hypothetical question embeds near the query. This is the recall lift the
    technique promises, so a regression to zero ranking contribution fails here.
    """
    plain = RuntimeBuilder(_diff_settings(settings, tmp_path, "plain", hype=False)).build()
    hyped = RuntimeBuilder(_diff_settings(settings, tmp_path, "hype", hype=True)).build()
    try:
        w_plain, c_plain, plain_target = _ingest_diff(plain)
        plain_packet = plain.retrieve(QueryRequest(workspace_id=w_plain, corpus_id=c_plain, query=_DIFF_QUERY, top_k=1))
        plain_ids = {hit.chunk.id for hit in plain_packet.hits}
        assert plain_target not in plain_ids
        assert "hype" not in plain_packet.routes

        w_hype, c_hype, hype_target = _ingest_diff(hyped)
        hype_packet = hyped.retrieve(QueryRequest(workspace_id=w_hype, corpus_id=c_hype, query=_DIFF_QUERY, top_k=1))
        hype_ids = {hit.chunk.id for hit in hype_packet.hits}
        assert hype_target in hype_ids
        assert "hype" in hype_packet.routes
    finally:
        plain.close()
        hyped.close()


class _StubGenerator:
    model_id = "stub-generator-v1"

    def __init__(self, answer: str) -> None:
        self._answer = answer

    def generate(self, request: object) -> object:
        from refaat.domain.models import GenerationResult

        return GenerationResult(answer=self._answer, cited_chunk_ids=())

    def stream(self, request: object) -> object:  # pragma: no cover - not exercised
        raise NotImplementedError


def test_model_generator_parses_lines_and_falls_back() -> None:
    fallback = DeterministicHypotheticalGenerator()
    model = ModelHypotheticalGenerator(_StubGenerator("What is X?\n1. How is X used?\n\n- What is X?"), fallback)
    questions = model.questions("Some chunk about a subject")
    assert questions == ["What is X?", "How is X used?"]
    blank = ModelHypotheticalGenerator(_StubGenerator("   "), fallback)
    assert blank.questions("Photosynthesis converts sunlight") == fallback.questions("Photosynthesis converts sunlight")
