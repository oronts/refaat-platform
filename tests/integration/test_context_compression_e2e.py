from __future__ import annotations

from collections.abc import Iterator

import pytest

from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.config import Settings
from refaat.domain.models import DocumentInput

pytestmark = pytest.mark.integration

_BOILERPLATE = "All deployments must pass peer review before promotion to production."
_REDUNDANT_DOC = "\n\n".join(
    [
        "Rollback restores the prior artifact for the deployment platform whenever a "
        f"promotion fails inside the staged release window. {_BOILERPLATE}",
        "Caching keeps the grounded answer identity stable for the deployment platform "
        f"across repeated retrieval requests over time. {_BOILERPLATE}",
        "Retrieval fuses semantic and lexical candidates for the deployment platform "
        f"before the budgeter packs the final context. {_BOILERPLATE}",
    ]
)
_QUERY = "how does peer review gate promotion to production"


def _build(settings: Settings, tag: str = "main", **overrides: object) -> Refaat:
    base = settings.database_url.rsplit("/", 1)[0]
    updates: dict[str, object] = {
        "chunk_overlap_tokens": 0,
        "database_url": f"{base}/refaat-{tag}.db",
        **overrides,
    }
    return RuntimeBuilder(settings.model_copy(update=updates)).build()


def _ingest(runtime: Refaat, content: str = _REDUNDANT_DOC, *, title: str = "Handbook") -> tuple[str, str, str]:
    project = runtime.project(workspace="acme", corpus="handbook")
    result = runtime.ingest(
        project.workspace_id,
        project.corpus_id,
        DocumentInput(source_id="policy.md", title=title, media_type="text/markdown", content=content.encode()),
    )
    return project.workspace_id, project.corpus_id, result.document_id


@pytest.fixture
def compressed_runtime(settings: Settings) -> Iterator[Refaat]:
    runtime = _build(settings, compression_enabled=True)
    try:
        yield runtime
    finally:
        runtime.close()


def test_compress_route_present_and_citations_resolve_to_uncompressed_stored_text(
    compressed_runtime: Refaat,
) -> None:
    workspace_id, corpus_id, _ = _ingest(compressed_runtime)
    project = compressed_runtime.project(workspace="acme", corpus="handbook")

    context = project.retrieve(_QUERY, top_k=10)
    assert "compress" in context.routes

    response = project.ask(_QUERY, top_k=10)
    assert response.citations
    hit_ids = {hit.chunk.id for hit in context.hits}
    stored = {
        chunk.id: chunk.text
        for chunk in compressed_runtime.store.get_chunks(
            workspace_id, corpus_id, [citation.chunk_id for citation in response.citations]
        )
    }
    for citation in response.citations:
        assert citation.chunk_id in hit_ids
        assert citation.quote in stored[citation.chunk_id]


def test_metadata_carries_heading_and_context_strategy(compressed_runtime: Refaat) -> None:
    workspace_id, corpus_id, document_id = _ingest(compressed_runtime)
    chunks = compressed_runtime.store.list_document_chunks(workspace_id, corpus_id, document_id)
    assert chunks
    for chunk in chunks:
        assert "heading" in chunk.metadata
        assert chunk.metadata["context_strategy"] == "flat"


def test_compression_lowers_context_tokens_while_citations_still_resolve(settings: Settings) -> None:
    off = _build(settings, tag="off", compression_enabled=False)
    on = _build(settings, tag="on", compression_enabled=True)
    try:
        _ingest(off)
        _ingest(on)
        off_context = off.project(workspace="acme", corpus="handbook").retrieve(_QUERY, top_k=10)
        on_context = on.project(workspace="acme", corpus="handbook").retrieve(_QUERY, top_k=10)
        assert "compress" not in off_context.routes
        assert "compress" in on_context.routes
        assert on_context.token_count < off_context.token_count
        answer = on.project(workspace="acme", corpus="handbook").ask(_QUERY, top_k=10)
        assert answer.citations
    finally:
        off.close()
        on.close()


def test_contextual_retrieval_changes_embeddings_but_not_chunk_ids(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Pin the document id so both fresh stores derive identical chunk ids; the
    # chunk id depends only on (document_id, checksum, ordinal), never on the
    # embedding, so a toggled embedding strategy must not move it.
    from refaat.application import services as services_module

    real_new_id = services_module.new_id
    monkeypatch.setattr(
        services_module,
        "new_id",
        lambda prefix: "doc_pinned_contextual" if prefix == "doc" else real_new_id(prefix),
    )
    flat = _build(settings, tag="flat", contextual_retrieval=False)
    contextual = _build(settings, tag="contextual", contextual_retrieval=True)
    try:
        flat_ws, flat_corpus, flat_doc = _ingest(flat)
        ctx_ws, ctx_corpus, ctx_doc = _ingest(contextual)
        flat_chunks = sorted(
            flat.store.list_document_chunks(flat_ws, flat_corpus, flat_doc), key=lambda chunk: chunk.ordinal
        )
        ctx_chunks = sorted(
            contextual.store.list_document_chunks(ctx_ws, ctx_corpus, ctx_doc), key=lambda chunk: chunk.ordinal
        )
        assert [chunk.id for chunk in flat_chunks] == [chunk.id for chunk in ctx_chunks]
        assert flat_chunks
        for flat_chunk, ctx_chunk in zip(flat_chunks, ctx_chunks, strict=True):
            assert flat_chunk.text == ctx_chunk.text
            assert flat_chunk.embedding != ctx_chunk.embedding
        assert all(chunk.metadata["context_strategy"] == "contextual" for chunk in ctx_chunks)
    finally:
        flat.close()
        contextual.close()


def test_changing_compression_min_salience_via_harness_changes_cache_scope(compressed_runtime: Refaat) -> None:
    workspace_id, corpus_id, _ = _ingest(compressed_runtime)
    project = compressed_runtime.project(workspace="acme", corpus="handbook")

    first = project.ask(_QUERY, top_k=10)
    repeated = project.ask(_QUERY, top_k=10)
    assert not first.cached
    assert repeated.cached

    compressed_runtime.harness.create_snapshot(
        workspace_id=workspace_id,
        corpus_id=corpus_id,
        supplemental={"retrieval": {"compression_min_salience": 0.5}},
        evidence={"run_id": "eval_salience"},
        activate=True,
    )
    after = project.ask(_QUERY, top_k=10)
    assert not after.cached
