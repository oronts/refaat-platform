"""Complete deletion (right to erasure): every auto-derived surface is removed or re-derived.

The whole feature is additive. The ordinary IngestionService.delete and the pinned grounded-rag
artifact id are asserted unchanged, so byte-identity holds where erasure is not used.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from sqlalchemy import func, select

from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.config import Settings
from refaat.domain.audit import AuditEvent
from refaat.domain.erasure import ErasureStatus
from refaat.domain.models import DocumentInput, Feedback, QueryRequest
from refaat.errors import NotFoundError
from refaat.infrastructure.database.schema import erasure_ledger, model_call_cache, outbox, query_cache
from refaat.infrastructure.embeddings import DeterministicEmbedder
from refaat.infrastructure.model_cache import cache_identity_of, embedding_cache_key
from refaat.programs.program import GroundedRagProgram

pytestmark = pytest.mark.integration

MARKER = "ZEBRAERASUREMARK"
PINNED_GROUNDED_RAG_ID = "sha256:b76d73f8d956bcd38ccc9d4c24372b07b387647a189f3a9e300cded186febbb4"


class _CacheIdentifiedEmbedder:
    """A real, cacheable embedder that produces deterministic vectors.

    It is not a DeterministicEmbedder subclass, so the runtime wraps it in the caching embedder
    and genuinely warms the embedding model-call cache during ingest.
    """

    def __init__(self) -> None:
        self._inner = DeterministicEmbedder()

    @property
    def model_id(self) -> str:
        return "erasure-test-embed"

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    @property
    def cache_identity(self) -> str:
        return "erasure-test-embed-v1"

    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return self._inner.embed(texts)


class _RecordingSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        self.events.append(event)


class _RaisingSink:
    def record(self, event: AuditEvent) -> None:
        del event
        raise RuntimeError("audit sink down")


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path / 'refaat.db'}",
        auto_create_schema=True,
        blob_root=tmp_path / "blobs",
        connector_file_root=tmp_path / "connectors",
        skill_root=tmp_path / "skills",
        api_key_pepper="test-pepper-with-enough-entropy",
        bootstrap_token="bootstrap-test-token",
        default_chunk_tokens=48,
        chunk_overlap_tokens=8,
        query_token_budget=600,
        raptor_enabled=True,
        hype_enabled=True,
        model_call_cache_enabled=True,
        log_json=False,
    )


_OTHER_ONE = "Quarterly logistics report covers warehouse throughput and carrier routing options."
_OTHER_TWO = "The onboarding handbook explains payroll setup, benefits enrolment, and equipment requests."
_TARGET = (
    f"The {MARKER} incident postmortem records that the payment gateway timeout cascaded into a "
    f"checkout outage, and the {MARKER} rollback restored service after the on-call engineer paged the team."
)


def _ingest(runtime: Refaat, ws: str, corpus: str) -> dict[str, str]:
    docs = {"other1": _OTHER_ONE, "other2": _OTHER_TWO, "target": _TARGET}
    ids: dict[str, str] = {}
    for key, body in docs.items():
        result = runtime.ingest(
            ws,
            corpus,
            DocumentInput(source_id=key, title=key, media_type="text/plain", content=body.encode()),
        )
        ids[key] = result.document_id
    return ids


def _query_cache_count(runtime: Refaat, ws: str, corpus: str) -> int:
    with runtime.store.engine.connect() as conn:
        return int(
            conn.execute(
                select(func.count())
                .select_from(query_cache)
                .where(query_cache.c.workspace_id == ws, query_cache.c.corpus_id == corpus)
            ).scalar_one()
        )


def _generation_cache_count(runtime: Refaat, ws: str) -> int:
    with runtime.store.engine.connect() as conn:
        return int(
            conn.execute(
                select(func.count())
                .select_from(model_call_cache)
                .where(model_call_cache.c.kind == "generation", model_call_cache.c.workspace_id == ws)
            ).scalar_one()
        )


@pytest.fixture
def wired(tmp_path: Path) -> Iterator[tuple[Refaat, _RecordingSink, _CacheIdentifiedEmbedder]]:
    sink = _RecordingSink()
    embedder = _CacheIdentifiedEmbedder()
    runtime = RuntimeBuilder(_settings(tmp_path)).with_embedder(embedder).with_audit_sink(sink).build()
    try:
        yield runtime, sink, embedder
    finally:
        runtime.close()


def test_erasure_removes_every_auto_derived_surface(
    wired: tuple[Refaat, _RecordingSink, _CacheIdentifiedEmbedder],
) -> None:
    runtime, sink, embedder = wired
    ws = runtime.create_workspace("acme", "Acme")
    corpus = runtime.create_corpus(ws, "handbook")
    ids = _ingest(runtime, ws, corpus)
    target = ids["target"]

    target_chunks = runtime.store.list_document_chunks(ws, corpus, target)
    assert target_chunks
    chunk_texts = [chunk.text for chunk in target_chunks]
    target_chunk_id = target_chunks[0].id
    blob_uri = str(runtime.store.get_document(ws, corpus, target)["blob_uri"])

    # A cached answer for the corpus.
    runtime.queries.query(QueryRequest(workspace_id=ws, corpus_id=corpus, query=f"{MARKER} rollback outage"))
    assert _query_cache_count(runtime, ws, corpus) >= 1

    # A learned preference keyed to the target document.
    runtime.submit_feedback(
        ws, corpus, Feedback(query_id=None, user_id="analyst", rating=1, selected_chunk_ids=(target_chunk_id,))
    )
    assert runtime.store.get_preferences(ws, corpus, "user:analyst")

    # A session with a summary and a turn that cites the target chunk, plus a non-citing turn.
    session = runtime.store.create_session(ws, corpus, user_id="analyst", goal=None)
    citing_turn = runtime.store.append_session_turn(
        ws, session, "assistant", "grounded answer", {"citations": [target_chunk_id]}
    )
    keep_turn = runtime.store.append_session_turn(ws, session, "user", "unrelated follow up", {})
    runtime.store.compact_session(ws, session, summary=f"{MARKER} conversation summary", remove_turn_ids=[])
    assert runtime.store.get_session(ws, session)["summary"] is not None

    # A warm generation model-call cache row for the workspace.
    runtime.store.put_model_call(ws, "gen-model", "gen-key", {"answer": "x", "cited_chunk_ids": [], "usage": {}}, 3600)
    assert _generation_cache_count(runtime, ws) >= 1

    # The embedding cache was genuinely warmed for the target chunk texts during ingest.
    identity = cache_identity_of(embedder)
    assert identity is not None
    embedding_keys = [embedding_cache_key(identity, text) for text in chunk_texts]
    assert all(runtime.store.get_embedding(identity, key) is not None for key in embedding_keys)

    receipt = runtime.project(workspace="acme", corpus="handbook").erase(target)

    # The receipt reports a complete, non-vacuous erasure.
    assert receipt.status is ErasureStatus.ERASED
    assert receipt.chunks == len(target_chunks)
    assert receipt.hypotheticals > 0
    assert receipt.query_cache_rows >= 1
    assert receipt.generation_cache_rows >= 1
    assert receipt.embedding_cache_rows >= len(chunk_texts)
    assert receipt.session_summaries_cleared >= 1
    assert receipt.session_turns_deleted == 1
    assert receipt.preferences >= 1
    assert receipt.outbox_rows >= 1
    assert receipt.blob_deleted is True

    # No chunk, entity, relation, hypothetical, or raptor row for the document survives.
    assert runtime.store.list_document_chunks(ws, corpus, target) == []
    assert runtime.store.list_document_hypothetical_questions(ws, corpus, target) == []
    entities, relations = runtime.store.list_graph(ws, corpus)
    assert all(entity.document_id != target for entity in entities)
    assert all(relation.document_id != target for relation in relations)
    surviving_chunk_ids = {chunk.id for chunk in runtime.store.list_chunks(ws, corpus)}
    assert target_chunk_id not in surviving_chunk_ids
    for _node_id, _embedding, leaf_ids in runtime.store.list_raptor_nodes(ws, corpus):
        assert not (set(leaf_ids) & {chunk.id for chunk in target_chunks})

    # The marker survives in no queryable text surface.
    assert all(MARKER not in chunk.text for chunk in runtime.store.list_chunks(ws, corpus))
    assert all(MARKER not in summary for summary in runtime.store.list_raptor_summaries(ws, corpus))

    # Caches are purged.
    assert _query_cache_count(runtime, ws, corpus) == 0
    assert _generation_cache_count(runtime, ws) == 0
    assert all(runtime.store.get_embedding(identity, key) is None for key in embedding_keys)

    # Session summary cleared, the citing turn deleted, the non-citing turn kept.
    assert runtime.store.get_session(ws, session)["summary"] is None
    remaining_turns = {turn["id"] for turn in runtime.store.list_session_turns(ws, session)}
    assert citing_turn not in remaining_turns
    assert keep_turn in remaining_turns

    # Preference, blob, and outbox gone.
    assert runtime.store.get_preferences(ws, corpus, "user:analyst") == {}
    with pytest.raises(NotFoundError):
        runtime.erasure.blob_store.open(blob_uri)
    with runtime.store.engine.connect() as conn:
        referencing = [
            row
            for row in conn.execute(select(outbox.c.payload)).all()
            if isinstance(row[0], dict) and row[0].get("document_id") == target
        ]
    assert referencing == []

    # A fresh query can no longer surface the marker.
    answer = runtime.queries.query(QueryRequest(workspace_id=ws, corpus_id=corpus, query=f"{MARKER} rollback outage"))
    assert MARKER not in answer.answer

    # The other documents are untouched.
    assert runtime.store.list_document_chunks(ws, corpus, ids["other1"])
    assert runtime.store.list_document_chunks(ws, corpus, ids["other2"])

    # Exactly one redacted document.erased audit event, carrying no marker text.
    erased_events = [event for event in sink.events if event.event_type == "document.erased"]
    assert len(erased_events) == 1
    event = erased_events[0]
    assert event.cited_document_ids == (target,)
    assert MARKER not in json.dumps(event.to_json())

    # The ledger row carries only ids and counts, never the marker text.
    ledger = runtime.store.get_erasure_record(receipt.request_id)
    assert ledger is not None
    assert ledger["document_id"] == target
    assert ledger["status"] == "erased"
    assert MARKER not in json.dumps(ledger, default=str)


def test_erasure_purges_contextualized_embedding_cache(tmp_path: Path) -> None:
    """Under contextual retrieval the ingester embeds contextualize(title, heading, text).

    The purge must key on that exact embedded input, not the raw chunk text, or the semantic
    vector survives in the embedding model-call cache.
    """
    from refaat.core.chunking import contextualize

    settings = _settings(tmp_path).model_copy(update={"contextual_retrieval": True})
    embedder = _CacheIdentifiedEmbedder()
    runtime = RuntimeBuilder(settings).with_embedder(embedder).build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "handbook")
        ids = _ingest(runtime, ws, corpus)
        target = ids["target"]

        target_chunks = runtime.store.list_document_chunks(ws, corpus, target)
        assert target_chunks
        assert all(chunk.metadata.get("context_strategy") == "contextual" for chunk in target_chunks)

        identity = cache_identity_of(embedder)
        assert identity is not None
        contextual_inputs = [
            contextualize(
                str(chunk.metadata.get("title", "")),
                str(chunk.metadata.get("heading", "")),
                chunk.text,
            )
            for chunk in target_chunks
        ]
        # The contextualized inputs really differ from the raw texts and are what warmed the cache.
        assert all(inp != chunk.text for inp, chunk in zip(contextual_inputs, target_chunks, strict=True))
        contextual_keys = [embedding_cache_key(identity, inp) for inp in contextual_inputs]
        assert all(runtime.store.get_embedding(identity, key) is not None for key in contextual_keys)

        receipt = runtime.project(workspace="acme", corpus="handbook").erase(target)

        assert receipt.status is ErasureStatus.ERASED
        assert receipt.embedding_cache_rows >= len(target_chunks)
        assert all(runtime.store.get_embedding(identity, key) is None for key in contextual_keys)
    finally:
        runtime.close()


def test_erasure_is_idempotent(wired: tuple[Refaat, _RecordingSink, _CacheIdentifiedEmbedder]) -> None:
    runtime, _sink, _embedder = wired
    ws = runtime.create_workspace("acme", "Acme")
    corpus = runtime.create_corpus(ws, "handbook")
    ids = _ingest(runtime, ws, corpus)
    project = runtime.project(workspace="acme", corpus="handbook")

    first = project.erase(ids["target"])
    assert first.status is ErasureStatus.ERASED

    second = project.erase(ids["target"])
    assert second.status is ErasureStatus.ERASED
    assert second.chunks == 0
    assert second.hypotheticals == 0
    assert second.query_cache_rows == 0
    assert second.preferences == 0
    assert second.blob_deleted is False
    assert second.request_id != first.request_id


def test_a_blob_delete_failure_keeps_the_document_row_for_a_retry(
    wired: tuple[Refaat, _RecordingSink, _CacheIdentifiedEmbedder],
) -> None:
    runtime, _sink, _embedder = wired
    ws = runtime.create_workspace("acme", "Acme")
    corpus = runtime.create_corpus(ws, "handbook")
    target = _ingest(runtime, ws, corpus)["target"]
    blob_uri = str(runtime.store.get_document(ws, corpus, target)["blob_uri"])

    class _FailOnceBlob:
        def __init__(self, inner: object) -> None:
            self.inner = inner
            self.failed = False

        def delete(self, uri: str) -> None:
            if not self.failed:
                self.failed = True
                raise OSError("transient blob delete failure")
            self.inner.delete(uri)  # type: ignore[attr-defined]

        def __getattr__(self, name: str) -> object:
            return getattr(self.inner, name)

    runtime.erasure.blob_store = _FailOnceBlob(runtime.erasure.blob_store)
    project = runtime.project(workspace="acme", corpus="handbook")

    # The blob delete runs before the row delete, so its failure aborts with the row intact.
    with pytest.raises(OSError, match="transient blob delete failure"):
        project.erase(target)
    assert runtime.store.get_document(ws, corpus, target) is not None

    # The retry re-resolves the blob from the still-present row and completes; the blob is gone.
    receipt = project.erase(target)
    assert receipt.status is ErasureStatus.ERASED
    assert receipt.blob_deleted is True
    with pytest.raises(NotFoundError):
        runtime.erasure.blob_store.open(blob_uri)


def test_erasure_deletes_a_session_turn_from_a_superseded_revision(
    wired: tuple[Refaat, _RecordingSink, _CacheIdentifiedEmbedder],
) -> None:
    runtime, _sink, _embedder = wired
    ws = runtime.create_workspace("acme", "Acme")
    corpus = runtime.create_corpus(ws, "handbook")
    target = _ingest(runtime, ws, corpus)["target"]

    session = runtime.store.create_session(ws, corpus, user_id="analyst", goal=None)
    runtime.queries.query(
        QueryRequest(workspace_id=ws, corpus_id=corpus, session_id=session, query=f"{MARKER} rollback outage")
    )
    turns = runtime.store.list_session_turns(ws, session)
    assistant = next(turn for turn in turns if turn["role"] == "assistant")
    assert target in assistant["metadata"]["cited_document_ids"]

    runtime.ingest(
        ws,
        corpus,
        DocumentInput(
            source_id="target",
            title="target",
            media_type="text/plain",
            content=f"The {MARKER} revised postmortem is a much shorter note now.".encode(),
        ),
    )
    live_chunk_ids = {chunk.id for chunk in runtime.store.list_document_chunks(ws, corpus, target)}
    assert not (set(assistant["metadata"]["citations"]) & live_chunk_ids)

    receipt = runtime.project(workspace="acme", corpus="handbook").erase(target)
    assert receipt.status is ErasureStatus.ERASED
    assert receipt.session_turns_deleted >= 1
    assert all(turn["id"] != assistant["id"] for turn in runtime.store.list_session_turns(ws, session))


def test_erasure_purges_raptor_when_the_builder_is_disabled(tmp_path: Path) -> None:
    built = RuntimeBuilder(_settings(tmp_path)).with_embedder(_CacheIdentifiedEmbedder()).build()
    ws = built.create_workspace("acme", "Acme")
    corpus = built.create_corpus(ws, "handbook")
    target = _ingest(built, ws, corpus)["target"]
    assert built.store.list_raptor_nodes(ws, corpus)
    built.close()

    disabled = _settings(tmp_path).model_copy(update={"raptor_enabled": False})
    runtime = RuntimeBuilder(disabled).with_embedder(_CacheIdentifiedEmbedder()).build()
    try:
        assert runtime.erasure.raptor_builder is None
        receipt = runtime.project(workspace="acme", corpus="handbook").erase(target)
        assert receipt.status is ErasureStatus.ERASED
        assert runtime.store.list_raptor_nodes(ws, corpus) == []
        assert runtime.store.list_raptor_summaries(ws, corpus) == []
    finally:
        runtime.close()


def test_a_post_row_failure_does_not_strand_a_legacy_citing_turn(
    wired: tuple[Refaat, _RecordingSink, _CacheIdentifiedEmbedder],
) -> None:
    runtime, _sink, _embedder = wired
    ws = runtime.create_workspace("acme", "Acme")
    corpus = runtime.create_corpus(ws, "handbook")
    target = _ingest(runtime, ws, corpus)["target"]
    target_chunk_id = runtime.store.list_document_chunks(ws, corpus, target)[0].id

    session = runtime.store.create_session(ws, corpus, user_id="analyst", goal=None)
    # A legacy-format turn: only the volatile chunk id, no stable cited_document_ids.
    legacy_turn = runtime.store.append_session_turn(
        ws, session, "assistant", "grounded answer", {"citations": [target_chunk_id]}
    )

    class _FailOnceRaptor:
        def __init__(self, inner: object) -> None:
            self.inner = inner
            self.failed = False

        def rebuild(self, workspace_id: str, corpus_id: str) -> int:
            if not self.failed:
                self.failed = True
                raise RuntimeError("transient raptor failure")
            return self.inner.rebuild(workspace_id, corpus_id)  # type: ignore[attr-defined]

        def __getattr__(self, name: str) -> object:
            return getattr(self.inner, name)

    runtime.erasure.raptor_builder = _FailOnceRaptor(runtime.erasure.raptor_builder)  # type: ignore[assignment]
    project = runtime.project(workspace="acme", corpus="handbook")

    # The RAPTOR rebuild runs after the row delete, so the first pass fails after the row is gone.
    with pytest.raises(RuntimeError, match="transient raptor failure"):
        project.erase(target)
    # The citing turn was deleted before the row, so the post-row failure did not strand it and the
    # chunk id it needs to match is no longer required.
    assert all(turn["id"] != legacy_turn for turn in runtime.store.list_session_turns(ws, session))

    # The retry converges even though the document row is already gone.
    receipt = project.erase(target)
    assert receipt.status is ErasureStatus.ERASED


def test_legal_hold_refuses_erasure(tmp_path: Path) -> None:
    runtime = (
        RuntimeBuilder(_settings(tmp_path)).with_embedder(_CacheIdentifiedEmbedder()).with_legal_hold_registry().build()
    )
    try:
        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "handbook")
        ids = _ingest(runtime, ws, corpus)
        before = runtime.store.list_document_chunks(ws, corpus, ids["target"])
        assert before

        project = runtime.project(workspace="acme", corpus="handbook")
        project.legal_holds.place(corpus_id=corpus, placed_by="counsel")

        receipt = project.erase(ids["target"])
        assert receipt.status is ErasureStatus.HELD
        assert receipt.chunks == 0
        assert receipt.blob_deleted is False

        # Nothing was erased: the document and its chunks survive.
        after = runtime.store.list_document_chunks(ws, corpus, ids["target"])
        assert {chunk.id for chunk in after} == {chunk.id for chunk in before}
    finally:
        runtime.close()


def test_normal_delete_is_unchanged(wired: tuple[Refaat, _RecordingSink, _CacheIdentifiedEmbedder]) -> None:
    """Byte-identity: the ordinary delete path still behaves exactly as before."""
    runtime, _sink, _embedder = wired
    ws = runtime.create_workspace("acme", "Acme")
    corpus = runtime.create_corpus(ws, "handbook")
    ids = _ingest(runtime, ws, corpus)

    version = runtime.ingestion.delete(ws, corpus, ids["other1"])
    assert isinstance(version, int)
    assert runtime.store.list_document_chunks(ws, corpus, ids["other1"]) == []
    # The other documents are unaffected.
    assert runtime.store.list_document_chunks(ws, corpus, ids["target"])


def test_erasure_completes_when_a_fail_closed_audit_sink_is_down(tmp_path: Path) -> None:
    """A completed erasure never raises on a sink outage: the ledger row is authoritative.

    Under audit_fail_closed the erasure EVENT is best-effort because the physical deletion is
    irreversible and already durably recorded in the erasure_ledger, so a failing sink must not
    hand the caller an exception over an already-ERASED ledger row.
    """
    settings = _settings(tmp_path).model_copy(update={"audit_fail_closed": True})
    runtime = RuntimeBuilder(settings).with_embedder(_CacheIdentifiedEmbedder()).with_audit_sink(_RaisingSink()).build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "handbook")
        target = _ingest(runtime, ws, corpus)["target"]

        receipt = runtime.project(workspace="acme", corpus="handbook").erase(target)
        assert receipt.status is ErasureStatus.ERASED

        with runtime.store.engine.connect() as conn:
            status = conn.execute(
                select(erasure_ledger.c.status).where(erasure_ledger.c.request_id == receipt.request_id)
            ).scalar_one()
        assert status == ErasureStatus.ERASED.value

        assert runtime.store.list_document_chunks(ws, corpus, target) == []
        with pytest.raises(NotFoundError):
            runtime.store.get_document(ws, corpus, target)
    finally:
        runtime.close()


def test_pinned_grounded_rag_artifact_id_is_unchanged() -> None:
    assert GroundedRagProgram().artifact().artifact_id == PINNED_GROUNDED_RAG_ID


def test_http_erase_route_requires_admin_scope(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from refaat.api.app import create_app

    settings = _settings(tmp_path)
    settings = settings.model_copy(update={"trusted_hosts": ["testserver", "localhost", "127.0.0.1"]})
    runtime = RuntimeBuilder(settings).with_embedder(_CacheIdentifiedEmbedder()).build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "handbook")
        ids = _ingest(runtime, ws, corpus)
        _, admin_key = runtime.issue_api_key(ws, "admin", scopes=("read", "write", "admin"))
        _, read_key = runtime.issue_api_key(ws, "reader", scopes=("read",))
        with TestClient(create_app(settings, runtime), raise_server_exceptions=False) as http:
            forbidden = http.post(
                f"/v1/corpora/{corpus}/documents/{ids['target']}/erase",
                headers={"Authorization": f"Bearer {read_key}"},
            )
            assert forbidden.status_code == 403

            response = http.post(
                f"/v1/corpora/{corpus}/documents/{ids['target']}/erase",
                headers={"Authorization": f"Bearer {admin_key}"},
            )
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "erased"
            assert body["document_id"] == ids["target"]
            assert body["counts"]["chunks"] >= 1
        assert runtime.store.list_document_chunks(ws, corpus, ids["target"]) == []
    finally:
        runtime.close()
