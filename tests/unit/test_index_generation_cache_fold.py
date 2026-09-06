"""The index-generation answer-cache fold: byte-identical off, invalidating on activation.

Activation here is a test-only direct row insert; this increment ships no serving flip, so the tests
exercise identity and the cache fold, not a governed activation path.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import insert, update
from tests.integration.test_admin_status_e2e import PINNED_CONTEXT_HASH, _default_context_hash

from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.config import Settings
from refaat.core.text import sha256_text
from refaat.domain.models import DocumentInput, QueryRequest
from refaat.ids import new_id
from refaat.infrastructure.database.schema import index_generations

_BODY = b"The release checklist requires reviewer approval before any production deployment proceeds."
_QUERY = "what does the release checklist require"


def _enabled(settings: Settings) -> Settings:
    return settings.model_copy(update={"index_generations_enabled": True})


def _corpus(runtime: Refaat) -> tuple[str, str]:
    project = runtime.project(workspace="acme", corpus="handbook")
    runtime.ingest(
        project.workspace_id,
        project.corpus_id,
        DocumentInput(source_id="a.txt", title="a", media_type="text/plain", content=_BODY),
    )
    return project.workspace_id, project.corpus_id


def _activate(runtime: Refaat, ws: str, corpus: str, *, key: str, checksum: str) -> None:
    now = datetime.now(UTC)
    with runtime.store.engine.begin() as conn:
        conn.execute(
            update(index_generations)
            .where(index_generations.c.workspace_id == ws, index_generations.c.corpus_id == corpus)
            .values(status="retired", retired_at=now)
        )
        conn.execute(
            insert(index_generations).values(
                id=new_id("gen"),
                workspace_id=ws,
                corpus_id=corpus,
                derivation_key=key,
                corpus_version=0,
                content_checksum=checksum,
                row_counts={},
                config={},
                embedding_model="emb",
                embedding_dimensions=8,
                status="active",
                created_at=now,
                updated_at=now,
                activated_at=now,
            )
        )


def test_default_context_hash_is_unmoved_by_the_fold() -> None:
    assert _default_context_hash() == PINNED_CONTEXT_HASH


def test_active_generation_resolves_to_the_content_identity_term(settings: Settings) -> None:
    runtime = RuntimeBuilder(_enabled(settings)).build()
    try:
        ws, corpus = _corpus(runtime)
        assert runtime.store.get_active_index_generation(ws, corpus) is None
        _activate(runtime, ws, corpus, key="k1", checksum="c1")
        assert runtime.store.get_active_index_generation(ws, corpus) == sha256_text("k1:c1")
        _activate(runtime, ws, corpus, key="k1", checksum="c2")
        assert runtime.store.get_active_index_generation(ws, corpus) == sha256_text("k1:c2")
    finally:
        runtime.close()


def test_read_degrades_to_none_never_empty_string(settings: Settings) -> None:
    runtime = RuntimeBuilder(_enabled(settings)).build()
    try:
        ws, corpus = _corpus(runtime)
        _activate(runtime, ws, corpus, key="", checksum="")
        assert runtime.store.get_active_index_generation(ws, corpus) is None
        with runtime.store.engine.begin() as conn:
            conn.exec_driver_sql("DROP TABLE index_generations")
        assert runtime.store.get_active_index_generation(ws, corpus) is None
    finally:
        runtime.close()


def test_activation_invalidates_the_answer_cache_when_enabled(settings: Settings) -> None:
    runtime = RuntimeBuilder(_enabled(settings)).build()
    try:
        ws, corpus = _corpus(runtime)
        request = QueryRequest(workspace_id=ws, corpus_id=corpus, query=_QUERY)
        assert runtime.query(request).cached is False
        assert runtime.query(request).cached is True
        _activate(runtime, ws, corpus, key="k1", checksum="c1")
        assert runtime.query(request).cached is False
        assert runtime.query(request).cached is True
    finally:
        runtime.close()


def test_off_path_is_byte_identical_even_with_an_active_row(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings).build()
    try:
        ws, corpus = _corpus(runtime)
        request = QueryRequest(workspace_id=ws, corpus_id=corpus, query=_QUERY)
        assert runtime.query(request).cached is False
        assert runtime.query(request).cached is True
        # An active row is present, but the flag gates the read off, so nothing folds and the hit stands.
        _activate(runtime, ws, corpus, key="k1", checksum="c1")
        assert runtime.query(request).cached is True
    finally:
        runtime.close()
