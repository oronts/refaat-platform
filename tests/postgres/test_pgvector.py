from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.config import Settings
from refaat.domain.models import DocumentInput, QueryRequest
from refaat.infrastructure.events import OutboxPublisher
from refaat.infrastructure.search import PgVectorSearchIndex, VectorReconcileSink

pytestmark = [pytest.mark.postgres, pytest.mark.integration]


def _pg_runtime(tmp_path: Path) -> Refaat:
    return RuntimeBuilder(
        Settings(
            environment="test",
            database_url=os.environ["REFAAT_TEST_POSTGRES_URL"],
            auto_create_schema=True,
            blob_root=tmp_path / "blobs",
            skill_root=tmp_path / "skills",
            api_key_pepper="postgres-test-pepper",
            embedding_dimensions=64,
            log_json=False,
        )
    ).build()


@pytest.mark.skipif(not os.getenv("REFAAT_TEST_POSTGRES_URL"), reason="PostgreSQL test URL not configured")
def test_postgres_pgvector_round_trip(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(
        Settings(
            environment="test",
            database_url=os.environ["REFAAT_TEST_POSTGRES_URL"],
            auto_create_schema=True,
            blob_root=tmp_path / "blobs",
            skill_root=tmp_path / "skills",
            api_key_pepper="postgres-test-pepper",
            embedding_dimensions=64,
            log_json=False,
        )
    ).build()
    try:
        workspace = runtime.create_workspace("postgres-test", "Postgres Test")
        corpus = runtime.create_corpus(workspace, "knowledge")
        runtime.ingest(
            workspace,
            corpus,
            DocumentInput(
                source_id="postgres.txt",
                title="Postgres",
                media_type="text/plain",
                content=b"PostgreSQL vector retrieval uses pgvector cosine distance.",
            ),
        )
        answer = runtime.query(
            QueryRequest(workspace_id=workspace, corpus_id=corpus, query="What uses cosine distance?")
        )
        assert answer.citations
    finally:
        runtime.close()


@pytest.mark.skipif(not os.getenv("REFAAT_TEST_POSTGRES_URL"), reason="PostgreSQL test URL not configured")
def test_pgvector_pool_is_bounded_and_disposed_on_close(tmp_path: Path) -> None:
    runtime = _pg_runtime(tmp_path)
    captured: list[object] = []
    try:
        workspace = runtime.create_workspace("pg-pool", "Pool Test")
        corpus = runtime.create_corpus(workspace, "knowledge")
        runtime.ingest(
            workspace,
            corpus,
            DocumentInput(
                source_id="pool.txt",
                title="Pool",
                media_type="text/plain",
                content=b"Connection pooling reuses established database connections.",
            ),
        )
        for _ in range(6):
            runtime.query(QueryRequest(workspace_id=workspace, corpus_id=corpus, query="What does pooling reuse?"))
        index = runtime.ingestion.search_index
        assert isinstance(index, PgVectorSearchIndex)
        assert len(index._all_connections) <= 5, "the pool must not grow past its bound under repeated queries"
        captured = list(index._all_connections)
    finally:
        runtime.close()
    assert captured, "the pool created at least one connection"
    assert all(bool(connection.closed) for connection in captured), "runtime.close() disposes every pooled connection"


@pytest.mark.skipif(not os.getenv("REFAAT_TEST_POSTGRES_URL"), reason="PostgreSQL test URL not configured")
def test_pgvector_document_pushdown_narrows_and_fails_closed(tmp_path: Path) -> None:
    runtime = _pg_runtime(tmp_path)
    try:
        workspace = runtime.create_workspace("pg-pushdown", "Pushdown Test")
        corpus = runtime.create_corpus(workspace, "knowledge")
        first = runtime.ingest(
            workspace,
            corpus,
            DocumentInput(
                source_id="alpha.txt",
                title="Alpha",
                media_type="text/plain",
                content=b"The alpha production release rollback plan needs approval.",
            ),
        )
        runtime.ingest(
            workspace,
            corpus,
            DocumentInput(
                source_id="bravo.txt",
                title="Bravo",
                media_type="text/plain",
                content=b"The bravo production release rollback plan needs approval.",
            ),
        )
        index = runtime.ingestion.search_index
        assert isinstance(index, PgVectorSearchIndex)
        embedding = runtime.ingestion.embedder.embed(["production release rollback plan"])[0]
        query = "production release rollback plan"

        for search in (
            lambda **kw: index.semantic_search(query_embedding=embedding, limit=10, **kw),
            lambda **kw: index.lexical_search(query=query, limit=10, **kw),
        ):
            baseline = search(workspace_id=workspace, corpus_id=corpus)
            assert baseline, "the mixed corpus returns hits"
            with_none = search(workspace_id=workspace, corpus_id=corpus, allowed_document_ids=None)
            assert [hit.chunk.id for hit in with_none] == [hit.chunk.id for hit in baseline]
            narrowed = search(
                workspace_id=workspace, corpus_id=corpus, allowed_document_ids=frozenset({first.document_id})
            )
            assert narrowed
            assert all(hit.chunk.document_id == first.document_id for hit in narrowed)
            empty = search(workspace_id=workspace, corpus_id=corpus, allowed_document_ids=frozenset())
            assert empty == []
    finally:
        runtime.close()


def test_reconcile_heals_vector_table_from_committed_chunks(tmp_path: Path) -> None:
    runtime = _pg_runtime(tmp_path)
    try:
        workspace = runtime.create_workspace("pg-reconcile", "Reconcile Test")
        corpus = runtime.create_corpus(workspace, "knowledge")
        runtime.ingest(
            workspace,
            corpus,
            DocumentInput(
                source_id="reconcile.txt",
                title="Reconcile",
                media_type="text/plain",
                content=b"Reconciliation rebuilds the vector table from the committed chunk rows.",
            ),
        )
        index = runtime.ingestion.search_index
        assert isinstance(index, PgVectorSearchIndex)
        embedding = runtime.ingestion.embedder.embed(["reconciliation rebuild"])[0]

        with index._borrow() as connection, connection.cursor() as cursor:
            cursor.execute("DELETE FROM refaat_chunk_vectors")
        assert (
            index.semantic_search(workspace_id=workspace, corpus_id=corpus, query_embedding=embedding, limit=5) == []
        ), "the out-of-band wipe removed every vector row"

        event = next(
            candidate
            for candidate in runtime.store.pending_outbox(50)
            if candidate["topic"] == "document.indexed" and candidate["payload"].get("workspace_id") == workspace
        )
        VectorReconcileSink(index, runtime.store).emit(str(event["topic"]), event["payload"])

        healed = index.semantic_search(workspace_id=workspace, corpus_id=corpus, query_embedding=embedding, limit=5)
        assert healed, "reconcile rebuilds the vectors from committed chunks so semantic search resolves again"
    finally:
        runtime.close()


@pytest.mark.skipif(not os.getenv("REFAAT_TEST_POSTGRES_URL"), reason="PostgreSQL test URL not configured")
def test_pgvector_total_connections_bounded_under_concurrency(tmp_path: Path) -> None:
    runtime = _pg_runtime(tmp_path)
    try:
        workspace = runtime.create_workspace("pg-bound", "Bound Test")
        corpus = runtime.create_corpus(workspace, "knowledge")
        runtime.ingest(
            workspace,
            corpus,
            DocumentInput(
                source_id="bound.txt",
                title="Bound",
                media_type="text/plain",
                content=b"A bounded connection pool caps total live connections with a semaphore.",
            ),
        )
        index = runtime.ingestion.search_index
        assert isinstance(index, PgVectorSearchIndex)
        request = QueryRequest(workspace_id=workspace, corpus_id=corpus, query="What caps live connections?")
        errors: list[BaseException] = []
        lock = threading.Lock()

        def hammer() -> None:
            try:
                for _ in range(8):
                    runtime.query(request)
            except BaseException as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=hammer) for _ in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors, f"concurrent queries must not fail: {errors}"
        assert len(index._all_connections) <= 5, (
            "the semaphore caps TOTAL live connections at the pool bound even under heavy concurrency, "
            "so many parallel searches cannot each open a fresh connection"
        )
    finally:
        runtime.close()


@pytest.mark.skipif(not os.getenv("REFAAT_TEST_POSTGRES_URL"), reason="PostgreSQL test URL not configured")
def test_reconcile_corrects_a_wrong_scope_vector_row(tmp_path: Path) -> None:
    runtime = _pg_runtime(tmp_path)
    try:
        workspace = runtime.create_workspace("pg-scope", "Scope Test")
        corpus = runtime.create_corpus(workspace, "knowledge")
        result = runtime.ingest(
            workspace,
            corpus,
            DocumentInput(
                source_id="scope.txt",
                title="Scope",
                media_type="text/plain",
                content=b"Reconcile rewrites the full vector row so a wrong workspace scope is corrected.",
            ),
        )
        index = runtime.ingestion.search_index
        assert isinstance(index, PgVectorSearchIndex)

        # Scope the corruption to this document: the vector table is shared across the suite.
        with index._borrow() as connection, connection.cursor() as cursor:
            cursor.execute(
                "UPDATE refaat_chunk_vectors SET workspace_id = %s WHERE document_id = %s",
                ("bogus-workspace", result.document_id),
            )
            cursor.execute(
                "SELECT count(*) FROM refaat_chunk_vectors WHERE workspace_id = %s AND document_id = %s",
                (workspace, result.document_id),
            )
            assert cursor.fetchone()[0] == 0, "the out-of-band update corrupted this document's vector scope"

        index.reconcile_document(runtime.store, workspace, corpus, result.document_id)

        with index._borrow() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM refaat_chunk_vectors WHERE workspace_id = %s AND document_id = %s",
                ("bogus-workspace", result.document_id),
            )
            assert cursor.fetchone()[0] == 0, "this document has no wrong-scope vector row after reconcile"
            cursor.execute(
                "SELECT count(*) FROM refaat_chunk_vectors WHERE workspace_id = %s AND document_id = %s",
                (workspace, result.document_id),
            )
            assert cursor.fetchone()[0] > 0, "reconcile rewrote the full row back to the correct workspace scope"
    finally:
        runtime.close()


@pytest.mark.skipif(not os.getenv("REFAAT_TEST_POSTGRES_URL"), reason="PostgreSQL test URL not configured")
def test_reconcile_runs_even_when_an_earlier_sink_fails(tmp_path: Path) -> None:
    runtime = _pg_runtime(tmp_path)
    try:
        workspace = runtime.create_workspace("pg-fanout", "Fanout Test")
        corpus = runtime.create_corpus(workspace, "knowledge")
        runtime.ingest(
            workspace,
            corpus,
            DocumentInput(
                source_id="fanout.txt",
                title="Fanout",
                media_type="text/plain",
                content=b"An earlier failing sink must not starve the reconcile sink in the fan-out.",
            ),
        )
        index = runtime.ingestion.search_index
        assert isinstance(index, PgVectorSearchIndex)
        embedding = runtime.ingestion.embedder.embed(["earlier failing sink reconcile"])[0]

        with index._borrow() as connection, connection.cursor() as cursor:
            cursor.execute("DELETE FROM refaat_chunk_vectors")
        assert index.semantic_search(workspace_id=workspace, corpus_id=corpus, query_embedding=embedding, limit=5) == []

        class FailingSink:
            def emit(self, topic: str, payload: dict[str, object], *, event_id: str | None = None) -> None:
                raise RuntimeError("unrelated custom sink is down")

        publisher = OutboxPublisher(
            runtime.store, [FailingSink(), VectorReconcileSink(index, runtime.store)], worker_id="w1"
        )
        publisher.publish(50)

        healed = index.semantic_search(workspace_id=workspace, corpus_id=corpus, query_embedding=embedding, limit=5)
        assert healed, "the reconcile sink still heals the vectors despite the earlier sink raising in the same batch"
    finally:
        runtime.close()


@pytest.mark.skipif(not os.getenv("REFAAT_TEST_POSTGRES_URL"), reason="PostgreSQL test URL not configured")
def test_pgvector_provision_flag_verifies_instead_of_creating(tmp_path: Path) -> None:
    from sqlalchemy import text

    from refaat.errors import ConfigurationError

    url = os.environ["REFAAT_TEST_POSTGRES_URL"]
    runtime = _pg_runtime(tmp_path)
    try:
        store = runtime.store
        # provision=True at build created the table, so a verify-only index succeeds.
        PgVectorSearchIndex(url, 64, store, provision=False).close()

        # With the table absent, a verify-only index fails closed with a clear error, never issuing DDL.
        with store.engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS refaat_chunk_vectors"))
        with pytest.raises(ConfigurationError):
            PgVectorSearchIndex(url, 64, store, provision=False)

        # A provisioning index re-creates it, after which verify passes again.
        PgVectorSearchIndex(url, 64, store, provision=True).close()
        PgVectorSearchIndex(url, 64, store, provision=False).close()

        # A provisioned dimension that disagrees with the embedder fails closed at startup, not at write time.
        with store.engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS refaat_chunk_vectors"))
            conn.execute(
                text("CREATE TABLE refaat_chunk_vectors (chunk_id varchar(96) PRIMARY KEY, embedding vector(32))")
            )
        with pytest.raises(ConfigurationError):
            PgVectorSearchIndex(url, 64, store, provision=False)

        # A relation of the right name that is not an ordinary table (a view) also fails closed.
        with store.engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS refaat_chunk_vectors"))
            conn.execute(text("CREATE VIEW refaat_chunk_vectors AS SELECT 1 AS embedding"))
        with pytest.raises(ConfigurationError):
            PgVectorSearchIndex(url, 64, store, provision=False)
        with store.engine.begin() as conn:
            conn.execute(text("DROP VIEW IF EXISTS refaat_chunk_vectors"))

        # A missing extension surfaces the actionable ConfigurationError, not an opaque adapter error,
        # because the bootstrap connection does not register the vector type before the check.
        with store.engine.begin() as conn:
            conn.execute(text("DROP EXTENSION IF EXISTS vector CASCADE"))
        with pytest.raises(ConfigurationError):
            PgVectorSearchIndex(url, 64, store, provision=False)

        # Re-provisioning restores the extension and table, and verify passes.
        PgVectorSearchIndex(url, 64, store, provision=True).close()
        PgVectorSearchIndex(url, 64, store, provision=False).close()

        # provision=True also verifies: a stale wrong-dimension table (CREATE TABLE IF NOT EXISTS keeps it)
        # fails closed at startup rather than at the first write. The stale table carries the full column set,
        # so _initialize's index build succeeds and only the dimension check refuses it.
        with store.engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS refaat_chunk_vectors"))
            conn.execute(
                text(
                    "CREATE TABLE refaat_chunk_vectors ("
                    "chunk_id varchar(96) PRIMARY KEY, workspace_id varchar(64) NOT NULL, "
                    "corpus_id varchar(64) NOT NULL, document_id varchar(64) NOT NULL, embedding vector(32))"
                )
            )
        with pytest.raises(ConfigurationError):
            PgVectorSearchIndex(url, 64, store, provision=True)

        # A same-typmod non-vector column (varchar of the matching length) does not pass the type check.
        with store.engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS refaat_chunk_vectors"))
            conn.execute(
                text("CREATE TABLE refaat_chunk_vectors (chunk_id varchar(96) PRIMARY KEY, embedding varchar(60))")
            )
        with pytest.raises(ConfigurationError):
            PgVectorSearchIndex(url, 64, store, provision=False)

        # Verification binds to the search-path relation: a valid table in another schema does not satisfy it.
        with store.engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS refaat_chunk_vectors"))
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS shadow"))
            conn.execute(
                text(
                    "CREATE TABLE shadow.refaat_chunk_vectors (chunk_id varchar(96) PRIMARY KEY, embedding vector(64))"
                )
            )
        with pytest.raises(ConfigurationError):
            PgVectorSearchIndex(url, 64, store, provision=False)
        with store.engine.begin() as conn:
            conn.execute(text("DROP SCHEMA IF EXISTS shadow CASCADE"))

        # A same-named 'vector' type in another schema must not break verification of a valid deployment:
        # the type is resolved by the column's own type id, never a free type-name lookup that could match
        # more than one row and raise a cardinality error.
        PgVectorSearchIndex(url, 64, store, provision=True).close()
        with store.engine.begin() as conn:
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS other"))
            conn.execute(text("CREATE TABLE other.vector (id int)"))  # creates a composite type other.vector
        PgVectorSearchIndex(url, 64, store, provision=False).close()
        with store.engine.begin() as conn:
            conn.execute(text("DROP SCHEMA IF EXISTS other CASCADE"))

        PgVectorSearchIndex(url, 64, store, provision=True).close()
        PgVectorSearchIndex(url, 64, store, provision=False).close()
    finally:
        runtime.close()


@pytest.mark.skipif(not os.getenv("REFAAT_TEST_POSTGRES_URL"), reason="PostgreSQL test URL not configured")
def test_pgvector_passes_the_search_index_conformance_contract(tmp_path: Path) -> None:
    import dataclasses

    from tests.integration._search_index_cases import build_search_index_case

    from refaat.conformance.search_index import search_index_conformance

    runtime = _pg_runtime(tmp_path)
    try:
        case = build_search_index_case(runtime, workspace="pg-conformance", corpus="handbook")
        chunks = runtime.store.list_chunks(case.workspace_id, case.corpus_id)
        probe_ids = frozenset(c.id for c in chunks if c.metadata.get("team") == "bravo")
        probe_doc = next(c.document_id for c in chunks if c.metadata.get("team") == "bravo")
        probed = dataclasses.replace(case, delete_probe_document_id=probe_doc, delete_probe_chunk_ids=probe_ids)
        assert search_index_conformance(probed) == []
    finally:
        runtime.close()
