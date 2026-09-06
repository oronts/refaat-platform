from __future__ import annotations

import contextlib
import queue
import re
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from refaat.core.math import cosine_similarity
from refaat.core.sparse import BM25Params, bm25_scores
from refaat.core.text import tokens
from refaat.domain.models import Chunk, JsonObject, MetadataFilter, SearchHit
from refaat.errors import ConfigurationError
from refaat.infrastructure.database.store import SQLStore

if TYPE_CHECKING:
    from refaat.ports import Embedder, SearchIndex

DEFAULT_BM25_PARAMS = BM25Params(k1=1.2, b=0.75)


@runtime_checkable
class SearchIndexFactory(Protocol):
    """Builds a SearchIndex that needs the runtime-owned store and embedder.

    A self-contained backend can be injected directly as a SearchIndex; a backend that needs the
    canonical store (for id to chunk hydration and the ACL and metadata pushdown) is injected as this
    factory instead, whose build is called at the single search-index construction point with the live
    store and embedder. Discriminated from a plain SearchIndex by the presence of ``build``.
    """

    def build(self, store: SQLStore, embedder: Embedder) -> SearchIndex: ...


@runtime_checkable
class ReconcilableIndex(Protocol):
    """A search backend that can rebuild a document's index rows from the committed chunks.

    The durable ``document.indexed`` outbox event drives this on redelivery, so a crash between the
    relational commit and the inline index write self-heals. Any backend exposing it is wired to the
    reconcile sink, so the durable healing path is backend-neutral rather than pgvector-only.
    """

    def reconcile_document(self, store: SQLStore, workspace_id: str, corpus_id: str, document_id: str) -> None: ...


class LocalSearchIndex:
    def __init__(self, store: SQLStore) -> None:
        self.store = store

    def replace_document(self, document_id: str, chunks: Sequence[Chunk]) -> None:
        del document_id, chunks

    def delete_document(self, document_id: str) -> None:
        del document_id

    def _candidate_chunks(
        self,
        workspace_id: str,
        corpus_id: str,
        metadata_filter: MetadataFilter | None,
        allowed_document_ids: frozenset[str] | None = None,
    ) -> list[Chunk]:
        corpus_chunks = self.store.list_chunks(workspace_id, corpus_id)
        if allowed_document_ids is not None:
            corpus_chunks = [chunk for chunk in corpus_chunks if chunk.document_id in allowed_document_ids]
        if metadata_filter is None:
            return corpus_chunks
        return [chunk for chunk in corpus_chunks if metadata_filter.matches(chunk.metadata)]

    def semantic_search(
        self,
        *,
        workspace_id: str,
        corpus_id: str,
        query_embedding: tuple[float, ...],
        limit: int,
        metadata_filter: MetadataFilter | None = None,
        allowed_document_ids: frozenset[str] | None = None,
    ) -> list[SearchHit]:
        values = [
            SearchHit(chunk=chunk, score=score, semantic_score=score, reasons=("semantic",))
            for chunk in self._candidate_chunks(workspace_id, corpus_id, metadata_filter, allowed_document_ids)
            if (score := max(0.0, cosine_similarity(query_embedding, chunk.embedding))) > 0.0
        ]
        return sorted(values, key=lambda hit: (-hit.score, hit.chunk.id))[:limit]

    def lexical_search(
        self,
        *,
        workspace_id: str,
        corpus_id: str,
        query: str,
        limit: int,
        metadata_filter: MetadataFilter | None = None,
        params: BM25Params | None = None,
        allowed_document_ids: frozenset[str] | None = None,
    ) -> list[SearchHit]:
        corpus_chunks = self._candidate_chunks(workspace_id, corpus_id, metadata_filter, allowed_document_ids)
        query_terms = tokens(query)
        if not query_terms or not corpus_chunks:
            return []
        documents = {chunk.id: tokens(chunk.text) for chunk in corpus_chunks}
        scores = bm25_scores(query_terms, documents, params or DEFAULT_BM25_PARAMS)
        hits = [
            SearchHit(chunk=chunk, score=scores[chunk.id], lexical_score=scores[chunk.id], reasons=("lexical",))
            for chunk in corpus_chunks
            if chunk.id in scores
        ]
        return sorted(hits, key=lambda hit: (-hit.score, hit.chunk.id))[:limit]


class PgVectorSearchIndex:
    def __init__(
        self, database_url: str, dimensions: int, store: SQLStore, *, pool_size: int = 5, provision: bool = True
    ) -> None:
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as exc:
            raise RuntimeError("Install refaat[postgres] for PostgreSQL vector search") from exc
        self.psycopg = psycopg
        self.register_vector = register_vector
        self.database_url = re.sub(r"^postgresql\+psycopg://", "postgresql://", database_url)
        self.dimensions = dimensions
        self.store = store
        bound = max(1, pool_size)
        self._pool: queue.LifoQueue[Any] = queue.LifoQueue(maxsize=bound)
        # Cap TOTAL live connections, not just idle ones: the LifoQueue alone bounds
        # only what is checked in, so many concurrent searches would each open a fresh
        # connection and exhaust PostgreSQL. A borrower must hold a permit before it
        # may acquire or open a connection, so at most ``bound`` exist at once.
        self._capacity = threading.BoundedSemaphore(bound)
        self._all_connections: list[Any] = []
        self._pool_lock = threading.Lock()
        # In production the schema is provisioned by migrations under a privileged role, so the runtime
        # verifies it exists instead of issuing DDL. When provisioning is enabled (dev, test, offline) the
        # runtime creates it, which keeps the single-process and gate paths unchanged. Verification runs on
        # both paths, so a stale table whose dimension no longer matches the embedder fails closed at startup.
        if provision:
            self._initialize()
        self._verify()

    def _new_connection(self) -> Any:
        connection = self.psycopg.connect(self.database_url)
        self.register_vector(connection)
        with self._pool_lock:
            self._all_connections.append(connection)
        return connection

    @contextmanager
    def _bootstrap_connection(self) -> Iterator[Any]:
        # A one-off connection that does NOT register the vector type: the extension may not exist yet
        # (provisioning) or its absence must surface as an actionable error (verification), and the pgvector
        # adapter's type lookup raises an opaque error when the extension is missing. It is separate from the
        # pool and closed immediately, so normal pooled connections still register the type once it exists.
        connection = self.psycopg.connect(self.database_url)
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _acquire(self) -> Any:
        try:
            return self._pool.get_nowait()
        except queue.Empty:
            return self._new_connection()

    def _release(self, connection: Any) -> None:
        try:
            self._pool.put_nowait(connection)
        except queue.Full:
            self._discard(connection)

    def _discard(self, connection: Any) -> None:
        with self._pool_lock:
            if connection in self._all_connections:
                self._all_connections.remove(connection)
        with contextlib.suppress(Exception):
            connection.close()

    @contextmanager
    def _borrow(self) -> Iterator[Any]:
        self._capacity.acquire()
        try:
            connection = self._acquire()
            try:
                yield connection
                connection.commit()
            except BaseException:
                try:
                    connection.rollback()
                except Exception:
                    self._discard(connection)
                else:
                    self._release(connection)
                raise
            else:
                self._release(connection)
        finally:
            self._capacity.release()

    def close(self) -> None:
        with self._pool_lock:
            connections = list(self._all_connections)
            self._all_connections.clear()
        for connection in connections:
            with contextlib.suppress(Exception):
                connection.close()

    def reconcile_document(self, store: SQLStore, workspace_id: str, corpus_id: str, document_id: str) -> None:
        """Rebuild this document's vector rows from the committed chunks table.

        Idempotent: replace_document deletes the document's vectors and upserts the
        current stored chunks, so a crash after the relational commit but before the
        inline vector write self-heals when the document.indexed outbox event is
        published. The upsert rewrites the full row (workspace, corpus, document,
        embedding), so a vector row that named the wrong scope is corrected, not just
        its embedding. Vectors are always derived from committed chunks, so a citation
        can never resolve against a vector for a missing chunk.
        """
        chunks = store.list_document_chunks(workspace_id, corpus_id, document_id)
        self.replace_document(document_id, chunks)

    def _initialize(self) -> None:
        with self._bootstrap_connection() as connection, connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS refaat_chunk_vectors (
                    chunk_id varchar(96) PRIMARY KEY,
                    workspace_id varchar(64) NOT NULL,
                    corpus_id varchar(64) NOT NULL,
                    document_id varchar(64) NOT NULL,
                    embedding vector({self.dimensions}) NOT NULL
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS ix_refaat_vectors_scope ON refaat_chunk_vectors (workspace_id, corpus_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS ix_refaat_vectors_hnsw "
                "ON refaat_chunk_vectors USING hnsw (embedding vector_cosine_ops)"
            )

    def _verify(self) -> None:
        with self._bootstrap_connection() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            if cursor.fetchone() is None:
                raise ConfigurationError(
                    "The pgvector 'vector' extension is not installed. Provision the vector schema with the "
                    "migrations (alembic upgrade head) before starting, or set REFAAT_AUTO_CREATE_SCHEMA=true "
                    "to let the runtime create it (not for production)."
                )
            # Bind every check to the relation the runtime's own unqualified SQL resolves to: to_regclass
            # honors search_path, so a same-named table in another schema can never satisfy verification for
            # the relation that reads and writes actually target.
            cursor.execute("SELECT to_regclass('refaat_chunk_vectors')::oid")
            resolved = cursor.fetchone()
            if resolved is None or resolved[0] is None:
                raise ConfigurationError(
                    "The refaat_chunk_vectors table is missing on the search path. Provision the vector "
                    "schema with the migrations (alembic upgrade head) before starting."
                )
            oid = resolved[0]
            cursor.execute("SELECT relkind FROM pg_class WHERE oid = %s", (oid,))
            relkind_row = cursor.fetchone()
            if relkind_row is None or relkind_row[0] != "r":
                raise ConfigurationError(
                    "refaat_chunk_vectors is not an ordinary table. Provision the vector schema with the "
                    "migrations (alembic upgrade head) before starting."
                )
            # Resolve the embedding column's own type by joining on its atttypid (no free-standing type-name
            # lookup, which could match a same-named type in another schema and raise a cardinality error),
            # and confirm that type is the pgvector extension's 'vector' type via its extension membership. Its
            # type modifier is the declared dimension: confirming the type prevents a same-typmod non-vector
            # column (a varchar of the matching length has the same modifier) from passing, and confirming the
            # dimension fails a stale or mismatched provisioning closed at startup rather than at the first write.
            cursor.execute(
                "SELECT a.atttypmod, t.typname, EXISTS ("
                "  SELECT 1 FROM pg_depend d JOIN pg_extension e ON d.refobjid = e.oid "
                "  WHERE d.objid = a.atttypid AND d.deptype = 'e' AND e.extname = 'vector'"
                ") FROM pg_attribute a JOIN pg_type t ON a.atttypid = t.oid "
                "WHERE a.attrelid = %s AND a.attname = 'embedding' AND NOT a.attisdropped",
                (oid,),
            )
            column = cursor.fetchone()
            if column is None or column[1] != "vector" or not column[2]:
                raise ConfigurationError(
                    "refaat_chunk_vectors has no pgvector 'embedding' column. Provision the vector schema "
                    "with the migrations (alembic upgrade head) before starting."
                )
            provisioned = int(column[0])
            if provisioned != self.dimensions:
                raise ConfigurationError(
                    f"The refaat_chunk_vectors embedding column is vector({provisioned}) but the embedder "
                    f"produces {self.dimensions}-dimensional vectors. Re-provision the vector schema for the "
                    "configured REFAAT_EMBEDDING_DIMENSIONS."
                )

    def replace_document(self, document_id: str, chunks: Sequence[Chunk]) -> None:
        with self._borrow() as connection, connection.cursor() as cursor:
            cursor.execute("DELETE FROM refaat_chunk_vectors WHERE document_id = %s", (document_id,))
            cursor.executemany(
                """
                INSERT INTO refaat_chunk_vectors
                    (chunk_id, workspace_id, corpus_id, document_id, embedding)
                VALUES (%s, %s, %s, %s, %s::vector)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    workspace_id = EXCLUDED.workspace_id,
                    corpus_id = EXCLUDED.corpus_id,
                    document_id = EXCLUDED.document_id,
                    embedding = EXCLUDED.embedding
                """,
                [
                    (chunk.id, str(chunk.metadata["workspace_id"]), chunk.corpus_id, document_id, list(chunk.embedding))
                    for chunk in chunks
                ],
            )

    def delete_document(self, document_id: str) -> None:
        with self._borrow() as connection, connection.cursor() as cursor:
            cursor.execute("DELETE FROM refaat_chunk_vectors WHERE document_id = %s", (document_id,))

    def semantic_search(
        self,
        *,
        workspace_id: str,
        corpus_id: str,
        query_embedding: tuple[float, ...],
        limit: int,
        metadata_filter: MetadataFilter | None = None,
        allowed_document_ids: frozenset[str] | None = None,
    ) -> list[SearchHit]:
        allowed = self.store.filter_chunk_ids(workspace_id, corpus_id, metadata_filter)
        if allowed is not None and not allowed:
            return []
        if allowed_document_ids is not None and not allowed_document_ids:
            return []
        with self._borrow() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT chunk_id, 1 - (embedding <=> %s::vector) AS score
                FROM refaat_chunk_vectors
                WHERE workspace_id = %s AND corpus_id = %s
                  AND (%s::boolean IS FALSE OR chunk_id = ANY(%s))
                  AND (%s::boolean IS FALSE OR document_id = ANY(%s))
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (
                    list(query_embedding),
                    workspace_id,
                    corpus_id,
                    allowed is not None,
                    list(allowed) if allowed is not None else [],
                    allowed_document_ids is not None,
                    list(allowed_document_ids) if allowed_document_ids is not None else [],
                    list(query_embedding),
                    limit,
                ),
            )
            rows = cursor.fetchall()
        chunk_ids = [str(row[0]) for row in rows]
        score_by_id = {str(row[0]): max(0.0, float(row[1])) for row in rows}
        return [
            SearchHit(
                chunk=chunk,
                score=score_by_id[chunk.id],
                semantic_score=score_by_id[chunk.id],
                reasons=("semantic", "pgvector"),
            )
            for chunk in self.store.get_chunks(workspace_id, corpus_id, chunk_ids)
        ]

    def lexical_search(
        self,
        *,
        workspace_id: str,
        corpus_id: str,
        query: str,
        limit: int,
        metadata_filter: MetadataFilter | None = None,
        params: BM25Params | None = None,
        allowed_document_ids: frozenset[str] | None = None,
    ) -> list[SearchHit]:
        del params
        allowed = self.store.filter_chunk_ids(workspace_id, corpus_id, metadata_filter)
        if allowed is not None and not allowed:
            return []
        if allowed_document_ids is not None and not allowed_document_ids:
            return []
        with self._borrow() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id,
                       ts_rank_cd(to_tsvector('simple', text), plainto_tsquery('simple', %s)) AS score
                FROM chunks
                WHERE workspace_id = %s
                  AND corpus_id = %s
                  AND (%s::boolean IS FALSE OR id = ANY(%s))
                  AND (%s::boolean IS FALSE OR document_id = ANY(%s))
                  AND to_tsvector('simple', text) @@ plainto_tsquery('simple', %s)
                ORDER BY score DESC
                LIMIT %s
                """,
                (
                    query,
                    workspace_id,
                    corpus_id,
                    allowed is not None,
                    list(allowed) if allowed is not None else [],
                    allowed_document_ids is not None,
                    list(allowed_document_ids) if allowed_document_ids is not None else [],
                    query,
                    limit,
                ),
            )
            rows = cursor.fetchall()
        chunk_ids = [str(row[0]) for row in rows]
        maximum = max((float(row[1]) for row in rows), default=1.0)
        score_by_id = {str(row[0]): float(row[1]) / maximum for row in rows}
        return [
            SearchHit(
                chunk=chunk,
                score=score_by_id[chunk.id],
                lexical_score=score_by_id[chunk.id],
                reasons=("lexical", "postgres_fts"),
            )
            for chunk in self.store.get_chunks(workspace_id, corpus_id, chunk_ids)
        ]


class VectorReconcileSink:
    """Heals a search backend's index from committed chunks on a ``document.indexed`` event.

    Wired for any backend that exposes :class:`ReconcilableIndex` (pgvector and the OpenSearch
    reference backend), not only pgvector. Because the relational rows commit before the inline
    index write, a crash in between leaves the document.indexed outbox event durable; when the
    at-least-once outbox replays it here, the index rows are rebuilt from the committed chunks.
    Reading the current committed chunks makes it version-safe as well as idempotent: an out-of-order
    replay of an older event still writes the document's latest committed state.
    """

    sink_name = "vector-reconcile"

    def __init__(self, index: ReconcilableIndex, store: SQLStore) -> None:
        self.index = index
        self.store = store

    def emit(self, topic: str, payload: JsonObject, *, event_id: str | None = None) -> None:
        del event_id
        if topic != "document.indexed":
            return
        workspace_id = str(payload["workspace_id"])
        corpus_id = str(payload["corpus_id"])
        # Hold the ingest overwrite lock so this read-then-write cannot clobber a newer concurrent write.
        with self.store.document_destroy_lock(workspace_id, corpus_id):
            self.index.reconcile_document(self.store, workspace_id, corpus_id, str(payload["document_id"]))
