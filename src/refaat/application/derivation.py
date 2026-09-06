"""Content-addressed derivation manifests: the recipe, counts, and checksum that materialize a corpus.

A derivation is the chunking, embedding, and graph recipe that produced a corpus's stored rows. The
manifest records that recipe as a content-addressed key plus row counts and an order-independent
checksum, so a materialization can be verified against committed state and two derivations compared.
Off by default and byte-identical when off: the service is built only when derivation_manifest_enabled
is set, and its refresh is best-effort so a manifest failure never breaks an ingest.
"""

from __future__ import annotations

from dataclasses import dataclass

from refaat.canonical import canonical_digest
from refaat.config import Settings
from refaat.domain.models import JsonObject
from refaat.infrastructure.database.store import SQLStore
from refaat.observability import get_logger
from refaat.ports import Embedder

logger = get_logger(__name__)

VERIFY_NO_MANIFEST = "no_manifest"
VERIFY_MATCH = "match"
VERIFY_DRIFT = "drift"
VERIFY_NO_DOCUMENT = "no_document"
VERIFY_NO_DIGEST = "no_digest"


def compare_manifests(left: DerivationManifest, right: DerivationManifest) -> JsonObject:
    """A pure field-by-field comparison of two manifests, for blue/green generation identity checks.

    Every DerivationManifest field has a same_* entry, so identical is true exactly when all of them are.
    """
    return {
        "identical": left == right,
        "same_derivation_key": left.derivation_key == right.derivation_key,
        "same_corpus_version": left.corpus_version == right.corpus_version,
        "same_config": left.config == right.config,
        "same_embedding_model": left.embedding_model == right.embedding_model,
        "same_embedding_dimensions": left.embedding_dimensions == right.embedding_dimensions,
        "same_row_counts": left.row_counts == right.row_counts,
        "same_content_checksum": left.content_checksum == right.content_checksum,
        "same_status": left.status == right.status,
    }


@dataclass(frozen=True, slots=True)
class DerivationManifest:
    derivation_key: str
    corpus_version: int
    config: JsonObject
    embedding_model: str
    embedding_dimensions: int
    row_counts: JsonObject
    content_checksum: str
    status: str


def _manifest_from_row(row: JsonObject) -> DerivationManifest:
    return DerivationManifest(
        derivation_key=str(row["derivation_key"]),
        corpus_version=int(row["corpus_version"]),
        config=dict(row["config"]),
        embedding_model=str(row["embedding_model"]),
        embedding_dimensions=int(row["embedding_dimensions"]),
        row_counts=dict(row["row_counts"]),
        content_checksum=str(row["content_checksum"]),
        status=str(row["status"]),
    )


def derivation_config_snapshot(
    settings: Settings, embedder: Embedder, *, search_backend: str, graph_backend: str, chunking_strategy_id: str
) -> JsonObject:
    # The fields that change the stored chunks, graph, or embeddings, so a recipe change yields a new
    # derivation_key and two runs of the same recipe share one manifest row. embedding_backend and
    # graph_backend fold the concrete adapter identity, so a same-model different-backend or an injected
    # custom extractor is a distinct derivation. Local import breaks an application/infrastructure cycle.
    from refaat.infrastructure.model_cache import resolved_cache_identity

    return {
        "chunk_tokens": settings.default_chunk_tokens,
        "chunk_overlap": settings.chunk_overlap_tokens,
        "chunking_strategy": chunking_strategy_id,
        "sentence_window_radius": settings.sentence_window_radius,
        "contextual_retrieval": settings.contextual_retrieval,
        "embedding_model": embedder.model_id,
        "embedding_dimensions": embedder.dimensions,
        "embedding_backend": resolved_cache_identity(embedder),
        "graph_provider": settings.graph_provider,
        "graph_backend": graph_backend,
        "graph_max_entities": settings.graph_max_entities,
        "graph_min_token_length": settings.graph_min_token_length,
        "default_language": settings.default_language,
        "stopwords_by_language": settings.stopwords_by_language,
        "hype_enabled": settings.hype_enabled,
        "raptor_enabled": settings.raptor_enabled,
        "search_backend": search_backend,
    }


class DerivationService:
    """Produce and verify the derivation manifest for a corpus."""

    def __init__(
        self,
        store: SQLStore,
        settings: Settings,
        embedder: Embedder,
        *,
        search_backend: str,
        graph_backend: str,
        chunking_strategy_id: str,
    ) -> None:
        self._store = store
        self._settings = settings
        self._embedder = embedder
        self._search_backend = search_backend
        self._graph_backend = graph_backend
        self._chunking_strategy_id = chunking_strategy_id

    def _snapshot(self) -> JsonObject:
        return derivation_config_snapshot(
            self._settings,
            self._embedder,
            search_backend=self._search_backend,
            graph_backend=self._graph_backend,
            chunking_strategy_id=self._chunking_strategy_id,
        )

    def derivation_key(self) -> str:
        return canonical_digest(self._snapshot())

    def refresh_quietly(self, workspace_id: str, corpus_id: str) -> None:
        # Best-effort: the manifest is descriptive, so a refresh failure is logged and swallowed rather
        # than breaking the ingest, delete, or erase that just committed.
        try:
            self.refresh(workspace_id, corpus_id)
        except Exception:
            logger.warning("derivation_manifest_refresh_failed", workspace_id=workspace_id, corpus_id=corpus_id)

    def refresh(self, workspace_id: str, corpus_id: str) -> DerivationManifest:
        """Recompute the manifest for the corpus's current derivation from committed state.

        Called after an ingest or delete commits. Reads the authoritative corpus version and the
        row-count and checksum rollup, then upserts the manifest keyed on the derivation recipe.
        """
        snapshot = self._snapshot()
        key = canonical_digest(snapshot)
        corpus = self._store.get_corpus(workspace_id, corpus_id)
        counts, checksum = self._store.derivation_rollup(workspace_id, corpus_id)
        self._store.upsert_derivation_manifest(
            workspace_id,
            corpus_id,
            derivation_key=key,
            corpus_version=corpus.version,
            config=snapshot,
            embedding_model=self._embedder.model_id,
            embedding_dimensions=self._embedder.dimensions,
            row_counts=counts,
            content_checksum=checksum,
            status="ready",
        )
        return DerivationManifest(
            derivation_key=key,
            corpus_version=corpus.version,
            config=snapshot,
            embedding_model=self._embedder.model_id,
            embedding_dimensions=self._embedder.dimensions,
            row_counts=counts,
            content_checksum=checksum,
            status="ready",
        )

    def verify(self, workspace_id: str, corpus_id: str) -> bool:
        """True when the stored manifest matches a freshly recomputed rollup of committed state."""
        stored = self._store.get_derivation_manifest(workspace_id, corpus_id, self.derivation_key())
        if stored is None:
            return False
        counts, checksum = self._store.derivation_rollup(workspace_id, corpus_id)
        return bool(stored["content_checksum"] == checksum and stored["row_counts"] == counts)

    def verify_detail(self, workspace_id: str, corpus_id: str) -> str:
        """The operator-facing tri-state: no_manifest, match, or drift, recomputed from committed state."""
        stored = self._store.get_derivation_manifest(workspace_id, corpus_id, self.derivation_key())
        if stored is None:
            return VERIFY_NO_MANIFEST
        counts, checksum = self._store.derivation_rollup(workspace_id, corpus_id)
        if stored["content_checksum"] == checksum and stored["row_counts"] == counts:
            return VERIFY_MATCH
        return VERIFY_DRIFT

    def current(self, workspace_id: str, corpus_id: str) -> DerivationManifest | None:
        """The stored manifest for the corpus's current derivation recipe, or None if none is recorded."""
        stored = self._store.get_derivation_manifest(workspace_id, corpus_id, self.derivation_key())
        return _manifest_from_row(stored) if stored is not None else None

    def manifest(self, workspace_id: str, corpus_id: str, derivation_key: str) -> DerivationManifest | None:
        """The stored manifest for one derivation key, or None if that key was never recorded."""
        stored = self._store.get_derivation_manifest(workspace_id, corpus_id, derivation_key)
        return _manifest_from_row(stored) if stored is not None else None

    def manifests(self, workspace_id: str, corpus_id: str) -> list[DerivationManifest]:
        """Every stored manifest for the corpus, one per recorded derivation recipe, newest first."""
        return [_manifest_from_row(row) for row in self._store.list_derivation_manifests(workspace_id, corpus_id)]

    def verify_document(self, workspace_id: str, corpus_id: str, document_id: str) -> str:
        """Per-document integrity: no_document, no_digest, match, or drift, recomputed from committed state.

        Localizes which document's chunk set drifted, complementing the corpus-wide verify_detail. A NULL
        stored digest (feature off at ingest, or a pre-migration row) reports no_digest, never drift.
        """
        exists, stored = self._store.get_document_derivation_digest(workspace_id, corpus_id, document_id)
        if not exists:
            return VERIFY_NO_DOCUMENT
        if stored is None:
            return VERIFY_NO_DIGEST
        recomputed = self._store.document_derivation_digest(workspace_id, corpus_id, document_id)
        return VERIFY_MATCH if stored == recomputed else VERIFY_DRIFT

    def drifted_documents(self, workspace_id: str, corpus_id: str) -> list[str]:
        """The ids of documents whose stored digest no longer matches a fresh recompute, sorted by id."""
        digests = self._store.list_document_derivation_digests(workspace_id, corpus_id)
        drifted = [
            document_id
            for document_id, stored in digests.items()
            if stored is not None
            and stored != self._store.document_derivation_digest(workspace_id, corpus_id, document_id)
        ]
        return sorted(drifted)
