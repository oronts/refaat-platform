"""A reference OpenSearch search backend.

OpenSearch and Elasticsearch generate candidates with a knn query for semantic search and a match
query for lexical search, both filtered on the workspace, corpus, and the ACL and metadata allow-sets
resolved through the canonical store, exactly like the pgvector backend. Candidates are hydrated back
to stored chunks and re-scored locally, so every returned hit resolves to a real stored chunk and the
scores share a basis with the built-in backends. The backend fails closed: a transport error raises
rather than falling back to a full store scan, so a degraded cluster never quietly bypasses the index
or its ACL pushdown. The opensearch-py client is an optional extra imported lazily, so the offline
gate never needs it, and an injected client makes the adapter testable without the dependency.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING, Any

from refaat.core.math import cosine_similarity
from refaat.core.sparse import BM25Params, bm25_scores
from refaat.core.text import tokens
from refaat.domain.models import Chunk, MetadataFilter, SearchHit
from refaat.errors import ConfigurationError, DependencyUnavailableError
from refaat.infrastructure.database.store import SQLStore
from refaat.infrastructure.search import DEFAULT_BM25_PARAMS

if TYPE_CHECKING:
    from refaat.ports import Embedder

# The ceiling for a single terms clause; an ACL allow-set larger than this needs a sharded index.
_MAX_TERMS = 200_000


class OpenSearchSearchIndex:
    def __init__(
        self,
        store: SQLStore,
        embedder: Embedder,
        *,
        index_name: str = "refaat_chunks",
        hosts: Sequence[str] | None = None,
        client: Any | None = None,
        provision: bool = True,
        candidate_multiplier: int = 4,
    ) -> None:
        self._store = store
        self._dimensions = int(embedder.dimensions)
        self._index_name = index_name
        self._candidate_multiplier = max(1, candidate_multiplier)
        if client is None:
            self._client = self._build_client(hosts)
            self._owns_client = True
        else:
            self._client = client
            self._owns_client = False
        if provision:
            self._ensure_index()

    @property
    def cache_id(self) -> str:
        return f"opensearch:{self._index_name}:{self._dimensions}"

    def _build_client(self, hosts: Sequence[str] | None) -> Any:
        try:
            from opensearchpy import OpenSearch
        except ImportError as exc:
            raise DependencyUnavailableError("Install refaat[opensearch] for the OpenSearch search backend") from exc
        if not hosts:
            raise ConfigurationError("OpenSearchSearchIndex needs hosts or an injected client")
        return OpenSearch(hosts=list(hosts))

    def _ensure_index(self) -> None:
        try:
            if self._client.indices.exists(index=self._index_name):
                self._verify_dimension()
                return
            self._client.indices.create(
                index=self._index_name,
                body={
                    "settings": {"index": {"knn": True, "max_terms_count": _MAX_TERMS}},
                    "mappings": {
                        "properties": {
                            "chunk_id": {"type": "keyword"},
                            "workspace_id": {"type": "keyword"},
                            "corpus_id": {"type": "keyword"},
                            "document_id": {"type": "keyword"},
                            "text": {"type": "text"},
                            "embedding": {
                                "type": "knn_vector",
                                "dimension": self._dimensions,
                                "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "lucene"},
                            },
                        }
                    },
                },
            )
        except ConfigurationError:
            raise
        except Exception as exc:
            raise DependencyUnavailableError("OpenSearch index provisioning failed") from exc

    def _verify_dimension(self) -> None:
        mapping = self._client.indices.get_mapping(index=self._index_name)
        properties = mapping.get(self._index_name, {}).get("mappings", {}).get("properties", {})
        declared = properties.get("embedding", {}).get("dimension")
        if declared is not None and int(declared) != self._dimensions:
            raise ConfigurationError(
                f"The OpenSearch index {self._index_name!r} embedding dimension is {declared} but the "
                f"embedder produces {self._dimensions}-dimensional vectors. Re-provision the index for the "
                "configured REFAAT_EMBEDDING_DIMENSIONS."
            )

    def replace_document(self, document_id: str, chunks: Sequence[Chunk]) -> None:
        try:
            self._client.delete_by_query(
                index=self._index_name,
                body={"query": {"term": {"document_id": document_id}}},
                refresh=True,
            )
            for chunk in chunks:
                self._client.index(
                    index=self._index_name,
                    id=chunk.id,
                    body={
                        "chunk_id": chunk.id,
                        "workspace_id": str(chunk.metadata["workspace_id"]),
                        "corpus_id": chunk.corpus_id,
                        "document_id": document_id,
                        "text": chunk.text,
                        "embedding": list(chunk.embedding),
                    },
                )
            self._client.indices.refresh(index=self._index_name)
        except Exception as exc:
            raise DependencyUnavailableError("OpenSearch index write failed") from exc

    def delete_document(self, document_id: str) -> None:
        try:
            self._client.delete_by_query(
                index=self._index_name,
                body={"query": {"term": {"document_id": document_id}}},
                refresh=True,
            )
        except Exception as exc:
            raise DependencyUnavailableError("OpenSearch index delete failed") from exc

    def reconcile_document(self, store: SQLStore, workspace_id: str, corpus_id: str, document_id: str) -> None:
        self.replace_document(document_id, store.list_document_chunks(workspace_id, corpus_id, document_id))

    def _filter_clauses(
        self,
        workspace_id: str,
        corpus_id: str,
        allowed: AbstractSet[str] | None,
        allowed_document_ids: AbstractSet[str] | None,
    ) -> list[dict[str, Any]]:
        clauses: list[dict[str, Any]] = [
            {"term": {"workspace_id": workspace_id}},
            {"term": {"corpus_id": corpus_id}},
        ]
        if allowed is not None:
            clauses.append({"terms": {"chunk_id": sorted(allowed)}})
        if allowed_document_ids is not None:
            clauses.append({"terms": {"document_id": sorted(allowed_document_ids)}})
        return clauses

    def _candidate_ids(self, body: dict[str, Any]) -> list[str]:
        try:
            response = self._client.search(index=self._index_name, body=body, allow_partial_search_results=False)
            shards = response.get("_shards")
            degraded = shards is None or bool(response.get("timed_out")) or int(shards.get("failed", 0) or 0) > 0
            if (
                shards is not None
                and "total" in shards
                and int(shards.get("successful", 0) or 0) != int(shards["total"])
            ):
                degraded = True
            if degraded:
                raise DependencyUnavailableError("OpenSearch returned a partial or timed-out result")
            hits = response.get("hits", {}).get("hits", [])
            return [str(hit["_source"]["chunk_id"]) for hit in hits]
        except DependencyUnavailableError:
            raise
        except Exception as exc:
            raise DependencyUnavailableError("OpenSearch search backend is unavailable") from exc

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
        allowed = self._store.filter_chunk_ids(workspace_id, corpus_id, metadata_filter)
        if allowed is not None and not allowed:
            return []
        if allowed_document_ids is not None and not allowed_document_ids:
            return []
        k = max(1, limit) * self._candidate_multiplier
        body = {
            "size": k,
            "_source": ["chunk_id"],
            "query": {
                "knn": {
                    "embedding": {
                        "vector": list(query_embedding),
                        "k": k,
                        "filter": {
                            "bool": {
                                "filter": self._filter_clauses(workspace_id, corpus_id, allowed, allowed_document_ids)
                            }
                        },
                    }
                }
            },
        }
        chunk_ids = self._candidate_ids(body)
        hits = [
            SearchHit(chunk=chunk, score=score, semantic_score=score, reasons=("semantic", "opensearch"))
            for chunk in self._store.get_chunks(workspace_id, corpus_id, chunk_ids)
            if (score := max(0.0, cosine_similarity(query_embedding, chunk.embedding))) > 0.0
        ]
        return sorted(hits, key=lambda hit: (-hit.score, hit.chunk.id))[:limit]

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
        allowed = self._store.filter_chunk_ids(workspace_id, corpus_id, metadata_filter)
        if allowed is not None and not allowed:
            return []
        if allowed_document_ids is not None and not allowed_document_ids:
            return []
        query_terms = tokens(query)
        if not query_terms:
            return []
        k = max(1, limit) * self._candidate_multiplier
        body = {
            "size": k,
            "_source": ["chunk_id"],
            "query": {
                "bool": {
                    "filter": self._filter_clauses(workspace_id, corpus_id, allowed, allowed_document_ids),
                    "must": [{"match": {"text": query}}],
                }
            },
        }
        candidates = self._store.get_chunks(workspace_id, corpus_id, self._candidate_ids(body))
        documents = {chunk.id: tokens(chunk.text) for chunk in candidates}
        scores = bm25_scores(query_terms, documents, params or DEFAULT_BM25_PARAMS)
        hits = [
            SearchHit(
                chunk=chunk, score=scores[chunk.id], lexical_score=scores[chunk.id], reasons=("lexical", "opensearch")
            )
            for chunk in candidates
            if chunk.id in scores
        ]
        return sorted(hits, key=lambda hit: (-hit.score, hit.chunk.id))[:limit]

    def close(self) -> None:
        if self._owns_client:
            with contextlib.suppress(Exception):
                self._client.close()


class OpenSearchIndexFactory:
    """Builds an OpenSearchSearchIndex at the composition root with the runtime-owned store."""

    def __init__(
        self,
        *,
        index_name: str = "refaat_chunks",
        hosts: Sequence[str] | None = None,
        client: Any | None = None,
        provision: bool = True,
        candidate_multiplier: int = 4,
    ) -> None:
        self._index_name = index_name
        self._hosts = hosts
        self._client = client
        self._provision = provision
        self._candidate_multiplier = candidate_multiplier

    def build(self, store: SQLStore, embedder: Embedder) -> OpenSearchSearchIndex:
        return OpenSearchSearchIndex(
            store,
            embedder,
            index_name=self._index_name,
            hosts=self._hosts,
            client=self._client,
            provision=self._provision,
            candidate_multiplier=self._candidate_multiplier,
        )
