from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from typing import Any, cast

from refaat.config import Settings
from refaat.core.retrieval import AdaptiveRetriever
from refaat.domain.models import (
    Chunk,
    GenerationRequest,
    GenerationResult,
    MetadataFilter,
    QueryRequest,
    SearchHit,
)
from refaat.infrastructure.embeddings import DeterministicEmbedder
from refaat.infrastructure.query_transform import ModelMultiQueryTransform, ModelStepBackQueryTransform

QUERY = "How does rollback restore the prior artifact in production?"
VARIANT_A = "what restores the artifact"
VARIANT_B = "how does a production rollback work"
BROADER = "What is artifact recovery?"
INDEXED = {
    "c1": "Rollback restores the prior artifact atomically.",
    "c2": "The registry keeps append-only lineage of every artifact.",
    "c3": "Production deployments advance the active alias.",
}


class _FakeStore:
    def list_graph(self, workspace_id: str, corpus_id: str) -> tuple[list[Any], list[Any]]:
        del workspace_id, corpus_id
        return [], []


class _RecordingIndex:
    """Returns a distinct semantic ranking per variant embedding, no lexical hits.

    Each variant's query embedding maps to its own ordered hit list, so the
    cross-variant RRF merge is observable: a chunk surfaced by every variant
    outranks one surfaced by a single variant. An unmapped embedding yields no
    hits, and lexical search records the query text but contributes nothing.
    """

    def __init__(self, semantic_by_embedding: dict[tuple[float, ...], list[SearchHit]]) -> None:
        self._semantic_by_embedding = semantic_by_embedding
        self.semantic_embeddings: list[tuple[float, ...]] = []
        self.lexical_queries: list[str] = []

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
        del workspace_id, corpus_id, limit, metadata_filter, allowed_document_ids
        self.semantic_embeddings.append(query_embedding)
        return [replace(hit) for hit in self._semantic_by_embedding.get(query_embedding, [])]

    def lexical_search(
        self,
        *,
        workspace_id: str,
        corpus_id: str,
        query: str,
        limit: int,
        metadata_filter: MetadataFilter | None = None,
        params: Any = None,
        allowed_document_ids: frozenset[str] | None = None,
    ) -> list[SearchHit]:
        del workspace_id, corpus_id, limit, metadata_filter, params, allowed_document_ids
        self.lexical_queries.append(query)
        return []


class _FakeGenerator:
    def __init__(self, answer: str) -> None:
        self._answer = answer

    @property
    def model_id(self) -> str:
        return "fake-tier2-model"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        del request
        return GenerationResult(answer=self._answer, cited_chunk_ids=())

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        yield self._answer


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        corpus_id="c",
        document_id=f"d-{chunk_id}",
        ordinal=0,
        text=text,
        token_count=len(text.split()),
        checksum="x",
        metadata={},
        embedding=(),
    )


def _embed(text: str) -> tuple[float, ...]:
    return DeterministicEmbedder().embed([text])[0]


def _hits(*scored: tuple[str, float]) -> list[SearchHit]:
    chunks = {cid: _chunk(cid, text) for cid, text in INDEXED.items()}
    return [SearchHit(chunk=chunks[cid], score=0.0, semantic_score=score) for cid, score in scored]


def _index() -> _RecordingIndex:
    return _RecordingIndex(
        {
            _embed(QUERY): _hits(("c1", 0.9), ("c3", 0.3)),
            _embed(VARIANT_A): _hits(("c2", 0.9), ("c3", 0.8)),
            _embed(VARIANT_B): _hits(("c2", 0.9), ("c3", 0.8)),
            _embed(BROADER): _hits(("c2", 0.9), ("c3", 0.8)),
        }
    )


def _retrieve(index: _RecordingIndex, transform: Any) -> Any:
    settings = Settings(
        api_key_pepper="test-pepper-with-enough-entropy",
        query_expansion=True,
        max_query_variants=4,
    )
    retriever = AdaptiveRetriever(
        cast(Any, _FakeStore()),
        cast(Any, index),
        settings,
        embedder=DeterministicEmbedder(),
        query_transform=transform,
    )
    request = QueryRequest(workspace_id="w", corpus_id="c", query=QUERY)
    return retriever.retrieve(request, _embed(QUERY), profile={}, supplemental_harness={})


def test_multi_query_drives_a_retrieval_per_variant_merged_by_chunk_id() -> None:
    generator = _FakeGenerator(f"{VARIANT_A}\n{VARIANT_B}")
    index = _index()
    packet = _retrieve(index, ModelMultiQueryTransform(generator))

    assert index.lexical_queries == [QUERY, VARIANT_A, VARIANT_B]
    assert len(index.semantic_embeddings) == 3
    assert "expansion" in packet.routes

    # c3 tops the merge only because every variant surfaces it, while c1 (top of
    # the original variant alone) sinks to last: an ordering the first variant's
    # list can never produce, so it pins the cross-variant RRF merge.
    assert [hit.chunk.id for hit in packet.hits] == ["c3", "c2", "c1"]

    variant_texts = set(index.lexical_queries)
    assert packet.hits
    for hit in packet.hits:
        assert hit.chunk.id in INDEXED
        assert hit.chunk.text == INDEXED[hit.chunk.id]
        assert hit.chunk.text not in variant_texts


def test_step_back_broader_variant_retrieves_real_chunks_only() -> None:
    index = _index()
    packet = _retrieve(index, ModelStepBackQueryTransform(_FakeGenerator(BROADER)))

    assert index.lexical_queries == [QUERY, BROADER]
    assert len(index.semantic_embeddings) == 2
    assert _embed(BROADER) in index.semantic_embeddings
    assert "expansion" in packet.routes

    # c3 leads because both the original and broader variants surface it; c2 comes
    # only from the broader variant, so its presence proves the merge ran.
    assert [hit.chunk.id for hit in packet.hits] == ["c3", "c1", "c2"]

    assert packet.hits
    for hit in packet.hits:
        assert hit.chunk.id in INDEXED
        assert hit.chunk.text == INDEXED[hit.chunk.id]
        assert hit.chunk.text != BROADER


def test_multi_query_blank_generation_stays_on_single_variant_path() -> None:
    index = _index()
    packet = _retrieve(index, ModelMultiQueryTransform(_FakeGenerator("   ")))

    assert index.lexical_queries == [QUERY]
    assert len(index.semantic_embeddings) == 1
    assert "expansion" not in packet.routes
