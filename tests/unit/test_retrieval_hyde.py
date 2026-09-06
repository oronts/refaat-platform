from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

from refaat.config import Settings
from refaat.core.retrieval import AdaptiveRetriever
from refaat.domain.models import Chunk, MetadataFilter, QueryRequest, SearchHit, TransformedQuery
from refaat.infrastructure.embeddings import DeterministicEmbedder
from refaat.infrastructure.query_transform import DeterministicHydeQueryTransform, _templated_hypothesis

QUERY = "How does rollback restore the prior artifact?"


class _FakeStore:
    def list_graph(self, workspace_id: str, corpus_id: str) -> tuple[list[Any], list[Any]]:
        del workspace_id, corpus_id
        return [], []


class _RecordingIndex:
    def __init__(self, semantic: list[SearchHit], lexical: list[SearchHit]) -> None:
        self._semantic = semantic
        self._lexical = lexical
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
        return [replace(hit) for hit in self._semantic]

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
        return [replace(hit) for hit in self._lexical]


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


def _build_retriever(index: _RecordingIndex) -> AdaptiveRetriever:
    settings = Settings(
        api_key_pepper="test-pepper-with-enough-entropy",
        query_expansion=True,
        max_query_variants=2,
    )
    return AdaptiveRetriever(
        cast(Any, _FakeStore()),
        cast(Any, index),
        settings,
        embedder=DeterministicEmbedder(),
        query_transform=DeterministicHydeQueryTransform(),
    )


def _run() -> tuple[_RecordingIndex, Any]:
    indexed = {
        "c1": _chunk("c1", "Rollback restores the prior artifact atomically."),
        "c2": _chunk("c2", "The registry keeps append-only lineage."),
    }
    semantic = [
        SearchHit(chunk=indexed["c1"], score=0.0, semantic_score=0.9),
        SearchHit(chunk=indexed["c2"], score=0.0, semantic_score=0.5),
    ]
    lexical = [SearchHit(chunk=indexed["c1"], score=0.0, lexical_score=0.8)]
    index = _RecordingIndex(semantic, lexical)
    retriever = _build_retriever(index)
    request = QueryRequest(workspace_id="w", corpus_id="c", query=QUERY)
    packet = retriever.retrieve(
        request,
        _embed(QUERY),
        profile={},
        supplemental_harness={},
    )
    return index, packet


def test_semantic_receives_query_and_hypothesis_embeddings() -> None:
    index, _ = _run()
    hypothesis = _templated_hypothesis(QUERY)
    query_vec = _embed(QUERY)
    hypothesis_vec = _embed(hypothesis)

    assert len(index.semantic_embeddings) == 2
    assert query_vec in index.semantic_embeddings
    assert hypothesis_vec in index.semantic_embeddings
    assert hypothesis_vec != query_vec


def test_lexical_only_ever_sees_the_original_query_text() -> None:
    index, _ = _run()
    assert index.lexical_queries == [QUERY, QUERY]
    hypothesis = _templated_hypothesis(QUERY)
    assert hypothesis not in index.lexical_queries


def test_routes_include_expansion_and_rrf() -> None:
    _, packet = _run()
    assert "expansion" in packet.routes
    assert "rrf" in packet.routes


def test_every_hit_is_an_indexed_chunk_and_never_the_hypothesis() -> None:
    _, packet = _run()
    hypothesis = _templated_hypothesis(QUERY)
    indexed_ids = {"c1", "c2"}
    assert packet.hits
    for hit in packet.hits:
        assert hit.chunk.id in indexed_ids
        assert hit.chunk.text != hypothesis


def test_embed_text_none_is_byte_identical_to_embedding_the_variant_text() -> None:
    embedder = DeterministicEmbedder()
    retriever = AdaptiveRetriever(
        cast(Any, _FakeStore()),
        cast(Any, _RecordingIndex([], [])),
        Settings(api_key_pepper="test-pepper-with-enough-entropy"),
        embedder=embedder,
        query_transform=_StubTransform(),
    )
    policy = retriever.resolve_policy(
        QueryRequest(workspace_id="w", corpus_id="c", query=QUERY),
        {},
    )
    plan = retriever._transform_plan(QUERY, _embed(QUERY), policy, "w")
    assert plan[1][0] == "restore prior artifact"
    assert plan[1][1] == embedder.embed(["restore prior artifact"])[0]


class _StubTransform:
    @property
    def cache_id(self) -> str:
        return "stub-v1"

    def expand(self, query: str, max_variants: int, *, workspace_id: str = "") -> tuple[TransformedQuery, ...]:
        del max_variants, workspace_id
        return (
            TransformedQuery(text=query, kind="original"),
            TransformedQuery(text="restore prior artifact", kind="subquery"),
        )
