from __future__ import annotations

from typing import Any, cast

from refaat.config import Settings
from refaat.core.retrieval import AdaptiveRetriever
from refaat.domain.models import Chunk, SearchHit


def _retriever(settings: Settings) -> AdaptiveRetriever:
    return AdaptiveRetriever(cast(Any, None), cast(Any, None), settings)


def _hit(chunk_id: str, document_id: str, *, semantic: float = 0.0, lexical: float = 0.0) -> SearchHit:
    chunk = Chunk(
        id=chunk_id,
        corpus_id="c",
        document_id=document_id,
        ordinal=0,
        text=chunk_id,
        token_count=1,
        checksum="x",
        metadata={},
        embedding=(0.0,),
    )
    return SearchHit(chunk=chunk, score=0.0, semantic_score=semantic, lexical_score=lexical)


def test_rrf_ranks_a_chunk_present_in_both_lists_first(settings: Settings) -> None:
    semantic = [_hit("c1", "d1", semantic=0.9), _hit("c2", "d2", semantic=0.8)]
    lexical = [_hit("c1", "d1", lexical=0.9), _hit("c3", "d3", lexical=0.7)]
    fused = _retriever(settings)._fuse(semantic, lexical, {}, {}, (0.5, 0.3, 0.2), fusion="rrf", rrf_k=60)
    assert fused[0].chunk.id == "c1"
    assert all(0.0 <= hit.score <= 1.0 for hit in fused)


def test_rrf_normalizes_top_to_one_and_keeps_all_positive(settings: Settings) -> None:
    semantic = [_hit("a", "d1", semantic=0.9), _hit("b", "d2", semantic=0.5)]
    lexical = [_hit("b", "d2", lexical=0.9), _hit("c", "d3", lexical=0.5)]
    fused = _retriever(settings)._fuse(semantic, lexical, {}, {}, (0.5, 0.3, 0.2), fusion="rrf", rrf_k=60)
    assert max(hit.score for hit in fused) == 1.0
    assert all(hit.score > 0.0 for hit in fused)


def test_rrf_breaks_ties_by_chunk_id(settings: Settings) -> None:
    semantic = [_hit("b", "d1", semantic=0.5)]
    lexical = [_hit("a", "d2", lexical=0.5)]
    fused = _retriever(settings)._fuse(semantic, lexical, {}, {}, (0.5, 0.5, 0.0), fusion="rrf", rrf_k=60)
    assert [hit.chunk.id for hit in fused] == ["a", "b"]


def test_rrf_graph_boost_promotes_its_document(settings: Settings) -> None:
    semantic = [_hit("a", "d1", semantic=0.5), _hit("b", "d2", semantic=0.5)]
    fused = _retriever(settings)._fuse(semantic, [], {"d2": 1.0}, {}, (0.4, 0.3, 0.3), fusion="rrf", rrf_k=60)
    assert fused[0].chunk.id == "b"
    graph_reasons = {hit.chunk.id: "graph" in hit.reasons for hit in fused}
    assert graph_reasons == {"a": False, "b": True}


def test_linear_reproduces_the_weighted_sum(settings: Settings) -> None:
    semantic = [_hit("a", "d1", semantic=0.8), _hit("b", "d2", semantic=0.4)]
    lexical = [_hit("a", "d1", lexical=0.6), _hit("c", "d3", lexical=0.9)]
    fused = _retriever(settings)._fuse(semantic, lexical, {}, {}, (0.6, 0.3, 0.1), fusion="linear", rrf_k=60)
    scores = {hit.chunk.id: round(hit.score, 6) for hit in fused}
    assert scores["a"] == round(0.6 * 0.8 + 0.3 * 0.6, 6)
    assert scores["b"] == round(0.6 * 0.4, 6)
    assert scores["c"] == round(0.3 * 0.9, 6)
    assert [hit.chunk.id for hit in fused] == ["a", "c", "b"]
