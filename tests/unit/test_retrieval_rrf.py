from __future__ import annotations

from typing import Any, cast

from refaat.config import Settings
from refaat.core.retrieval import AdaptiveRetriever
from refaat.domain.models import Chunk, SearchHit


def _retriever(settings: Settings) -> AdaptiveRetriever:
    return AdaptiveRetriever(cast(Any, None), cast(Any, None), settings)


def _hit(chunk_id: str, *, score: float = 0.0) -> SearchHit:
    chunk = Chunk(
        id=chunk_id,
        corpus_id="c",
        document_id=f"d-{chunk_id}",
        ordinal=0,
        text=chunk_id,
        token_count=1,
        checksum="x",
        metadata={},
        embedding=(0.0,),
    )
    return SearchHit(chunk=chunk, score=score)


def test_chunk_ranked_high_in_both_lists_outranks_single_list_chunks(settings: Settings) -> None:
    list_one = [_hit("c1"), _hit("c2")]
    list_two = [_hit("c1"), _hit("c3")]
    merged = _retriever(settings)._rrf_merge([list_one, list_two], 60)
    assert merged[0].chunk.id == "c1"
    assert [hit.chunk.id for hit in merged] == ["c1", "c2", "c3"]


def test_scores_are_normalized_to_the_unit_interval_with_top_one(settings: Settings) -> None:
    merged = _retriever(settings)._rrf_merge([[_hit("a"), _hit("b")], [_hit("a"), _hit("c")]], 60)
    assert max(hit.score for hit in merged) == 1.0
    assert all(0.0 < hit.score <= 1.0 for hit in merged)


def test_ties_break_by_chunk_id(settings: Settings) -> None:
    merged = _retriever(settings)._rrf_merge([[_hit("b")], [_hit("a")]], 60)
    assert [hit.chunk.id for hit in merged] == ["a", "b"]


def test_single_list_preserves_positional_order(settings: Settings) -> None:
    merged = _retriever(settings)._rrf_merge([[_hit("z", score=0.1), _hit("y", score=0.9), _hit("x", score=0.5)]], 60)
    assert [hit.chunk.id for hit in merged] == ["z", "y", "x"]


def test_single_candidate_normalizes_without_dividing_by_zero(settings: Settings) -> None:
    merged = _retriever(settings)._rrf_merge([[_hit("only")]], 60)
    assert len(merged) == 1
    assert merged[0].score == 1.0


def test_uniform_scores_do_not_divide_by_zero(settings: Settings) -> None:
    merged = _retriever(settings)._rrf_merge([[_hit("a")], [_hit("b")]], 60)
    assert {hit.chunk.id: hit.score for hit in merged} == {"a": 1.0, "b": 1.0}


def test_empty_input_returns_empty(settings: Settings) -> None:
    assert _retriever(settings)._rrf_merge([], 60) == []
    assert _retriever(settings)._rrf_merge([[], []], 60) == []
