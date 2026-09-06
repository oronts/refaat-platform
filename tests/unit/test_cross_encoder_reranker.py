from __future__ import annotations

from collections.abc import Sequence

import pytest

from refaat.application.runtime import RuntimeBuilder
from refaat.config import Settings
from refaat.domain.models import Chunk, SearchHit
from refaat.errors import ConfigurationError, DependencyUnavailableError
from refaat.infrastructure.reranking import CrossEncoderReranker


class _KeywordScorer:
    """Deterministic test double: scores a document by whether it contains the keyword."""

    def __init__(self, keyword: str, model_id: str = "test-cross-encoder-v1") -> None:
        self.keyword = keyword
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        return [1.0 if self.keyword in document else 0.0 for document in documents]


class _FailingScorer:
    @property
    def model_id(self) -> str:
        return "failing-v1"

    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        raise DependencyUnavailableError("cross-encoder backend unavailable")


class _ShortScorer:
    @property
    def model_id(self) -> str:
        return "short-v1"

    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        return [1.0]


def _hit(chunk_id: str, text: str, score: float) -> SearchHit:
    chunk = Chunk(
        id=chunk_id,
        corpus_id="c",
        document_id="d",
        ordinal=0,
        text=text,
        token_count=len(text.split()),
        checksum="x",
        metadata={},
        embedding=(0.0,),
    )
    return SearchHit(chunk=chunk, score=score)


def test_cross_encoder_reranker_reorders_by_scorer() -> None:
    reranker = CrossEncoderReranker(_KeywordScorer("rollback"))
    hits = [
        _hit("a", "unrelated filler content", 0.9),
        _hit("b", "rollback restores the prior artifact", 0.1),
    ]
    ranked = reranker.rerank("how does rollback work", hits)
    assert [hit.chunk.id for hit in ranked] == ["b", "a"]
    assert ranked[0].score == 1.0
    assert ranked[1].score == 0.0


def test_cross_encoder_cache_id_includes_model_id() -> None:
    reranker = CrossEncoderReranker(_KeywordScorer("x", model_id="bge-reranker-large"))
    assert reranker.cache_id == "cross_encoder:bge-reranker-large"


def test_cross_encoder_reranker_empty_candidates() -> None:
    reranker = CrossEncoderReranker(_KeywordScorer("x"))
    assert reranker.rerank("q", ()) == ()


def test_cross_encoder_reranker_degrades_on_scorer_failure() -> None:
    reranker = CrossEncoderReranker(_FailingScorer())
    hits = (_hit("a", "first", 0.9), _hit("b", "second", 0.1))
    assert reranker.rerank("q", hits) == hits


def test_cross_encoder_reranker_degrades_on_length_mismatch() -> None:
    reranker = CrossEncoderReranker(_ShortScorer())
    hits = (_hit("a", "first", 0.9), _hit("b", "second", 0.1))
    assert reranker.rerank("q", hits) == hits


def test_builder_cross_encoder_requires_scorer() -> None:
    with pytest.raises(ConfigurationError):
        RuntimeBuilder(Settings(reranker_provider="cross_encoder"))._reranker()


def test_builder_cross_encoder_uses_injected_scorer() -> None:
    scorer = _KeywordScorer("x")
    reranker = RuntimeBuilder(Settings(reranker_provider="cross_encoder")).with_cross_encoder_scorer(scorer)._reranker()
    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.scorer is scorer


def test_builder_custom_reranker_takes_precedence() -> None:
    custom = CrossEncoderReranker(_KeywordScorer("x"))
    assert RuntimeBuilder(Settings()).with_reranker(custom)._reranker() is custom
