from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from refaat.core.text import tokens
from refaat.domain.models import SearchHit
from refaat.errors import DependencyUnavailableError
from refaat.ports import CrossEncoderScorer


class LexicalReranker:
    """Deterministic, offline reranker that refines fused candidates by query-term coverage.

    It rescreens the fused ranking without a model, so it runs in CI. A model-based reranker,
    for example a cross-encoder or a hosted rerank endpoint, replaces it by implementing the
    Reranker port and being injected through RuntimeBuilder.with_reranker.
    """

    def __init__(self, weight: float = 0.5) -> None:
        self.weight = weight

    @property
    def cache_id(self) -> str:
        return f"lexical:{self.weight}"

    def rerank(self, query: str, hits: Sequence[SearchHit]) -> tuple[SearchHit, ...]:
        query_terms = set(tokens(query))
        if not query_terms:
            return tuple(hits)
        rescored: list[SearchHit] = []
        for hit in hits:
            coverage = len(query_terms & set(tokens(hit.chunk.text))) / len(query_terms)
            rescored.append(replace(hit, score=hit.score + self.weight * coverage))
        return tuple(sorted(rescored, key=lambda hit: (-hit.score, hit.chunk.id)))


class CrossEncoderReranker:
    """Model-backed reranker that rescreens fused candidates with an injected cross-encoder.

    The cross-encoder model library never enters core: the scorer is supplied through the
    :class:`CrossEncoderScorer` seam and injected via RuntimeBuilder.with_cross_encoder_scorer
    or with_reranker. Only real retrieved candidates are reordered, so citations still resolve
    to stored chunks. A scorer failure or an empty candidate set degrades to the input order.
    """

    def __init__(self, scorer: CrossEncoderScorer) -> None:
        self.scorer = scorer

    @property
    def cache_id(self) -> str:
        return f"cross_encoder:{self.scorer.model_id}"

    def rerank(self, query: str, hits: Sequence[SearchHit]) -> tuple[SearchHit, ...]:
        if not hits:
            return tuple(hits)
        try:
            scores = self.scorer.score(query, [hit.chunk.text for hit in hits])
        except DependencyUnavailableError:
            return tuple(hits)
        if len(scores) != len(hits):
            return tuple(hits)
        rescored = [replace(hit, score=float(score)) for hit, score in zip(hits, scores, strict=True)]
        return tuple(sorted(rescored, key=lambda hit: (-hit.score, hit.chunk.id)))
