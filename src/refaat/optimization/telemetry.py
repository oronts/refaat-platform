from __future__ import annotations

from dataclasses import dataclass

from refaat.domain.models import JsonObject
from refaat.programs.program import ProgramResult


@dataclass(frozen=True, slots=True)
class ShadowComparison:
    """Deterministic offline comparison of an active vs candidate program run.

    Every signal is a pure function of the two :class:`ProgramResult` values, so
    the comparison is reproducible with no model, network, or wall-clock input.
    Latency is intentionally excluded so the verdict stays deterministic.
    """

    answers_equal: bool
    citation_jaccard: float
    citation_overlap: int
    active_grounded: bool
    candidate_grounded: bool
    routes_equal: bool
    active_context_tokens: int
    candidate_context_tokens: int
    token_delta: int

    def to_json(self) -> JsonObject:
        return {
            "answers_equal": self.answers_equal,
            "citation_jaccard": self.citation_jaccard,
            "citation_overlap": self.citation_overlap,
            "active_grounded": self.active_grounded,
            "candidate_grounded": self.candidate_grounded,
            "routes_equal": self.routes_equal,
            "active_context_tokens": self.active_context_tokens,
            "candidate_context_tokens": self.candidate_context_tokens,
            "token_delta": self.token_delta,
        }


def compare_results(active: ProgramResult, candidate: ProgramResult) -> ShadowComparison:
    active_ids = {citation.chunk_id for citation in active.citations}
    candidate_ids = {citation.chunk_id for citation in candidate.citations}
    union = active_ids | candidate_ids
    intersection = active_ids & candidate_ids
    jaccard = len(intersection) / len(union) if union else 1.0
    active_tokens = int(active.usage.get("context_tokens", 0))
    candidate_tokens = int(candidate.usage.get("context_tokens", 0))
    return ShadowComparison(
        answers_equal=active.answer == candidate.answer,
        citation_jaccard=jaccard,
        citation_overlap=len(intersection),
        active_grounded=active.grounded,
        candidate_grounded=candidate.grounded,
        routes_equal=tuple(active.routes) == tuple(candidate.routes),
        active_context_tokens=active_tokens,
        candidate_context_tokens=candidate_tokens,
        token_delta=candidate_tokens - active_tokens,
    )
