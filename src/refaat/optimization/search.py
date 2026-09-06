from __future__ import annotations

from collections.abc import Sequence
from itertools import product

from refaat.programs.program import Tunables


def retrieval_search_space(
    *,
    top_k: Sequence[int] = (4, 8, 12, 16),
    token_budget: Sequence[int | None] = (None,),
    minimum_citation_coverage: Sequence[float] = (0.0,),
) -> list[Tunables]:
    return [
        Tunables(top_k=size, token_budget=budget, minimum_citation_coverage=coverage)
        for size, budget, coverage in product(top_k, token_budget, minimum_citation_coverage)
    ]


def default_search_space() -> list[Tunables]:
    return retrieval_search_space()


def context_packing_search_space(
    *,
    top_k: Sequence[int] = (4, 8),
    token_budget: Sequence[int | None] = (256, 512, 1024, None),
) -> list[Tunables]:
    """Search token budgets and top_k to find the cheapest context that preserves quality."""
    return retrieval_search_space(top_k=top_k, token_budget=token_budget)


def fusion_search_space(
    *,
    fusion: Sequence[str] = ("rrf", "linear"),
    rrf_k: Sequence[int] = (10, 60),
    top_k: Sequence[int] = (8,),
) -> list[Tunables]:
    """Search the fusion strategy and RRF constant to compare ranking regimes."""
    return [
        Tunables(top_k=size, fusion=strategy, rrf_k=constant)
        for size, strategy, constant in product(top_k, fusion, rrf_k)
    ]
