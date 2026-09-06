from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from refaat.evaluation.bundle import MetricBundle, default_bundle
from refaat.evaluation.dataset import Dataset
from refaat.optimization.candidates import OptimizationResult, build_ranked_result_budgeted
from refaat.optimization.search import default_search_space
from refaat.programs.program import Program, Tunables

if TYPE_CHECKING:
    from refaat.application.runtime import RefaatProject


class RetrievalOptimizer:
    """Deterministic, model-free search over a program's tunable parameters.

    Every candidate is evaluated with the same metric bundle production uses, so
    the search never spends model calls on a weak configuration. Optimization
    never mutates the active program: it returns ranked, immutable candidate
    artifacts and a recommendation for explicit promotion.
    """

    def __init__(self, project: RefaatProject) -> None:
        self._project = project

    def optimize(
        self,
        program: Program,
        dataset: Dataset,
        *,
        bundle: MetricBundle | None = None,
        search_space: Sequence[Tunables] | None = None,
        budget: int | None = None,
        token_budget: int | None = None,
        cost_budget_usd: float | None = None,
    ) -> OptimizationResult:
        active_bundle = bundle or default_bundle()
        space = list(search_space) if search_space is not None else default_search_space()
        if budget is not None:
            space = space[:budget]
        return build_ranked_result_budgeted(
            self._project,
            program,
            space,
            dataset,
            active_bundle,
            token_budget=token_budget,
            cost_budget_usd=cost_budget_usd,
        )
