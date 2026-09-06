from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from refaat.core.router import RouterConfig
from refaat.evaluation.bundle import MetricBundle, default_bundle
from refaat.evaluation.dataset import Dataset
from refaat.optimization.candidates import OptimizationResult, build_ranked_result_budgeted
from refaat.programs.program import Program
from refaat.programs.router import RouterProgram

if TYPE_CHECKING:
    from refaat.application.runtime import RefaatProject


def default_router_grid() -> list[RouterConfig]:
    """A small, distinct grid of router threshold configurations to search."""
    return [
        RouterConfig(long_query_tokens=16, multi_clause_threshold=2),
        RouterConfig(long_query_tokens=24, multi_clause_threshold=2),
        RouterConfig(long_query_tokens=32, multi_clause_threshold=3),
    ]


class RouterOptimizer:
    """Deterministic, model-free search over a router's threshold configurations.

    Each grid entry becomes a ``router`` tunable on a clone of the program, so the
    active program, corpus, and graph are never mutated: the candidates are the
    same immutable, content-addressed artifacts every native optimizer produces,
    ranked by the shared :func:`build_ranked_result` body.
    """

    def __init__(self, project: RefaatProject) -> None:
        self._project = project

    def optimize(
        self,
        program: Program,
        dataset: Dataset,
        *,
        bundle: MetricBundle | None = None,
        grid: Sequence[RouterConfig] | None = None,
        budget: int | None = None,
    ) -> OptimizationResult:
        active_bundle = bundle or default_bundle()
        router_program = program if isinstance(program, RouterProgram) else RouterProgram(program)
        configs = list(grid) if grid is not None else default_router_grid()
        base = router_program.tunables
        space = [replace(base, router=config.as_json()) for config in configs]
        if budget is not None:
            space = space[:budget]
        return build_ranked_result_budgeted(self._project, router_program, space, dataset, active_bundle)
