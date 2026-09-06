from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from refaat.core.guidance import sanitize_demonstration_answer
from refaat.domain.models import Demonstration
from refaat.evaluation.bundle import MetricBundle, default_bundle
from refaat.evaluation.dataset import Dataset, EvaluationCase
from refaat.optimization.candidates import OptimizationResult, build_ranked_result_budgeted
from refaat.programs.program import Program

if TYPE_CHECKING:
    from refaat.application.runtime import RefaatProject


def select_demonstrations(cases: Sequence[EvaluationCase], k: int) -> tuple[Demonstration, ...]:
    """Pick the first ``k`` labeled cases, id-sorted, as sanitized demonstrations.

    Only cases carrying a ``reference_answer`` are eligible, so unlabeled cases
    are skipped. Selection is id-sorted for determinism, and every answer is run
    through :func:`sanitize_demonstration_answer` so a real corpus chunk id can
    never be carried from a label into a provider prompt.
    """
    labeled = sorted((case for case in cases if case.reference_answer is not None), key=lambda case: case.id)
    return tuple(
        Demonstration(
            question=case.query,
            answer=sanitize_demonstration_answer(str(case.reference_answer)),
        )
        for case in labeled[:k]
    )


class LabeledFewShotOptimizer:
    """Deterministic, model-free selection of K labeled demonstrations.

    Each candidate pins a different shot count into the program's tunables and is
    scored on a held-out evaluation set: every case eligible to be a demonstration
    is excluded from scoring, so a demonstration is never silently reused as its
    own test. The active program is never mutated. Ranking, the score, and the
    Pareto front reuse the shared build-and-rank helper. The offline
    ExtractiveGenerator does not itself consume guidance, so the shared helper
    folds a deterministic ``render_guidance`` token estimate into each candidate's
    ``mean_tokens``: the Pareto token axis is load-bearing here on any backend,
    separating higher-shot candidates by their real prompt cost.
    """

    def __init__(self, project: RefaatProject) -> None:
        self._project = project

    def optimize(
        self,
        program: Program,
        dataset: Dataset,
        *,
        bundle: MetricBundle | None = None,
        shots: Sequence[int] = (1, 2, 4),
        instructions: str | None = None,
        budget: int | None = None,
    ) -> OptimizationResult:
        active_bundle = bundle or default_bundle()
        labeled = sorted(
            (case for case in dataset.cases if case.reference_answer is not None),
            key=lambda case: case.id,
        )
        if not labeled:
            raise ValueError("LabeledFewShot optimization requires at least one labeled case")
        shot_counts = [count for count in shots if 0 < count <= len(labeled)]
        if budget is not None:
            shot_counts = shot_counts[:budget]
        if not shot_counts:
            raise ValueError("No usable shot counts for the labeled case count")
        max_shots = max(shot_counts)
        demonstration_ids = {case.id for case in labeled[:max_shots]}
        held_out = tuple(case for case in dataset.cases if case.id not in demonstration_ids)
        if not held_out:
            raise ValueError("No held-out evaluation cases remain after excluding demonstrations")
        eval_dataset = Dataset(name=f"{dataset.name}:held-out", cases=held_out)
        tunables_space = [
            replace(program.tunables, instructions=instructions, demonstrations=select_demonstrations(labeled, count))
            for count in shot_counts
        ]
        return build_ranked_result_budgeted(self._project, program, tunables_space, eval_dataset, active_bundle)
