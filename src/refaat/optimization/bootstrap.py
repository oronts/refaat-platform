from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from refaat.application.budget import active_meter, budget_scope, prices_from
from refaat.core.guidance import sanitize_demonstration_answer
from refaat.domain.models import Demonstration
from refaat.errors import ValidationError
from refaat.evaluation.bundle import MetricBundle, default_bundle
from refaat.evaluation.dataset import Dataset, EvaluationCase
from refaat.evaluation.metrics import AnswerContains, Grounded, Metric, TokenF1
from refaat.evaluation.runner import RunnableProgram
from refaat.optimization.candidates import OptimizationResult, build_ranked_result
from refaat.programs.program import Program
from refaat.programs.registry import BoundProgram

if TYPE_CHECKING:
    from refaat.application.runtime import RefaatProject

BOOTSTRAP = "bootstrap"
RANDOM_SEARCH = "random_search"


@dataclass(frozen=True, slots=True)
class BootstrapTrace:
    """One labeled case whose teacher run passed the metric gate.

    ``case_id`` is retained so the source case can be excluded from the held-out
    evaluation set: a bootstrapped demonstration is never scored as its own test.
    """

    case_id: str
    demonstration: Demonstration


def _has_reference(case: EvaluationCase) -> bool:
    return bool(case.reference_answer and case.reference_answer.strip())


def _case_gate(case: EvaluationCase, extra_filters: Sequence[Metric]) -> list[Metric]:
    """Build the per-case metric gate.

    Grounding is always mandatory, so a caller can never admit an ungrounded
    demonstration. The label-aware metrics ensure the teacher answer actually
    matches the case labels: AnswerContains for expected terms and a reference-
    aware TokenF1 for a reference answer, so a grounded but off-reference answer is
    rejected rather than passing AnswerContains vacuously on empty terms. Caller
    filters are additive, never a replacement.
    """
    gate: list[Metric] = [Grounded()]
    if case.expected_terms:
        gate.append(AnswerContains())
    if _has_reference(case):
        gate.append(TokenF1())
    gate.extend(extra_filters)
    return gate


def _bootstrap_traces(
    runner: RunnableProgram,
    cases: Sequence[EvaluationCase],
    extra_filters: Sequence[Metric],
    max_demos: int,
) -> tuple[BootstrapTrace, ...]:
    """Run the teacher over labeled cases, keeping only runs that pass the gate.

    The teacher is a :class:`RunnableProgram`, so a unit test can substitute a
    deterministic fake and drive the gate directly. The gate always requires
    grounding plus a label-aware match (expected terms or a reference answer), so
    an ungrounded or off-label teacher run can never become a demonstration. Every
    kept run is sanitized through :func:`sanitize_demonstration_answer`, so a real
    corpus chunk id in the teacher's question or answer can never reach a prompt.
    """
    traces: list[BootstrapTrace] = []
    ran = False
    for case in sorted(cases, key=lambda case: case.id):
        # A case with no expected terms and no usable reference answer has nothing
        # to gate against, so it can never be a meaningful demonstration.
        if not case.expected_terms and not _has_reference(case):
            continue
        # Stop before running the next teacher case once the shared meter is
        # exhausted; the first teacher run always executes so the loop can produce
        # at least one candidate demonstration before the boundary can bite.
        meter = active_meter()
        if meter is not None and ran and meter.exhausted():
            break
        result = runner.run(question=case.query)
        ran = True
        if not all(metric.evaluate(case, result).passed for metric in _case_gate(case, extra_filters)):
            continue
        demonstration = Demonstration(
            question=sanitize_demonstration_answer(case.query),
            answer=sanitize_demonstration_answer(result.answer),
        )
        traces.append(BootstrapTrace(case_id=case.id, demonstration=demonstration))
        if len(traces) >= max_demos:
            break
    return tuple(traces)


def bootstrap_demonstrations(
    project: RefaatProject,
    program: Program,
    cases: Sequence[EvaluationCase],
    *,
    extra_filters: Sequence[Metric] = (),
    max_demos: int = 4,
) -> tuple[BootstrapTrace, ...]:
    """Bootstrap demonstrations by running ``program`` as a deterministic teacher.

    ``BoundProgram.run`` reuses the exact runtime path :class:`ProgramEvaluator`
    uses, so the teacher answers are produced by the same retrieval, generation,
    caching, and citation pipeline production uses. The mandatory grounding plus
    label-aware gate is built per case; ``extra_filters`` are additive.
    """
    return _bootstrap_traces(BoundProgram(project, program), cases, extra_filters, max_demos)


def random_demonstration_subsets(
    pool: Sequence[Demonstration], *, trials: int, seed: int
) -> tuple[tuple[Demonstration, ...], ...]:
    """Draw deterministic, deduplicated subsets of a demonstration pool.

    Pure and seeded: a fixed ``seed`` reproduces the same subsets. Each trial
    draws a size in ``[1, len(pool)]`` then a sorted index sample, so every
    subset preserves pool order. Identical index tuples are dropped through a
    seen-set, so a run yields at most ``trials`` distinct subsets.
    """
    if not pool:
        return ()
    rng = random.Random(seed)  # noqa: S311 - subset sampling is not a security context
    seen: set[tuple[int, ...]] = set()
    subsets: list[tuple[Demonstration, ...]] = []
    for _ in range(trials):
        size = rng.randint(1, len(pool))
        idx = tuple(sorted(rng.sample(range(len(pool)), size)))
        if idx in seen:
            continue
        seen.add(idx)
        subsets.append(tuple(pool[index] for index in idx))
    return tuple(subsets)


class BootstrapFewShotOptimizer:
    """Model-free BootstrapFewShot: teach demonstrations, then rank by the shared helper.

    It mirrors :class:`LabeledFewShotOptimizer` and differs only in how the
    demonstrations are produced. Instead of copying labeled reference answers, it
    runs the active program as a deterministic teacher over the labeled cases,
    keeps only the runs whose result passes the metric gate, sanitizes them, and
    hands the resulting demonstration set to the same build-and-rank helper. The
    baseline is candidate 0 with unchanged (empty) demonstrations, so the
    zero-shot variant is always in the ranked set and the active program is never
    mutated. ``RANDOM_SEARCH`` evaluates several random subsets of the same pool.
    """

    def __init__(self, project: RefaatProject) -> None:
        self._project = project

    def optimize(
        self,
        program: Program,
        dataset: Dataset,
        *,
        bundle: MetricBundle | None = None,
        extra_filters: Sequence[Metric] = (),
        max_demos: int = 4,
        mode: str = BOOTSTRAP,
        trials: int = 8,
        seed: int = 0,
        budget: int | None = None,
    ) -> OptimizationResult:
        if mode not in {BOOTSTRAP, RANDOM_SEARCH}:
            raise ValidationError(f"Unknown BootstrapFewShot mode: {mode!r}")
        active_bundle = bundle or default_bundle()
        settings = self._project.runtime.settings
        # One shared meter over the teacher phase and the candidate ranking, so a
        # configured run budget bounds the whole optimization, not just the ranking.
        with budget_scope(
            settings.run_token_budget,
            settings.run_cost_budget_usd,
            prices=prices_from(settings.model_prices),
        ):
            traces = bootstrap_demonstrations(
                self._project, program, dataset.cases, extra_filters=extra_filters, max_demos=max_demos
            )
            if not traces:
                raise ValueError("BootstrapFewShot produced no metric-passing demonstrations")
            pool = tuple(trace.demonstration for trace in traces)
            source_ids = {trace.case_id for trace in traces}
            held_out = tuple(case for case in dataset.cases if case.id not in source_ids)
            if not held_out:
                raise ValueError("No held-out evaluation cases remain after excluding bootstrap sources")
            eval_dataset = Dataset(name=f"{dataset.name}:held-out", cases=held_out)
            if mode == BOOTSTRAP:
                tunables_space = [program.tunables, replace(program.tunables, demonstrations=pool)]
            else:
                tunables_space = [
                    program.tunables,
                    *(
                        replace(program.tunables, demonstrations=subset)
                        for subset in random_demonstration_subsets(pool, trials=trials, seed=seed)
                    ),
                ]
            if budget is not None:
                tunables_space = tunables_space[:budget]
            return build_ranked_result(self._project, program, tunables_space, eval_dataset, active_bundle)
