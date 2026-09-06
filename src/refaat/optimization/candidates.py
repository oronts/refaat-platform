from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from refaat.application.budget import active_meter, budget_scope, prices_from
from refaat.config import _priced
from refaat.core.guidance import render_guidance
from refaat.core.text import tokens
from refaat.domain.models import JsonObject
from refaat.evaluation.runner import ProgramEvaluator
from refaat.programs.registry import BoundProgram

if TYPE_CHECKING:
    from refaat.application.runtime import RefaatProject
    from refaat.evaluation.bundle import MetricBundle
    from refaat.evaluation.dataset import Dataset
    from refaat.programs.program import Program, Tunables


class _Objective(Protocol):
    @property
    def hard_constraints_passed(self) -> bool: ...

    @property
    def score(self) -> float: ...

    @property
    def mean_tokens(self) -> float: ...


def pareto_front[T: _Objective](candidates: Sequence[T]) -> tuple[T, ...]:
    """Compliant candidates not dominated by another on both answer score and token cost.

    A candidate is dominated when another compliant candidate is at least as good on both
    objectives (higher-or-equal score, lower-or-equal mean tokens) and strictly better on one.
    """

    compliant = [candidate for candidate in candidates if candidate.hard_constraints_passed]
    front: list[T] = []
    for candidate in compliant:
        dominated = any(
            other is not candidate
            and other.score >= candidate.score
            and other.mean_tokens <= candidate.mean_tokens
            and (other.score > candidate.score or other.mean_tokens < candidate.mean_tokens)
            for other in compliant
        )
        if not dominated:
            front.append(candidate)
    return tuple(front)


@dataclass(frozen=True, slots=True)
class Candidate:
    tunables: JsonObject
    artifact_id: str
    hard_constraints_passed: bool
    score: float
    aggregate: dict[str, float]
    mean_tokens: float
    definition_hash: str | None = None
    immutable_policy_hash: str | None = None

    def to_json(self) -> JsonObject:
        payload: JsonObject = {
            "tunables": self.tunables,
            "artifact_id": self.artifact_id,
            "hard_constraints_passed": self.hard_constraints_passed,
            "score": self.score,
            "aggregate": self.aggregate,
            "mean_tokens": self.mean_tokens,
        }
        # Emitted only when set so a bare Candidate (no rehydrated variant behind
        # it) serializes byte-identically to before this field existed.
        if self.definition_hash is not None:
            payload["definition_hash"] = self.definition_hash
        if self.immutable_policy_hash is not None:
            payload["immutable_policy_hash"] = self.immutable_policy_hash
        return payload

    @classmethod
    def from_json(cls, payload: JsonObject) -> Candidate:
        definition_hash = payload.get("definition_hash")
        immutable_policy_hash = payload.get("immutable_policy_hash")
        return cls(
            tunables=payload["tunables"],
            artifact_id=str(payload["artifact_id"]),
            hard_constraints_passed=bool(payload["hard_constraints_passed"]),
            score=float(payload["score"]),
            aggregate={str(key): float(value) for key, value in payload["aggregate"].items()},
            mean_tokens=float(payload["mean_tokens"]),
            definition_hash=str(definition_hash) if definition_hash is not None else None,
            immutable_policy_hash=str(immutable_policy_hash) if immutable_policy_hash is not None else None,
        )


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    program: str
    dataset_version: str
    candidate_count: int
    baseline: Candidate
    best: Candidate
    improvement: float
    candidates: tuple[Candidate, ...]
    pareto_front: tuple[Candidate, ...] = ()

    def to_json(self) -> JsonObject:
        return {
            "program": self.program,
            "dataset_version": self.dataset_version,
            "candidate_count": self.candidate_count,
            "improvement": self.improvement,
            "baseline": self.baseline.to_json(),
            "best": self.best.to_json(),
            "candidates": [candidate.to_json() for candidate in self.candidates],
            "pareto_front": [candidate.to_json() for candidate in self.pareto_front],
        }


def _guidance_token_estimate(tunables: Tunables) -> int:
    """Deterministic prompt-token cost of a variant's rendered guidance block.

    Returns 0 for a guidance-free variant so unguided candidates (every
    ``RetrievalOptimizer`` candidate) keep their pure retrieval token cost. When
    a variant carries instructions or demonstrations, the estimate is the token
    count of the actual :func:`render_guidance` block that a guidance-consuming
    backend would prepend, so the Pareto token axis genuinely separates
    LabeledFewShot candidates by shot count on any backend, including the offline
    ExtractiveGenerator that does not itself consume the block.
    """
    if not tunables.instructions and not tunables.demonstrations:
        return 0
    return len(tokens(render_guidance(tunables.instructions, tunables.demonstrations)))


def evaluate_candidate(
    project: RefaatProject,
    program: Program,
    tunables: Tunables,
    dataset: Dataset,
    bundle: MetricBundle,
) -> Candidate:
    """Evaluate one immutable variant and score it into a :class:`Candidate`.

    The single per-variant body, extracted so both the synchronous
    :func:`build_ranked_result` loop and the durable, resumable runner evaluate a
    candidate through exactly the same path. The active program is never mutated:
    the variant is an immutable ``with_tunables`` clone.
    """
    variant = program.with_tunables(tunables)
    report = ProgramEvaluator(BoundProgram(project, variant)).evaluate(dataset, bundle)
    artifact = variant.artifact()
    return Candidate(
        tunables=tunables.as_json(),
        artifact_id=artifact.artifact_id,
        hard_constraints_passed=report.hard_constraints_passed,
        score=report.score,
        aggregate=report.aggregate,
        mean_tokens=report.mean_tokens + _guidance_token_estimate(tunables),
        definition_hash=artifact.definition_hash,
        immutable_policy_hash=artifact.immutable_policy_hash,
    )


def assemble_result(
    program_name: str,
    dataset_version: str,
    candidates: Sequence[Candidate],
) -> OptimizationResult:
    """Rank the evaluated candidates and assemble the final result.

    The single ranking-and-assembly body: hard constraints first, then score,
    then lower measured token cost. Extracted so the synchronous path and the
    durable runner reassemble a byte-identical result from the same ordered
    candidate set.
    """
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            candidate.hard_constraints_passed,
            candidate.score,
            -candidate.mean_tokens,
        ),
        reverse=True,
    )
    baseline = candidates[0]
    best = ranked[0]
    return OptimizationResult(
        program=program_name,
        dataset_version=dataset_version,
        candidate_count=len(candidates),
        baseline=baseline,
        best=best,
        improvement=best.score - baseline.score,
        candidates=tuple(ranked),
        pareto_front=pareto_front(candidates),
    )


def build_ranked_result(
    project: RefaatProject,
    program: Program,
    tunables_space: Sequence[Tunables],
    dataset: Dataset,
    bundle: MetricBundle,
) -> OptimizationResult:
    """Evaluate each tunable variant, rank the candidates, and assemble a result.

    This is the single build-and-rank body shared by every native optimizer so
    the code-audit duplicate-body guarantee is structural, not incidental to the
    differing search-space code around it. The active program is never mutated:
    each variant is an immutable ``with_tunables`` clone, and the ranking key is
    hard constraints first, then score, then lower measured token cost.
    """
    if not tunables_space:
        raise ValueError("Optimization search space is empty")
    candidates: list[Candidate] = []
    for tunables in tunables_space:
        meter = active_meter()
        # Stop before evaluating the next candidate once the meter is exhausted; the
        # first candidate always runs so a search never returns nothing.
        if meter is not None and candidates and meter.exhausted():
            break
        candidates.append(evaluate_candidate(project, program, tunables, dataset, bundle))
    return assemble_result(program.name, dataset.version, candidates)


def build_ranked_result_budgeted(
    project: RefaatProject,
    program: Program,
    tunables_space: Sequence[Tunables],
    dataset: Dataset,
    bundle: MetricBundle,
    *,
    token_budget: int | None = None,
    cost_budget_usd: float | None = None,
) -> OptimizationResult:
    """Build-and-rank under a run budget scope, so every native optimizer's candidate
    loop is bounded the same way.

    The scope shares an already-active meter, so an optimizer invoked inside another
    budgeted run does not reset the ceiling. Explicit ``token_budget`` and
    ``cost_budget_usd`` overrides fall back to the configured run budget.
    """
    settings = project.runtime.settings
    # A call-time cost budget bypasses the startup guard, so fail closed here: an
    # empty or output-less price map cannot price the active generation model and
    # the cost meter would never trip.
    if cost_budget_usd is not None and not _priced(
        settings.model_prices.get(settings.generation_model), require_output=True
    ):
        raise ValueError(
            "A call-time optimizer cost budget requires REFAAT_MODEL_PRICES to price the active generation "
            f"model {settings.generation_model!r} with positive input and output rates"
        )
    token_ceiling = token_budget if token_budget is not None else settings.run_token_budget
    cost_ceiling = cost_budget_usd if cost_budget_usd is not None else settings.run_cost_budget_usd
    with budget_scope(token_ceiling, cost_ceiling, prices=prices_from(settings.model_prices)):
        return build_ranked_result(project, program, tunables_space, dataset, bundle)
