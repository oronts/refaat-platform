from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol

from refaat.application.budget import active_meter, budget_scope, prices_from
from refaat.domain.authz import Principal
from refaat.domain.enums import SideEffectClass
from refaat.domain.models import JsonObject
from refaat.errors import ValidationError
from refaat.programs.program import Program, ProgramResult, Validator

if TYPE_CHECKING:
    from refaat.application.runtime import RefaatProject

BEST_OF = "best_of"
REFINE = "refine"
_MODES = (BEST_OF, REFINE)

_GROUNDED_WEIGHT = 1_000_000.0


class ResultScore(Protocol):
    def __call__(self, result: ProgramResult) -> float: ...


def default_result_score(result: ProgramResult) -> float:
    """Deterministic quality score over ``ProgramResult`` fields only.

    Grounding (all validators passing) dominates, then the citation count, then
    a bounded length signal as a deterministic tiebreak. The weight exceeds any
    realistic citation count, so a grounded result always outranks an ungrounded
    one. The function is pure, so identical results score identically, which is
    what makes offline best-of ranking reproducible.
    """
    grounded = _GROUNDED_WEIGHT if result.grounded else 0.0
    citations = float(len(result.citations))
    length_tiebreak = 1.0 - 1.0 / (1.0 + len(result.answer))
    return grounded + citations + length_tiebreak


def select_best(
    results: Sequence[ProgramResult],
    *,
    score: ResultScore = default_result_score,
) -> ProgramResult:
    """Select the best result: grounded first, then score, earliest attempt wins ties.

    The grounded flag is the explicit primary key, so selection prefers a
    grounded result even under a custom ``score`` that ignores grounding. Pure
    and deterministic over the given results.
    """
    if not results:
        raise ValidationError("select_best requires at least one result")
    best_index = 0
    best_key = (results[0].grounded, score(results[0]))
    for index in range(1, len(results)):
        key = (results[index].grounded, score(results[index]))
        if key > best_key:
            best_key = key
            best_index = index
    return results[best_index]


def _aggregate_usage(results: Sequence[ProgramResult], selected: ProgramResult) -> JsonObject:
    """Sum the numeric usage counters across every attempt the loop executed.

    A refine or best-of run spends tokens on each attempt it runs, even when the
    selected attempt itself served from cache with zero usage, so reporting only
    the selected attempt understates the real spend. The non-numeric usage fields
    of the selected result (for example the model id) are preserved, and the
    numeric counters are replaced by the per-attempt sums. Offline the extractive
    generator reports no counters, so the sum stays empty and the result is
    unchanged.
    """
    total: dict[str, Any] = dict(selected.usage or {})
    summed: dict[str, Any] = {}
    for result in results:
        for key, value in (result.usage or {}).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            summed[key] = summed.get(key, 0) + value
    total.update(summed)
    return total


class BoundedInferencePolicy(Program):
    """Bounded multi-attempt wrapper around an inner program.

    ``best_of`` runs exactly ``attempts`` attempts and selects the best result.
    ``refine`` runs until a grounded result appears, up to ``attempts`` attempts,
    then selects the best of what it has. Retrieval, generation, caching, and
    citation logic are untouched: the policy only re-runs the inner program and
    selects among standard results. The mode and attempt cap are folded into the
    content-addressed artifact, so best-of-3 and best-of-5 are distinct,
    reproducible artifacts.

    On the deterministic offline stack repeated attempts are identical, so the
    selection is proven over crafted result sets rather than model variance; the
    value appears once the inner program calls a sampling model.
    """

    def __init__(
        self,
        inner: Program,
        *,
        mode: str = BEST_OF,
        attempts: int = 3,
        score: ResultScore = default_result_score,
    ) -> None:
        if mode not in _MODES:
            raise ValidationError(f"Unknown bounded-inference mode: {mode!r}")
        if attempts < 1:
            raise ValidationError("attempts must be at least 1")
        super().__init__(tunables=inner.tunables)
        self._inner = inner
        self._mode = mode
        self._attempts = attempts
        self._score = score
        self.name = f"{inner.name}+{mode}"
        self.input_signature = inner.input_signature
        self.output_signature = inner.output_signature

    @property
    def shadow_safe(self) -> bool:
        # A best-of or refine loop re-runs the inner program; its side-effect profile is the inner's.
        return self._inner.shadow_safe

    def declared_side_effect(self) -> SideEffectClass:
        return self._inner.declared_side_effect()

    def steps(self) -> tuple[str, ...]:
        return (*self._inner.steps(), self._mode)

    def validators(self) -> tuple[Validator, ...]:
        return (*self._inner.validators(), *self._extra_validators)

    def _definition_extra(self) -> JsonObject:
        return {
            "policy": {
                "kind": "bounded_inference",
                "mode": self._mode,
                "attempts": self._attempts,
                "inner": self._inner.artifact().artifact_id,
            }
        }

    def execute(
        self,
        project: RefaatProject,
        *,
        inputs: Mapping[str, Any],
        user_id: str | None,
        session_id: str | None,
        principal: Principal | None = None,
    ) -> ProgramResult:
        settings = project.runtime.settings
        with budget_scope(
            settings.run_token_budget,
            settings.run_cost_budget_usd,
            prices=prices_from(settings.model_prices),
        ):
            meter = active_meter()
            results: list[ProgramResult] = []
            for _ in range(self._attempts):
                # Enforce at the attempt boundary: once the meter is already
                # exhausted, stop and select from what ran. The first attempt always
                # runs (results is empty), so a run never returns nothing, and the
                # charge-only meter never raises mid-attempt.
                if meter is not None and results and meter.exhausted():
                    break
                result = self._inner.execute(
                    project, inputs=inputs, user_id=user_id, session_id=session_id, principal=principal
                )
                results.append(result)
                if self._mode == REFINE and result.grounded:
                    return replace(result, usage=_aggregate_usage(results, result))
            best = select_best(results, score=self._score)
            return replace(best, usage=_aggregate_usage(results, best))

    def stream(
        self,
        project: RefaatProject,
        *,
        inputs: Mapping[str, Any],
        user_id: str | None,
        session_id: str | None,
        principal: Principal | None = None,
    ) -> Iterator[dict[str, Any]]:
        # No single request reproduces a selection over several attempts, so refuse
        # streaming rather than answer as one bare attempt with the wrong cache identity.
        raise ValidationError("BoundedInferencePolicy does not support streaming; call run() for the selected result")
