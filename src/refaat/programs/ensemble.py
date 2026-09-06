from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from refaat.domain.authz import Principal
from refaat.domain.enums import SideEffectClass
from refaat.domain.models import JsonObject
from refaat.errors import ValidationError
from refaat.programs.policy import ResultScore, default_result_score, select_best
from refaat.programs.program import Program, ProgramResult

if TYPE_CHECKING:
    from refaat.application.runtime import RefaatProject

BEST = "best"
MAJORITY = "majority"
_POLICIES = (BEST, MAJORITY)


def _context_tokens(result: ProgramResult) -> float:
    value = result.usage.get("context_tokens", 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


class EnsembleProgram(Program):
    """Compose several member programs and combine their results deterministically.

    Each member runs on the same inputs. The ``best`` policy selects via the
    shared ``select_best`` (grounded first, then ``default_result_score``); the
    ``majority`` policy picks the result whose citation set is most common, ties
    broken by ``select_best``. Members are never mutated, and the member artifact
    ids and the policy are pinned into the ensemble's content-addressed artifact.
    The ensemble usage sums each member's context tokens so evaluation reflects
    the ensemble's cost rather than a single member's.
    """

    def __init__(
        self,
        members: Sequence[Program],
        *,
        policy: str = BEST,
        score: ResultScore = default_result_score,
    ) -> None:
        if not members:
            raise ValidationError("EnsembleProgram requires at least one member")
        if policy not in _POLICIES:
            raise ValidationError(f"Unknown ensemble policy: {policy!r}")
        super().__init__()
        self._members = tuple(members)
        self._policy = policy
        self._score = score
        first = self._members[0]
        self.name = "ensemble"
        self.input_signature = first.input_signature
        self.output_signature = first.output_signature

    def declared_side_effect(self) -> SideEffectClass:
        return max(
            (member.declared_side_effect() for member in self._members),
            key=lambda side_effect: side_effect.rank,
            default=SideEffectClass.READ_ONLY,
        )

    @property
    def shadow_safe(self) -> bool:
        # An ensemble runs every member, so it is shadow-safe only when all of them are.
        return all(member.shadow_safe for member in self._members)

    def steps(self) -> tuple[str, ...]:
        return ("ensemble", self._policy)

    def _definition_extra(self) -> JsonObject:
        return {
            "ensemble": {
                "policy": self._policy,
                "members": [member.artifact().artifact_id for member in self._members],
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
        results = [
            member.execute(project, inputs=inputs, user_id=user_id, session_id=session_id, principal=principal)
            for member in self._members
        ]
        chosen = self._combine(results)
        member_tokens = sum(_context_tokens(result) for result in results)
        return replace(
            chosen,
            program=self.name,
            artifact_id=self.artifact().artifact_id,
            outputs={
                **chosen.outputs,
                "ensemble": {
                    "policy": self._policy,
                    "members": [result.program for result in results],
                    "selected": chosen.program,
                },
            },
            usage={
                **chosen.usage,
                "ensemble_member_tokens": member_tokens,
                "ensemble_members": len(results),
            },
        )

    def stream(
        self,
        project: RefaatProject,
        *,
        inputs: Mapping[str, Any],
        user_id: str | None,
        session_id: str | None,
        principal: Principal | None = None,
    ) -> Iterator[dict[str, Any]]:
        # No single request reproduces a selection over several member results, so
        # refuse streaming rather than answer as one bare attempt with the wrong cache identity.
        raise ValidationError("EnsembleProgram does not support streaming; call run() for the combined result")

    def _combine(self, results: Sequence[ProgramResult]) -> ProgramResult:
        if self._policy == MAJORITY:
            return self._majority(results)
        return select_best(results, score=self._score)

    @staticmethod
    def _majority(results: Sequence[ProgramResult]) -> ProgramResult:
        counts: Counter[frozenset[str]] = Counter(
            frozenset(citation.chunk_id for citation in result.citations) for result in results
        )
        winner = counts.most_common(1)[0][0]
        agreeing = [result for result in results if frozenset(c.chunk_id for c in result.citations) == winner]
        return select_best(agreeing)
