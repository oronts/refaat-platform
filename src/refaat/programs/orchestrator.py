from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from refaat.domain.authz import Principal
from refaat.domain.enums import SideEffectClass
from refaat.domain.models import JsonObject
from refaat.errors import ValidationError
from refaat.programs.ensemble import _context_tokens
from refaat.programs.policy import ResultScore, default_result_score, select_best
from refaat.programs.program import Program, ProgramResult

if TYPE_CHECKING:
    from refaat.application.runtime import RefaatProject

SELECT_BEST = "select_best"


class Reducer(Protocol):
    """Typed injection seam for merging member results into one.

    A reducer carries a stable ``name`` so distinct reducers pin distinct
    orchestrator artifact ids, and its ``reduce`` must return one of the input
    results or a result whose citations are drawn only from the member results,
    so the citation contract is never widened.
    """

    @property
    def name(self) -> str: ...

    def reduce(self, results: Sequence[ProgramResult]) -> ProgramResult: ...


class TaskRouter:
    """Deterministic dispatcher from specialist names to keyword tuples.

    Routes are held in sorted-key order, so iteration, matching, and the
    no-match fallback are reproducible regardless of caller dict insertion
    order. ``route`` returns every specialist whose keyword substring-matches
    the lowercased task, in sorted order, and falls back to the full sorted
    specialist set when nothing matches, so a merge is always possible.
    """

    def __init__(self, routes: Mapping[str, Sequence[str]]) -> None:
        if not routes:
            raise ValidationError("TaskRouter requires at least one specialist")
        self._routes: dict[str, tuple[str, ...]] = {name: tuple(routes[name]) for name in sorted(routes)}

    def names(self) -> tuple[str, ...]:
        return tuple(self._routes)

    def route(self, task: str) -> tuple[str, ...]:
        lowered = task.lower()
        matched = tuple(
            name for name, keywords in self._routes.items() if any(keyword.lower() in lowered for keyword in keywords)
        )
        return matched if matched else tuple(self._routes)

    def as_json(self) -> JsonObject:
        return {name: list(keywords) for name, keywords in self._routes.items()}


class MultiAgentOrchestrator(Program):
    """Route a task to specialist member programs and merge their results.

    The router selects a subset of members in sorted-key order, each selected
    member answers through its own ``project.ask`` path so retrieval, caching,
    and the citation contract are unchanged and no answer-cache identity shifts,
    and the results are merged by an injected :class:`Reducer` or, when none is
    given, by the shipped ``select_best`` (grounded first, then the result
    score, earliest index winning ties). Members are never mutated. The member
    artifact ids, the router keywords, a stable reducer identity, and the handoff
    flag are pinned into the content-addressed artifact; handoff creates durable
    agents and messages, a real behavioral difference, so it is part of the id and
    a handoff orchestrator is a distinct artifact from a no-handoff one. The
    orchestrator usage sums each member's context tokens.
    """

    def __init__(
        self,
        members: Mapping[str, Program],
        router: TaskRouter,
        *,
        reducer: Reducer | None = None,
        handoff: bool = False,
        score: ResultScore = default_result_score,
    ) -> None:
        if not members:
            raise ValidationError("MultiAgentOrchestrator requires at least one member")
        member_names = set(members)
        unknown = tuple(sorted(name for name in router.names() if name not in member_names))
        if unknown:
            raise ValidationError(f"Router references unknown specialists: {', '.join(unknown)}")
        super().__init__()
        self._members = {name: members[name] for name in sorted(members)}
        self._router = router
        self._reducer = reducer
        self._handoff = handoff
        self._score = score
        self._reducer_kind = SELECT_BEST if reducer is None else reducer.name
        if reducer is not None and reducer.name == SELECT_BEST:
            raise ValidationError(f"Reducer name {SELECT_BEST!r} is reserved for the default reducer")
        first = next(iter(self._members.values()))
        self._require_homogeneous_signatures(first)
        self.name = "multi-agent"
        self.input_signature = first.input_signature
        self.output_signature = first.output_signature

    def _require_homogeneous_signatures(self, first: Program) -> None:
        input_hash = first.input_signature.signature_hash()
        output_hash = first.output_signature.signature_hash()
        for member in self._members.values():
            if member.input_signature.signature_hash() != input_hash:
                raise ValidationError("MultiAgentOrchestrator members must share an input signature")
            if member.output_signature.signature_hash() != output_hash:
                raise ValidationError("MultiAgentOrchestrator members must share an output signature")

    def steps(self) -> tuple[str, ...]:
        return ("route", "dispatch", "merge")

    @property
    def shadow_safe(self) -> bool:
        """False when handoffs are enabled or any member is not shadow-safe.

        A handoff creates orchestrator and specialist agents and sends messages,
        durable rows a write-suppression scope cannot neutralize, so a canary of a
        handoff orchestrator is refused. Without handoff the members answer directly,
        so the orchestrator is shadow-safe only when every member it dispatches is.
        """
        return not self._handoff and all(member.shadow_safe for member in self._members.values())

    def declared_side_effect(self) -> SideEffectClass:
        # A handoff creates durable agents and sends messages, a write side effect independent of the
        # members, so a handoff orchestrator is at least a write even over purely read-only members.
        effects = [member.declared_side_effect() for member in self._members.values()]
        if self._handoff:
            effects.append(SideEffectClass.WRITE)
        return max(effects, key=lambda side_effect: side_effect.rank, default=SideEffectClass.READ_ONLY)

    def _definition_extra(self) -> JsonObject:
        return {
            "orchestrator": {
                "members": [
                    {"name": name, "artifact_id": member.artifact().artifact_id}
                    for name, member in self._members.items()
                ],
                "router": self._router.as_json(),
                "reducer": self._reducer_kind,
                "score": getattr(self._score, "name", None) or getattr(self._score, "__qualname__", "custom"),
                # Handoff creates durable orchestrator and specialist agents and sends messages, a real
                # behavioral difference, so it is part of the artifact identity: the registry, a
                # promotion, an approval, or a rollback must never treat a handoff and a no-handoff
                # orchestrator as the same artifact.
                "handoff": self._handoff,
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
        task = self._task_text(inputs)
        selected = self._router.route(task)
        handoff = (
            self._record_handoff(project, task=task, selected=selected, principal=principal) if self._handoff else None
        )
        results = [
            self._members[name].execute(
                project, inputs=inputs, user_id=user_id, session_id=session_id, principal=principal
            )
            for name in selected
        ]
        chosen = self._reduce(results)
        member_tokens = sum(_context_tokens(result) for result in results)
        orchestration: JsonObject = {
            "selected": list(selected),
            "members": [result.program for result in results],
            "reducer": self._reducer_kind,
            "chosen": chosen.program,
        }
        if handoff is not None:
            orchestration["handoff"] = handoff
        return replace(
            chosen,
            program=self.name,
            artifact_id=self.artifact().artifact_id,
            outputs={
                **chosen.outputs,
                "orchestrator": orchestration,
            },
            usage={
                **chosen.usage,
                "orchestrator_member_tokens": member_tokens,
                "orchestrator_members": len(results),
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
        # No single request reproduces routing plus a reducer over several member
        # results, so refuse streaming rather than answer as one bare attempt.
        raise ValidationError("MultiAgentOrchestrator does not support streaming; call run() for the merged result")

    def _task_text(self, inputs: Mapping[str, Any]) -> str:
        fields = self.input_signature.input_fields()
        key = fields[0].name if fields else "question"
        value = inputs.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValidationError(f"Program input {key!r} is required and must be a non-empty string")
        return value

    def _reduce(self, results: Sequence[ProgramResult]) -> ProgramResult:
        if self._reducer is None:
            return select_best(results, score=self._score)
        # Snapshot the members and their allowed citations BEFORE the reducer runs, and pass
        # an immutable tuple, so a reducer cannot widen the union it is validated against.
        # Compare whole Citation identity, not just the chunk id, so a fabricated citation
        # reusing a member chunk id cannot pass.
        snapshot = tuple(results)
        allowed = [citation for result in snapshot for citation in result.citations]
        chosen = self._reducer.reduce(snapshot)
        outside = [citation for citation in chosen.citations if citation not in allowed]
        if outside:
            raise ValidationError(
                "Reducer produced citations outside the member results: "
                + ", ".join(sorted(citation.chunk_id for citation in outside))
            )
        return chosen

    def _record_handoff(
        self, project: RefaatProject, *, task: str, selected: Sequence[str], principal: Principal | None = None
    ) -> JsonObject:
        agents = project.runtime.agents
        workspace_id = project.workspace_id
        correlation_id = self._correlation_id(workspace_id, task, selected)
        # Agent names are unique per workspace, so each dispatch gets a fresh nonce; the
        # correlation id stays deterministic because it is derived from the task, not the run.
        nonce = uuid4().hex[:12]
        orchestrator_agent = agents.create(
            workspace_id, f"{self.name}:{nonce}", "orchestrator", goal=task, principal=principal
        )
        specialists: dict[str, str] = {}
        for name in selected:
            specialist_agent = agents.create(
                workspace_id, f"{name}:{nonce}", "specialist", parent_agent_id=orchestrator_agent, principal=principal
            )
            agents.send(
                workspace_id,
                orchestrator_agent,
                specialist_agent,
                "handoff",
                {"task": task, "specialist": name},
                correlation_id=correlation_id,
                principal=principal,
            )
            specialists[name] = specialist_agent
        return {
            "orchestrator_agent": orchestrator_agent,
            "specialists": specialists,
            "correlation_id": correlation_id,
        }

    def _correlation_id(self, workspace_id: str, task: str, selected: Sequence[str]) -> str:
        payload = json.dumps(
            {
                "workspace": workspace_id,
                "orchestrator": self.artifact().artifact_id,
                "task": task,
                "selected": list(selected),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return "corr:" + hashlib.sha256(payload.encode()).hexdigest()[:32]
