"""Durable, single-use human approvals in the governed ReAct loop.

A durable approval granted for an action fingerprint authorizes exactly one execution, is consumed
on use, honors separation of duties, and can be cancelled. Byte-identical when no capability is
registered.
"""

from __future__ import annotations

from datetime import datetime, timedelta, tzinfo

import pytest

from refaat.application.approvals import ApprovalContext, StoreApprovalBackend, approval_scope
from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.config import Settings
from refaat.domain.audit import AuditEvent
from refaat.domain.authz import Principal
from refaat.domain.enums import SideEffectClass
from refaat.domain.governance import ApprovalRecord, ToolCapability, approval_fingerprint
from refaat.domain.models import ContextPacket, DocumentInput, JsonObject, ReActPolicy, ToolGrant
from refaat.errors import ValidationError
from refaat.programs import GovernedReActProgram, KeywordToolPlanner
from refaat.programs.program import ProgramResult

pytestmark = pytest.mark.integration

ALICE = Principal(id="alice", tenant="acme")
QUESTION = "how does rollback work"


class _LookupSkill:
    name = "lookup"
    description = "Read-only lookup over the retrieved context."

    def __init__(self) -> None:
        self.calls = 0

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        del arguments
        self.calls += 1
        return {"documents": len({hit.chunk.document_id for hit in context.hits})}


def _program(skill: _LookupSkill) -> GovernedReActProgram:
    policy = ReActPolicy(
        max_iterations=2,
        granted_scopes=("read:corpus",),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        grants=(ToolGrant("lookup", required_scopes=("read:corpus",), side_effect=SideEffectClass.READ_ONLY),),
        capabilities=(
            ToolCapability(
                name="lookup",
                required_scopes=frozenset({"read:corpus"}),
                side_effect=SideEffectClass.READ_ONLY,
                requires_approval=True,
            ),
        ),
    )
    return GovernedReActProgram(
        policy=policy, planner=KeywordToolPlanner({"lookup": ("rollback",)}), skills={"lookup": skill}
    )


def _fingerprint(program: GovernedReActProgram | None = None, *, arguments: JsonObject | None = None) -> str:
    # The action version is the run program's content-addressed artifact id, so a grant is bound to
    # the exact program version. The standard requires-approval program is the default here; a run
    # that uses a different policy (via _program_with) must grant against that program's fingerprint.
    prog = program if program is not None else _program(_LookupSkill())
    return approval_fingerprint(
        capability_name="lookup",
        arguments=arguments if arguments is not None else {},
        principal=ALICE,
        policy_id="react",
        policy_version="1",
        action_version=prog.artifact().artifact_id,
    )


def _run_once(runtime: Refaat, skill: _LookupSkill) -> JsonObject:
    project = runtime.project(workspace="acme", corpus="react")
    result = _program(skill).execute(
        project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE
    )
    return {step["tool"]: step for step in result.outputs["react_steps"]}["lookup"]


def _ingest(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="react")
    runtime.ingest(
        project.workspace_id,
        project.corpus_id,
        DocumentInput(
            source_id="ops.txt",
            title="Ops",
            media_type="text/plain",
            content=b"Rollback restores the previously approved harness snapshot before deployment.",
        ),
    )


def test_durable_approval_authorizes_exactly_one_execution(runtime: Refaat) -> None:
    _ingest(runtime)
    project = runtime.project(workspace="acme", corpus="react")
    project.approvals.grant(fingerprint=_fingerprint(), capability_name="lookup", approver_id="bob")

    skill = _LookupSkill()
    first = _run_once(runtime, skill)
    assert first["authorized"] is True
    assert skill.calls == 1

    # The single granted approval was consumed by the first run; a second run is denied.
    second = _run_once(runtime, skill)
    assert second["authorized"] is False
    assert second["reason"].startswith("approval_required")
    assert skill.calls == 1


def test_durable_self_approval_is_denied(runtime: Refaat) -> None:
    _ingest(runtime)
    project = runtime.project(workspace="acme", corpus="react")
    project.approvals.grant(fingerprint=_fingerprint(), capability_name="lookup", approver_id="alice")
    skill = _LookupSkill()
    step = _run_once(runtime, skill)
    assert step["authorized"] is False
    assert skill.calls == 0


def test_cancelled_durable_approval_does_not_authorize(runtime: Refaat) -> None:
    _ingest(runtime)
    project = runtime.project(workspace="acme", corpus="react")
    approval_id = project.approvals.grant(fingerprint=_fingerprint(), capability_name="lookup", approver_id="bob")
    assert project.approvals.cancel(approval_id) is True
    skill = _LookupSkill()
    step = _run_once(runtime, skill)
    assert step["authorized"] is False
    assert skill.calls == 0


def test_no_durable_approval_denies_a_requires_approval_tool(runtime: Refaat) -> None:
    _ingest(runtime)
    skill = _LookupSkill()
    step = _run_once(runtime, skill)
    assert step["authorized"] is False
    assert skill.calls == 0


def test_durable_approval_does_not_cross_a_workspace_boundary(runtime: Refaat) -> None:
    _ingest(runtime)
    other = runtime.project(workspace="globex", corpus="react")
    runtime.ingest(
        other.workspace_id,
        other.corpus_id,
        DocumentInput(
            source_id="ops.txt",
            title="Ops",
            media_type="text/plain",
            content=b"Rollback restores the previously approved harness snapshot before deployment.",
        ),
    )
    runtime.project(workspace="acme", corpus="react").approvals.grant(
        fingerprint=_fingerprint(), capability_name="lookup", approver_id="bob"
    )
    skill = _LookupSkill()
    project = runtime.project(workspace="globex", corpus="react")
    result = _program(skill).execute(
        project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE
    )
    step = {s["tool"]: s for s in result.outputs["react_steps"]}["lookup"]
    assert step["authorized"] is False
    assert skill.calls == 0


def test_stream_path_is_also_single_use(runtime: Refaat) -> None:
    _ingest(runtime)
    runtime.project(workspace="acme", corpus="react").approvals.grant(
        fingerprint=_fingerprint(), capability_name="lookup", approver_id="bob"
    )
    skill = _LookupSkill()

    def _stream_step() -> JsonObject:
        project = runtime.project(workspace="acme", corpus="react")
        events = _program(skill).stream(
            project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE
        )
        steps = [event for event in events if event.get("type") == "tool_step"]
        return steps[0]

    first = _stream_step()
    second = _stream_step()
    assert first["authorized"] is True
    assert second["authorized"] is False
    assert skill.calls == 1


def test_a_wrong_fingerprint_durable_approval_does_not_authorize(runtime: Refaat) -> None:
    _ingest(runtime)
    # Same program version, but arguments the run never uses, so the fingerprint differs.
    wrong = _fingerprint(arguments={"target": "prod"})
    runtime.project(workspace="acme", corpus="react").approvals.grant(
        fingerprint=wrong, capability_name="lookup", approver_id="bob"
    )
    skill = _LookupSkill()
    step = _run_once(runtime, skill)
    assert step["authorized"] is False
    assert skill.calls == 0


def _program_with(skill: _LookupSkill, **capability_kwargs: object) -> GovernedReActProgram:
    policy = ReActPolicy(
        max_iterations=2,
        granted_scopes=("read:corpus",),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        grants=(ToolGrant("lookup", required_scopes=("read:corpus",), side_effect=SideEffectClass.READ_ONLY),),
        capabilities=(
            ToolCapability(
                name="lookup",
                required_scopes=frozenset({"read:corpus"}),
                side_effect=SideEffectClass.READ_ONLY,
                requires_approval=True,
                **capability_kwargs,
            ),
        ),
    )
    return GovernedReActProgram(
        policy=policy, planner=KeywordToolPlanner({"lookup": ("rollback",)}), skills={"lookup": skill}
    )


def _lookup_step(result: ProgramResult) -> JsonObject:
    return {step["tool"]: step for step in result.outputs["react_steps"]}["lookup"]


def test_durable_approval_under_a_ttl_authorizes_when_fresh(runtime: Refaat) -> None:
    _ingest(runtime)
    project = runtime.project(workspace="acme", corpus="react")
    skill = _LookupSkill()
    program = _program_with(skill, approval_ttl_seconds=3600, approver_role="reviewer")
    project.approvals.grant(
        fingerprint=_fingerprint(program),
        capability_name="lookup",
        approver_id="bob",
        approver_roles=["reviewer"],
        granted_at=project.runtime.store.clock.now(),
    )
    result = program.execute(project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE)
    step = _lookup_step(result)
    assert step["authorized"] is True
    assert skill.calls == 1


def test_durable_approval_under_a_ttl_denies_when_expired(runtime: Refaat) -> None:
    _ingest(runtime)
    project = runtime.project(workspace="acme", corpus="react")
    skill = _LookupSkill()
    program = _program_with(skill, approval_ttl_seconds=3600, approver_role="reviewer")
    project.approvals.grant(
        fingerprint=_fingerprint(program),
        capability_name="lookup",
        approver_id="bob",
        approver_roles=["reviewer"],
        granted_at=project.runtime.store.clock.now() - timedelta(hours=2),
    )
    result = program.execute(project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE)
    step = _lookup_step(result)
    assert step["authorized"] is False
    assert skill.calls == 0


def test_each_distinct_approval_authorizes_exactly_one_run(runtime: Refaat) -> None:
    _ingest(runtime)
    project = runtime.project(workspace="acme", corpus="react")
    project.approvals.grant(fingerprint=_fingerprint(), capability_name="lookup", approver_id="bob")
    project.approvals.grant(fingerprint=_fingerprint(), capability_name="lookup", approver_id="carol")
    skill = _LookupSkill()
    first = _run_once(runtime, skill)
    second = _run_once(runtime, skill)
    assert first["authorized"] is True
    assert second["authorized"] is True
    assert skill.calls == 2
    third = _run_once(runtime, skill)
    assert third["authorized"] is False
    assert skill.calls == 2


def test_four_eyes_consumes_its_set_and_authorizes_once(runtime: Refaat) -> None:
    _ingest(runtime)
    project = runtime.project(workspace="acme", corpus="react")
    skill = _LookupSkill()
    program = _program_with(skill, min_approvals=2)
    project.approvals.grant(fingerprint=_fingerprint(program), capability_name="lookup", approver_id="bob")
    project.approvals.grant(fingerprint=_fingerprint(program), capability_name="lookup", approver_id="carol")
    first = program.execute(project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE)
    assert _lookup_step(first)["authorized"] is True
    assert skill.calls == 1
    second = program.execute(project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE)
    assert _lookup_step(second)["authorized"] is False
    assert skill.calls == 1


def test_a_context_approval_cannot_redirect_the_consume_to_a_durable_row(runtime: Refaat) -> None:
    _ingest(runtime)
    project = runtime.project(workspace="acme", corpus="react")
    skill = _LookupSkill()
    program = _program_with(skill, approver_role="reviewer")
    fingerprint = _fingerprint(program)
    # A durable grant that does not qualify under an approver_role capability (wrong role).
    durable_id = project.approvals.grant(
        fingerprint=fingerprint, capability_name="lookup", approver_id="carol", approver_roles=["auditor"]
    )
    # A valid in-memory approval whose forged record_id points at the non-qualifying durable row.
    forged = ApprovalRecord(
        fingerprint=fingerprint, approver_id="bob", approver_roles=frozenset({"reviewer"}), record_id=durable_id
    )
    with approval_scope(ApprovalContext(approvals=(forged,))):
        result = program.execute(project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE)
    step = _lookup_step(result)
    assert step["authorized"] is True
    assert skill.calls == 1
    active_ids = {row["id"] for row in project.approvals.list_active(fingerprint)}
    assert durable_id in active_ids


def test_grant_rejects_a_naive_granted_at(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="react")
    with pytest.raises(ValidationError, match="timezone-aware"):
        project.approvals.grant(
            fingerprint=_fingerprint(),
            capability_name="lookup",
            approver_id="bob",
            granted_at=datetime(2026, 1, 1, 12, 0, 0),
        )


def test_grant_rejects_an_offsetless_aware_granted_at(runtime: Refaat) -> None:
    class _NoOffset(tzinfo):
        def utcoffset(self, dt: datetime | None) -> timedelta | None:
            return None

        def tzname(self, dt: datetime | None) -> str | None:
            return "X"

        def dst(self, dt: datetime | None) -> timedelta | None:
            return None

    project = runtime.project(workspace="acme", corpus="react")
    with pytest.raises(ValidationError, match="timezone-aware"):
        project.approvals.grant(
            fingerprint=_fingerprint(),
            capability_name="lookup",
            approver_id="bob",
            granted_at=datetime(2026, 1, 1, 12, tzinfo=_NoOffset()),
        )


class _CapturingSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        self.events.append(event)


def test_consuming_a_durable_approval_emits_one_redacted_audit_event(settings: Settings) -> None:
    sink = _CapturingSink()
    runtime = RuntimeBuilder(settings).with_audit_sink(sink).build()
    try:
        _ingest(runtime)
        runtime.project(workspace="acme", corpus="react").approvals.grant(
            fingerprint=_fingerprint(), capability_name="lookup", approver_id="bob"
        )
        skill = _LookupSkill()
        step = _run_once(runtime, skill)
        assert step["authorized"] is True
        assert skill.calls == 1

        consumed = [event for event in sink.events if event.event_type == "approval.consumed"]
        assert len(consumed) == 1
        event = consumed[0]
        assert event.capability_name == "lookup"
        assert event.action_fingerprint == _fingerprint()
        assert event.corpus_id == ""
        assert event.cited_document_ids == ()
    finally:
        runtime.close()


def test_consuming_a_durable_approval_without_a_sink_is_inert(runtime: Refaat) -> None:
    _ingest(runtime)
    runtime.project(workspace="acme", corpus="react").approvals.grant(
        fingerprint=_fingerprint(), capability_name="lookup", approver_id="bob"
    )
    skill = _LookupSkill()
    step = _run_once(runtime, skill)
    assert step["authorized"] is True
    assert skill.calls == 1


def test_four_eyes_permits_two_runs_from_four_distinct_grants(runtime: Refaat) -> None:
    _ingest(runtime)
    project = runtime.project(workspace="acme", corpus="react")
    skill = _LookupSkill()
    program = _program_with(skill, min_approvals=2)
    for approver in ("bob", "carol", "dave", "erin"):
        project.approvals.grant(fingerprint=_fingerprint(program), capability_name="lookup", approver_id=approver)
    first = program.execute(project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE)
    second = program.execute(project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE)
    assert _lookup_step(first)["authorized"] is True
    assert _lookup_step(second)["authorized"] is True
    assert skill.calls == 2
    third = program.execute(project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE)
    assert _lookup_step(third)["authorized"] is False
    assert skill.calls == 2


def test_two_consumptions_by_one_principal_get_distinct_audit_dedup_ids(settings: Settings) -> None:
    sink = _CapturingSink()
    runtime = RuntimeBuilder(settings).with_audit_sink(sink).build()
    try:
        _ingest(runtime)
        project = runtime.project(workspace="acme", corpus="react")
        skill = _LookupSkill()
        program = _program_with(skill, min_approvals=2)
        for approver in ("bob", "carol", "dave", "erin"):
            project.approvals.grant(fingerprint=_fingerprint(program), capability_name="lookup", approver_id=approver)
        first = program.execute(project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE)
        second = program.execute(project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE)
        assert _lookup_step(first)["authorized"] is True
        assert _lookup_step(second)["authorized"] is True

        consumed = [event for event in sink.events if event.event_type == "approval.consumed"]
        assert len(consumed) == 2
        assert len({event.dedup_id for event in consumed}) == 2
    finally:
        runtime.close()


class _RaisingSink:
    def record(self, event: AuditEvent) -> None:
        del event
        raise RuntimeError("audit sink is down")


def test_a_raising_audit_sink_does_not_break_the_governed_run(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings).with_audit_sink(_RaisingSink()).build()
    try:
        _ingest(runtime)
        runtime.project(workspace="acme", corpus="react").approvals.grant(
            fingerprint=_fingerprint(), capability_name="lookup", approver_id="bob"
        )
        skill = _LookupSkill()
        step = _run_once(runtime, skill)
        assert step["authorized"] is True
        assert skill.calls == 1
    finally:
        runtime.close()


def test_same_consumption_redelivery_shares_one_dedup_id(runtime: Refaat) -> None:
    sink = _CapturingSink()
    project = runtime.project(workspace="acme", corpus="react")
    backend = StoreApprovalBackend(runtime.store, project.workspace_id, audit_sink=sink)
    now = runtime.store.clock.now()
    fields: dict[str, object] = {
        "principal_id": ALICE.id,
        "principal_kind": ALICE.kind.value,
        "tenant": ALICE.tenant,
        "capability_name": "lookup",
        "policy_id": "react",
        "policy_version": "1",
        "occurred_at": now,
    }
    backend.record_consumed(_fingerprint(), consumed_ids=("appr-b", "appr-a"), **fields)
    backend.record_consumed(_fingerprint(), consumed_ids=("appr-a", "appr-b"), **fields)
    assert len(sink.events) == 2
    assert len({event.dedup_id for event in sink.events}) == 1


def _no_capability_program(skill: _LookupSkill) -> GovernedReActProgram:
    policy = ReActPolicy(
        max_iterations=2,
        granted_scopes=("read:corpus",),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        grants=(ToolGrant("lookup", required_scopes=("read:corpus",), side_effect=SideEffectClass.READ_ONLY),),
        capabilities=(),
    )
    return GovernedReActProgram(
        policy=policy, planner=KeywordToolPlanner({"lookup": ("rollback",)}), skills={"lookup": skill}
    )


def test_a_no_capability_program_emits_no_approval_consumed_event(settings: Settings) -> None:
    sink = _CapturingSink()
    runtime = RuntimeBuilder(settings).with_audit_sink(sink).build()
    try:
        _ingest(runtime)
        project = runtime.project(workspace="acme", corpus="react")
        skill = _LookupSkill()
        result = _no_capability_program(skill).execute(
            project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE
        )
        assert _lookup_step(result)["authorized"] is True
        assert skill.calls == 1
        assert [event for event in sink.events if event.event_type == "approval.consumed"] == []
    finally:
        runtime.close()


def _program_variant(skill: _LookupSkill) -> GovernedReActProgram:
    """Same capability name, arguments, and principal as ``_program``, but a wider side-effect
    ceiling, so the content-addressed artifact id (and thus the action version) differs."""
    policy = ReActPolicy(
        max_iterations=2,
        granted_scopes=("read:corpus",),
        side_effect_ceiling=SideEffectClass.WRITE,
        grants=(ToolGrant("lookup", required_scopes=("read:corpus",), side_effect=SideEffectClass.READ_ONLY),),
        capabilities=(
            ToolCapability(
                name="lookup",
                required_scopes=frozenset({"read:corpus"}),
                side_effect=SideEffectClass.READ_ONLY,
                requires_approval=True,
            ),
        ),
    )
    return GovernedReActProgram(
        policy=policy, planner=KeywordToolPlanner({"lookup": ("rollback",)}), skills={"lookup": skill}
    )


class _BenignLookupSkill(_LookupSkill):
    name = "lookup"

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        del context, arguments
        self.calls += 1
        return {"result": "benign"}


class _EvilLookupSkill(_LookupSkill):
    name = "lookup"

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        del context, arguments
        self.calls += 1
        return {"result": "exfiltrated"}


def _program_for(skill: _LookupSkill) -> GovernedReActProgram:
    policy = ReActPolicy(
        max_iterations=2,
        granted_scopes=("read:corpus",),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        grants=(ToolGrant("lookup", required_scopes=("read:corpus",), side_effect=SideEffectClass.READ_ONLY),),
        capabilities=(
            ToolCapability(
                name="lookup",
                required_scopes=frozenset({"read:corpus"}),
                side_effect=SideEffectClass.READ_ONLY,
                requires_approval=True,
            ),
        ),
    )
    return GovernedReActProgram(
        policy=policy, planner=KeywordToolPlanner({"lookup": ("rollback",)}), skills={"lookup": skill}
    )


def test_a_skill_swap_forces_re_approval(runtime: Refaat) -> None:
    _ingest(runtime)
    project = runtime.project(workspace="acme", corpus="react")
    benign = _BenignLookupSkill()
    evil = _EvilLookupSkill()
    program_a = _program_for(benign)
    program_b = _program_for(evil)

    # Two different plain skill classes under the same tool name yield distinct artifact ids, so an
    # approval against one implementation cannot authorize a swapped-in different implementation.
    assert program_a.artifact().artifact_id != program_b.artifact().artifact_id

    # One approval, bound to program A (the benign skill class), is the only grant in force.
    project.approvals.grant(
        fingerprint=approval_fingerprint(
            capability_name="lookup",
            arguments={},
            principal=ALICE,
            policy_id="react",
            policy_version="1",
            action_version=program_a.artifact().artifact_id,
        ),
        capability_name="lookup",
        approver_id="bob",
    )

    # Program B runs a different skill implementation class, so its action version differs and the
    # approval does not authorize it.
    result_b = program_b.execute(project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE)
    step_b = _lookup_step(result_b)
    assert step_b["authorized"] is False
    assert step_b["reason"].startswith("approval_required")
    assert evil.calls == 0

    # Program A, the exact implementation the approval authorizes, runs.
    result_a = program_a.execute(project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE)
    step_a = _lookup_step(result_a)
    assert step_a["authorized"] is True
    assert benign.calls == 1


def test_an_approval_bound_to_one_program_does_not_authorize_a_changed_program(runtime: Refaat) -> None:
    _ingest(runtime)
    project = runtime.project(workspace="acme", corpus="react")
    skill = _LookupSkill()
    program_a = _program(skill)
    program_b = _program_variant(skill)
    assert program_a.artifact().artifact_id != program_b.artifact().artifact_id

    # One approval, bound to program A's artifact id, is the only grant in force.
    project.approvals.grant(fingerprint=_fingerprint(program_a), capability_name="lookup", approver_id="bob")

    # Program B has the same capability, arguments, and principal but a different policy, so its
    # artifact-derived action version differs and the approval does not authorize it.
    result_b = program_b.execute(project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE)
    step_b = _lookup_step(result_b)
    assert step_b["authorized"] is False
    assert step_b["reason"].startswith("approval_required")
    assert skill.calls == 0

    # Program A, the exact program the approval authorizes, runs.
    result_a = program_a.execute(project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE)
    step_a = _lookup_step(result_a)
    assert step_a["authorized"] is True
    assert skill.calls == 1
