"""Break-glass approvals end to end: persistence, the governed consume path, and audit redaction.

A break-glass grant is stored with kind "break_glass" and its justification on the row, satisfies
the explicit alternate approval path (a bound role and a dual-approval floor) in the governed loop,
and never leaks its reason into an audit event. A legacy or default-kind row still satisfies the
normal path, so an approval written before the kind column existed keeps working.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa

from refaat.application.approvals import StoreApprovalBackend
from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.config import Settings
from refaat.domain.audit import AuditEvent
from refaat.domain.authz import Principal
from refaat.domain.enums import SideEffectClass
from refaat.domain.governance import ToolCapability, approval_fingerprint
from refaat.domain.models import ContextPacket, DocumentInput, JsonObject, ReActPolicy, ToolGrant
from refaat.infrastructure.database.schema import tool_approvals
from refaat.programs import GovernedReActProgram, KeywordToolPlanner
from refaat.programs.program import ProgramResult

pytestmark = pytest.mark.integration

ALICE = Principal(id="alice", tenant="acme")
QUESTION = "how does rollback work"
BREAK_GLASS_ROLE = "incident-commander"
REASON = "SECRET-INCIDENT-JUSTIFICATION-42"


class _LookupSkill:
    name = "lookup"
    description = "Read-only lookup over the retrieved context."

    def __init__(self) -> None:
        self.calls = 0

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        del arguments
        self.calls += 1
        return {"documents": len({hit.chunk.document_id for hit in context.hits})}


def _bg_program(skill: _LookupSkill) -> GovernedReActProgram:
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
                break_glass_enabled=True,
                break_glass_approver_role=BREAK_GLASS_ROLE,
            ),
        ),
    )
    return GovernedReActProgram(
        policy=policy, planner=KeywordToolPlanner({"lookup": ("rollback",)}), skills={"lookup": skill}
    )


def _normal_program(skill: _LookupSkill) -> GovernedReActProgram:
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


def _fingerprint(program: GovernedReActProgram) -> str:
    return approval_fingerprint(
        capability_name="lookup",
        arguments={},
        principal=ALICE,
        policy_id="react",
        policy_version="1",
        action_version=program.artifact().artifact_id,
    )


def _lookup_step(result: ProgramResult) -> JsonObject:
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


def _row(runtime: Refaat, approval_id: str) -> sa.RowMapping:
    with runtime.store.engine.connect() as conn:
        return conn.execute(sa.select(tool_approvals).where(tool_approvals.c.id == approval_id)).mappings().one()


def test_break_glass_grant_records_kind_and_reason_on_the_row(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="react")
    approval_id = project.approvals.grant(
        fingerprint=_fingerprint(_bg_program(_LookupSkill())),
        capability_name="lookup",
        approver_id="bob",
        approver_roles=[BREAK_GLASS_ROLE],
        kind="break_glass",
        reason=REASON,
    )
    row = _row(runtime, approval_id)
    assert row["kind"] == "break_glass"
    assert row["reason"] == REASON


def test_a_normal_grant_stores_no_reason_even_if_supplied(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="react")
    approval_id = project.approvals.grant(
        fingerprint=_fingerprint(_normal_program(_LookupSkill())),
        capability_name="lookup",
        approver_id="bob",
        reason=REASON,
    )
    row = _row(runtime, approval_id)
    assert row["kind"] == "normal"
    assert row["reason"] is None


class _CapturingSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        self.events.append(event)


def _string_values(event: AuditEvent) -> list[str]:
    values: list[str] = []
    for value in event.to_json().values():
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend(item for item in value if isinstance(item, str))
    return values


def test_break_glass_grant_emits_a_redacted_break_glass_event(settings: Settings) -> None:
    sink = _CapturingSink()
    runtime = RuntimeBuilder(settings).with_audit_sink(sink).build()
    try:
        project = runtime.project(workspace="acme", corpus="react")
        fingerprint = _fingerprint(_bg_program(_LookupSkill()))
        project.approvals.grant(
            fingerprint=fingerprint,
            capability_name="lookup",
            approver_id="bob",
            approver_roles=[BREAK_GLASS_ROLE],
            kind="break_glass",
            reason=REASON,
        )
        events = [event for event in sink.events if event.event_type == "approval.break_glass"]
        assert len(events) == 1
        assert events[0].capability_name == "lookup"
        assert events[0].action_fingerprint == fingerprint
        assert events[0].principal_id == "bob"
        # The reason must never appear in any string field of any emitted event.
        for event in sink.events:
            assert all(REASON not in value for value in _string_values(event))
    finally:
        runtime.close()


def test_break_glass_two_approvers_authorize_the_governed_run(runtime: Refaat) -> None:
    _ingest(runtime)
    project = runtime.project(workspace="acme", corpus="react")
    skill = _LookupSkill()
    program = _bg_program(skill)
    fingerprint = _fingerprint(program)
    for approver in ("bob", "carol"):
        project.approvals.grant(
            fingerprint=fingerprint,
            capability_name="lookup",
            approver_id=approver,
            approver_roles=[BREAK_GLASS_ROLE],
            kind="break_glass",
            reason=REASON,
        )
    result = program.execute(project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE)
    step = _lookup_step(result)
    assert step["authorized"] is True
    assert skill.calls == 1
    # The break-glass approvals were single-use: a second run has none left and is denied.
    second = program.execute(project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE)
    assert _lookup_step(second)["authorized"] is False
    assert skill.calls == 1


def test_a_single_break_glass_approver_fails_the_dual_floor(runtime: Refaat) -> None:
    _ingest(runtime)
    project = runtime.project(workspace="acme", corpus="react")
    skill = _LookupSkill()
    program = _bg_program(skill)
    project.approvals.grant(
        fingerprint=_fingerprint(program),
        capability_name="lookup",
        approver_id="bob",
        approver_roles=[BREAK_GLASS_ROLE],
        kind="break_glass",
        reason=REASON,
    )
    result = program.execute(project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE)
    assert _lookup_step(result)["authorized"] is False
    assert skill.calls == 0


def test_a_normal_grant_still_satisfies_the_normal_path(runtime: Refaat) -> None:
    _ingest(runtime)
    project = runtime.project(workspace="acme", corpus="react")
    skill = _LookupSkill()
    program = _normal_program(skill)
    project.approvals.grant(fingerprint=_fingerprint(program), capability_name="lookup", approver_id="bob")
    result = program.execute(project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE)
    assert _lookup_step(result)["authorized"] is True
    assert skill.calls == 1


def test_a_loaded_record_defaults_a_missing_kind_to_normal(runtime: Refaat) -> None:
    # Defense in depth: a row whose kind reads back as None (a legacy row) loads as a normal approval,
    # so the normal path still counts it. The backend surfaces the kind on every loaded record.
    project = runtime.project(workspace="acme", corpus="react")
    fingerprint = _fingerprint(_normal_program(_LookupSkill()))
    project.approvals.grant(fingerprint=fingerprint, capability_name="lookup", approver_id="bob")
    backend = StoreApprovalBackend(runtime.store, project.workspace_id)
    records = backend.active(fingerprint)
    assert len(records) == 1
    assert records[0].kind == "normal"
