"""Time-based approval escalation end to end: the anchor store and the governed consume path.

An open approval request records an anchor timestamp. After the capability's escalation delay has
passed since the earliest open request, the governed loop widens the acceptable approver role from
the base role to also include the escalation role, so an escalation-role approval that would not
count on the un-escalated path now satisfies. Before the delay, or with no request at all, the
acceptable role stays the base role and the escalation-role approval does not satisfy.
"""

from __future__ import annotations

import hashlib
from datetime import timedelta

import pytest

from refaat.application.runtime import Refaat
from refaat.domain.authz import Principal
from refaat.domain.enums import SideEffectClass
from refaat.domain.governance import ToolCapability, approval_fingerprint
from refaat.domain.models import ContextPacket, DocumentInput, JsonObject, ReActPolicy, ToolGrant
from refaat.programs import GovernedReActProgram, KeywordToolPlanner
from refaat.programs.program import ProgramResult

pytestmark = pytest.mark.integration

ALICE = Principal(id="alice", tenant="acme")
QUESTION = "how does rollback work"
BASE_ROLE = "reviewer"
ESCALATION_ROLE = "incident-commander"


class _LookupSkill:
    name = "lookup"
    description = "Read-only lookup over the retrieved context."

    def __init__(self) -> None:
        self.calls = 0

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        del arguments
        self.calls += 1
        return {"documents": len({hit.chunk.document_id for hit in context.hits})}


def _program(skill: _LookupSkill, *, escalation_after_seconds: int) -> GovernedReActProgram:
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
                approver_role=BASE_ROLE,
                escalation_role=ESCALATION_ROLE,
                escalation_after_seconds=escalation_after_seconds,
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


def _fp(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


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


def test_earliest_open_request_anchor_returns_the_earliest_and_is_tz_aware(runtime: Refaat) -> None:
    workspace_id = runtime.create_workspace("acme", "Acme")
    store = runtime.store
    fingerprint = _fp("escalate")
    now = store.clock.now()
    earlier = now - timedelta(hours=2)
    later = now - timedelta(hours=1)
    store.record_approval_request(
        workspace_id, fingerprint=fingerprint, capability_name="lookup", requested_by="alice", created_at=later
    )
    store.record_approval_request(
        workspace_id, fingerprint=fingerprint, capability_name="lookup", requested_by="alice", created_at=earlier
    )

    anchor = store.earliest_open_request_anchor(workspace_id, fingerprint)
    assert anchor is not None
    assert anchor.tzinfo is not None
    assert anchor == earlier
    assert store.earliest_open_request_anchor(workspace_id, _fp("no-request")) is None


def test_time_based_escalation_widens_the_acceptable_role(runtime: Refaat) -> None:
    _ingest(runtime)
    project = runtime.project(workspace="acme", corpus="react")
    skill = _LookupSkill()
    program = _program(skill, escalation_after_seconds=1)
    fingerprint = _fingerprint(program)
    # An approval from an escalation-role approver only, which does not hold the base role.
    project.approvals.grant(
        fingerprint=fingerprint,
        capability_name="lookup",
        approver_id="bob",
        approver_roles=[ESCALATION_ROLE],
    )
    # An open request whose anchor is well past the one-second escalation delay.
    project.runtime.store.record_approval_request(
        project.workspace_id,
        fingerprint=fingerprint,
        capability_name="lookup",
        requested_by="alice",
        created_at=project.runtime.store.clock.now() - timedelta(hours=1),
    )

    result = program.execute(project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE)
    step = _lookup_step(result)
    assert step["authorized"] is True
    assert skill.calls == 1


def test_escalation_role_approval_is_denied_without_a_request_anchor(runtime: Refaat) -> None:
    _ingest(runtime)
    project = runtime.project(workspace="acme", corpus="react")
    skill = _LookupSkill()
    program = _program(skill, escalation_after_seconds=1)
    project.approvals.grant(
        fingerprint=_fingerprint(program),
        capability_name="lookup",
        approver_id="bob",
        approver_roles=[ESCALATION_ROLE],
    )

    # No open request means no anchor, so escalation never activates and the base role is required.
    result = program.execute(project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE)
    step = _lookup_step(result)
    assert step["authorized"] is False
    assert step["reason"].startswith("approval_required")
    assert skill.calls == 0


def test_escalation_role_approval_is_denied_when_the_request_is_too_recent(runtime: Refaat) -> None:
    _ingest(runtime)
    project = runtime.project(workspace="acme", corpus="react")
    skill = _LookupSkill()
    program = _program(skill, escalation_after_seconds=3600)
    fingerprint = _fingerprint(program)
    project.approvals.grant(
        fingerprint=fingerprint,
        capability_name="lookup",
        approver_id="bob",
        approver_roles=[ESCALATION_ROLE],
    )
    # A request recorded now is inside the one-hour delay, so escalation is not active yet.
    project.runtime.store.record_approval_request(
        project.workspace_id,
        fingerprint=fingerprint,
        capability_name="lookup",
        requested_by="alice",
        created_at=project.runtime.store.clock.now(),
    )

    result = program.execute(project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE)
    step = _lookup_step(result)
    assert step["authorized"] is False
    assert step["reason"].startswith("approval_required")
    assert skill.calls == 0
