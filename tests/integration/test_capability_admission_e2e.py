"""Capability admission wired into the governed ReAct loop.

`capability_permits` runs after grant authorization and before the approval gate, so the measurable
dimensions (scopes, the side-effect ceiling, the evidence-classification ceiling, and the payload
bound) gate a tool step. The classification default stays PUBLIC, so an untagged corpus and a
no-capability program are byte-identical, and an over-classified corpus or an over-cap payload fails
closed without consuming a single-use durable approval.
"""

from __future__ import annotations

import pytest
from test_react_approval_e2e import GOVERNED_CAP_ID
from test_react_approval_e2e import _LookupSkill as _PinnedSkill
from test_react_approval_e2e import _program as _pinned_program

from refaat.application.runtime import Refaat
from refaat.canonical import canonical_json
from refaat.domain.authz import Classification, Principal
from refaat.domain.enums import SideEffectClass
from refaat.domain.governance import ToolCapability, approval_fingerprint
from refaat.domain.models import ContextPacket, DocumentInput, JsonObject, ReActPolicy, ToolGrant, ToolProposal
from refaat.programs import GovernedReActProgram, KeywordToolPlanner

pytestmark = pytest.mark.integration

ALICE = Principal(id="alice", tenant="acme")
QUESTION = "how does rollback work"
CONTENT = b"Rollback restores the previously approved harness snapshot before deployment."

GROUNDED_RAG_ID = "sha256:b76d73f8d956bcd38ccc9d4c24372b07b387647a189f3a9e300cded186febbb4"
FIXTURE_ID = "sha256:0aa531a8e454fc3b2852b38211d96e7678c254b1c2ab2301fea1848d2dd3d4c5"


class _LookupSkill:
    name = "lookup"
    description = "Read-only lookup over the retrieved context."

    def __init__(self) -> None:
        self.calls = 0

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        del arguments
        self.calls += 1
        return {"documents": len({hit.chunk.document_id for hit in context.hits})}


class _FixedArgsPlanner:
    """Keyword-family planner that proposes ``lookup`` once with a fixed argument payload.

    The keyword planner always proposes empty arguments, so a payload-cap test needs a planner that
    supplies a concrete argument dict; ``kind`` stays keyword so no model argument validation runs.
    """

    kind = "keyword"

    def __init__(self, arguments: JsonObject) -> None:
        self._arguments = dict(arguments)

    def tools(self) -> tuple[str, ...]:
        return ("lookup",)

    def plan(self, query: str, used: frozenset[str]) -> ToolProposal | None:
        del query
        if "lookup" in used:
            return None
        return ToolProposal(tool="lookup", arguments=dict(self._arguments))


def _capability(
    *,
    allowed_classification: Classification = Classification.PUBLIC,
    requires_approval: bool = False,
    max_payload_bytes: int | None = None,
) -> ToolCapability:
    return ToolCapability(
        name="lookup",
        required_scopes=frozenset({"read:corpus"}),
        side_effect=SideEffectClass.READ_ONLY,
        allowed_classification=allowed_classification,
        max_payload_bytes=max_payload_bytes,
        requires_approval=requires_approval,
    )


def _program(
    skill: _LookupSkill,
    capability: ToolCapability | None,
    *,
    planner: object | None = None,
) -> GovernedReActProgram:
    policy = ReActPolicy(
        max_iterations=2,
        granted_scopes=("read:corpus",),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        grants=(ToolGrant("lookup", required_scopes=("read:corpus",), side_effect=SideEffectClass.READ_ONLY),),
        capabilities=(capability,) if capability is not None else (),
    )
    return GovernedReActProgram(
        policy=policy,
        planner=planner if planner is not None else KeywordToolPlanner({"lookup": ("rollback",)}),
        skills={"lookup": skill},
    )


def _ingest(runtime: Refaat, *, metadata: JsonObject | None = None) -> None:
    project = runtime.project(workspace="acme", corpus="react")
    runtime.ingest(
        project.workspace_id,
        project.corpus_id,
        DocumentInput(
            source_id="ops.txt",
            title="Ops",
            media_type="text/plain",
            content=CONTENT,
            metadata=dict(metadata) if metadata else {},
        ),
    )


def _run_step(runtime: Refaat, program: GovernedReActProgram, *, principal: Principal | None = ALICE) -> JsonObject:
    project = runtime.project(workspace="acme", corpus="react")
    result = program.execute(project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=principal)
    return {step["tool"]: step for step in result.outputs["react_steps"]}["lookup"]


def _fingerprint(program: GovernedReActProgram) -> str:
    return approval_fingerprint(
        capability_name="lookup",
        arguments={},
        principal=ALICE,
        policy_id="react",
        policy_version="1",
        action_version=program.artifact().artifact_id,
    )


CONFIDENTIAL_ACL: JsonObject = {"acl": {"classification": "confidential"}}
MALFORMED_ACL: JsonObject = {"acl": {"classification": "top_secret"}}


def test_public_ceiling_denies_confidential_evidence_and_burns_no_approval(runtime: Refaat) -> None:
    _ingest(runtime, metadata=CONFIDENTIAL_ACL)
    skill = _LookupSkill()
    program = _program(skill, _capability(requires_approval=True))
    fingerprint = _fingerprint(program)
    project = runtime.project(workspace="acme", corpus="react")
    project.approvals.grant(fingerprint=fingerprint, capability_name="lookup", approver_id="bob")
    assert len(project.approvals.list_active(fingerprint)) == 1

    step = _run_step(runtime, program)
    assert step["authorized"] is False
    assert step["reason"] == "classification_exceeds_capability"
    assert skill.calls == 0
    # Admission denies before the approval gate, so the single-use durable approval is untouched.
    assert len(project.approvals.list_active(fingerprint)) == 1


def test_malformed_acl_fails_closed_to_restricted(runtime: Refaat) -> None:
    _ingest(runtime, metadata=MALFORMED_ACL)
    skill = _LookupSkill()
    step = _run_step(runtime, _program(skill, _capability()))
    assert step["authorized"] is False
    assert step["reason"] == "classification_exceeds_capability"
    assert skill.calls == 0


def test_untagged_corpus_admits_and_matches_unenforced_behavior(runtime: Refaat) -> None:
    _ingest(runtime)
    skill = _LookupSkill()
    program = _program(skill, _capability())
    project = runtime.project(workspace="acme", corpus="react")
    result = program.execute(project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE)
    step = {s["tool"]: s for s in result.outputs["react_steps"]}["lookup"]
    assert step["authorized"] is True
    assert skill.calls == 1

    # A no-capability governed program over the same corpus produces the identical answer and step.
    baseline_skill = _LookupSkill()
    baseline = _program(baseline_skill, None)
    base_result = baseline.execute(
        project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE
    )
    base_step = {s["tool"]: s for s in base_result.outputs["react_steps"]}["lookup"]
    assert result.answer == base_result.answer
    assert step == base_step


def test_payload_cap_denies_over_cap_and_admits_in_cap(runtime: Refaat) -> None:
    _ingest(runtime)
    arguments: JsonObject = {"note": "payload-value"}
    size = len(canonical_json(arguments).encode())

    over_skill = _LookupSkill()
    over = _program(over_skill, _capability(max_payload_bytes=size - 1), planner=_FixedArgsPlanner(arguments))
    over_step = _run_step(runtime, over)
    assert over_step["authorized"] is False
    assert over_step["reason"] == "payload_exceeds_max"
    assert over_skill.calls == 0

    in_skill = _LookupSkill()
    in_cap = _program(in_skill, _capability(max_payload_bytes=size), planner=_FixedArgsPlanner(arguments))
    in_step = _run_step(runtime, in_cap)
    assert in_step["authorized"] is True
    assert in_skill.calls == 1


def test_higher_ceiling_admits_confidential_and_moves_its_own_id(runtime: Refaat) -> None:
    _ingest(runtime, metadata=CONFIDENTIAL_ACL)
    skill = _LookupSkill()
    program = _program(skill, _capability(allowed_classification=Classification.CONFIDENTIAL))
    step = _run_step(runtime, program)
    assert step["authorized"] is True
    assert skill.calls == 1

    public_id = (
        _program(_LookupSkill(), _capability(allowed_classification=Classification.PUBLIC)).artifact().artifact_id
    )
    assert program.artifact().artifact_id != public_id


def test_no_capability_program_is_byte_identical_over_confidential_evidence(runtime: Refaat) -> None:
    _ingest(runtime, metadata=CONFIDENTIAL_ACL)
    skill = _LookupSkill()
    step = _run_step(runtime, _program(skill, None))
    assert step["authorized"] is True
    assert skill.calls == 1


def test_pinned_ids_unchanged() -> None:
    from refaat.programs.artifact import ProgramArtifact
    from refaat.programs.program import GroundedRagProgram

    assert GroundedRagProgram().artifact().artifact_id == GROUNDED_RAG_ID
    fixture = ProgramArtifact(
        program="p",
        backend="native",
        definition_hash="dh",
        immutable_policy_hash="ph",
        signatures={"input": "i", "output": "o"},
        tunables={"top_k": None},
        validators=("grounding",),
        dependency_versions={"refaat": "0.1.0"},
    )
    assert fixture.artifact_id == FIXTURE_ID
    # The requires-approval governed program's id must not move: no fold or dataclass change was made.
    assert _pinned_program(_PinnedSkill()).artifact().artifact_id == GOVERNED_CAP_ID
