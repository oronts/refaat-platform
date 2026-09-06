"""Human-approval enforcement in the governed ReAct loop.

A tool whose registered ToolCapability requires approval never executes without a valid approval,
the approval is bound to an immutable fingerprint, and separation of duties holds. Byte-identical
when no capability is registered.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from refaat.application.approvals import ApprovalContext, approval_scope
from refaat.application.runtime import Refaat
from refaat.domain.authz import Principal
from refaat.domain.enums import AuditLevel, SideEffectClass
from refaat.domain.governance import ApprovalRecord, ToolCapability, approval_fingerprint
from refaat.domain.models import ContextPacket, DocumentInput, JsonObject, ReActPolicy, ToolGrant
from refaat.programs import GovernedReActProgram, KeywordToolPlanner

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


def _ingest(runtime: Refaat, workspace_id: str, corpus_id: str) -> None:
    runtime.ingest(
        workspace_id,
        corpus_id,
        DocumentInput(
            source_id="ops.txt",
            title="Ops",
            media_type="text/plain",
            content=b"Rollback restores the previously approved harness snapshot before deployment.",
        ),
    )


def _program(skill: _LookupSkill, *, requires_approval: bool = True) -> GovernedReActProgram:
    capabilities = (
        (
            ToolCapability(
                name="lookup",
                required_scopes=frozenset({"read:corpus"}),
                side_effect=SideEffectClass.READ_ONLY,
                requires_approval=True,
            ),
        )
        if requires_approval
        else ()
    )
    policy = ReActPolicy(
        max_iterations=2,
        granted_scopes=("read:corpus",),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        grants=(ToolGrant("lookup", required_scopes=("read:corpus",), side_effect=SideEffectClass.READ_ONLY),),
        capabilities=capabilities,
    )
    return GovernedReActProgram(
        policy=policy, planner=KeywordToolPlanner({"lookup": ("rollback",)}), skills={"lookup": skill}
    )


def _fingerprint(program: GovernedReActProgram | None = None) -> str:
    # The action version is the run program's content-addressed artifact id, so a grant is bound to
    # the exact program version. The standard requires-approval program is the default here.
    prog = program if program is not None else _program(_LookupSkill())
    return approval_fingerprint(
        capability_name="lookup",
        arguments={},
        principal=ALICE,
        policy_id="react",
        policy_version="1",
        action_version=prog.artifact().artifact_id,
    )


def _lookup_step(runtime: Refaat, skill: _LookupSkill, *, principal: Principal | None) -> JsonObject:
    project = runtime.project(workspace="acme", corpus="react")
    _ingest(runtime, project.workspace_id, project.corpus_id)
    result = _program(skill).execute(
        project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=principal
    )
    return {step["tool"]: step for step in result.outputs["react_steps"]}["lookup"]


def test_requires_approval_tool_is_denied_without_an_approval(runtime: Refaat) -> None:
    skill = _LookupSkill()
    step = _lookup_step(runtime, skill, principal=ALICE)
    assert skill.calls == 0
    assert step["authorized"] is False
    assert step["reason"].startswith("approval_required")


def test_no_principal_denies_a_requires_approval_tool(runtime: Refaat) -> None:
    skill = _LookupSkill()
    step = _lookup_step(runtime, skill, principal=None)
    assert skill.calls == 0
    assert step["authorized"] is False


def test_requires_approval_tool_runs_with_a_valid_approval(runtime: Refaat) -> None:
    skill = _LookupSkill()
    approval = ApprovalRecord(fingerprint=_fingerprint(), approver_id="bob")
    with approval_scope(ApprovalContext(approvals=(approval,))):
        step = _lookup_step(runtime, skill, principal=ALICE)
    assert skill.calls == 1
    assert step["authorized"] is True


def test_initiator_cannot_self_approve(runtime: Refaat) -> None:
    skill = _LookupSkill()
    self_approval = ApprovalRecord(fingerprint=_fingerprint(), approver_id="alice")
    with approval_scope(ApprovalContext(approvals=(self_approval,))):
        step = _lookup_step(runtime, skill, principal=ALICE)
    assert skill.calls == 0
    assert step["authorized"] is False


def test_a_stale_fingerprint_approval_does_not_authorize(runtime: Refaat) -> None:
    skill = _LookupSkill()
    stale = ApprovalRecord(fingerprint="not-the-right-fingerprint", approver_id="bob")
    with approval_scope(ApprovalContext(approvals=(stale,))):
        step = _lookup_step(runtime, skill, principal=ALICE)
    assert skill.calls == 0
    assert step["authorized"] is False


def test_dangling_capability_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="no matching tool grant"):
        ReActPolicy(
            grants=(ToolGrant("lookup", required_scopes=("read:corpus",)),),
            capabilities=(ToolCapability(name="lookup_v2", requires_approval=True),),
        )


def test_duplicate_capability_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate capability"):
        ReActPolicy(
            grants=(ToolGrant("lookup", required_scopes=("read:corpus",)),),
            capabilities=(
                ToolCapability(name="lookup", requires_approval=True),
                ToolCapability(name="lookup", requires_approval=False),
            ),
        )


def test_capability_folds_into_the_artifact_and_is_byte_identical_when_absent() -> None:
    skill = _LookupSkill()
    without_program = _program(skill, requires_approval=False)
    with_program = _program(skill, requires_approval=True)
    # A registered capability changes the governed artifact id; no capability leaves the artifact
    # definition without a capabilities key, so an existing governed artifact id is unchanged.
    assert without_program.artifact().artifact_id != with_program.artifact().artifact_id
    assert "capabilities" not in without_program._definition_extra()["react"]
    assert "capabilities" in with_program._definition_extra()["react"]


GOVERNED_CAP_ID = "sha256:07fa28eaa397ee0e42fdc7450f7b29f48cfff621cbc8092f28be85133201a469"
GROUNDED_RAG_ID = "sha256:b76d73f8d956bcd38ccc9d4c24372b07b387647a189f3a9e300cded186febbb4"
FIXTURE_ID = "sha256:0aa531a8e454fc3b2852b38211d96e7678c254b1c2ab2301fea1848d2dd3d4c5"


def _base_capability() -> ToolCapability:
    return ToolCapability(
        name="lookup",
        required_scopes=frozenset({"read:corpus"}),
        side_effect=SideEffectClass.READ_ONLY,
        requires_approval=True,
    )


def _program_with_capability(capability: ToolCapability) -> GovernedReActProgram:
    policy = ReActPolicy(
        max_iterations=2,
        granted_scopes=("read:corpus",),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        grants=(ToolGrant("lookup", required_scopes=("read:corpus",), side_effect=SideEffectClass.READ_ONLY),),
        capabilities=(capability,),
    )
    return GovernedReActProgram(
        policy=policy, planner=KeywordToolPlanner({"lookup": ("rollback",)}), skills={"lookup": _LookupSkill()}
    )


def _capability_entry(program: GovernedReActProgram) -> JsonObject:
    entry = program._definition_extra()["react"]["capabilities"][0]
    assert isinstance(entry, dict)
    return entry


def _tripwire_pinned_ids() -> None:
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


def test_declared_governance_fields_byte_identical_when_unset() -> None:
    program = _program(_LookupSkill())
    assert program.artifact().artifact_id == GOVERNED_CAP_ID
    entry = _capability_entry(program)
    assert "allowed_network" not in entry
    assert "resource_namespace" not in entry
    assert "required_secrets" not in entry
    _tripwire_pinned_ids()


def test_each_declared_governance_field_folds_and_flips_the_id() -> None:
    unset_id = _program(_LookupSkill()).artifact().artifact_id

    network = _program_with_capability(replace(_base_capability(), allowed_network=frozenset({"api.acme.com:443"})))
    assert network.artifact().artifact_id != unset_id
    assert _capability_entry(network)["allowed_network"] == ["api.acme.com:443"]

    namespace = _program_with_capability(replace(_base_capability(), resource_namespace="corpus:acme"))
    assert namespace.artifact().artifact_id != unset_id
    assert _capability_entry(namespace)["resource_namespace"] == "corpus:acme"

    secrets = _program_with_capability(replace(_base_capability(), required_secrets=frozenset({"acme/deploy-key"})))
    assert secrets.artifact().artifact_id != unset_id
    assert _capability_entry(secrets)["required_secrets"] == ["acme/deploy-key"]

    # Two allowlists built in different insertion orders fold to the same sorted list and the same id.
    order_a = _program_with_capability(replace(_base_capability(), allowed_network=frozenset({"a:1", "b:2", "c:3"})))
    order_b = _program_with_capability(replace(_base_capability(), allowed_network=frozenset(["c:3", "b:2", "a:1"])))
    assert _capability_entry(order_a)["allowed_network"] == ["a:1", "b:2", "c:3"]
    assert order_a.artifact().artifact_id == order_b.artifact().artifact_id


def test_audit_level_folds_and_flips_the_id() -> None:
    standard = _program(_LookupSkill())
    assert standard.artifact().artifact_id == GOVERNED_CAP_ID
    assert "audit_level" not in _capability_entry(standard)
    enhanced = _program_with_capability(replace(_base_capability(), audit_level=AuditLevel.ENHANCED))
    assert enhanced.artifact().artifact_id != GOVERNED_CAP_ID
    assert _capability_entry(enhanced)["audit_level"] == "enhanced"
    _tripwire_pinned_ids()


def _executing_capability(audit_level: AuditLevel = AuditLevel.STANDARD) -> ToolCapability:
    # requires_approval is False so the tool actually executes, isolating the observability layer.
    return ToolCapability(
        name="lookup",
        required_scopes=frozenset({"read:corpus"}),
        side_effect=SideEffectClass.READ_ONLY,
        requires_approval=False,
        audit_level=audit_level,
    )


class _RecordCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _run_governed_capturing(
    runtime: Refaat, capability: ToolCapability, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[object], JsonObject, list[logging.LogRecord]]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr("refaat.observability._program_tracer", lambda: provider.get_tracer("test"))
    # Capture on the react logger directly, not root: RuntimeBuilder.build() clears root handlers, so a
    # root-attached handler (caplog included) would be dropped. Re-enable the logger because Alembic's
    # fileConfig, run by an earlier migration test, disables every existing logger by default.
    collector = _RecordCollector()
    react_logger = logging.getLogger("refaat.programs.react")
    react_logger.addHandler(collector)
    previous_level = react_logger.level
    previous_disabled = react_logger.disabled
    react_logger.setLevel(logging.INFO)
    react_logger.disabled = False
    project = runtime.project(workspace="acme", corpus="react")
    _ingest(runtime, project.workspace_id, project.corpus_id)
    try:
        result = _program_with_capability(capability).execute(
            project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE
        )
    finally:
        react_logger.removeHandler(collector)
        react_logger.setLevel(previous_level)
        react_logger.disabled = previous_disabled
    tool_spans = [span for span in exporter.get_finished_spans() if span.name == "refaat.tool.step"]
    step = {s["tool"]: s for s in result.outputs["react_steps"]}["lookup"]
    step_logs = [record for record in collector.records if record.msg == "governed_tool_step"]
    return tool_spans, step, step_logs


def test_standard_capability_emits_no_step_observability(runtime: Refaat, monkeypatch: pytest.MonkeyPatch) -> None:
    tool_spans, step, step_logs = _run_governed_capturing(runtime, _executing_capability(), monkeypatch)
    assert step["authorized"] is True  # the tool ran, so a STANDARD run is genuinely inert, not skipped
    assert tool_spans == []
    assert step_logs == []


def test_enhanced_capability_emits_redacted_step_observability(
    runtime: Refaat, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool_spans, step, step_logs = _run_governed_capturing(
        runtime, _executing_capability(AuditLevel.ENHANCED), monkeypatch
    )
    assert step["authorized"] is True
    assert len(tool_spans) == 1
    attrs = tool_spans[0].attributes  # type: ignore[attr-defined]
    assert attrs["tool"] == "lookup"
    assert attrs["capability"] == "lookup"
    assert attrs["authorized"] is True
    assert attrs["reason_code"] == "authorized"
    assert attrs["side_effect"] == "read_only"
    assert attrs["arguments_count"] == 0
    assert len(step_logs) == 1
    fields = step_logs[0].fields  # type: ignore[attr-defined]
    assert set(fields) == {
        "tool",
        "capability",
        "authorized",
        "reason_code",
        "side_effect",
        "classification",
        "requires_approval",
        "arguments_count",
        "observation_present",
        "iteration",
    }
    assert fields["reason_code"] == "authorized"
