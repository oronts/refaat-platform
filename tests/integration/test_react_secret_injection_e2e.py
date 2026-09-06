"""Execution-time secret injection in the governed ReAct loop.

A governed tool whose capability declares required_secrets receives a scoped SecretBundle only when
a provider resolves them, revealed at the side effect and never present in the recorded step. With
no provider, or an unresolvable secret, the step is denied and no single-use approval is burned.
Byte-identical for a program that declares no secret.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

from refaat.application.approvals import ApprovalService
from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.config import Settings
from refaat.domain.authz import Principal
from refaat.domain.enums import SideEffectClass
from refaat.domain.governance import ToolCapability, approval_fingerprint
from refaat.domain.models import ContextPacket, DocumentInput, JsonObject, ReActPolicy, ToolGrant
from refaat.domain.secrets import SecretBundle, SecretScope, SecretValue
from refaat.infrastructure.secrets import StaticSecretProvider
from refaat.programs import GovernedReActProgram, KeywordToolPlanner

pytestmark = pytest.mark.integration

ALICE = Principal(id="alice", tenant="acme")
QUESTION = "how does rollback work"
SECRET_NAME = "api_key"
CONTENT = b"Rollback restores the previously approved harness snapshot before deployment."


class _SecretUsingSkill:
    name = "push"
    description = "Governed tool that reveals a scoped secret to a fake outbound sink."
    wants_secrets = True

    def __init__(self) -> None:
        self.calls = 0
        self.received_none = False
        self.observed_names: frozenset[str] = frozenset()
        self.sink: list[str] = []

    def run(self, context: ContextPacket, arguments: JsonObject, *, secrets: SecretBundle | None) -> JsonObject:
        del context, arguments
        self.calls += 1
        self.received_none = secrets is None
        if secrets is not None:
            self.observed_names = secrets.names
            self.sink.append(secrets.reveal(SECRET_NAME))
        return {"called": True}


class _LyingProvider:
    """Reports every requested name as available but resolves none: an availability-to-resolution lie."""

    provider_id = "lying"

    def available(self, names: frozenset[str], *, scope: SecretScope) -> frozenset[str]:
        del scope
        return frozenset(names)

    def resolve(self, name: str, *, scope: SecretScope) -> SecretValue | None:
        del name, scope
        return None


def _program(
    skill: _SecretUsingSkill,
    *,
    required_secrets: frozenset[str] = frozenset({SECRET_NAME}),
    requires_approval: bool = False,
) -> GovernedReActProgram:
    capability = ToolCapability(
        name="push",
        required_scopes=frozenset({"write:corpus"}),
        side_effect=SideEffectClass.WRITE,
        requires_approval=requires_approval,
        required_secrets=required_secrets,
    )
    policy = ReActPolicy(
        max_iterations=2,
        granted_scopes=("write:corpus",),
        side_effect_ceiling=SideEffectClass.WRITE,
        grants=(ToolGrant("push", required_scopes=("write:corpus",), side_effect=SideEffectClass.WRITE),),
        capabilities=(capability,),
    )
    return GovernedReActProgram(
        policy=policy, planner=KeywordToolPlanner({"push": ("rollback",)}), skills={"push": skill}
    )


def _built(settings: Settings, provider: object | None) -> Iterator[Refaat]:
    builder = RuntimeBuilder(settings)
    if provider is not None:
        builder = builder.with_secret_provider(provider)  # type: ignore[arg-type]
    value = builder.build()
    try:
        yield value
    finally:
        value.close()


def _run(runtime: Refaat, program: GovernedReActProgram) -> JsonObject:
    project = runtime.project(workspace="acme", corpus="react")
    runtime.ingest(
        project.workspace_id,
        project.corpus_id,
        DocumentInput(source_id="ops.txt", title="Ops", media_type="text/plain", content=CONTENT),
    )
    result = program.execute(project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE)
    return {step["tool"]: step for step in result.outputs["react_steps"]}["push"]


def test_scoped_secret_is_injected_and_absent_from_the_step(settings: Settings) -> None:
    skill = _SecretUsingSkill()
    for runtime in _built(settings, StaticSecretProvider({SECRET_NAME: "PLANT-abc123"})):
        step = _run(runtime, _program(skill))
    assert step["authorized"] is True
    assert skill.calls == 1
    assert skill.observed_names == frozenset({SECRET_NAME})
    assert skill.sink == ["PLANT-abc123"]
    assert "PLANT-abc123" not in json.dumps(step)


def test_no_provider_denies_a_secret_declaring_capability(settings: Settings) -> None:
    skill = _SecretUsingSkill()
    for runtime in _built(settings, None):
        step = _run(runtime, _program(skill))
    assert step["authorized"] is False
    assert step["reason"].startswith("required_secret_unavailable")
    assert skill.calls == 0


def test_unavailable_secret_denies(settings: Settings) -> None:
    skill = _SecretUsingSkill()
    for runtime in _built(settings, StaticSecretProvider({"other_key": "x"})):
        step = _run(runtime, _program(skill))
    assert step["authorized"] is False
    assert step["reason"].startswith("required_secret_unavailable")
    assert skill.calls == 0


def test_available_but_unresolvable_secret_denies(settings: Settings) -> None:
    skill = _SecretUsingSkill()
    for runtime in _built(settings, _LyingProvider()):
        step = _run(runtime, _program(skill))
    assert step["authorized"] is False
    assert step["reason"] == "required_secret_unavailable:unresolved"
    assert skill.calls == 0


def test_capability_declaring_no_secret_receives_none(settings: Settings) -> None:
    skill = _SecretUsingSkill()
    for runtime in _built(settings, StaticSecretProvider({SECRET_NAME: "PLANT-abc123"})):
        step = _run(runtime, _program(skill, required_secrets=frozenset()))
    assert step["authorized"] is True
    assert skill.calls == 1
    assert skill.received_none is True
    assert skill.sink == []


def _grant(runtime: Refaat, program: GovernedReActProgram, workspace_id: str) -> tuple[ApprovalService, str]:
    approvals = ApprovalService(runtime.store, workspace_id)
    fingerprint = approval_fingerprint(
        capability_name="push",
        arguments={},
        principal=ALICE,
        policy_id="react",
        policy_version="1",
        action_version=program.artifact().artifact_id,
    )
    approvals.grant(fingerprint=fingerprint, capability_name="push", approver_id="bob")
    return approvals, fingerprint


def test_durable_approval_not_consumed_when_required_secret_unresolvable(settings: Settings) -> None:
    skill = _SecretUsingSkill()
    program = _program(skill, requires_approval=True)
    for runtime in _built(settings, _LyingProvider()):
        project = runtime.project(workspace="acme", corpus="react")
        runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(source_id="ops.txt", title="Ops", media_type="text/plain", content=CONTENT),
        )
        approvals, fingerprint = _grant(runtime, program, project.workspace_id)
        result = program.execute(project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE)
        step = {s["tool"]: s for s in result.outputs["react_steps"]}["push"]
        assert step["authorized"] is False
        assert step["reason"] == "required_secret_unavailable:unresolved"
        assert skill.calls == 0
        # R2: the materialize runs before the approval consume, so the single-use approval survives.
        assert len(approvals.list_active(fingerprint)) == 1


def test_durable_approval_consumed_when_secret_resolves(settings: Settings) -> None:
    skill = _SecretUsingSkill()
    program = _program(skill, requires_approval=True)
    for runtime in _built(settings, StaticSecretProvider({SECRET_NAME: "PLANT-xyz"})):
        project = runtime.project(workspace="acme", corpus="react")
        runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(source_id="ops.txt", title="Ops", media_type="text/plain", content=CONTENT),
        )
        approvals, fingerprint = _grant(runtime, program, project.workspace_id)
        result = program.execute(project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE)
        step = {s["tool"]: s for s in result.outputs["react_steps"]}["push"]
        assert step["authorized"] is True
        assert skill.calls == 1
        assert skill.sink == ["PLANT-xyz"]
        assert approvals.list_active(fingerprint) == []


def test_pinned_governed_id_unchanged_with_a_provider_installed(settings: Settings) -> None:
    from test_react_approval_e2e import GOVERNED_CAP_ID
    from test_react_approval_e2e import _LookupSkill as _PinnedSkill
    from test_react_approval_e2e import _program as _pinned_program

    program = _pinned_program(_PinnedSkill())
    assert program.artifact().artifact_id == GOVERNED_CAP_ID
    # A provider is a runtime input, never folded into the artifact, so installing one cannot move the id.
    for runtime in _built(settings, StaticSecretProvider({SECRET_NAME: "PLANT-abc123"})):
        assert runtime.secret_provider is not None
        assert program.artifact().artifact_id == GOVERNED_CAP_ID
