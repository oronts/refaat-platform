"""Governed outbound HTTP in the ReAct loop.

A wants_outbound skill receives a governed outbound handle only when its capability declares the
EXTERNAL side effect; a READ_ONLY capability or no capability denies the step, and the direct skill
path refuses it outright. A skill declaring both wants_secrets and wants_outbound receives both. A
skill with no marker is byte-identical.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.config import Settings
from refaat.domain.authz import Principal
from refaat.domain.enums import SideEffectClass
from refaat.domain.governance import ToolCapability
from refaat.domain.models import ContextPacket, DocumentInput, JsonObject, QueryRequest, ReActPolicy, ToolGrant
from refaat.domain.secrets import SecretBundle
from refaat.errors import AuthorizationError
from refaat.infrastructure.secrets import StaticSecretProvider
from refaat.programs import GovernedReActProgram, KeywordToolPlanner

pytestmark = pytest.mark.integration

ALICE = Principal(id="alice", tenant="acme")
QUESTION = "how does rollback work"
CONTENT = b"Rollback restores the previously approved harness snapshot before deployment."


class _OutboundSkill:
    name = "push"
    description = "A governed tool that would reach an external system through the governed outbound handle."
    wants_outbound = True

    def __init__(self) -> None:
        self.calls = 0
        self.received_outbound = False

    def run(self, context: ContextPacket, arguments: JsonObject, *, outbound: object) -> JsonObject:
        del context, arguments
        self.calls += 1
        self.received_outbound = outbound is not None
        return {"called": True}


class _SecretAndOutboundSkill:
    name = "push"
    description = "A governed tool that reveals a scoped secret and reaches an external system."
    wants_secrets = True
    wants_outbound = True

    def __init__(self) -> None:
        self.received_outbound = False
        self.received_secret = False

    def run(
        self, context: ContextPacket, arguments: JsonObject, *, secrets: SecretBundle | None, outbound: object
    ) -> JsonObject:
        del context, arguments
        self.received_outbound = outbound is not None
        self.received_secret = secrets is not None and bool(secrets.names)
        return {"called": True, "leak": secrets.reveal("api_key") if secrets is not None else ""}


def _program(
    skill: object, *, side_effect: SideEffectClass, required_secrets: frozenset[str] = frozenset()
) -> GovernedReActProgram:
    capability = ToolCapability(
        name="push",
        required_scopes=frozenset({"reach:external"}),
        side_effect=side_effect,
        required_secrets=required_secrets,
        allowed_network=frozenset({"api.example.com"}),
    )
    policy = ReActPolicy(
        max_iterations=2,
        granted_scopes=("reach:external",),
        side_effect_ceiling=SideEffectClass.EXTERNAL,
        grants=(ToolGrant("push", required_scopes=("reach:external",), side_effect=side_effect),),
        capabilities=(capability,),
    )
    return GovernedReActProgram(
        policy=policy, planner=KeywordToolPlanner({"push": ("rollback",)}), skills={"push": skill}
    )


def _program_no_capability(skill: object) -> GovernedReActProgram:
    policy = ReActPolicy(
        max_iterations=2,
        granted_scopes=("reach:external",),
        side_effect_ceiling=SideEffectClass.EXTERNAL,
        grants=(ToolGrant("push", required_scopes=("reach:external",), side_effect=SideEffectClass.EXTERNAL),),
    )
    return GovernedReActProgram(
        policy=policy, planner=KeywordToolPlanner({"push": ("rollback",)}), skills={"push": skill}
    )


def _built(settings: Settings, *, provider: object | None = None, skill: object | None = None) -> Iterator[Refaat]:
    builder = RuntimeBuilder(settings)
    if provider is not None:
        builder = builder.with_secret_provider(provider)  # type: ignore[arg-type]
    if skill is not None:
        builder = builder.with_skill(skill)  # type: ignore[arg-type]
    value = builder.build()
    try:
        yield value
    finally:
        value.close()


def _run(runtime: Refaat, program: GovernedReActProgram, *, principal: Principal | None = ALICE) -> JsonObject:
    project = runtime.project(workspace="acme", corpus="react")
    runtime.ingest(
        project.workspace_id,
        project.corpus_id,
        DocumentInput(source_id="ops.txt", title="Ops", media_type="text/plain", content=CONTENT),
    )
    result = program.execute(project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=principal)
    return {step["tool"]: step for step in result.outputs["react_steps"]}["push"]


def test_external_capability_grants_a_governed_outbound_handle(settings: Settings) -> None:
    skill = _OutboundSkill()
    for runtime in _built(settings):
        step = _run(runtime, _program(skill, side_effect=SideEffectClass.EXTERNAL))
    assert step["authorized"] is True
    assert skill.calls == 1
    assert skill.received_outbound is True


def test_read_only_capability_denies_outbound(settings: Settings) -> None:
    skill = _OutboundSkill()
    for runtime in _built(settings):
        step = _run(runtime, _program(skill, side_effect=SideEffectClass.READ_ONLY))
    assert step["authorized"] is False
    assert step["reason"] == "network_destination_denied:requires_external_capability"
    assert skill.calls == 0


def test_no_capability_denies_outbound(settings: Settings) -> None:
    skill = _OutboundSkill()
    for runtime in _built(settings):
        step = _run(runtime, _program_no_capability(skill))
    assert step["authorized"] is False
    assert step["reason"] == "network_destination_denied:requires_external_capability"
    assert skill.calls == 0


def test_secret_and_outbound_compose_and_the_observation_is_scrubbed(settings: Settings) -> None:
    skill = _SecretAndOutboundSkill()
    for runtime in _built(settings, provider=StaticSecretProvider({"api_key": "PLANT-xyz789"})):
        step = _run(
            runtime, _program(skill, side_effect=SideEffectClass.EXTERNAL, required_secrets=frozenset({"api_key"}))
        )
    assert step["authorized"] is True
    assert skill.received_outbound is True
    assert skill.received_secret is True
    assert "PLANT-xyz789" not in json.dumps(step)


def test_direct_skill_path_refuses_a_wants_outbound_skill(settings: Settings) -> None:
    skill = _OutboundSkill()
    for runtime in _built(settings, skill=skill):
        workspace = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(workspace, "react")
        with pytest.raises(AuthorizationError, match="governed outbound HTTP"):
            runtime.skills.run("push", QueryRequest(workspace_id=workspace, corpus_id=corpus, query=QUESTION))
