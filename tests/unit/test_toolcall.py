from __future__ import annotations

import json

import httpx

from refaat.domain.enums import SideEffectClass
from refaat.domain.models import (
    ContextPacket,
    JsonObject,
    ReActPolicy,
    ToolGrant,
    ToolProposal,
    ToolSpec,
)
from refaat.infrastructure.generation import OpenAICompatibleGenerator
from refaat.programs import (
    GovernedReActProgram,
    KeywordToolPlanner,
    ModelToolPlanner,
    build_tool_manifest,
    validate_tool_arguments,
)

_LOOKUP_SCHEMA: JsonObject = {
    "type": "object",
    "properties": {"topic": {"type": "string"}},
    "required": ["topic"],
    "additionalProperties": False,
}


class _ArgSkill:
    name = "lookup"
    description = "Arg-taking lookup."
    parameters = _LOOKUP_SCHEMA

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        del context
        return {"echo": arguments}


class _NoArgSkill:
    name = "health"
    description = "No-argument health check."

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        del context, arguments
        return {"ok": True}


class _FakeToolCallingGenerator:
    def __init__(self, proposal: ToolProposal | None) -> None:
        self._proposal = proposal
        self.seen: list[tuple[str, ...]] = []

    @property
    def model_id(self) -> str:
        return "fake-tool-model"

    def propose_tool_call(self, query: str, tools: tuple[ToolSpec, ...], used: frozenset[str]) -> ToolProposal | None:
        del query, used
        self.seen.append(tuple(spec.name for spec in tools))
        return self._proposal


def test_validate_tool_arguments_accepts_a_conforming_object() -> None:
    ok, reason = validate_tool_arguments({"topic": "rollback"}, _LOOKUP_SCHEMA)
    assert ok
    assert reason == "valid"


def test_validate_tool_arguments_rejects_missing_required() -> None:
    ok, reason = validate_tool_arguments({}, _LOOKUP_SCHEMA)
    assert not ok
    assert reason == "missing required field: topic"


def test_validate_tool_arguments_rejects_wrong_type() -> None:
    ok, reason = validate_tool_arguments({"topic": 3}, _LOOKUP_SCHEMA)
    assert not ok
    assert reason == "field 'topic' must be of type string"


def test_validate_tool_arguments_rejects_extra_property() -> None:
    ok, reason = validate_tool_arguments({"topic": "x", "extra": 1}, _LOOKUP_SCHEMA)
    assert not ok
    assert reason == "unexpected field: extra"


def test_validate_tool_arguments_rejects_a_non_object() -> None:
    ok, reason = validate_tool_arguments([], _LOOKUP_SCHEMA)  # type: ignore[arg-type]
    assert not ok
    assert reason == "value must be an object"


def test_boolean_is_not_an_integer() -> None:
    schema: JsonObject = {"type": "object", "properties": {"count": {"type": "integer"}}}
    ok, _ = validate_tool_arguments({"count": True}, schema)
    assert not ok


def test_build_tool_manifest_reads_parameters_and_defaults_empty() -> None:
    manifest = build_tool_manifest(
        {"lookup": _ArgSkill(), "health": _NoArgSkill()},
        (ToolGrant("lookup"), ToolGrant("health"), ToolGrant("missing")),
    )
    by_name = {spec.name: spec for spec in manifest}
    assert set(by_name) == {"lookup", "health"}
    assert by_name["lookup"].parameters == _LOOKUP_SCHEMA
    assert by_name["health"].parameters == {"type": "object", "properties": {}, "required": []}


def test_model_planner_returns_proposal_then_stops() -> None:
    proposal = ToolProposal(tool="lookup", arguments={"topic": "rollback"})
    generator = _FakeToolCallingGenerator(proposal)
    manifest = (ToolSpec(name="lookup", description="Arg-taking lookup.", parameters=_LOOKUP_SCHEMA),)
    planner = ModelToolPlanner(generator, manifest)

    assert planner.tools() == ("lookup",)
    assert planner.plan("how does rollback work", frozenset()) == proposal
    assert planner.plan("how does rollback work", frozenset({"lookup"})) is None
    assert generator.seen == [("lookup",)]


def test_model_manifest_pins_a_distinct_artifact_id() -> None:
    policy = ReActPolicy(
        granted_scopes=("read:corpus",),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        grants=(ToolGrant("lookup", required_scopes=("read:corpus",)),),
    )
    skills = {"lookup": _ArgSkill()}
    manifest_a = build_tool_manifest(skills, policy.grants)
    manifest_b = (
        ToolSpec(
            name="lookup",
            description="Arg-taking lookup.",
            parameters={"type": "object", "properties": {"topic": {"type": "integer"}}},
        ),
    )
    program_a = GovernedReActProgram(
        policy=policy, planner=ModelToolPlanner(_FakeToolCallingGenerator(None), manifest_a), skills=skills
    )
    program_b = GovernedReActProgram(
        policy=policy, planner=ModelToolPlanner(_FakeToolCallingGenerator(None), manifest_b), skills=skills
    )
    keyword = GovernedReActProgram(policy=policy, planner=KeywordToolPlanner({"lookup": ("rollback",)}), skills=skills)

    id_a = program_a.artifact().artifact_id
    id_b = program_b.artifact().artifact_id
    id_keyword = keyword.artifact().artifact_id
    assert id_a != id_b
    assert id_keyword != id_a
    assert id_keyword != id_b


def test_openai_generator_proposes_a_tool_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read().decode())
        assert body["tools"][0]["function"]["name"] == "lookup"
        assert body["tool_choice"] == "auto"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [{"function": {"name": "lookup", "arguments": '{"topic": "rollback"}'}}]
                        }
                    }
                ]
            },
        )

    generator = OpenAICompatibleGenerator(
        base_url="https://models.test/v1",
        api_key="secret",
        model="model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    spec = ToolSpec(name="lookup", description="Arg-taking lookup.", parameters=_LOOKUP_SCHEMA)
    proposal = generator.propose_tool_call("how does rollback work", (spec,), frozenset())
    assert proposal is not None
    assert proposal.tool == "lookup"
    assert proposal.arguments == {"topic": "rollback"}


def test_openai_generator_returns_none_without_a_tool_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "no tool needed"}}]})

    generator = OpenAICompatibleGenerator(
        base_url="https://models.test/v1",
        api_key="secret",
        model="model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    spec = ToolSpec(name="lookup", description="Arg-taking lookup.", parameters=_LOOKUP_SCHEMA)
    assert generator.propose_tool_call("no tools", (spec,), frozenset()) is None
    assert generator.propose_tool_call("empty manifest", (), frozenset()) is None
