from __future__ import annotations

from collections.abc import Iterator

import pytest

from refaat.application.runtime import Refaat, RefaatProject
from refaat.domain.enums import SideEffectClass
from refaat.domain.models import (
    ContextPacket,
    DocumentInput,
    JsonObject,
    ReActPolicy,
    ToolCallDelta,
    ToolGrant,
    ToolProposal,
    ToolSpec,
)
from refaat.programs import (
    REFINE,
    BoundedInferencePolicy,
    GovernedReActProgram,
    GroundedRagProgram,
    KeywordToolPlanner,
    ModelToolPlanner,
    StreamingModelToolPlanner,
    build_tool_manifest,
)

pytestmark = pytest.mark.integration

_ARG_SCHEMA: JsonObject = {
    "type": "object",
    "properties": {"topic": {"type": "string"}},
    "required": ["topic"],
    "additionalProperties": False,
}


class _RecordingSkill:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = f"{name} over the retrieved context."
        self.parameters = _ARG_SCHEMA
        self.calls: list[JsonObject] = []

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        del context
        self.calls.append(arguments)
        return {"echoed": arguments}


class _NotifySkill:
    name = "notify"
    description = "External notification."

    def __init__(self) -> None:
        self.calls = 0

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        del context, arguments
        self.calls += 1
        return {"sent": True}


class _ScriptedToolGenerator:
    def __init__(self, proposals: list[ToolProposal]) -> None:
        self._queue = list(proposals)

    @property
    def model_id(self) -> str:
        return "scripted-tool-model"

    def propose_tool_call(self, query: str, tools: tuple[ToolSpec, ...], used: frozenset[str]) -> ToolProposal | None:
        del query, tools, used
        if not self._queue:
            return None
        return self._queue.pop(0)


def _ingest(runtime: Refaat, project: RefaatProject) -> None:
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


def _policy() -> ReActPolicy:
    return ReActPolicy(
        max_iterations=4,
        granted_scopes=("read:corpus", "write:crm"),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        grants=(
            ToolGrant("lookup", required_scopes=("read:corpus",), side_effect=SideEffectClass.READ_ONLY),
            ToolGrant("search", required_scopes=("read:corpus",), side_effect=SideEffectClass.READ_ONLY),
            ToolGrant("notify", required_scopes=("write:crm",), side_effect=SideEffectClass.EXTERNAL),
        ),
    )


def _skills() -> dict[str, object]:
    return {"lookup": _RecordingSkill("lookup"), "search": _RecordingSkill("search"), "notify": _NotifySkill()}


def _program(skills: dict[str, object]) -> GovernedReActProgram:
    policy = _policy()
    manifest = build_tool_manifest(skills, policy.grants)  # type: ignore[arg-type]
    generator = _ScriptedToolGenerator(
        [
            ToolProposal(tool="lookup", arguments={"topic": "rollback"}),
            ToolProposal(tool="search", arguments={"topic": 5}),
            ToolProposal(tool="notify", arguments={}),
        ]
    )
    planner = ModelToolPlanner(generator, manifest)
    return GovernedReActProgram(policy=policy, planner=planner, skills=skills)  # type: ignore[arg-type]


def test_model_planned_loop_validates_governs_and_executes(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="toolcall")
    _ingest(runtime, project)
    skills = _skills()
    program = _program(skills)

    result = program.execute(project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None)

    by_tool = {step["tool"]: step for step in result.outputs["react_steps"]}
    lookup = skills["lookup"]
    assert isinstance(lookup, _RecordingSkill)
    assert lookup.calls == [{"topic": "rollback"}]
    assert by_tool["lookup"]["authorized"] is True
    assert by_tool["lookup"]["arguments"] == {"topic": "rollback"}

    assert by_tool["search"]["authorized"] is False
    assert "argument validation failed" in by_tool["search"]["reason"]
    assert "must be of type string" in by_tool["search"]["reason"]
    search = skills["search"]
    assert isinstance(search, _RecordingSkill)
    assert search.calls == []

    assert by_tool["notify"]["authorized"] is False
    assert "exceeds ceiling" in by_tool["notify"]["reason"]
    notify = skills["notify"]
    assert isinstance(notify, _NotifySkill)
    assert notify.calls == 0

    assert result.grounded
    assert result.citations


def test_observation_only_invariant_matches_plain_grounded_answer(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="toolcall")
    _ingest(runtime, project)
    react = _program(_skills()).execute(
        project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None
    )
    plain = GroundedRagProgram().execute(
        project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None
    )
    assert react.answer == plain.answer
    assert [c.chunk_id for c in react.citations] == [c.chunk_id for c in plain.citations]


class _ScriptedStreamingToolGenerator:
    def __init__(self, scripts: list[list[ToolCallDelta]]) -> None:
        self._scripts = list(scripts)

    @property
    def model_id(self) -> str:
        return "scripted-tool-model"

    def stream_tool_call(
        self, query: str, tools: tuple[ToolSpec, ...], used: frozenset[str]
    ) -> Iterator[ToolCallDelta]:
        del query, tools, used
        if not self._scripts:
            return
        yield from self._scripts.pop(0)


def _streaming_scripts() -> list[list[ToolCallDelta]]:
    return [
        [
            ToolCallDelta(tool="lookup"),
            ToolCallDelta(arguments_fragment='{"topic": '),
            ToolCallDelta(arguments_fragment='"rollback"}'),
            ToolCallDelta(done=True),
        ],
        [
            ToolCallDelta(tool="search"),
            ToolCallDelta(arguments_fragment='{"topic": 5}'),
            ToolCallDelta(done=True),
        ],
        [ToolCallDelta(tool="notify"), ToolCallDelta(done=True)],
    ]


def _streaming_program(skills: dict[str, object]) -> GovernedReActProgram:
    return _streaming_program_with(skills, _streaming_scripts())


def _streaming_program_with(skills: dict[str, object], scripts: list[list[ToolCallDelta]]) -> GovernedReActProgram:
    policy = _policy()
    manifest = build_tool_manifest(skills, policy.grants)  # type: ignore[arg-type]
    generator = _ScriptedStreamingToolGenerator(scripts)
    planner = StreamingModelToolPlanner(generator, manifest)
    return GovernedReActProgram(policy=policy, planner=planner, skills=skills)  # type: ignore[arg-type]


def _keyword_program(skills: dict[str, object]) -> GovernedReActProgram:
    policy = _policy()
    planner = KeywordToolPlanner({"lookup": ("rollback",)})
    return GovernedReActProgram(policy=policy, planner=planner, skills=skills)  # type: ignore[arg-type]


def test_streamed_model_loop_orders_deltas_steps_then_a_single_grounded_final(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="toolcall")
    _ingest(runtime, project)
    skills = _skills()
    program = _streaming_program(skills)

    events = list(program.stream(project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None))
    types = [event["type"] for event in events]

    assert "tool_call_delta" in types
    # Every tool_call_delta precedes the first answer delta: fragments stream
    # before the governed answer is produced.
    assert types.index("tool_call_delta") < types.index("delta")

    steps = [event for event in events if event["type"] == "tool_step"]
    by_tool = {step["tool"]: step for step in steps}
    assert list(by_tool) == ["lookup", "search", "notify"]

    lookup = skills["lookup"]
    assert isinstance(lookup, _RecordingSkill)
    assert lookup.calls == [{"topic": "rollback"}]
    assert by_tool["lookup"]["authorized"] is True

    assert by_tool["search"]["authorized"] is False
    assert "argument validation failed" in by_tool["search"]["reason"]
    search = skills["search"]
    assert isinstance(search, _RecordingSkill)
    assert search.calls == []

    assert by_tool["notify"]["authorized"] is False
    assert "exceeds ceiling" in by_tool["notify"]["reason"]
    notify = skills["notify"]
    assert isinstance(notify, _NotifySkill)
    assert notify.calls == 0

    finals = [event for event in events if event["type"] == "final"]
    assert len(finals) == 1
    final = finals[0]
    assert final["citations"]
    assert len(final["react_steps"]) == 3


def test_streamed_answer_matches_execute_and_plain_grounded(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="toolcall")
    _ingest(runtime, project)

    events = list(
        _streaming_program(_skills()).stream(
            project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None
        )
    )
    streamed_answer = "".join(event["text"] for event in events if event["type"] == "delta")
    final = next(event for event in events if event["type"] == "final")

    executed = _program(_skills()).execute(
        project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None
    )
    plain = GroundedRagProgram().execute(
        project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None
    )

    assert streamed_answer == executed.answer == plain.answer
    streamed_citation_ids = [citation["chunk_id"] for citation in final["citations"]]
    assert streamed_citation_ids == [c.chunk_id for c in executed.citations] == [c.chunk_id for c in plain.citations]


def test_offline_keyword_default_stream_emits_no_tool_call_delta(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="toolcall")
    _ingest(runtime, project)
    skills = _skills()
    program = _keyword_program(skills)

    events = list(program.stream(project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None))
    types = [event["type"] for event in events]

    assert "tool_call_delta" not in types
    steps = [event for event in events if event["type"] == "tool_step"]
    assert [step["tool"] for step in steps] == ["lookup"]
    assert steps[0]["authorized"] is True

    streamed_answer = "".join(event["text"] for event in events if event["type"] == "delta")
    executed = _keyword_program(_skills()).execute(
        project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None
    )
    assert streamed_answer == executed.answer


def test_streaming_and_non_streaming_model_planners_share_one_artifact_id(runtime: Refaat) -> None:
    streamed = _streaming_program(_skills())
    executed = _program(_skills())
    assert streamed.artifact().artifact_id == executed.artifact().artifact_id


def test_streamed_answer_folds_into_the_same_cache_key(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="toolcall")
    _ingest(runtime, project)

    for _ in _streaming_program(_skills()).stream(
        project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None
    ):
        pass

    follow_up = GroundedRagProgram().execute(
        project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None
    )
    assert follow_up.cached is True
    assert "cag:exact" in follow_up.routes


def test_streamed_incomplete_tool_call_without_done_executes_nothing(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="toolcall")
    _ingest(runtime, project)
    skills = _skills()
    program = _streaming_program_with(
        skills,
        [
            [
                ToolCallDelta(tool="lookup"),
                ToolCallDelta(arguments_fragment='{"topic": "rollback"}'),
            ]
        ],
    )

    events = list(program.stream(project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None))

    assert not [event for event in events if event["type"] == "tool_step"]
    lookup = skills["lookup"]
    assert isinstance(lookup, _RecordingSkill)
    assert lookup.calls == []
    final = next(event for event in events if event["type"] == "final")
    assert final["react_steps"] == []
    assert final["citations"]


def test_streamed_post_done_fragments_do_not_corrupt_the_executed_arguments(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="toolcall")
    _ingest(runtime, project)
    skills = _skills()
    program = _streaming_program_with(
        skills,
        [
            [
                ToolCallDelta(tool="lookup"),
                ToolCallDelta(arguments_fragment='{"topic": "rollback"}'),
                ToolCallDelta(done=True),
                ToolCallDelta(arguments_fragment='{"topic": "TAMPERED"}'),
            ]
        ],
    )

    list(program.stream(project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None))

    lookup = skills["lookup"]
    assert isinstance(lookup, _RecordingSkill)
    assert lookup.calls == [{"topic": "rollback"}]


def test_streamed_duplicate_tool_runs_at_most_once(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="toolcall")
    _ingest(runtime, project)
    skills = _skills()
    complete = [
        ToolCallDelta(tool="lookup"),
        ToolCallDelta(arguments_fragment='{"topic": "rollback"}'),
        ToolCallDelta(done=True),
    ]
    program = _streaming_program_with(skills, [list(complete), list(complete)])

    events = list(program.stream(project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None))

    steps = [event for event in events if event["type"] == "tool_step"]
    assert [step["tool"] for step in steps] == ["lookup"]
    lookup = skills["lookup"]
    assert isinstance(lookup, _RecordingSkill)
    assert lookup.calls == [{"topic": "rollback"}]


def test_streamed_unadvertised_tool_never_executes(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="toolcall")
    _ingest(runtime, project)
    skills = _skills()
    program = _streaming_program_with(
        skills,
        [
            [
                ToolCallDelta(tool="ghost"),
                ToolCallDelta(arguments_fragment='{"topic": "rollback"}'),
                ToolCallDelta(done=True),
            ]
        ],
    )

    events = list(program.stream(project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None))

    assert not [event for event in events if event["type"] == "tool_step"]
    for skill in skills.values():
        if isinstance(skill, _RecordingSkill):
            assert skill.calls == []
    notify = skills["notify"]
    assert isinstance(notify, _NotifySkill)
    assert notify.calls == 0


def test_refine_policy_wraps_the_model_planned_program(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="toolcall")
    _ingest(runtime, project)
    wrapped = BoundedInferencePolicy(_program(_skills()), mode=REFINE, attempts=2)

    result = wrapped.execute(project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None)

    assert result.grounded
    assert result.citations


def test_streamed_react_events_never_expose_raw_observations(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="toolcall")
    _ingest(runtime, project)
    program = _streaming_program(_skills())
    events = list(program.stream(project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None))
    assert all("observation" not in event for event in events)
    final = next(event for event in events if event["type"] == "final")
    assert all("observation" not in step for step in final["react_steps"])
