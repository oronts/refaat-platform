from __future__ import annotations

import pytest

from refaat.application.runtime import Refaat, RefaatProject
from refaat.domain.enums import SideEffectClass
from refaat.domain.models import DocumentInput, JsonObject, ReActPolicy
from refaat.mcp.remote import MCPRemoteSkill, mcp_tool_grant
from refaat.programs import BoundedInferencePolicy, GovernedReActProgram, KeywordToolPlanner
from refaat.programs.policy import REFINE

pytestmark = pytest.mark.integration


class _FakeInvoker:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, name: str, arguments: JsonObject) -> JsonObject:
        del arguments
        self.calls += 1
        return {"remote": name, "status": "ok"}


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


def _policy(ceiling: SideEffectClass) -> ReActPolicy:
    return ReActPolicy(
        max_iterations=4,
        granted_scopes=("call:weather",),
        side_effect_ceiling=ceiling,
        grants=(mcp_tool_grant("weather", required_scopes=("call:weather",)),),
    )


def _program(policy: ReActPolicy, invoker: _FakeInvoker) -> GovernedReActProgram:
    skill = MCPRemoteSkill(name="weather", description="External weather tool.", invoker=invoker)
    planner = KeywordToolPlanner({"weather": ("rollback",)})
    return GovernedReActProgram(policy=policy, planner=planner, skills={"weather": skill})


def test_external_mcp_tool_runs_under_an_external_ceiling(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="mcp")
    _ingest(runtime, project)
    invoker = _FakeInvoker()
    program = _program(_policy(SideEffectClass.EXTERNAL), invoker)

    result = program.execute(project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None)

    step = result.outputs["react_steps"][0]
    assert step["authorized"] is True
    assert step["observation"] == {"remote": "weather", "status": "ok"}
    assert invoker.calls == 1
    assert result.grounded
    assert result.citations


def test_external_mcp_tool_denied_under_read_only_ceiling(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="mcp")
    _ingest(runtime, project)
    invoker = _FakeInvoker()
    external = _program(_policy(SideEffectClass.EXTERNAL), invoker)
    read_only_invoker = _FakeInvoker()
    read_only = _program(_policy(SideEffectClass.READ_ONLY), read_only_invoker)

    external_result = external.execute(
        project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None
    )
    read_only_result = read_only.execute(
        project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None
    )

    denied = read_only_result.outputs["react_steps"][0]
    assert denied["authorized"] is False
    assert "exceeds ceiling" in denied["reason"]
    assert read_only_invoker.calls == 0
    assert read_only_result.answer == external_result.answer


def test_external_grant_is_pinned_into_the_artifact(runtime: Refaat) -> None:
    invoker = _FakeInvoker()
    external = _program(_policy(SideEffectClass.EXTERNAL), invoker).artifact().artifact_id
    read_only = _program(_policy(SideEffectClass.READ_ONLY), invoker).artifact().artifact_id

    assert external != read_only


def test_external_mcp_tool_composes_under_bounded_refine(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="mcp")
    _ingest(runtime, project)
    invoker = _FakeInvoker()
    program = _program(_policy(SideEffectClass.EXTERNAL), invoker)
    bounded = BoundedInferencePolicy(program, mode=REFINE, attempts=2)

    result = bounded.execute(project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None)

    assert result.grounded
    assert invoker.calls >= 1
