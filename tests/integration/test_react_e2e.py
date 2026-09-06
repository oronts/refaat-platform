from __future__ import annotations

import pytest

from refaat.application.runtime import Refaat, RefaatProject
from refaat.domain.enums import SideEffectClass
from refaat.domain.models import ContextPacket, DocumentInput, JsonObject, ReActPolicy, ToolGrant
from refaat.programs import GovernedReActProgram, GroundedRagProgram, KeywordToolPlanner

pytestmark = pytest.mark.integration


class _LookupSkill:
    name = "lookup"
    description = "Read-only lookup over the retrieved context."

    def __init__(self) -> None:
        self.calls = 0
        self.saw_context = False

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        del arguments
        self.calls += 1
        self.saw_context = isinstance(context, ContextPacket)
        return {"documents": len({hit.chunk.document_id for hit in context.hits})}


class _NotifySkill:
    name = "notify"
    description = "External notification, denied under a read-only ceiling."

    def __init__(self) -> None:
        self.calls = 0

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        del context, arguments
        self.calls += 1
        return {"sent": True}


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


def test_governed_react_runs_the_authorized_tool_and_denies_the_external_one(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="react")
    _ingest(runtime, project)
    lookup, notify = _LookupSkill(), _NotifySkill()
    policy = ReActPolicy(
        max_iterations=4,
        granted_scopes=("read:corpus", "write:crm"),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        grants=(
            ToolGrant("lookup", required_scopes=("read:corpus",), side_effect=SideEffectClass.READ_ONLY),
            ToolGrant("notify", required_scopes=("write:crm",), side_effect=SideEffectClass.EXTERNAL),
        ),
    )
    planner = KeywordToolPlanner({"lookup": ("rollback",), "notify": ("rollback",)})
    program = GovernedReActProgram(policy=policy, planner=planner, skills={"lookup": lookup, "notify": notify})

    result = program.execute(project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None)

    by_tool = {step["tool"]: step for step in result.outputs["react_steps"]}
    assert lookup.calls == 1
    assert lookup.saw_context
    assert notify.calls == 0
    assert by_tool["lookup"]["authorized"] is True
    assert by_tool["notify"]["authorized"] is False
    assert "exceeds ceiling" in by_tool["notify"]["reason"]
    assert result.grounded
    assert result.citations


def test_governed_react_stops_at_the_iteration_cap(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="react")
    _ingest(runtime, project)
    skills = {f"t{index}": _LookupSkill() for index in range(5)}
    grants = tuple(ToolGrant(name, required_scopes=("read:corpus",)) for name in skills)
    policy = ReActPolicy(max_iterations=2, granted_scopes=("read:corpus",), grants=grants)
    planner = KeywordToolPlanner(dict.fromkeys(skills, ("rollback",)))
    program = GovernedReActProgram(policy=policy, planner=planner, skills=skills)

    result = program.execute(project, inputs={"question": "rollback please"}, user_id=None, session_id=None)

    assert len(result.outputs["react_steps"]) == 2


def test_governance_is_pinned_into_the_artifact() -> None:
    planner = KeywordToolPlanner({"lookup": ("rollback",)})
    skills = {"lookup": _LookupSkill()}
    grant = (ToolGrant("lookup", required_scopes=("read:corpus",)),)
    read_only = (
        GovernedReActProgram(
            policy=ReActPolicy(
                granted_scopes=("read:corpus",), side_effect_ceiling=SideEffectClass.READ_ONLY, grants=grant
            ),
            planner=planner,
            skills=skills,
        )
        .artifact()
        .artifact_id
    )
    external = (
        GovernedReActProgram(
            policy=ReActPolicy(
                granted_scopes=("read:corpus",), side_effect_ceiling=SideEffectClass.EXTERNAL, grants=grant
            ),
            planner=planner,
            skills=skills,
        )
        .artifact()
        .artifact_id
    )

    assert read_only != external
    assert read_only != GroundedRagProgram().artifact().artifact_id
