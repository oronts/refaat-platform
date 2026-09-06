from __future__ import annotations

import pytest

from refaat.application.runtime import Refaat, RefaatProject
from refaat.domain.enums import SideEffectClass
from refaat.domain.governance import ToolCapability
from refaat.domain.models import ContextPacket, DocumentInput, JsonObject, ReActPolicy, ToolGrant
from refaat.programs import GovernedReActProgram, GroundedRagProgram, KeywordToolPlanner, Tunables
from refaat.programs.registry import BoundProgram
from refaat.skills import ProgramSkill, SkillRegistry

pytestmark = pytest.mark.integration


class _PagerSkill:
    name = "pager"
    description = "External escalation, denied under a read-only ceiling."

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        del context, arguments
        return {"paged": True}


class _DigestSkill:
    name = "evidence_digest"
    description = "Read-only digest over the retrieved context."

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        del arguments
        return {"chunks": len(context.hits)}


class _DigestWithIdSkill:
    name = "evidence_digest"
    description = "Plain read-only skill that also exposes an artifact_id attribute."
    artifact_id = "sha256:not-a-program-skill"

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        del arguments
        return {"chunks": len(context.hits)}


def _ops_project(runtime: Refaat) -> RefaatProject:
    project = runtime.project(workspace="acme", corpus="ops")
    for source_id, title, text in (
        ("rollback.txt", "Rollback", "Rollback restores the previously approved harness snapshot before deployment."),
        (
            "incident.txt",
            "Incident",
            "During an incident the on-call engineer coordinates a rollback and posts updates.",
        ),
        ("deploy.txt", "Deploy", "A deployment proceeds only after a rollback plan and a passing canary."),
    ):
        runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(source_id=source_id, title=title, media_type="text/plain", content=text.encode()),
        )
    return project


def _program_skill(
    project: RefaatProject, *, tunables: Tunables | None = None, name: str = "ops_program"
) -> ProgramSkill:
    program = GroundedRagProgram(tunables=tunables) if tunables is not None else GroundedRagProgram()
    return ProgramSkill(BoundProgram(project, program), name=name)


def test_program_skill_runs_as_a_governed_tool_resolved_from_the_registry(runtime: Refaat) -> None:
    project = _ops_project(runtime)
    registry = SkillRegistry()
    registry.register(_program_skill(project))
    registry.register(_PagerSkill())

    policy = ReActPolicy(
        max_iterations=4,
        granted_scopes=("read",),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        grants=(
            ToolGrant("ops_program", required_scopes=("read",), side_effect=SideEffectClass.READ_ONLY),
            ToolGrant("pager", required_scopes=("read",), side_effect=SideEffectClass.EXTERNAL),
        ),
    )
    planner = KeywordToolPlanner({"ops_program": ("rollback",), "pager": ("rollback",)})
    react = GovernedReActProgram.from_registry(policy=policy, planner=planner, registry=registry)

    result = react.execute(
        project, inputs={"question": "how does rollback work during an incident"}, user_id=None, session_id=None
    )

    by_tool = {step["tool"]: step for step in result.outputs["react_steps"]}
    program_step = by_tool["ops_program"]
    assert program_step["authorized"] is True
    observation = program_step["observation"]
    assert observation["answer"].strip()
    assert observation["citations"]
    expected_id = project.programs.get("grounded-rag").program.artifact().artifact_id
    assert observation["artifact_id"] == expected_id

    pager_step = by_tool["pager"]
    assert pager_step["authorized"] is False
    assert "exceeds ceiling" in pager_step["reason"]
    assert pager_step["observation"] == {}


def test_program_skill_artifact_id_uses_the_property_not_a_to_json_key(runtime: Refaat) -> None:
    project = _ops_project(runtime)
    skill = _program_skill(project)
    bound = project.programs.get("grounded-rag")
    assert skill.artifact_id == bound.program.artifact().artifact_id
    assert "artifact_id" not in bound.program.artifact().to_json()


def _react(project: RefaatProject, skill: object) -> GovernedReActProgram:
    policy = ReActPolicy(
        granted_scopes=("read",),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        grants=(ToolGrant("ops_program", required_scopes=("read",), side_effect=SideEffectClass.READ_ONLY),),
    )
    planner = KeywordToolPlanner({"ops_program": ("rollback",)})
    return GovernedReActProgram(policy=policy, planner=planner, skills={"ops_program": skill})


def test_granting_a_program_skill_changes_the_outer_artifact_id(runtime: Refaat) -> None:
    project = _ops_project(runtime)
    base = _react(project, _program_skill(project)).artifact().artifact_id
    tuned = _react(project, _program_skill(project, tunables=Tunables(top_k=4))).artifact().artifact_id
    assert base != tuned


def test_granting_only_plain_skills_leaves_the_artifact_id_stable() -> None:
    policy = ReActPolicy(
        granted_scopes=("read",),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        grants=(ToolGrant("evidence_digest", required_scopes=("read",), side_effect=SideEffectClass.READ_ONLY),),
    )
    planner = KeywordToolPlanner({"evidence_digest": ("rollback",)})
    first = GovernedReActProgram(policy=policy, planner=planner, skills={"evidence_digest": _DigestSkill()})
    second = GovernedReActProgram(policy=policy, planner=planner, skills={"evidence_digest": _DigestSkill()})
    assert first.artifact().artifact_id == second.artifact().artifact_id


def test_a_plain_skill_folds_its_class_identity_not_its_forged_artifact_id() -> None:
    policy = ReActPolicy(
        granted_scopes=("read",),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        grants=(ToolGrant("evidence_digest", required_scopes=("read",), side_effect=SideEffectClass.READ_ONLY),),
        capabilities=(
            ToolCapability(
                name="evidence_digest",
                required_scopes=frozenset({"read"}),
                side_effect=SideEffectClass.READ_ONLY,
                requires_approval=True,
            ),
        ),
    )
    planner = KeywordToolPlanner({"evidence_digest": ("rollback",)})
    with_attr = GovernedReActProgram(policy=policy, planner=planner, skills={"evidence_digest": _DigestWithIdSkill()})
    grant = with_attr._definition_extra()["react"]["grants"][0]
    assert grant.get("skill_identity", {}).get("qualname", "").endswith("_DigestWithIdSkill")
    assert "artifact_id" not in grant


def _plain_skill_class(module: str, qualname: str) -> type:
    def run(self: object, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        del context, arguments
        return {}

    cls = type(qualname.rsplit(".", 1)[-1], (), {"name": "evidence_digest", "description": "d", "run": run})
    cls.__module__ = module
    cls.__qualname__ = qualname
    return cls


def _governed_with(skill: object, *, capabilities: bool) -> GovernedReActProgram:
    policy = ReActPolicy(
        granted_scopes=("read",),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        grants=(ToolGrant("evidence_digest", required_scopes=("read",), side_effect=SideEffectClass.READ_ONLY),),
        capabilities=(
            (
                ToolCapability(
                    name="evidence_digest",
                    required_scopes=frozenset({"read"}),
                    side_effect=SideEffectClass.READ_ONLY,
                    requires_approval=True,
                ),
            )
            if capabilities
            else ()
        ),
    )
    planner = KeywordToolPlanner({"evidence_digest": ("rollback",)})
    return GovernedReActProgram(policy=policy, planner=planner, skills={"evidence_digest": skill})


def test_a_no_capability_plain_skill_program_folds_no_skill_identity() -> None:
    skill = _DigestSkill()
    base = _governed_with(skill, capabilities=False)._definition_extra()["react"]["grants"][0]
    governed = _governed_with(skill, capabilities=True)._definition_extra()["react"]["grants"][0]
    assert "skill_identity" not in base
    assert "skill_identity" in governed
    # Without capabilities the class identity never enters the artifact, so two different plain skill
    # classes yield the same no-capability id.
    a = _governed_with(_plain_skill_class("m.a", "SkillA")(), capabilities=False).artifact().artifact_id
    b = _governed_with(_plain_skill_class("m.b", "SkillB")(), capabilities=False).artifact().artifact_id
    assert a == b


def test_structured_skill_identity_avoids_a_module_qualname_flatten_collision() -> None:
    # "pkg" + "Outer.Inner" and "pkg.Outer" + "Inner" both flatten to "pkg.Outer.Inner"; separate
    # module and qualname fields keep the two classes distinct.
    x = _governed_with(_plain_skill_class("pkg", "Outer.Inner")(), capabilities=True)
    y = _governed_with(_plain_skill_class("pkg.Outer", "Inner")(), capabilities=True)
    assert x.artifact().artifact_id != y.artifact().artifact_id
