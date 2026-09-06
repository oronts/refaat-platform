from __future__ import annotations

import pytest

from refaat.application.runtime import Refaat, RefaatProject
from refaat.domain.enums import SideEffectClass
from refaat.domain.models import ContextPacket, DocumentInput, JsonObject, ReActPolicy, ToolGrant
from refaat.evaluation import Dataset, ProgramEvaluator, trajectory_bundle
from refaat.programs import BEST_OF, BoundedInferencePolicy, GovernedReActProgram, KeywordToolPlanner
from refaat.programs.registry import BoundProgram

pytestmark = pytest.mark.integration


class _LookupSkill:
    name = "lookup"
    description = "Read-only lookup over the retrieved context."

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        del arguments
        return {"documents": len({hit.chunk.document_id for hit in context.hits})}


class _NotifySkill:
    name = "notify"
    description = "External notification, denied under a read-only ceiling."

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        del context, arguments
        return {"sent": True}


def _ops_project(runtime: Refaat) -> RefaatProject:
    project = runtime.project(workspace="acme", corpus="trajectory")
    for source_id, text in (
        ("rollback.txt", "Rollback restores the previously approved harness snapshot before deployment."),
        ("incident.txt", "During an incident the on-call engineer runs a rollback and posts updates."),
    ):
        runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(source_id=source_id, title=source_id, media_type="text/plain", content=text.encode()),
        )
    return project


def _react(project: RefaatProject) -> GovernedReActProgram:
    policy = ReActPolicy(
        max_iterations=4,
        granted_scopes=("read",),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        grants=(
            ToolGrant("lookup", required_scopes=("read",), side_effect=SideEffectClass.READ_ONLY),
            ToolGrant("notify", required_scopes=("read",), side_effect=SideEffectClass.EXTERNAL),
        ),
    )
    planner = KeywordToolPlanner({"lookup": ("rollback",), "notify": ("rollback",)})
    return GovernedReActProgram(
        policy=policy, planner=planner, skills={"lookup": _LookupSkill(), "notify": _NotifySkill()}
    )


def _dataset() -> Dataset:
    return Dataset.from_records(
        "trajectory-eval",
        [
            {"id": "t1", "query": "how does rollback work"},
            {"id": "t2", "query": "who runs a rollback during an incident"},
        ],
    )


def test_trajectory_bundle_scores_a_governed_multi_step_run(runtime: Refaat) -> None:
    project = _ops_project(runtime)
    bound = BoundProgram(project, _react(project))
    report = ProgramEvaluator(bound).evaluate(_dataset(), trajectory_bundle())

    assert report.hard_constraints_passed
    assert report.aggregate["tool_safety"] == 1.0
    assert report.aggregate["tool_authorization"] == 1.0
    assert report.aggregate["step_budget"] == 1.0
    assert report.aggregate["token_cost"] == 1.0
    assert report.aggregate["grounded"] == 1.0
    assert report.mean_tokens > 0


def test_a_tight_step_budget_lowers_step_budget_without_failing_hard_constraints(runtime: Refaat) -> None:
    project = _ops_project(runtime)
    bound = BoundProgram(project, _react(project))
    report = ProgramEvaluator(bound).evaluate(_dataset(), trajectory_bundle(max_steps=1))

    assert report.hard_constraints_passed
    assert report.aggregate["step_budget"] < 1.0


def test_best_of_wrapper_still_yields_react_step_metrics(runtime: Refaat) -> None:
    project = _ops_project(runtime)
    wrapped = BoundedInferencePolicy(_react(project), mode=BEST_OF, attempts=2)
    bound = BoundProgram(project, wrapped)
    report = ProgramEvaluator(bound).evaluate(_dataset(), trajectory_bundle())

    assert report.hard_constraints_passed
    assert report.aggregate["tool_safety"] == 1.0
    assert report.aggregate["tool_authorization"] == 1.0
