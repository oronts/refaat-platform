from __future__ import annotations

import pytest

from refaat.application.runtime import Refaat, RefaatProject
from refaat.domain.models import DocumentInput
from refaat.evaluation.dataset import Dataset, EvaluationCase
from refaat.evaluation.runner import ProgramEvaluator
from refaat.programs import (
    BEST_OF,
    BoundedInferencePolicy,
    GroundedRagProgram,
    MultiAgentOrchestrator,
    TaskRouter,
    Tunables,
)

pytestmark = pytest.mark.integration


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
    runtime.ingest(
        project.workspace_id,
        project.corpus_id,
        DocumentInput(
            source_id="billing.txt",
            title="Billing",
            media_type="text/plain",
            content=b"A refund reverses a paid invoice and credits the original payment method.",
        ),
    )


def _members() -> dict[str, GroundedRagProgram]:
    return {
        "ops": GroundedRagProgram(tunables=Tunables(top_k=1)),
        "billing": GroundedRagProgram(tunables=Tunables(top_k=3)),
    }


def _router() -> TaskRouter:
    return TaskRouter({"ops": ("rollback", "deploy"), "billing": ("refund", "invoice")})


def test_orchestrator_routes_and_merges_a_grounded_answer(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="orchestrator")
    _ingest(runtime, project)
    members = _members()
    orchestrator = MultiAgentOrchestrator(members, _router())

    result = orchestrator.execute(project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None)

    assert result.program == "multi-agent"
    assert result.outputs["orchestrator"]["selected"] == ["ops"]
    assert result.grounded
    assert result.citations
    known = {chunk.id for chunk in runtime.store.list_chunks(project.workspace_id, project.corpus_id)}
    assert {citation.chunk_id for citation in result.citations} <= known
    ops_result = members["ops"].execute(
        project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None
    )
    ops_tokens = float(ops_result.usage.get("context_tokens", 0.0))
    assert result.usage["orchestrator_member_tokens"] == ops_tokens
    extra = orchestrator._definition_extra()["orchestrator"]
    assert {entry["artifact_id"] for entry in extra["members"]} == {
        members["ops"].artifact().artifact_id,
        members["billing"].artifact().artifact_id,
    }


def test_handoff_records_typed_messages_and_changes_the_artifact_id(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="orchestrator-handoff")
    _ingest(runtime, project)
    members = _members()
    router = _router()
    off = MultiAgentOrchestrator(members, router, handoff=False)
    on = MultiAgentOrchestrator(members, router, handoff=True)
    # Handoff creates durable agents and messages, a real behavioral difference, so it is part of the
    # artifact identity, while the id stays deterministic across reconstruction of an identical program.
    assert off.artifact().artifact_id != on.artifact().artifact_id
    assert on.artifact().artifact_id == MultiAgentOrchestrator(members, router, handoff=True).artifact().artifact_id

    result = on.execute(project, inputs={"question": "process a refund"}, user_id=None, session_id=None)
    assert result.outputs["orchestrator"]["selected"] == ["billing"]
    handoff = result.outputs["orchestrator"]["handoff"]
    orchestrator_agent_id = handoff["orchestrator_agent"]
    specialist_agent_id = handoff["specialists"]["billing"]

    specialist = runtime.agents.store.get_agent(project.workspace_id, specialist_agent_id)
    assert specialist["kind"] == "specialist"
    assert specialist["parent_agent_id"] == orchestrator_agent_id

    messages = runtime.agents.receive(project.workspace_id, specialist_agent_id)
    assert len(messages) == 1
    message = messages[0]
    assert message.kind == "handoff"
    assert message.sender_agent_id == orchestrator_agent_id
    assert message.recipient_agent_id == specialist_agent_id
    assert set(message.payload) == {"task", "specialist"}
    assert message.payload["specialist"] == "billing"
    assert message.correlation_id == handoff["correlation_id"]

    again = on.execute(project, inputs={"question": "process a refund"}, user_id=None, session_id=None)
    # The correlation id is deterministic: an identical task rewrites the same id.
    assert again.outputs["orchestrator"]["handoff"]["correlation_id"] == message.correlation_id
    assert again.outputs["orchestrator"]["selected"] == ["billing"]


def test_orchestrator_evaluates_and_composes_under_bounded_inference(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="orchestrator-eval")
    _ingest(runtime, project)
    orchestrator = MultiAgentOrchestrator(_members(), _router())
    bound = project.programs.register(orchestrator)

    dataset = Dataset(
        name="ops",
        cases=(
            EvaluationCase(id="c0", query="how does rollback work", expected_terms=("rollback",)),
            EvaluationCase(id="c1", query="how does a refund work", expected_terms=("refund",)),
        ),
    )
    report = ProgramEvaluator(bound).evaluate(dataset)
    assert report.case_count == 2
    assert report.program == "multi-agent"

    policy = BoundedInferencePolicy(orchestrator, mode=BEST_OF, attempts=2)
    assert policy.artifact().artifact_id != orchestrator.artifact().artifact_id
    result = policy.execute(project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None)
    assert result.grounded
    assert result.citations
