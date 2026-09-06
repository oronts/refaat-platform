from __future__ import annotations

import pytest
from sqlalchemy import update

from refaat.application.worker import DurableWorker
from refaat.core.harness import BASE_SYSTEM_PROMPT_HASH, system_prompt
from refaat.domain.models import DocumentInput, Feedback, QueryRequest
from refaat.errors import ConflictError, NotFoundError, ValidationError
from refaat.infrastructure.database.schema import schedules
from refaat.infrastructure.events import OutboxPublisher

pytestmark = pytest.mark.integration


def ingest_policy(runtime, workspace_id: str, corpus_id: str, source_id: str = "policy.md"):
    return runtime.ingest(
        workspace_id,
        corpus_id,
        DocumentInput(
            source_id=source_id,
            title="Policy",
            media_type="text/markdown",
            content=b"Production changes require peer review and a rollback plan.",
        ),
    )


def test_feedback_learning_session_compaction_and_harness(
    runtime,
    workspace_id: str,
    corpus_id: str,
) -> None:
    ingested = ingest_policy(runtime, workspace_id, corpus_id)
    chunk = runtime.store.list_chunks(workspace_id, corpus_id)[0]

    runtime.submit_feedback(
        workspace_id,
        corpus_id,
        Feedback(
            query_id=None,
            user_id="alice",
            rating=1,
            selected_chunk_ids=(chunk.id,),
        ),
    )
    runtime.submit_feedback(
        workspace_id,
        corpus_id,
        Feedback(query_id=None, user_id=None, rating=1, selected_chunk_ids=(chunk.id,)),
    )
    profile = runtime.store.get_preferences(workspace_id, corpus_id, "user:alice")
    assert profile[f"document:{ingested.document_id}"]["confidence"] >= 0.6

    snapshot = runtime.harness.create_snapshot(
        workspace_id=workspace_id,
        corpus_id=corpus_id,
        supplemental={
            "retrieval": {"semantic_weight": 0.6, "lexical_weight": 0.3, "graph_weight": 0.1},
            "instruction_suffix": "Prefer a compact answer.",
        },
        evidence={"evaluation_run": "eval_1"},
        activate=True,
    )
    assert snapshot.base_prompt_hash == BASE_SYSTEM_PROMPT_HASH
    assert "Supplemental behavior" in system_prompt(snapshot.supplemental)

    session_id = runtime.store.create_session(
        workspace_id,
        corpus_id,
        user_id="alice",
        goal="Understand production controls",
    )
    for index in range(3):
        runtime.query(
            QueryRequest(
                workspace_id=workspace_id,
                corpus_id=corpus_id,
                query=f"What does the policy require? detail {index}",
                user_id="alice",
                session_id=session_id,
            )
        )
    session = runtime.store.get_session(workspace_id, session_id)
    turns = runtime.store.list_session_turns(workspace_id, session_id)
    assert session["summary"]
    assert len(turns) == runtime.settings.retained_session_turns

    context = runtime.retrieve(
        QueryRequest(
            workspace_id=workspace_id,
            corpus_id=corpus_id,
            query="What does production require?",
            user_id="alice",
            session_id=session_id,
        )
    )
    assert {"profile", "harness", "session"} <= set(context.routes)


def test_harness_validation_proposals_and_rollback(runtime, workspace_id: str, corpus_id: str) -> None:
    with pytest.raises(ValidationError, match="recorded evidence"):
        runtime.harness.create_snapshot(
            workspace_id=workspace_id,
            corpus_id=corpus_id,
            supplemental={},
            evidence={},
        )
    with pytest.raises(ValidationError, match="Unsupported harness"):
        runtime.harness.create_snapshot(
            workspace_id=workspace_id,
            corpus_id=corpus_id,
            supplemental={"base_prompt": "rewrite"},
            evidence={"reason": "unsafe"},
        )
    with pytest.raises(ValidationError, match="At least 10"):
        runtime.harness.propose_from_metrics({"case_count": 9})

    retrieval_proposal = runtime.harness.propose_from_metrics(
        {"case_count": 10, "citation_recall": 0.5, "answer_hit_rate": 0.9}
    )
    answer_proposal = runtime.harness.propose_from_metrics(
        {"case_count": 10, "citation_recall": 0.9, "answer_hit_rate": 0.5}
    )
    assert retrieval_proposal["retrieval"]["top_k"] == 16
    assert answer_proposal["retrieval"]["semantic_weight"] == 0.6
    assert (
        runtime.harness.propose_from_metrics({"case_count": 10, "citation_recall": 0.9, "answer_hit_rate": 0.9}) == {}
    )

    first = runtime.harness.create_snapshot(
        workspace_id=workspace_id,
        corpus_id=corpus_id,
        supplemental={"answer_style": "concise"},
        evidence={"run": "one"},
        activate=True,
    )
    second = runtime.harness.create_snapshot(
        workspace_id=workspace_id,
        corpus_id=corpus_id,
        supplemental={"answer_style": "detailed"},
        evidence={"run": "two"},
        activate=True,
    )
    assert second.parent_id == first.id
    rolled_back = runtime.store.activate_harness_snapshot(workspace_id, first.id)
    assert rolled_back.id == first.id
    assert runtime.harness.active(workspace_id, corpus_id)["answer_style"] == "concise"
    with pytest.raises(NotFoundError):
        runtime.store.activate_harness_snapshot(workspace_id, "missing")


def test_worker_jobs_schedules_evaluations_and_outbox(
    runtime,
    workspace_id: str,
    corpus_id: str,
) -> None:
    ingest_policy(runtime, workspace_id, corpus_id)
    delivered: list[tuple[str, dict[str, object]]] = []

    class Sink:
        def emit(self, topic: str, payload: dict[str, object]) -> None:
            delivered.append((topic, payload))

    assert OutboxPublisher(runtime.store, [Sink()]).publish() == 1
    assert delivered[0][0] == "document.indexed"
    assert OutboxPublisher(runtime.store, [Sink()]).publish() == 0

    worker = DurableWorker(runtime)
    worker.register("echo", lambda _workspace, payload: {"echo": payload["value"]})
    with pytest.raises(ValueError, match="already registered"):
        worker.register("echo", lambda _workspace, payload: payload)

    job_id = runtime.store.enqueue_job(workspace_id, "echo", {"value": "ok"})
    assert worker.run_once()
    job = runtime.store.get_job(workspace_id, job_id)
    assert str(job["status"]) == "succeeded"
    assert job["result"] == {"echo": "ok"}

    missing_handler = runtime.store.enqueue_job(workspace_id, "unknown", {})
    assert worker.run_once()
    assert "No handler" in str(runtime.store.get_job(workspace_id, missing_handler)["last_error"])

    schedule_id = runtime.store.create_schedule(
        workspace_id,
        "scheduled-echo",
        "echo",
        {"value": "scheduled"},
        interval_seconds=60,
    )
    with runtime.store.engine.begin() as connection:
        connection.execute(
            update(schedules).where(schedules.c.id == schedule_id).values(next_run_at=runtime.store.clock.now())
        )
    assert runtime.store.dispatch_due_schedules() == 1
    assert worker.run_once()

    suite_id = runtime.store.create_evaluation_suite(
        workspace_id,
        "smoke",
        [{"query": "What is required?", "expected_terms": ["review"]}],
    )
    evaluation_job = runtime.store.enqueue_job(
        workspace_id,
        "evaluation.run",
        {"suite_id": suite_id, "corpus_id": corpus_id},
    )
    assert worker.run_once()
    evaluation_result = runtime.store.get_job(workspace_id, evaluation_job)["result"]
    run = runtime.store.get_evaluation_run(workspace_id, evaluation_result["run_id"])
    assert str(run["status"]) == "succeeded"
    assert run["metrics"]["case_count"] == 1


def test_store_conflicts_and_authorization_boundaries(runtime, workspace_id: str, corpus_id: str) -> None:
    with pytest.raises(ConflictError):
        runtime.create_workspace("acme", "Duplicate")
    with pytest.raises(ConflictError):
        runtime.create_corpus(workspace_id, "handbook")
    with pytest.raises(NotFoundError):
        runtime.store.get_job(workspace_id, "missing")
