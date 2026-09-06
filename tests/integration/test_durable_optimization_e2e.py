from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from sqlalchemy import func, select, update

from refaat.application.runtime import Refaat, RefaatProject
from refaat.application.worker import DurableWorker
from refaat.domain.models import DocumentInput, JsonObject
from refaat.errors import NotFoundError
from refaat.infrastructure.database.schema import jobs, optimization_runs
from refaat.optimization import durable as durable_module
from refaat.optimization import retrieval_search_space
from refaat.optimization.durable import resume_optimization_run
from refaat.programs import GroundedRagProgram

pytestmark = pytest.mark.integration


def _seed(runtime: Refaat) -> RefaatProject:
    project = runtime.project(workspace="acme", corpus="handbook")
    for index in range(4):
        runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id=f"doc{index}.txt",
                title=f"Doc {index}",
                media_type="text/plain",
                content=(
                    f"Runbook {index}: a production release requires sign-off before deploy; "
                    "escalate a Sev1 incident to on-call within fifteen minutes."
                ).encode(),
            ),
        )
    return project


def _dataset(runtime: Refaat) -> Any:
    from refaat.evaluation import Dataset

    return Dataset.from_records(
        "durable-optimization",
        [
            {"query": "production release sign-off", "expected_terms": ["release"]},
            {"query": "incident escalation", "expected_terms": ["incident"]},
        ],
    )


def test_durable_run_matches_synchronous(runtime: Refaat) -> None:
    project = _seed(runtime)
    dataset = _dataset(runtime)
    space = retrieval_search_space(top_k=(2, 6, 12))

    synchronous = project.programs.get("grounded-rag").optimize(dataset, search_space=space)
    run_id = project.programs.get("grounded-rag").enqueue_optimization(dataset, search_space=space)

    assert DurableWorker(runtime).run_once() is True
    run = runtime.store.get_optimization_run(project.workspace_id, run_id)

    assert run["status"] == "succeeded"
    assert run["candidate_total"] == 3
    assert len(runtime.store.list_optimization_candidates(run_id)) == 3
    assert run["result"] == synchronous.to_json()
    assert run["best_artifact_id"] == synchronous.best.artifact_id


def test_resume_skips_checkpointed_ordinals(runtime: Refaat, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _seed(runtime)
    dataset = _dataset(runtime)
    space = retrieval_search_space(top_k=(2, 6, 12))

    synchronous = project.programs.get("grounded-rag").optimize(dataset, search_space=space)
    run_id = project.programs.get("grounded-rag").enqueue_optimization(dataset, search_space=space)

    real_evaluate = durable_module.evaluate_candidate
    first_pass = {"count": 0}

    def failing(*args: Any, **kwargs: Any) -> Any:
        if first_pass["count"] >= 2:
            raise RuntimeError("simulated crash")
        first_pass["count"] += 1
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(durable_module, "evaluate_candidate", failing)
    with pytest.raises(RuntimeError):
        resume_optimization_run(project, run_id)
    assert len(runtime.store.list_optimization_candidates(run_id)) == 2

    second_pass = {"count": 0}

    def counting(*args: Any, **kwargs: Any) -> Any:
        second_pass["count"] += 1
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(durable_module, "evaluate_candidate", counting)
    result = resume_optimization_run(project, run_id)

    assert second_pass["count"] == 1
    assert len(runtime.store.list_optimization_candidates(run_id)) == 3
    assert result == synchronous.to_json()


def test_lease_reclaim_resumes_after_worker_death(runtime: Refaat) -> None:
    project = _seed(runtime)
    dataset = _dataset(runtime)
    space = retrieval_search_space(top_k=(2, 6))

    synchronous = project.programs.get("grounded-rag").optimize(dataset, search_space=space)
    run_id = project.programs.get("grounded-rag").enqueue_optimization(dataset, search_space=space)

    first = runtime.store.claim_job("worker-1", 1)
    assert first is not None
    attempts_first = int(first["attempts"])

    with runtime.store.engine.begin() as conn:
        conn.execute(
            update(jobs)
            .where(jobs.c.id == first["id"])
            .values(leased_until=runtime.store.clock.now() - timedelta(seconds=30))
        )

    reclaimed = runtime.store.claim_job("worker-2", 60)
    assert reclaimed is not None
    assert reclaimed["id"] == first["id"]
    assert int(reclaimed["attempts"]) == attempts_first + 1

    result = DurableWorker(runtime)._run_optimization(project.workspace_id, reclaimed["payload"])
    assert result["run_id"] == run_id

    run = runtime.store.get_optimization_run(project.workspace_id, run_id)
    assert run["status"] == "succeeded"
    assert run["result"] == synchronous.to_json()


def test_run_registers_artifacts_but_never_promotes(runtime: Refaat) -> None:
    project = _seed(runtime)
    dataset = _dataset(runtime)
    space = retrieval_search_space(top_k=(2, 6, 12))

    run_id = project.programs.get("grounded-rag").enqueue_optimization(dataset, search_space=space)
    DurableWorker(runtime).run_once()
    run = runtime.store.get_optimization_run(project.workspace_id, run_id)

    best_artifact_id = str(run["best_artifact_id"])
    assert runtime.registry.get(project.workspace_id, best_artifact_id)["program"] == "grounded-rag"

    with pytest.raises(NotFoundError):
        runtime.registry.resolve(project.workspace_id, "grounded-rag", "production")


def _force_single_attempt(runtime: Refaat, workspace_id: str) -> None:
    """Cap the run's job at one attempt so a single failure exhausts it."""
    with runtime.store.engine.begin() as conn:
        conn.execute(
            update(jobs)
            .where(jobs.c.workspace_id == workspace_id, jobs.c.kind == "optimization.run")
            .values(max_attempts=1)
        )


# A candidate that raises with an empty message fails the run once the
# job is exhausted (never a false SUCCEEDED), and a later manual resume still recovers,
# proving the empty error did not short-circuit on a false success.
def test_empty_error_marks_run_failed_and_survives_retry(runtime: Refaat, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _seed(runtime)
    dataset = _dataset(runtime)
    space = retrieval_search_space(top_k=(2, 6))
    run_id = project.programs.get("grounded-rag").enqueue_optimization(dataset, search_space=space)
    _force_single_attempt(runtime, project.workspace_id)

    def raise_empty(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("")

    monkeypatch.setattr(durable_module, "evaluate_candidate", raise_empty)
    assert DurableWorker(runtime).run_once() is True

    run = runtime.store.get_optimization_run(project.workspace_id, run_id)
    assert run["status"] == "failed"
    assert run["error"]  # non-empty (the exception type name), so no false success

    # A retry with a working evaluator recovers instead of short-circuiting on a false success.
    monkeypatch.undo()
    result = resume_optimization_run(project, run_id)
    assert runtime.store.get_optimization_run(project.workspace_id, run_id)["status"] == "succeeded"
    assert result["best"]["artifact_id"]


# A mid-retry attempt failure leaves the run resumable (never a transient FAILED a
# client could observe), and a subsequent successful attempt ends SUCCEEDED.
def test_transient_attempt_failure_leaves_run_resumable(runtime: Refaat, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _seed(runtime)
    dataset = _dataset(runtime)
    space = retrieval_search_space(top_k=(2, 6))
    run_id = project.programs.get("grounded-rag").enqueue_optimization(dataset, search_space=space)

    real_evaluate = durable_module.evaluate_candidate
    calls = {"count": 0}

    def flaky(*args: Any, **kwargs: Any) -> Any:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("transient blip")
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(durable_module, "evaluate_candidate", flaky)
    worker = DurableWorker(runtime)

    assert worker.run_once() is True
    # The job is not exhausted (default max_attempts), so the run stays non-terminal.
    mid = runtime.store.get_optimization_run(project.workspace_id, run_id)
    assert mid["status"] in {"pending", "running"}

    # Clear the retry backoff so the next claim runs the recovering attempt.
    with runtime.store.engine.begin() as conn:
        conn.execute(
            update(jobs)
            .where(jobs.c.workspace_id == project.workspace_id, jobs.c.kind == "optimization.run")
            .values(run_after=runtime.store.clock.now())
        )
    assert worker.run_once() is True
    assert runtime.store.get_optimization_run(project.workspace_id, run_id)["status"] == "succeeded"


# A resume after the program definition drifted is refused; the run is FAILED
# only once the job exhausts its attempts, never mid-retry.
def test_resume_refused_on_program_drift(runtime: Refaat) -> None:
    project = _seed(runtime)
    dataset = _dataset(runtime)
    space = retrieval_search_space(top_k=(2, 6))
    run_id = project.programs.get("grounded-rag").enqueue_optimization(dataset, search_space=space)

    with runtime.store.engine.begin() as conn:
        conn.execute(
            update(optimization_runs)
            .where(optimization_runs.c.id == run_id)
            .values(program_definition_hash="drifted-definition-hash")
        )

    # The resume itself refuses (unit-level), and no candidates are registered.
    with pytest.raises(ValueError, match="drifted"):
        resume_optimization_run(project, run_id)
    assert runtime.store.list_optimization_candidates(run_id) == []

    # Through the worker, the run terminalizes to FAILED once the job is exhausted.
    _force_single_attempt(runtime, project.workspace_id)
    assert DurableWorker(runtime).run_once() is True
    assert runtime.store.get_optimization_run(project.workspace_id, run_id)["status"] == "failed"
    assert runtime.store.list_optimization_candidates(run_id) == []


# Two enqueues of the same plan reuse the in-flight run and create no second job.
def test_enqueue_optimization_is_idempotent_per_plan(runtime: Refaat) -> None:
    project = _seed(runtime)
    dataset = _dataset(runtime)
    space = retrieval_search_space(top_k=(2, 6))

    first = project.programs.get("grounded-rag").enqueue_optimization(dataset, search_space=space)
    second = project.programs.get("grounded-rag").enqueue_optimization(dataset, search_space=space)
    assert first == second

    with runtime.store.engine.connect() as conn:
        job_count = conn.execute(
            select(func.count())
            .select_from(jobs)
            .where(jobs.c.workspace_id == project.workspace_id, jobs.c.kind == "optimization.run")
        ).scalar_one()
    assert job_count == 1


# Two enqueues that differ only in the corpus version (a re-ingest between them) must NOT dedup.
def test_enqueue_is_distinct_after_the_corpus_version_advances(runtime: Refaat) -> None:
    project = _seed(runtime)
    dataset = _dataset(runtime)
    space = retrieval_search_space(top_k=(2, 6))

    first = project.programs.get("grounded-rag").enqueue_optimization(dataset, search_space=space)
    runtime.ingest(
        project.workspace_id,
        project.corpus_id,
        DocumentInput(
            source_id="doc4.txt",
            title="Doc 4",
            media_type="text/plain",
            content=b"Runbook 4: a new production runbook advancing the corpus version.",
        ),
    )
    second = project.programs.get("grounded-rag").enqueue_optimization(dataset, search_space=space)

    # The corpus version is folded into the plan digest, so the identical dataset and search space over a
    # newer evidence version is a distinct run rather than a wrong merge onto the first in-flight one.
    assert first != second
    with runtime.store.engine.connect() as conn:
        job_count = conn.execute(
            select(func.count())
            .select_from(jobs)
            .where(jobs.c.workspace_id == project.workspace_id, jobs.c.kind == "optimization.run")
        ).scalar_one()
    assert job_count == 2


# A durable run for a program that cannot resolve after restart is refused at enqueue.
def test_enqueue_optimization_refuses_unresolvable_program(runtime: Refaat) -> None:
    project = _seed(runtime)
    dataset = _dataset(runtime)

    class _Custom(GroundedRagProgram):
        name = "custom-inproc"

    bound = project.programs.register(_Custom())
    with pytest.raises(ValueError, match="resolvable after a restart"):
        bound.enqueue_optimization(dataset, search_space=retrieval_search_space(top_k=(2, 6)))


# After evaluating, the committed checkpoint is used, so all resumers converge on it.
def test_resume_reads_back_committed_checkpoint(runtime: Refaat, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _seed(runtime)
    dataset = _dataset(runtime)
    space = retrieval_search_space(top_k=(2, 6))
    run_id = project.programs.get("grounded-rag").enqueue_optimization(dataset, search_space=space)

    real_record = runtime.store.record_optimization_candidate
    winners: dict[int, JsonObject] = {}

    def winning_record(rid: str, ordinal: int, candidate: JsonObject) -> None:
        # Simulate a concurrent resumer that won the ordinal with a different score.
        winner = {**candidate, "score": candidate["score"] + 100.0}
        winners[ordinal] = winner
        real_record(rid, ordinal, winner)

    monkeypatch.setattr(runtime.store, "record_optimization_candidate", winning_record)
    result = resume_optimization_run(project, run_id)

    stored = {int(row["ordinal"]): row for row in runtime.store.list_optimization_candidates(run_id)}
    assert {int(o): w["score"] for o, w in winners.items()} == {o: row["score"] for o, row in stored.items()}
    assembled = {c["artifact_id"]: c["score"] for c in result["candidates"]}
    for row in stored.values():
        assert assembled[str(row["artifact_id"])] == row["score"]


def test_reinvoke_on_succeeded_run_is_idempotent(runtime: Refaat, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _seed(runtime)
    dataset = _dataset(runtime)
    space = retrieval_search_space(top_k=(2, 6))

    run_id = project.programs.get("grounded-rag").enqueue_optimization(dataset, search_space=space)
    first = resume_optimization_run(project, run_id)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("candidate re-evaluated on an already-succeeded run")

    monkeypatch.setattr(durable_module, "evaluate_candidate", forbidden)
    again = resume_optimization_run(project, run_id)

    assert again == first


# A version straddle that shifts the winner's full artifact_id is refused on finalize, so a
# SUCCEEDED run can never persist a best_artifact_id that resolves to nothing. The winner's
# stored id was content-addressed under the old version and cannot be rebuilt at the new one,
# so registering only the new-version artifact would leave best_artifact_id dangling; refusing
# terminalizes the run instead, and a fresh run at the new version re-optimizes cleanly.
def test_resume_refused_when_version_bump_straddles_winner(runtime: Refaat, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _seed(runtime)
    dataset = _dataset(runtime)
    space = retrieval_search_space(top_k=(2, 6, 12))
    run_id = project.programs.get("grounded-rag").enqueue_optimization(dataset, search_space=space)

    # Phase 1: checkpoint the first two ordinals at the current version, then crash.
    real_evaluate = durable_module.evaluate_candidate
    seen = {"count": 0}

    def crash_after_two(*args: Any, **kwargs: Any) -> Any:
        if seen["count"] >= 2:
            raise RuntimeError("simulated crash before register")
        seen["count"] += 1
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(durable_module, "evaluate_candidate", crash_after_two)
    with pytest.raises(RuntimeError):
        resume_optimization_run(project, run_id)
    checkpoints = runtime.store.list_optimization_candidates(run_id)
    assert len(checkpoints) == 2
    sample = checkpoints[0]

    # Phase 2: bump the library version so a stored checkpoint's rebuilt full artifact_id
    # shifts (via dependency_versions) while its version-independent definition hash holds.
    monkeypatch.setattr(durable_module, "evaluate_candidate", real_evaluate)
    monkeypatch.setattr("refaat.programs.program.__version__", "99.9.9-test")

    from refaat.programs.program import Tunables

    rebuilt = (
        project.programs.get("grounded-rag").program.with_tunables(Tunables.from_json(sample["tunables"])).artifact()
    )
    assert rebuilt.artifact_id != str(sample["artifact_id"])
    assert rebuilt.definition_hash == str(sample["definition_hash"])

    # The lowest-token candidate (a pre-bump checkpoint) wins, so its rebuilt id no longer
    # matches its stored id: finalize refuses rather than persist a dangling best_artifact_id.
    with pytest.raises(ValueError, match="best_artifact_id would not resolve"):
        resume_optimization_run(project, run_id)
    assert runtime.store.get_optimization_run(project.workspace_id, run_id)["status"] != "succeeded"

    # Through the worker the run terminalizes to FAILED once the job exhausts its attempts,
    # and no best_artifact_id is ever persisted.
    _force_single_attempt(runtime, project.workspace_id)
    with runtime.store.engine.begin() as conn:
        conn.execute(
            update(jobs)
            .where(jobs.c.workspace_id == project.workspace_id, jobs.c.kind == "optimization.run")
            .values(run_after=runtime.store.clock.now())
        )
    assert DurableWorker(runtime).run_once() is True
    failed = runtime.store.get_optimization_run(project.workspace_id, run_id)
    assert failed["status"] == "failed"
    assert failed["best_artifact_id"] is None


# Two enqueues of the same plan that both miss the pre-check converge on one run and
# one job, because the partial-unique guard rejects the second create and the caller re-reads
# the winner.
def test_concurrent_enqueue_converges_on_one_run(runtime: Refaat, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _seed(runtime)
    dataset = _dataset(runtime)
    space = retrieval_search_space(top_k=(2, 6))

    first = project.programs.get("grounded-rag").enqueue_optimization(dataset, search_space=space)

    # Force the second enqueue's pre-check to miss the in-flight run so it races into create;
    # the DB guard then rejects it and the caller converges on the winner via a re-read.
    real_lookup = runtime.store.get_optimization_run_by_plan
    lookups = {"n": 0}

    def flaky_lookup(*args: Any, **kwargs: Any) -> Any:
        lookups["n"] += 1
        if lookups["n"] == 1:
            return None
        return real_lookup(*args, **kwargs)

    monkeypatch.setattr(runtime.store, "get_optimization_run_by_plan", flaky_lookup)
    second = project.programs.get("grounded-rag").enqueue_optimization(dataset, search_space=space)

    assert second == first
    with runtime.store.engine.connect() as conn:
        job_count = conn.execute(
            select(func.count())
            .select_from(jobs)
            .where(jobs.c.workspace_id == project.workspace_id, jobs.c.kind == "optimization.run")
        ).scalar_one()
        run_count = conn.execute(
            select(func.count())
            .select_from(optimization_runs)
            .where(optimization_runs.c.workspace_id == project.workspace_id)
        ).scalar_one()
    assert job_count == 1
    assert run_count == 1


# A legacy run with no pinned program identity cannot be verified, so a resume
# refuses to run against an unverifiable definition.
def test_resume_refused_on_null_identity(runtime: Refaat) -> None:
    project = _seed(runtime)
    dataset = _dataset(runtime)
    run_id = runtime.store.create_optimization_run(
        project.workspace_id,
        program="grounded-rag",
        corpus_id=project.corpus_id,
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        dataset_cases=dataset.to_records(),
        search_space=[{"top_k": 4}],
        bundle_id="grounded-rag",
        budget=None,
        plan_hash="legacy-null-hash",
        program_definition_hash=None,
        program_immutable_policy_hash=None,
    )

    with pytest.raises(ValueError, match="no pinned program identity"):
        resume_optimization_run(project, run_id)
    assert runtime.store.get_optimization_run(project.workspace_id, run_id)["status"] == "pending"
