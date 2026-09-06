from __future__ import annotations

import threading
from datetime import timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError

from refaat.application.runtime import Refaat
from refaat.application.worker import DurableWorker, LeaseHeartbeat, _error_message
from refaat.config import Settings
from refaat.domain.enums import EvaluationStatus, JobStatus
from refaat.errors import ConflictError
from refaat.infrastructure.database.schema import jobs, optimization_runs, outbox, program_shadow_runs


def _run(store: object, workspace_id: str, corpus_id: str, plan_hash: str = "plan-a") -> str:
    return store.create_optimization_run(  # type: ignore[attr-defined]
        workspace_id,
        program="grounded-rag",
        corpus_id=corpus_id,
        dataset_name="d",
        dataset_version="sha256:v",
        dataset_cases=[{"query": "q"}],
        search_space=[{"top_k": 4}],
        bundle_id="grounded-rag",
        budget=None,
        plan_hash=plan_hash,
    )


# Two default settings must never share a lease owner.
def test_default_worker_id_is_per_process_unique() -> None:
    first = Settings(_env_file=None)
    second = Settings(_env_file=None)
    assert first.worker_id != second.worker_id
    assert first.worker_id != "worker-local"
    assert second.worker_id != "worker-local"


# An explicit env value still wins so the helm downward-API injection controls it.
def test_explicit_worker_id_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REFAAT_WORKER_ID", "refaat-worker-7")
    assert Settings(_env_file=None).worker_id == "refaat-worker-7"


# Only the lease owner can advance or complete a leased job.
def test_lease_owner_guard_blocks_a_non_owner(runtime: Refaat) -> None:
    workspace_id = runtime.create_workspace("acme", "Acme")
    job_id = runtime.store.enqueue_job(workspace_id, "cache.purge", {})
    claimed = runtime.store.claim_job("worker-1", 60)
    assert claimed is not None
    assert str(claimed["id"]) == job_id

    assert runtime.store.heartbeat_job(job_id, "worker-2", 60) is False
    with pytest.raises(ConflictError):
        runtime.store.complete_job(job_id, "worker-2", {})
    with pytest.raises(ConflictError):
        runtime.store.fail_job(job_id, "worker-2", "boom", 0)
    # The real owner still controls it.
    assert runtime.store.heartbeat_job(job_id, "worker-1", 60) is True


# A transient heartbeat error keeps the daemon thread beating.
def test_heartbeat_tolerates_a_transient_error() -> None:
    class _FlakyStore:
        def __init__(self) -> None:
            self.calls = 0

        def heartbeat_job(self, job_id: str, worker_id: str, lease_seconds: int) -> bool:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("db blip")
            return True

    store = _FlakyStore()
    heartbeat = LeaseHeartbeat(SimpleNamespace(store=store), "job-1", "worker-1", 60)  # type: ignore[arg-type]
    assert heartbeat._beat() is True  # raised, swallowed, keep going
    assert heartbeat._beat() is True  # owned, keep going
    assert store.calls == 2


# A heartbeat that reports the lease is lost stops the loop.
def test_heartbeat_stops_when_lease_lost() -> None:
    store = SimpleNamespace(heartbeat_job=lambda *a: False)
    heartbeat = LeaseHeartbeat(SimpleNamespace(store=store), "job-1", "worker-1", 60)  # type: ignore[arg-type]
    assert heartbeat._beat() is False


def test_error_message_never_empty() -> None:
    assert _error_message(RuntimeError("")) == "RuntimeError"
    assert _error_message(RuntimeError("boom")) == "boom"


# An empty error string still fails the run (not a false success).
def test_finish_optimization_run_empty_error_is_failure(runtime: Refaat) -> None:
    workspace_id = runtime.create_workspace("acme", "Acme")
    corpus_id = runtime.create_corpus(workspace_id, "handbook")
    run_id = _run(runtime.store, workspace_id, corpus_id)

    runtime.store.finish_optimization_run(workspace_id, run_id, result={}, best_artifact_id=None, error="")
    run = runtime.store.get_optimization_run(workspace_id, run_id)
    assert run["status"] == EvaluationStatus.FAILED
    assert run["error"] == ""


# A run whose job is never enqueued is visibly PENDING, never falsely RUNNING.
def test_created_run_is_pending_until_started(runtime: Refaat) -> None:
    workspace_id = runtime.create_workspace("acme", "Acme")
    corpus_id = runtime.create_corpus(workspace_id, "handbook")
    run_id = _run(runtime.store, workspace_id, corpus_id)
    assert runtime.store.get_optimization_run(workspace_id, run_id)["status"] == EvaluationStatus.PENDING


def test_enqueue_job_requires_a_dedup_key_for_a_reserved_kind(runtime: Refaat) -> None:
    workspace_id = runtime.create_workspace("acme", "Acme")
    for kind in ("fine_tune.run", "optimization.run"):
        with pytest.raises(ValueError, match="dedup key"):
            runtime.store.enqueue_job(workspace_id, kind, {"run_id": "x"})
    # A non-reserved kind needs no dedup key.
    runtime.store.enqueue_job(workspace_id, "cache.purge", {})


# A RUNNING job that exhausted its attempts is reaped, terminalizing its run.
def test_reaper_terminalizes_abandoned_optimization_run(runtime: Refaat) -> None:
    workspace_id = runtime.create_workspace("acme", "Acme")
    corpus_id = runtime.create_corpus(workspace_id, "handbook")
    run_id = _run(runtime.store, workspace_id, corpus_id)
    job_id = runtime.store.enqueue_job(workspace_id, "optimization.run", {"run_id": run_id}, dedup_key=run_id)

    past = runtime.store.clock.now() - timedelta(seconds=120)
    with runtime.store.engine.begin() as conn:
        conn.execute(
            update(jobs)
            .where(jobs.c.id == job_id)
            .values(status=JobStatus.RUNNING, attempts=5, max_attempts=5, leased_until=past)
        )

    DurableWorker(runtime)._reap_abandoned()

    with runtime.store.engine.connect() as conn:
        job_status = conn.execute(select(jobs.c.status).where(jobs.c.id == job_id)).scalar_one()
    assert job_status == JobStatus.FAILED
    run = runtime.store.get_optimization_run(workspace_id, run_id)
    assert run["status"] == EvaluationStatus.FAILED
    assert run["error"]


# Expired shadow rows are purged, recent ones retained.
def test_purge_expired_shadow_runs(runtime: Refaat) -> None:
    workspace_id = runtime.create_workspace("acme", "Acme")
    old_id = runtime.store.record_shadow_run(
        workspace_id, "grounded-rag", "staging", "sha256:a", "sha256:b", "active", 0, "qh-old", None
    )
    recent_id = runtime.store.record_shadow_run(
        workspace_id, "grounded-rag", "staging", "sha256:a", "sha256:b", "active", 0, "qh-new", None
    )
    old_created = runtime.store.clock.now() - timedelta(days=45)
    with runtime.store.engine.begin() as conn:
        conn.execute(
            update(program_shadow_runs).where(program_shadow_runs.c.id == old_id).values(created_at=old_created)
        )

    purged = runtime.store.purge_expired_shadow_runs(30)
    assert purged == 1
    with runtime.store.engine.connect() as conn:
        remaining = conn.execute(select(program_shadow_runs.c.id)).scalars().all()
    assert recent_id in remaining
    assert old_id not in remaining


# Once claimed, fail_job folds ownership into the write, so a job reclaimed by
# another worker (the lease expired and a new owner took it) is never clobbered.
def test_fail_job_refuses_after_reclaim(runtime: Refaat) -> None:
    workspace_id = runtime.create_workspace("acme", "Acme")
    job_id = runtime.store.enqueue_job(workspace_id, "cache.purge", {})
    assert runtime.store.claim_job("worker-1", 60) is not None

    # Simulate a reclaim between worker-1's read and its write.
    with runtime.store.engine.begin() as conn:
        conn.execute(update(jobs).where(jobs.c.id == job_id).values(lease_owner="worker-2"))

    with pytest.raises(ConflictError):
        runtime.store.fail_job(job_id, "worker-1", "boom", 0)

    with runtime.store.engine.connect() as conn:
        row = (
            conn.execute(select(jobs.c.status, jobs.c.lease_owner, jobs.c.last_error).where(jobs.c.id == job_id))
            .mappings()
            .one()
        )
    # worker-1's blind write did not land: the job is still RUNNING under worker-2.
    assert row["status"] == JobStatus.RUNNING
    assert row["lease_owner"] == "worker-2"
    assert row["last_error"] is None


# The reaper's terminalize is guarded, so a job an owner completed after its lease
# expired (but before the reaper's write) is not overwritten with FAILED.
def test_reaper_does_not_clobber_a_completed_job(runtime: Refaat) -> None:
    workspace_id = runtime.create_workspace("acme", "Acme")
    job_id = runtime.store.enqueue_job(workspace_id, "cache.purge", {})
    assert runtime.store.claim_job("worker-1", 60) is not None

    past = runtime.store.clock.now() - timedelta(seconds=120)
    with runtime.store.engine.begin() as conn:
        conn.execute(update(jobs).where(jobs.c.id == job_id).values(attempts=5, max_attempts=5, leased_until=past))
    # The owner completes it despite the expired lease (complete_job guards ownership only).
    runtime.store.complete_job(job_id, "worker-1", {"ok": True})

    reaped = runtime.store.reap_abandoned_jobs()
    assert reaped == []
    with runtime.store.engine.connect() as conn:
        status = conn.execute(select(jobs.c.status).where(jobs.c.id == job_id)).scalar_one()
    assert status == JobStatus.SUCCEEDED


# A committed SUCCEEDED optimization run is never overwritten by a later FAILED.
def test_finish_optimization_run_is_monotonic(runtime: Refaat) -> None:
    workspace_id = runtime.create_workspace("acme", "Acme")
    corpus_id = runtime.create_corpus(workspace_id, "handbook")
    run_id = _run(runtime.store, workspace_id, corpus_id)

    runtime.store.finish_optimization_run(
        workspace_id, run_id, result={"ok": True}, best_artifact_id="sha256:winner", error=None
    )
    # A racing resumer's later FAILED must not clobber the committed success (idempotent).
    runtime.store.finish_optimization_run(workspace_id, run_id, result={}, best_artifact_id=None, error="late failure")

    run = runtime.store.get_optimization_run(workspace_id, run_id)
    assert run["status"] == EvaluationStatus.SUCCEEDED
    assert run["best_artifact_id"] == "sha256:winner"
    assert run["error"] is None


# A PENDING run whose job was never enqueued (a crash between create and enqueue)
# is re-enqueued once older than the grace window; a fresh one and one with a live job are not.
def test_orphan_pending_run_is_reenqueued(runtime: Refaat) -> None:
    workspace_id = runtime.create_workspace("acme", "Acme")
    corpus_id = runtime.create_corpus(workspace_id, "handbook")
    orphan = _run(runtime.store, workspace_id, corpus_id, plan_hash="orphan")
    fresh = _run(runtime.store, workspace_id, corpus_id, plan_hash="fresh")
    with_job = _run(runtime.store, workspace_id, corpus_id, plan_hash="withjob")
    runtime.store.enqueue_job(workspace_id, "optimization.run", {"run_id": with_job}, dedup_key=with_job)

    old = runtime.store.clock.now() - timedelta(seconds=3600)
    with runtime.store.engine.begin() as conn:
        conn.execute(
            update(optimization_runs).where(optimization_runs.c.id.in_((orphan, with_job))).values(updated_at=old)
        )

    repaired = runtime.store.reenqueue_orphan_optimization_runs(300)
    assert repaired == [orphan]

    with runtime.store.engine.connect() as conn:
        payloads = conn.execute(select(jobs.c.payload).where(jobs.c.kind == "optimization.run")).scalars().all()
    referenced = [str(p["run_id"]) for p in payloads]
    assert referenced.count(orphan) == 1  # exactly one job now
    assert referenced.count(with_job) == 1  # not double-enqueued
    assert fresh not in referenced  # too young to reap


# Without a job id the shadow idempotency key folds the question, so two different
# questions at the same index never collide into one deduped record.
def test_shadow_idempotency_key_includes_question(runtime: Refaat) -> None:
    worker = DurableWorker(runtime)
    worker._current_job_id = None
    key_a = worker._shadow_idempotency_key("ws", "grounded-rag", "staging", 0, "how does rollback work")
    key_b = worker._shadow_idempotency_key("ws", "grounded-rag", "staging", 0, "how does deploy work")
    assert key_a != key_b

    # With a real job id the key is (job, index) and independent of the question, so an
    # at-least-once redelivery of the same batch still dedupes.
    worker._current_job_id = "job-7"
    assert worker._shadow_idempotency_key("ws", "grounded-rag", "staging", 0, "q1") == "job-7:0"
    assert worker._shadow_idempotency_key("ws", "grounded-rag", "staging", 0, "q2") == "job-7:0"


# The retention purge drops shadow rows but leaves their outbox events; a retried
# keyed insert after a purge must not collide on the outbox primary key.
def test_shadow_insert_survives_outbox_after_purge(runtime: Refaat) -> None:
    workspace_id = runtime.create_workspace("acme", "Acme")
    run_id = runtime.store.record_shadow_run(
        workspace_id, "grounded-rag", "staging", "sha256:a", "sha256:b", "active", 0, "qh", None, idempotency_key="k1"
    )

    old = runtime.store.clock.now() - timedelta(days=45)
    with runtime.store.engine.begin() as conn:
        conn.execute(update(program_shadow_runs).where(program_shadow_runs.c.id == run_id).values(created_at=old))
    assert runtime.store.purge_expired_shadow_runs(30) == 1

    # The outbox event survived the purge.
    with runtime.store.engine.connect() as conn:
        events = conn.execute(select(outbox.c.id).where(outbox.c.topic == "program.shadow_compared")).scalars().all()
    assert len(events) == 1

    # The retried keyed insert re-creates the shadow row without colliding on the event id.
    again = runtime.store.record_shadow_run(
        workspace_id, "grounded-rag", "staging", "sha256:a", "sha256:b", "active", 0, "qh", None, idempotency_key="k1"
    )
    assert again == run_id
    with runtime.store.engine.connect() as conn:
        rows = conn.execute(select(program_shadow_runs.c.id).where(program_shadow_runs.c.id == run_id)).scalars().all()
        events_after = (
            conn.execute(select(outbox.c.id).where(outbox.c.topic == "program.shadow_compared")).scalars().all()
        )
    assert rows == [run_id]
    assert len(events_after) == 1  # still one event, not duplicated


# Two enqueues of the same optimization run converge on one job: the partial-unique dedup
# index rejects the second insert and the caller re-reads the in-flight job.
def test_optimization_job_dedup_key_prevents_double_enqueue(runtime: Refaat) -> None:
    workspace_id = runtime.create_workspace("acme", "Acme")
    corpus_id = runtime.create_corpus(workspace_id, "handbook")
    run_id = _run(runtime.store, workspace_id, corpus_id)

    first = runtime.store.enqueue_job(workspace_id, "optimization.run", {"run_id": run_id}, dedup_key=run_id)
    second = runtime.store.enqueue_job(workspace_id, "optimization.run", {"run_id": run_id}, dedup_key=run_id)

    assert second == first  # converged, not a duplicate
    with runtime.store.engine.connect() as conn:
        count = conn.execute(select(func.count()).select_from(jobs).where(jobs.c.dedup_key == run_id)).scalar_one()
    assert count == 1

    # A null dedup_key is outside the partial index, so ordinary jobs never contend.
    a = runtime.store.enqueue_job(workspace_id, "cache.purge", {})
    b = runtime.store.enqueue_job(workspace_id, "cache.purge", {})
    assert a != b


# The orphan reaper racing a legitimate enqueue of the same run yields exactly one job: the
# reaper's insert either wins outright or hits the dedup index as a no-op, never a duplicate.
def test_reaper_reenqueue_races_legitimate_enqueue_yields_one_job(runtime: Refaat) -> None:
    workspace_id = runtime.create_workspace("acme", "Acme")
    corpus_id = runtime.create_corpus(workspace_id, "handbook")
    orphan = _run(runtime.store, workspace_id, corpus_id, plan_hash="orphan")
    old = runtime.store.clock.now() - timedelta(seconds=3600)
    with runtime.store.engine.begin() as conn:
        conn.execute(update(optimization_runs).where(optimization_runs.c.id == orphan).values(updated_at=old))

    barrier = threading.Barrier(2)
    errors: list[BaseException] = []
    lock = threading.Lock()

    def enqueuer() -> None:
        barrier.wait()
        try:
            runtime.store.enqueue_job(workspace_id, "optimization.run", {"run_id": orphan}, dedup_key=orphan)
        except BaseException as exc:
            with lock:
                errors.append(exc)

    def reaper() -> None:
        barrier.wait()
        try:
            runtime.store.reenqueue_orphan_optimization_runs(300)
        except BaseException as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=enqueuer), threading.Thread(target=reaper)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not any(isinstance(error, IntegrityError) for error in errors), "a raw IntegrityError must not bubble"
    with runtime.store.engine.connect() as conn:
        payloads = conn.execute(select(jobs.c.payload).where(jobs.c.kind == "optimization.run")).scalars().all()
    referenced = [str(payload["run_id"]) for payload in payloads if isinstance(payload, dict)]
    assert referenced.count(orphan) == 1


# The keyed shadow path re-creates a purged outbox event even when the shadow row still
# exists: it never returns early before ensuring the event, so a retry heals a lost event.
def test_shadow_keyed_retry_recreates_purged_outbox_event(runtime: Refaat) -> None:
    workspace_id = runtime.create_workspace("acme", "Acme")
    run_id = runtime.store.record_shadow_run(
        workspace_id, "grounded-rag", "staging", "sha256:a", "sha256:b", "active", 0, "qh", None, idempotency_key="k1"
    )

    # Outbox retention drops the event while the shadow row survives (the inverse of the
    # shadow-row purge): the row remains, only its event is gone.
    with runtime.store.engine.begin() as conn:
        conn.execute(delete(outbox).where(outbox.c.topic == "program.shadow_compared"))
    with runtime.store.engine.connect() as conn:
        assert conn.execute(select(func.count()).select_from(outbox)).scalar_one() == 0
        assert conn.execute(select(func.count()).select_from(program_shadow_runs)).scalar_one() == 1

    again = runtime.store.record_shadow_run(
        workspace_id, "grounded-rag", "staging", "sha256:a", "sha256:b", "active", 0, "qh", None, idempotency_key="k1"
    )
    assert again == run_id
    with runtime.store.engine.connect() as conn:
        events = conn.execute(select(outbox.c.id).where(outbox.c.topic == "program.shadow_compared")).scalars().all()
        rows = conn.execute(select(program_shadow_runs.c.id)).scalars().all()
    assert len(events) == 1  # the lost event was re-created
    assert rows == [run_id]  # still exactly one shadow row


# Two concurrent first writers of the same keyed shadow record produce one row and one event:
# each insert is an atomic upsert, so the loser's conflict is a silent no-op.
def test_concurrent_keyed_shadow_writers_yield_one_row_and_event(runtime: Refaat) -> None:
    workspace_id = runtime.create_workspace("acme", "Acme")

    barrier = threading.Barrier(2)
    outcomes: list[object] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        try:
            result: object = runtime.store.record_shadow_run(
                workspace_id,
                "grounded-rag",
                "staging",
                "sha256:a",
                "sha256:b",
                "active",
                0,
                "qh",
                None,
                idempotency_key="race-key",
            )
        except BaseException as exc:
            result = exc
        with lock:
            outcomes.append(result)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not any(isinstance(outcome, IntegrityError) for outcome in outcomes), "a raw IntegrityError must not bubble"
    assert not any(isinstance(outcome, BaseException) for outcome in outcomes), "both writers succeed"
    assert len(set(outcomes)) == 1  # both converge on the same content-addressed id
    with runtime.store.engine.connect() as conn:
        rows = conn.execute(select(func.count()).select_from(program_shadow_runs)).scalar_one()
        events = conn.execute(
            select(func.count()).select_from(outbox).where(outbox.c.topic == "program.shadow_compared")
        ).scalar_one()
    assert rows == 1
    assert events == 1
