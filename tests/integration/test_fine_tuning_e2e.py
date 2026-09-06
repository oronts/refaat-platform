from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import func, select, update

from refaat.application.runtime import Refaat, RefaatProject, RuntimeBuilder
from refaat.application.worker import DurableWorker
from refaat.config import Settings
from refaat.domain.fine_tuning import FineTuneRequest, FineTuneResult
from refaat.domain.models import DocumentInput
from refaat.errors import AuthorizationError, ConflictError, NotFoundError, ValidationError
from refaat.evaluation import Dataset
from refaat.infrastructure.database.schema import fine_tune_runs

pytestmark = pytest.mark.integration


def _seed(runtime: Refaat) -> RefaatProject:
    project = runtime.project(workspace="acme", corpus="handbook")
    runtime.ingest(
        project.workspace_id,
        project.corpus_id,
        DocumentInput(
            source_id="runbook.txt",
            title="Runbook",
            media_type="text/plain",
            content=b"A production release requires sign-off; escalate a Sev1 incident to on-call.",
        ),
    )
    return project


def _dataset() -> Dataset:
    return Dataset.from_records(
        "fine-tune",
        [
            {"query": "production release sign-off", "reference_answer": "A checklist."},
            {"query": "incident escalation", "reference_answer": "On-call."},
        ],
    )


def test_full_governed_durable_journey(runtime: Refaat) -> None:
    project = _seed(runtime)
    dataset = _dataset()

    run_id = project.fine_tuning.request(dataset, base_model="refaat-base", hyperparameters={"epochs": 3})
    created = project.fine_tuning.get(run_id)
    assert created["status"] == "pending"
    assert created["approved"] is False
    assert created["provenance"]["dataset_version"] == dataset.version

    # Governance: an unapproved run is refused.
    with pytest.raises(ValidationError):
        project.fine_tuning.run(run_id)

    approved = project.fine_tuning.approve(run_id, approver="founder")
    assert approved["approved"] is True
    assert approved["approved_by"] == "founder"

    project.fine_tuning.enqueue(run_id)
    assert DurableWorker(runtime).run_once() is True

    run = project.fine_tuning.get(run_id)
    assert run["status"] == "succeeded"
    model_ref = str(run["result_model_ref"])
    assert model_ref.startswith("ft:refaat:")
    assert run["provenance"]["trained"] is False

    # The produced model is registered as an immutable artifact but never promoted.
    program = f"fine-tune:{run['base_model']}"
    from refaat.application.fine_tuning import fine_tune_artifact

    artifact = fine_tune_artifact(
        base_model=str(run["base_model"]),
        dataset_version=str(run["dataset_version"]),
        hyperparameters=dict(run["hyperparameters"]),
        model_ref=model_ref,
    )
    assert runtime.registry.get(project.workspace_id, artifact.artifact_id)["program"] == program
    with pytest.raises(NotFoundError):
        runtime.registry.resolve(project.workspace_id, program, "production")


def test_duplicate_active_request_is_rejected(runtime: Refaat) -> None:
    project = _seed(runtime)
    dataset = _dataset()
    project.fine_tuning.request(dataset, base_model="refaat-base", hyperparameters={"epochs": 3})
    with pytest.raises(ConflictError):
        project.fine_tuning.request(dataset, base_model="refaat-base", hyperparameters={"epochs": 3})


def test_run_is_idempotent_and_resumable_from_running(runtime: Refaat) -> None:
    project = _seed(runtime)
    run_id = project.fine_tuning.request(_dataset(), base_model="refaat-base")
    project.fine_tuning.approve(run_id, approver="founder")

    # Simulate a crash after the run was marked RUNNING: resume must complete it.
    runtime.store.mark_fine_tune_run_running(project.workspace_id, run_id)
    first = project.fine_tuning.run(run_id)
    assert first["status"] == "succeeded"

    # A second run is a no-op that returns the same model reference.
    second = project.fine_tuning.run(run_id)
    assert second["model_ref"] == first["model_ref"]

    # Exactly one artifact was registered (insert-if-absent), not two.
    with runtime.store.engine.connect() as conn:
        from refaat.infrastructure.database.schema import program_artifacts

        count = conn.execute(
            select(func.count())
            .select_from(program_artifacts)
            .where(program_artifacts.c.workspace_id == project.workspace_id)
        ).scalar_one()
    assert count == 1


def test_failed_provider_marks_run_failed_and_registers_nothing(settings: Settings) -> None:
    class FailingProvider:
        provider_id = "failing-test-provider"

        def submit(self, request: FineTuneRequest) -> FineTuneResult:
            return FineTuneResult(model_ref="", status="failed", provenance={"reason": "provider rejected the request"})

    runtime = RuntimeBuilder(settings).with_fine_tuning_provider(FailingProvider()).build()
    try:
        project = _seed(runtime)
        run_id = project.fine_tuning.request(_dataset(), base_model="refaat-base")
        project.fine_tuning.approve(run_id, approver="founder")
        project.fine_tuning.enqueue(run_id)
        assert DurableWorker(runtime).run_once() is True

        run = project.fine_tuning.get(run_id)
        assert run["status"] == "failed"
        assert run["result_model_ref"] is None
        assert run["provenance"]["error"]

        with runtime.store.engine.connect() as conn:
            from refaat.infrastructure.database.schema import program_artifacts

            count = conn.execute(select(func.count()).select_from(program_artifacts)).scalar_one()
        assert count == 0
    finally:
        runtime.close()


def test_disabled_path_leaves_the_table_empty_and_pipeline_intact(runtime: Refaat) -> None:
    project = _seed(runtime)
    first = project.ask("What does a production release require?")
    second = project.ask("What does a production release require?")
    assert first.citations
    assert second.cached is True

    with runtime.store.engine.connect() as conn:
        count = conn.execute(select(func.count()).select_from(fine_tune_runs)).scalar_one()
    assert count == 0


def test_running_provider_leaves_run_running_registers_nothing_then_resumes(settings: Settings) -> None:
    class RunningThenSucceedProvider:
        provider_id = "running-test-provider"

        def submit(self, request: FineTuneRequest) -> FineTuneResult:
            if request.provider_job_ref is None:
                return FineTuneResult(model_ref="", status="running", provenance={}, provider_job_ref="ftjob-xyz")
            return FineTuneResult(
                model_ref="ft:real:trained",
                status="succeeded",
                provenance={"trained": True},
                provider_job_ref=request.provider_job_ref,
            )

    runtime = RuntimeBuilder(settings).with_fine_tuning_provider(RunningThenSucceedProvider()).build()
    try:
        project = _seed(runtime)
        run_id = project.fine_tuning.request(_dataset(), base_model="refaat-base")
        project.fine_tuning.approve(run_id, approver="founder")

        first = project.fine_tuning.run(run_id)
        assert first["status"] == "running"
        running = project.fine_tuning.get(run_id)
        assert running["status"] == "running"
        assert running["provider_job_ref"] == "ftjob-xyz"
        assert running["result_model_ref"] is None
        from refaat.infrastructure.database.schema import program_artifacts

        with runtime.store.engine.connect() as conn:
            assert conn.execute(select(func.count()).select_from(program_artifacts)).scalar_one() == 0

        second = project.fine_tuning.run(run_id)
        assert second["status"] == "succeeded"
        assert str(second["model_ref"]) == "ft:real:trained"
    finally:
        runtime.close()


def test_worker_re_enqueues_a_delayed_poll_while_a_run_is_running(settings: Settings) -> None:
    class AlwaysRunningProvider:
        provider_id = "always-running-provider"

        def submit(self, request: FineTuneRequest) -> FineTuneResult:
            return FineTuneResult(model_ref="", status="running", provenance={}, provider_job_ref="ftjob-poll")

    runtime = RuntimeBuilder(settings).with_fine_tuning_provider(AlwaysRunningProvider()).build()
    try:
        project = _seed(runtime)
        run_id = project.fine_tuning.request(_dataset(), base_model="refaat-base")
        project.fine_tuning.approve(run_id, approver="founder")
        project.fine_tuning.enqueue(run_id)

        started = runtime.store.clock.now().replace(tzinfo=None)
        assert DurableWorker(runtime).run_once() is True

        assert project.fine_tuning.get(run_id)["status"] == "running"
        from refaat.infrastructure.database.schema import jobs, program_artifacts

        with runtime.store.engine.connect() as conn:
            assert conn.execute(select(func.count()).select_from(program_artifacts)).scalar_one() == 0
            pending = conn.execute(
                select(jobs.c.run_after, jobs.c.dedup_key).where(
                    jobs.c.kind == "fine_tune.run", jobs.c.status == "pending"
                )
            ).all()
        assert len(pending) == 1
        run_after, dedup_key = pending[0]
        assert dedup_key == run_id
        assert run_after.replace(tzinfo=None) > started
    finally:
        runtime.close()


def test_approve_rejects_a_blank_approver(runtime: Refaat) -> None:
    project = _seed(runtime)
    run_id = project.fine_tuning.request(_dataset(), base_model="refaat-base")
    with pytest.raises(AuthorizationError):
        project.fine_tuning.approve(run_id, approver="")
    assert project.fine_tuning.get(run_id)["approved"] is False


def test_approve_rejects_a_whitespace_padded_self_approval(runtime: Refaat) -> None:
    project = _seed(runtime)
    run_id = project.fine_tuning.request(_dataset(), base_model="refaat-base", requester="alice")
    with pytest.raises(AuthorizationError):
        project.fine_tuning.approve(run_id, approver=" alice ")
    assert project.fine_tuning.get(run_id)["approved"] is False


class _EmptyModel:
    provider_id = "no-model-provider"

    def submit(self, request: FineTuneRequest) -> FineTuneResult:
        return FineTuneResult(model_ref="", status="succeeded", provenance={"trained": True})


class _WhitespaceModel:
    provider_id = "whitespace-model-provider"

    def submit(self, request: FineTuneRequest) -> FineTuneResult:
        return FineTuneResult(model_ref="   ", status="succeeded", provenance={"trained": True})


class _JobIdAsModel:
    provider_id = "job-id-as-model-provider"

    def submit(self, request: FineTuneRequest) -> FineTuneResult:
        return FineTuneResult(
            model_ref="ftjob-1", status="succeeded", provenance={"trained": True}, provider_job_ref="ftjob-1"
        )


class _PaddedJobIdAsModel:
    provider_id = "padded-job-id-provider"

    def submit(self, request: FineTuneRequest) -> FineTuneResult:
        return FineTuneResult(
            model_ref=" ftjob-1 ", status="succeeded", provenance={"trained": True}, provider_job_ref="ftjob-1"
        )


class _PersistedJobIdAsModel:
    """First call queues a job; the resumed poll returns the persisted job id as the model."""

    provider_id = "persisted-job-id-provider"

    def submit(self, request: FineTuneRequest) -> FineTuneResult:
        if request.provider_job_ref is None:
            return FineTuneResult(model_ref="", status="running", provenance={}, provider_job_ref="ftjob-persist")
        return FineTuneResult(
            model_ref=request.provider_job_ref, status="succeeded", provenance={"trained": True}, provider_job_ref=None
        )


@pytest.mark.parametrize("provider", [_EmptyModel(), _WhitespaceModel(), _JobIdAsModel(), _PaddedJobIdAsModel()])
def test_success_without_a_valid_model_reference_is_failed_and_registers_nothing(
    settings: Settings, provider: object
) -> None:
    runtime = RuntimeBuilder(settings).with_fine_tuning_provider(provider).build()
    try:
        project = _seed(runtime)
        run_id = project.fine_tuning.request(_dataset(), base_model="refaat-base")
        project.fine_tuning.approve(run_id, approver="founder")
        project.fine_tuning.enqueue(run_id)
        assert DurableWorker(runtime).run_once() is True

        assert project.fine_tuning.get(run_id)["status"] == "failed"
        from refaat.infrastructure.database.schema import program_artifacts

        with runtime.store.engine.connect() as conn:
            assert conn.execute(select(func.count()).select_from(program_artifacts)).scalar_one() == 0
    finally:
        runtime.close()


def test_resumed_poll_that_returns_the_persisted_job_id_as_the_model_is_failed(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings).with_fine_tuning_provider(_PersistedJobIdAsModel()).build()
    try:
        project = _seed(runtime)
        run_id = project.fine_tuning.request(_dataset(), base_model="refaat-base")
        project.fine_tuning.approve(run_id, approver="founder")
        assert project.fine_tuning.run(run_id)["status"] == "running"
        # The resumed poll returns model_ref equal to the persisted job id (provider_job_ref omitted).
        assert project.fine_tuning.run(run_id)["status"] == "failed"
        from refaat.infrastructure.database.schema import program_artifacts

        with runtime.store.engine.connect() as conn:
            assert conn.execute(select(func.count()).select_from(program_artifacts)).scalar_one() == 0
    finally:
        runtime.close()


def test_distinct_padded_approver_is_stored_stripped(runtime: Refaat) -> None:
    project = _seed(runtime)
    run_id = project.fine_tuning.request(_dataset(), base_model="refaat-base", requester="alice")
    approved = project.fine_tuning.approve(run_id, approver=" bob ")
    assert approved["approved"] is True
    assert project.fine_tuning.get(run_id)["approved_by"] == "bob"


def test_dispatch_never_enqueues_a_reserved_kind_schedule(runtime: Refaat) -> None:
    project = _seed(runtime)
    from refaat.infrastructure.database.schema import jobs, schedules

    # Created directly, bypassing the API guard, to prove the effect-point (dispatch) guard.
    reserved = runtime.store.create_schedule(project.workspace_id, "sneaky", "fine_tune.run", {"run_id": "x"}, 3600)
    control = runtime.store.create_schedule(project.workspace_id, "maintenance", "cache.purge", {}, 3600)
    past = runtime.store.clock.now() - timedelta(seconds=1)
    with runtime.store.engine.begin() as conn:
        conn.execute(update(schedules).where(schedules.c.id.in_((reserved, control))).values(next_run_at=past))

    assert runtime.store.dispatch_due_schedules() == 1  # only the control kind is dispatched

    with runtime.store.engine.connect() as conn:
        kinds = [row[0] for row in conn.execute(select(jobs.c.kind)).all()]
    assert "fine_tune.run" not in kinds
    assert "cache.purge" in kinds


def test_orphan_running_run_is_re_driven_by_the_reaper(runtime: Refaat) -> None:
    project = _seed(runtime)
    run_id = project.fine_tuning.request(_dataset(), base_model="refaat-base")
    project.fine_tuning.approve(run_id, approver="founder")
    runtime.store.mark_fine_tune_run_running(project.workspace_id, run_id)
    runtime.store.set_fine_tune_provider_job_ref(project.workspace_id, run_id, "ftjob-stranded")

    # A crash lost the next poll: the run is RUNNING with a provider job but no live poll job.
    repaired = runtime.store.reenqueue_orphan_fine_tune_runs(0)
    assert run_id in repaired
    from refaat.infrastructure.database.schema import jobs

    with runtime.store.engine.connect() as conn:
        pending = conn.execute(
            select(jobs.c.dedup_key).where(jobs.c.kind == "fine_tune.run", jobs.c.status == "pending")
        ).all()
    assert [row[0] for row in pending] == [run_id]

    # Idempotent: a run already referenced by a live poll job is not re-driven again.
    assert runtime.store.reenqueue_orphan_fine_tune_runs(0) == []


def test_mark_running_refuses_a_terminalized_run(runtime: Refaat) -> None:
    project = _seed(runtime)
    run_id = project.fine_tuning.request(_dataset(), base_model="refaat-base")
    project.fine_tuning.approve(run_id, approver="founder")
    runtime.store.finish_fine_tune_run(project.workspace_id, run_id, error="terminalized for the test")
    with pytest.raises(ConflictError):
        runtime.store.mark_fine_tune_run_running(project.workspace_id, run_id)
