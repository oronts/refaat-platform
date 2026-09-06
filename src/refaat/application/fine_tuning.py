from __future__ import annotations

from refaat.application.registry import RegistryService
from refaat.canonical import canonical_digest
from refaat.domain.enums import EvaluationStatus
from refaat.domain.fine_tuning import FineTuneRequest
from refaat.domain.kill_switch import KillSwitchRegistry, ensure_workspace_active
from refaat.domain.models import JsonObject
from refaat.errors import AuthorizationError, ValidationError
from refaat.evaluation.dataset import Dataset
from refaat.infrastructure.database.store import SQLStore
from refaat.observability import get_logger
from refaat.ports import FineTuningProvider
from refaat.programs.artifact import ProgramArtifact

logger = get_logger(__name__)


def fine_tune_program_name(base_model: str) -> str:
    """The registry grouping key for fine-tunes of one base model.

    Promoting a fine-tuned model uses the ordinary registry promotion under this
    name, so the models produced from one base model share one promotion history.
    """
    return f"fine-tune:{base_model}"


def fine_tune_artifact(
    *, base_model: str, dataset_version: str, hyperparameters: JsonObject, model_ref: str
) -> ProgramArtifact:
    """An immutable, content-addressed registry record for a produced model.

    It is registered but never promoted: using the model requires the existing
    explicit registry promotion. The identity folds only the deterministic inputs
    and the model reference, so the artifact address is reproducible.
    """
    tunables = {
        "model_ref": model_ref,
        "base_model": base_model,
        "dataset_version": dataset_version,
        "hyperparameters": hyperparameters,
    }
    definition_hash = canonical_digest(tunables)
    return ProgramArtifact(
        program=fine_tune_program_name(base_model),
        backend="fine_tune",
        definition_hash=definition_hash,
        immutable_policy_hash=canonical_digest({"kind": "fine_tune"}),
        signatures={},
        tunables=tunables,
        validators=(),
        dependency_versions={},
    )


class FineTuningService:
    """Governed, durable fine-tuning: request, approve, execute, register, never promote.

    A run is created PENDING and unapproved. Nothing executes until an explicit
    approval gates the training data. The durable worker then runs the configured
    provider (the deterministic one offline), records the result and provenance, and
    registers the produced model reference as an immutable registry artifact that is
    never auto-promoted. A failed provider marks the run FAILED and inspectable and
    registers no artifact.
    """

    def __init__(
        self,
        store: SQLStore,
        registry: RegistryService,
        provider: FineTuningProvider,
        kill_switches: KillSwitchRegistry | None = None,
    ) -> None:
        self.store = store
        self.kill_switches = kill_switches
        self.registry = registry
        self.provider = provider

    def request(
        self,
        workspace_id: str,
        dataset: Dataset,
        *,
        base_model: str,
        hyperparameters: JsonObject | None = None,
        requester: str = "",
    ) -> str:
        params = dict(hyperparameters or {})
        dataset_version = dataset.version
        provenance = {
            "provider_id": self.provider.provider_id,
            "base_model": base_model,
            "dataset_name": dataset.name,
            "dataset_version": dataset_version,
            "dataset_case_count": len(dataset.cases),
            "hyperparameters": params,
            "requester": requester,
            "requested_at": self.store.clock.now().isoformat(),
        }
        content_key = canonical_digest(
            {
                "workspace_id": workspace_id,
                "dataset_version": dataset_version,
                "base_model": base_model,
                "hyperparameters": params,
            }
        )
        return self.store.create_fine_tune_run(
            workspace_id,
            dataset_version=dataset_version,
            base_model=base_model,
            hyperparameters=params,
            requester=requester,
            provenance=provenance,
            content_key=content_key,
        )

    def approve(self, workspace_id: str, run_id: str, approver: str) -> JsonObject:
        approver = approver.strip()
        if not approver:
            raise AuthorizationError("A fine-tune run requires a named approver")
        run = self.store.get_fine_tune_run(workspace_id, run_id)
        if approver == str(run["requester"]).strip():
            raise AuthorizationError("A fine-tune run cannot be approved by its requester")
        return self.store.approve_fine_tune_run(workspace_id, run_id, approver)

    def enqueue(self, workspace_id: str, run_id: str) -> str:
        # Confirm the run exists and is in this workspace before enqueuing a job.
        self.store.get_fine_tune_run(workspace_id, run_id)
        return self.store.enqueue_job(
            workspace_id,
            "fine_tune.run",
            {"run_id": run_id},
            dedup_key=run_id,
        )

    def run(self, workspace_id: str, run_id: str) -> JsonObject:
        ensure_workspace_active(self.kill_switches, workspace_id)
        run = self.store.get_fine_tune_run(workspace_id, run_id)
        if run["status"] == EvaluationStatus.SUCCEEDED:
            return {"run_id": run_id, "model_ref": run["result_model_ref"], "status": run["status"]}
        if not run["approved"]:
            # Governance: the job refuses to execute an unapproved run. The training
            # data must be explicitly approved before any model is produced.
            raise ValidationError("Fine-tune run must be approved before it can execute")
        self.store.mark_fine_tune_run_running(workspace_id, run_id)
        request = FineTuneRequest(
            workspace_id=workspace_id,
            dataset_version=str(run["dataset_version"]),
            base_model=str(run["base_model"]),
            hyperparameters=dict(run["hyperparameters"]),
            requester=str(run["requester"]),
            provider_job_ref=str(run["provider_job_ref"]) if run.get("provider_job_ref") else None,
        )
        result = self.provider.submit(request)
        if result.provider_job_ref:
            self.store.set_fine_tune_provider_job_ref(workspace_id, run_id, result.provider_job_ref)
        status = str(result.status)
        if status == EvaluationStatus.RUNNING:
            # The provider job is not terminal: keep the run RUNNING and register nothing. A re-drive
            # (re-enqueue of fine_tune.run) polls the persisted provider_job_ref instead of creating a job.
            return {"run_id": run_id, "status": EvaluationStatus.RUNNING, "provider_job_ref": result.provider_job_ref}
        if status != EvaluationStatus.SUCCEEDED:
            self.store.finish_fine_tune_run(
                workspace_id,
                run_id,
                provenance=dict(result.provenance),
                error=f"Fine-tune provider returned status {result.status!r}",
            )
            return {"run_id": run_id, "status": EvaluationStatus.FAILED}
        model_ref = result.model_ref.strip()
        known_job_refs = {
            ref.strip() for ref in (request.provider_job_ref, result.provider_job_ref) if ref and ref.strip()
        }
        if not model_ref or model_ref in known_job_refs:
            # A successful run must carry a real trained model reference, and no provider job id (the one
            # passed into this poll or the one the provider returned) may stand in as that model. Enforced
            # here for any injected provider so an empty, blank, or job-id-shaped reference fails the run.
            self.store.finish_fine_tune_run(
                workspace_id,
                run_id,
                provenance=dict(result.provenance),
                error="Fine-tune provider reported success without a valid model reference",
            )
            return {"run_id": run_id, "status": EvaluationStatus.FAILED}
        artifact = fine_tune_artifact(
            base_model=str(run["base_model"]),
            dataset_version=str(run["dataset_version"]),
            hyperparameters=dict(run["hyperparameters"]),
            model_ref=result.model_ref,
        )
        # Register the produced model as an immutable artifact WITHOUT promoting it:
        # using it requires the operator's explicit registry promotion.
        self.registry.save(workspace_id, artifact)
        self.store.finish_fine_tune_run(
            workspace_id,
            run_id,
            result_model_ref=result.model_ref,
            provenance=dict(result.provenance),
        )
        return {
            "run_id": run_id,
            "status": EvaluationStatus.SUCCEEDED,
            "model_ref": result.model_ref,
            "artifact_id": artifact.artifact_id,
            "program": artifact.program,
        }


class BoundFineTuning:
    """A :class:`FineTuningService` bound to a single workspace."""

    def __init__(self, service: FineTuningService, workspace_id: str) -> None:
        self._service = service
        self._workspace_id = workspace_id

    def request(
        self,
        dataset: Dataset,
        *,
        base_model: str,
        hyperparameters: JsonObject | None = None,
        requester: str = "",
    ) -> str:
        return self._service.request(
            self._workspace_id,
            dataset,
            base_model=base_model,
            hyperparameters=hyperparameters,
            requester=requester,
        )

    def approve(self, run_id: str, approver: str) -> JsonObject:
        return self._service.approve(self._workspace_id, run_id, approver)

    def enqueue(self, run_id: str) -> str:
        return self._service.enqueue(self._workspace_id, run_id)

    def run(self, run_id: str) -> JsonObject:
        return self._service.run(self._workspace_id, run_id)

    def get(self, run_id: str) -> JsonObject:
        return self._service.store.get_fine_tune_run(self._workspace_id, run_id)
