from __future__ import annotations

from typing import Any

import httpx

from refaat.canonical import canonical_digest
from refaat.domain.egress import (
    EgressDestination,
    EgressPolicy,
    EgressPurpose,
    ensure_egress_allowed,
)
from refaat.domain.enums import EvaluationStatus
from refaat.domain.fine_tuning import FineTuneRequest, FineTuneResult
from refaat.errors import ConfigurationError, DependencyUnavailableError
from refaat.infrastructure.egress_client import build_http_client
from refaat.infrastructure.http_retry import post_with_retry

_TERMINAL_FAILURE = frozenset({"failed", "cancelled"})


def deterministic_model_ref(request: FineTuneRequest) -> str:
    """A reproducible content address for the requested fine-tune.

    Two identical requests (same base model, approved data version, and
    hyperparameters) address the same reference, so the offline path is
    deterministic and idempotent.
    """
    return "ft:refaat:" + canonical_digest(
        {
            "base_model": request.base_model,
            "dataset_version": request.dataset_version,
            "hyperparameters": request.hyperparameters,
        }
    )


def provider_idempotency_key(request: FineTuneRequest) -> str:
    """A create key that is stable per run content yet scoped to the workspace.

    Stable per content so a create retry after a crash reuses the provider's job; scoped to the
    workspace so two tenants with byte-identical content on a shared provider account get distinct
    jobs rather than colliding onto one.
    """
    return canonical_digest({"workspace_id": request.workspace_id, "model_ref": deterministic_model_ref(request)})


class DeterministicFineTuner:
    """The offline default: honest provenance, no training and no network.

    It records the request and returns a content-addressed model reference. The
    provenance payload states plainly that no model was trained: this is a
    governance and provenance record, not a trained model. It exists so the whole
    fine-tuning machinery (approval gate, durable run, immutable registration,
    promotion) is exercised deterministically without a training provider.
    """

    provider_id = "refaat-deterministic-fine-tuner-v1"

    def submit(self, request: FineTuneRequest) -> FineTuneResult:
        model_ref = deterministic_model_ref(request)
        provenance = {
            "provider_id": self.provider_id,
            "trained": False,
            "note": (
                "Deterministic offline provider: no model was trained and no network call was made. "
                "This is a governance and provenance record addressing the approved training request, "
                "not a trained model. Configure a real fine-tuning provider to produce a trained model."
            ),
            "workspace_id": request.workspace_id,
            "base_model": request.base_model,
            "dataset_version": request.dataset_version,
            "hyperparameters": request.hyperparameters,
            "requester": request.requester,
        }
        return FineTuneResult(model_ref=model_ref, status=EvaluationStatus.SUCCEEDED, provenance=provenance)


class OpenAIFineTuningProvider:
    """A real OpenAI fine-tuning adapter, used only when explicitly configured.

    It reads its key from config like the other model adapters and calls the
    OpenAI fine-tuning REST API. It is never exercised in the gate: the offline
    default is the deterministic provider, so no network call ever enters a gate
    path. A real fine-tune needs an uploaded training file; its id is supplied in
    ``hyperparameters["training_file"]`` and the remaining hyperparameters are
    forwarded to the provider.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        retries: int = 2,
        retry_base_delay: float = 0.25,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
        egress_policy: EgressPolicy | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._retries = retries
        self._retry_base_delay = retry_base_delay
        self._owns_client = client is None
        self.client = build_http_client(
            timeout=timeout, egress_policy=egress_policy, purpose=EgressPurpose.FINE_TUNING, client=client
        )
        if egress_policy is not None:
            ensure_egress_allowed(egress_policy, EgressDestination.from_url(self.base_url, EgressPurpose.FINE_TUNING))

    provider_id = "openai-fine-tuning"

    def submit(self, request: FineTuneRequest) -> FineTuneResult:
        # Create the job on the first call, poll an existing one on a re-drive: a fine-tune is long, so a
        # run reaches SUCCEEDED only when the provider confirms a terminal success with a trained model,
        # never on job creation, and never registers the queued job id as the model reference.
        job = self._poll_job(request.provider_job_ref) if request.provider_job_ref else self._create_job(request)
        job_ref = str(job.get("id") or request.provider_job_ref or "")
        if not job_ref:
            raise DependencyUnavailableError("Fine-tuning provider returned no job reference")
        status = str(job.get("status") or "")
        provenance = {
            "provider_id": self.provider_id,
            "trained": status == "succeeded",
            "base_model": request.base_model,
            "dataset_version": request.dataset_version,
            "job_id": job_ref,
            "provider_status": status,
            "requester": request.requester,
        }
        if status in _TERMINAL_FAILURE:
            return FineTuneResult(
                model_ref="", status=EvaluationStatus.FAILED, provenance=provenance, provider_job_ref=job_ref
            )
        if status == "succeeded":
            model_ref = str(job.get("fine_tuned_model") or "")
            if not model_ref:
                raise DependencyUnavailableError("Fine-tune succeeded without a trained model reference")
            return FineTuneResult(
                model_ref=model_ref, status=EvaluationStatus.SUCCEEDED, provenance=provenance, provider_job_ref=job_ref
            )
        return FineTuneResult(
            model_ref="", status=EvaluationStatus.RUNNING, provenance=provenance, provider_job_ref=job_ref
        )

    def _create_job(self, request: FineTuneRequest) -> dict[str, Any]:
        hyperparameters = dict(request.hyperparameters)
        training_file = hyperparameters.pop("training_file", None)
        if not training_file:
            raise ConfigurationError(
                "The OpenAI fine-tuning provider requires an uploaded training file id in "
                "hyperparameters['training_file']; a real fine-tune trains on an uploaded dataset"
            )
        body: dict[str, Any] = {"model": request.base_model, "training_file": str(training_file)}
        if hyperparameters:
            body["hyperparameters"] = hyperparameters
        try:
            response = post_with_retry(
                self.client,
                f"{self.base_url}/fine_tuning/jobs",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    # A re-drive after a crash between create and persisting the job reference reuses the
                    # provider's existing job instead of minting a duplicate, and the key is workspace-scoped.
                    "Idempotency-Key": provider_idempotency_key(request),
                },
                json=body,
                retries=self._retries,
                base_delay=self._retry_base_delay,
            )
            return dict(response.json())
        except (httpx.HTTPError, ValueError) as exc:
            raise DependencyUnavailableError("Fine-tuning provider request failed") from exc

    def _poll_job(self, job_ref: str) -> dict[str, Any]:
        try:
            response = self.client.get(
                f"{self.base_url}/fine_tuning/jobs/{job_ref}",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
            return dict(response.json())
        except (httpx.HTTPError, ValueError) as exc:
            raise DependencyUnavailableError("Fine-tuning provider poll failed") from exc

    def close(self) -> None:
        if self._owns_client:
            self.client.close()
