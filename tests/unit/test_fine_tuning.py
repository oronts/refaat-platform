from __future__ import annotations

import httpx
import pytest

from refaat.application.fine_tuning import fine_tune_artifact, fine_tune_program_name
from refaat.domain.enums import EvaluationStatus
from refaat.domain.fine_tuning import FineTuneRequest
from refaat.errors import DependencyUnavailableError
from refaat.infrastructure.fine_tuning import (
    DeterministicFineTuner,
    OpenAIFineTuningProvider,
    deterministic_model_ref,
    provider_idempotency_key,
)


def _request(**overrides: object) -> FineTuneRequest:
    base = {
        "workspace_id": "ws_1",
        "dataset_version": "sha256:abc",
        "base_model": "refaat-base",
        "hyperparameters": {"epochs": 3},
        "requester": "founder",
    }
    base.update(overrides)
    return FineTuneRequest(**base)  # type: ignore[arg-type]


def test_deterministic_provider_is_reproducible() -> None:
    provider = DeterministicFineTuner()
    first = provider.submit(_request())
    second = provider.submit(_request())
    assert first.model_ref == second.model_ref
    assert first.model_ref.startswith("ft:refaat:")


def test_model_ref_changes_with_inputs() -> None:
    base = deterministic_model_ref(_request())
    assert deterministic_model_ref(_request(base_model="other")) != base
    assert deterministic_model_ref(_request(hyperparameters={"epochs": 5})) != base
    assert deterministic_model_ref(_request(dataset_version="sha256:zzz")) != base
    # The requester and workspace are provenance, not part of the model address.
    assert deterministic_model_ref(_request(requester="someone-else")) == base


def test_deterministic_provenance_is_honest() -> None:
    result = DeterministicFineTuner().submit(_request())
    assert result.status == "succeeded"
    assert result.provenance["trained"] is False
    assert "no model was trained" in result.provenance["note"].lower()
    assert result.provenance["dataset_version"] == "sha256:abc"
    assert result.provenance["provider_id"] == DeterministicFineTuner.provider_id


def test_artifact_is_content_addressed_and_grouped_by_base_model() -> None:
    artifact = fine_tune_artifact(
        base_model="refaat-base",
        dataset_version="sha256:abc",
        hyperparameters={"epochs": 3},
        model_ref="ft:refaat:deadbeef",
    )
    same = fine_tune_artifact(
        base_model="refaat-base",
        dataset_version="sha256:abc",
        hyperparameters={"epochs": 3},
        model_ref="ft:refaat:deadbeef",
    )
    assert artifact.artifact_id == same.artifact_id
    assert artifact.program == fine_tune_program_name("refaat-base")
    assert artifact.backend == "fine_tune"
    # No credentials or executable payload, only canonical JSON.
    assert artifact.to_json()["tunables"]["model_ref"] == "ft:refaat:deadbeef"


def _openai_provider(handler: object) -> OpenAIFineTuningProvider:
    return OpenAIFineTuningProvider(
        base_url="https://api.test",
        api_key="k",
        client=httpx.Client(transport=httpx.MockTransport(handler)),  # type: ignore[arg-type]
    )


def test_openai_provider_never_registers_a_queued_job_as_a_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "ftjob-1", "status": "queued", "fine_tuned_model": None})

    result = _openai_provider(handler).submit(_request(hyperparameters={"training_file": "file-1"}))
    assert result.status == EvaluationStatus.RUNNING
    assert result.provider_job_ref == "ftjob-1"
    assert result.model_ref == ""


def test_openai_provider_create_carries_a_content_addressed_idempotency_key() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["key"] = request.headers["Idempotency-Key"]
        return httpx.Response(200, json={"id": "ftjob-1", "status": "queued", "fine_tuned_model": None})

    request = _request(hyperparameters={"training_file": "file-1", "epochs": 3})
    _openai_provider(handler).submit(request)
    assert seen["key"] == provider_idempotency_key(request)


def test_idempotency_key_is_workspace_scoped_but_content_stable() -> None:
    a = _request(workspace_id="ws_a", hyperparameters={"training_file": "file-1", "epochs": 3})
    b = _request(workspace_id="ws_b", hyperparameters={"training_file": "file-1", "epochs": 3})
    # Same content, different workspace: distinct provider jobs, so no cross-tenant collision.
    assert provider_idempotency_key(a) != provider_idempotency_key(b)
    # Same run content is stable, so a create retry reuses the provider's existing job.
    assert provider_idempotency_key(a) == provider_idempotency_key(
        _request(workspace_id="ws_a", hyperparameters={"training_file": "file-1", "epochs": 3})
    )


def test_openai_provider_polls_an_existing_job_to_terminal_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        return httpx.Response(200, json={"id": "ftjob-1", "status": "succeeded", "fine_tuned_model": "ft:gpt:xyz"})

    result = _openai_provider(handler).submit(_request(provider_job_ref="ftjob-1"))
    assert result.status == EvaluationStatus.SUCCEEDED
    assert result.model_ref == "ft:gpt:xyz"


def test_openai_provider_succeeded_without_a_model_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "ftjob-1", "status": "succeeded", "fine_tuned_model": None})

    with pytest.raises(DependencyUnavailableError):
        _openai_provider(handler).submit(_request(provider_job_ref="ftjob-1"))


def test_openai_provider_failed_job_is_failed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "ftjob-1", "status": "failed", "fine_tuned_model": None})

    result = _openai_provider(handler).submit(_request(provider_job_ref="ftjob-1"))
    assert result.status == EvaluationStatus.FAILED
    assert result.model_ref == ""
