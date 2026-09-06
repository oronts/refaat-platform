from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from refaat.domain.models import JsonObject


@dataclass(frozen=True, slots=True)
class FineTuneRequest:
    """The inputs a fine-tune provider needs to produce a model reference.

    ``dataset_version`` is the content-addressed sha256 of the approved training
    data (an ``evaluation.Dataset`` version), so provenance ties a produced model
    to the exact data it was trained on.
    """

    workspace_id: str
    dataset_version: str
    base_model: str
    hyperparameters: JsonObject = field(default_factory=dict)
    requester: str = ""
    # A previously created provider job to poll instead of creating a new one, so a re-drive is idempotent.
    provider_job_ref: str | None = None


@dataclass(frozen=True, slots=True)
class FineTuneResult:
    """A provider's outcome: a model reference plus a provenance payload.

    The deterministic offline provider returns a content-addressed reference and a
    provenance record that plainly states no training happened; a real provider
    returns the trained-model reference its API produced.
    """

    model_ref: str
    status: str
    provenance: JsonObject = field(default_factory=dict)
    provider_job_ref: str | None = None


@dataclass(frozen=True, slots=True)
class FineTuneRun:
    """The durable, governed record of one fine-tune request through its lifecycle."""

    run_id: str
    workspace_id: str
    dataset_version: str
    base_model: str
    hyperparameters: JsonObject
    status: str
    approved: bool
    approved_by: str | None
    requester: str
    result_model_ref: str | None
    provenance: JsonObject
    created_at: datetime | None = None
    updated_at: datetime | None = None
