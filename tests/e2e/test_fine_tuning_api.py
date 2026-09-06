from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from refaat.application.runtime import Refaat
from refaat.application.worker import DurableWorker

pytestmark = pytest.mark.e2e


def test_fine_tuning_http_surface(
    client: TestClient, runtime: Refaat, api_key: str, workspace_id: str, corpus_id: str
) -> None:
    del corpus_id
    headers = {"Authorization": f"Bearer {api_key}"}
    cases = [{"query": "release sign-off", "reference_answer": "A checklist."}]

    requested = client.post(
        "/v1/fine-tunes",
        headers=headers,
        json={"cases": cases, "base_model": "refaat-base", "hyperparameters": {"epochs": 2}},
    )
    assert requested.status_code == 202, requested.text
    run_id = requested.json()["id"]

    pending = client.get(f"/v1/fine-tunes/{run_id}", headers=headers)
    assert pending.status_code == 200
    assert pending.json()["status"] == "pending"
    assert pending.json()["approved"] is False

    # Four-eyes: the requester cannot approve their own run.
    self_approval = client.post(f"/v1/fine-tunes/{run_id}/approve", headers=headers, json={})
    assert self_approval.status_code == 403, self_approval.text

    _, approver_key = runtime.issue_api_key(workspace_id, "ft-approver", scopes=("admin",))
    approver_headers = {"Authorization": f"Bearer {approver_key}"}
    approved = client.post(f"/v1/fine-tunes/{run_id}/approve", headers=approver_headers, json={})
    assert approved.status_code == 200, approved.text
    assert approved.json()["approved"] is True

    enqueued = client.post(f"/v1/fine-tunes/{run_id}/enqueue", headers=headers)
    assert enqueued.status_code == 202, enqueued.text

    assert DurableWorker(runtime).run_once() is True

    done = client.get(f"/v1/fine-tunes/{run_id}", headers=headers)
    assert done.status_code == 200
    body = done.json()
    assert body["status"] == "succeeded"
    assert str(body["result_model_ref"]).startswith("ft:refaat:")
    assert body["provenance"]["trained"] is False


@pytest.mark.parametrize("kind", ["fine_tune.run", "optimization.run"])
def test_generic_job_endpoint_rejects_a_reserved_kind(client: TestClient, api_key: str, kind: str) -> None:
    headers = {"Authorization": f"Bearer {api_key}"}
    response = client.post("/v1/jobs", headers=headers, json={"kind": kind, "payload": {"run_id": "whatever"}})
    assert response.status_code == 422, response.text


@pytest.mark.parametrize("kind", ["fine_tune.run", "optimization.run"])
def test_schedule_endpoint_rejects_a_reserved_kind(client: TestClient, api_key: str, kind: str) -> None:
    headers = {"Authorization": f"Bearer {api_key}"}
    response = client.post(
        "/v1/schedules",
        headers=headers,
        json={"name": "sneaky", "job_kind": kind, "payload": {"run_id": "x"}, "interval_seconds": 60},
    )
    assert response.status_code == 422, response.text


def test_request_requires_admin_scope(client: TestClient, runtime: Refaat, workspace_id: str) -> None:
    _, read_only = runtime.issue_api_key(workspace_id, "reader", scopes=("read",))
    response = client.post(
        "/v1/fine-tunes",
        headers={"Authorization": f"Bearer {read_only}"},
        json={"cases": [{"query": "q"}], "base_model": "refaat-base"},
    )
    assert response.status_code == 403, response.text
