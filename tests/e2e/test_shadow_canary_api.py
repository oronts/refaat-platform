from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from refaat.application.runtime import Refaat
from refaat.programs import GroundedRagProgram, Tunables
from refaat.sdk import RefaatClient

pytestmark = pytest.mark.e2e


def _register_candidate(runtime: Refaat) -> str:
    project = runtime.project(workspace="acme", corpus="handbook")
    candidate = GroundedRagProgram(tunables=Tunables(top_k=2)).artifact()
    project.registry.save(candidate)
    return candidate.artifact_id


def _ingest_and_promote(client: TestClient, headers: dict[str, str], corpus_id: str) -> None:
    ingest = client.post(
        f"/v1/corpora/{corpus_id}/documents/text",
        headers=headers,
        json={
            "source_id": "policy.txt",
            "title": "Policy",
            "content": "Release approval requires two reviewers before deployment to production.",
        },
    )
    assert ingest.status_code in (200, 201), ingest.text
    promote = client.post(
        f"/v1/corpora/{corpus_id}/programs/grounded-rag/promote",
        headers=headers,
        json={"environment": "staging"},
    )
    assert promote.status_code == 200, promote.text


def test_canary_http_surface(client: TestClient, runtime: Refaat, api_key: str, corpus_id: str) -> None:
    headers = {"Authorization": f"Bearer {api_key}"}
    _ingest_and_promote(client, headers, corpus_id)
    candidate_id = _register_candidate(runtime)

    attach = client.post(
        f"/v1/corpora/{corpus_id}/programs/grounded-rag/canary",
        headers=headers,
        json={"environment": "staging", "candidate_artifact_id": candidate_id, "percent": 0},
    )
    assert attach.status_code == 200, attach.text
    assert attach.json()["action"] == "set"

    run = client.post(
        f"/v1/corpora/{corpus_id}/programs/grounded-rag/deployments/staging/run",
        headers=headers,
        json={"question": "release approval reviewers"},
    )
    assert run.status_code == 200, run.text
    body = run.json()
    assert body["served"] == "active"
    assert body["candidate_artifact_id"] == candidate_id
    assert body["comparison"] is not None
    assert body["result"]["citations"]

    status = client.get("/v1/programs/grounded-rag/canary/staging", headers=headers)
    assert status.status_code == 200
    assert status.json()["canary"]["candidate_artifact_id"] == candidate_id
    assert len(status.json()["shadow_runs"]) == 1

    cleared = client.delete(
        f"/v1/corpora/{corpus_id}/programs/grounded-rag/canary",
        headers=headers,
        params={"environment": "staging"},
    )
    assert cleared.status_code == 200
    assert cleared.json()["action"] == "clear"

    reverted = client.get("/v1/programs/grounded-rag/canary/staging", headers=headers)
    assert reverted.json()["canary"] is None


def test_canary_sdk_surface(client: TestClient, runtime: Refaat, api_key: str, corpus_id: str) -> None:
    headers = {"Authorization": f"Bearer {api_key}"}
    _ingest_and_promote(client, headers, corpus_id)
    candidate_id = _register_candidate(runtime)

    with RefaatClient("http://testserver", api_key, client=client) as sdk:
        attached = sdk.set_canary("grounded-rag", corpus_id, "staging", candidate_id, percent=100)
        assert attached["action"] == "set"

        served = sdk.deploy_run("grounded-rag", corpus_id, "staging", "release approval reviewers", user_id="u1")
        assert served["served"] == "candidate"
        assert served["result"]["citations"]

        status = sdk.canary_status("grounded-rag", "staging")
        assert status["canary"]["percent"] == 100

        cleared = sdk.clear_canary("grounded-rag", corpus_id, "staging")
        assert cleared["action"] == "clear"


def test_shadow_compare_worker(runtime: Refaat, workspace_id: str, corpus_id: str) -> None:
    from refaat.application.worker import DurableWorker
    from refaat.domain.models import DocumentInput

    runtime.ingest(
        workspace_id,
        corpus_id,
        DocumentInput(
            source_id="policy.txt",
            title="Policy",
            media_type="text/plain",
            content=b"Release approval requires two reviewers before deployment to production.",
        ),
    )
    project = runtime.project(workspace="acme", corpus="handbook")
    base = GroundedRagProgram().artifact()
    candidate = GroundedRagProgram(tunables=Tunables(top_k=2)).artifact()
    project.registry.save(base)
    project.registry.save(candidate)
    project.registry.promote("grounded-rag", "staging", base.artifact_id)
    project.deployment.set_canary("grounded-rag", "staging", candidate.artifact_id, percent=0)

    runtime.store.enqueue_job(
        workspace_id,
        "program.shadow_compare",
        {
            "corpus_id": corpus_id,
            "program": "grounded-rag",
            "environment": "staging",
            "questions": ["release approval"],
        },
    )
    assert DurableWorker(runtime).run_once() is True

    runs = project.deployment.shadow_runs("grounded-rag", "staging")
    assert len(runs) == 1
    assert runs[0]["comparison"] is not None
