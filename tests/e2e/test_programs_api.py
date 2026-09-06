from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.e2e


def test_program_http_surface(client: TestClient, api_key: str, corpus_id: str) -> None:
    headers = {"Authorization": f"Bearer {api_key}"}
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

    listed = client.get(f"/v1/corpora/{corpus_id}/programs", headers=headers)
    assert listed.status_code == 200
    assert any(program["program"] == "grounded-rag" for program in listed.json())

    run = client.post(
        f"/v1/corpora/{corpus_id}/programs/grounded-rag/run",
        headers=headers,
        json={"question": "release approval reviewers"},
    )
    assert run.status_code == 200, run.text
    assert run.json()["grounded"] is True
    assert run.json()["citations"]

    evaluate = client.post(
        f"/v1/corpora/{corpus_id}/programs/grounded-rag/evaluate",
        headers=headers,
        json={"cases": [{"query": "release approval", "expected_terms": ["release"]}]},
    )
    assert evaluate.status_code == 200
    assert evaluate.json()["hard_constraints_passed"] is True

    approve = client.post(
        f"/v1/corpora/{corpus_id}/programs/grounded-rag/approve",
        headers=headers,
        json={"environment": "production"},
    )
    assert approve.status_code == 200, approve.text

    promote = client.post(
        f"/v1/corpora/{corpus_id}/programs/grounded-rag/promote",
        headers=headers,
        json={"environment": "production"},
    )
    assert promote.status_code == 200, promote.text

    resolved = client.get("/v1/programs/grounded-rag/deployments/production", headers=headers)
    assert resolved.status_code == 200
    assert resolved.json()["program"] == "grounded-rag"

    rollback = client.post("/v1/programs/grounded-rag/rollback", headers=headers, json={"environment": "production"})
    assert rollback.status_code >= 400


def test_program_run_requires_authentication(client: TestClient, corpus_id: str) -> None:
    response = client.post(
        f"/v1/corpora/{corpus_id}/programs/grounded-rag/run",
        json={"question": "anything"},
    )
    assert response.status_code == 401


def test_durable_optimization_http_surface(client: TestClient, runtime, api_key: str, corpus_id: str) -> None:
    from refaat.application.worker import DurableWorker

    headers = {"Authorization": f"Bearer {api_key}"}
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

    enqueue = client.post(
        "/v1/programs/grounded-rag/optimizations",
        headers=headers,
        json={
            "corpus_id": corpus_id,
            "cases": [{"query": "release approval reviewers", "expected_terms": ["release"]}],
            "search_space": [
                {"top_k": 2, "token_budget": None, "minimum_citation_coverage": 0.0},
                {"top_k": 6, "token_budget": None, "minimum_citation_coverage": 0.0},
            ],
        },
    )
    assert enqueue.status_code == 202, enqueue.text
    run_id = enqueue.json()["id"]

    assert DurableWorker(runtime).run_once() is True

    got = client.get(f"/v1/optimizations/{run_id}", headers=headers)
    assert got.status_code == 200, got.text
    assert got.json()["status"] == "succeeded"
    assert got.json()["best_artifact_id"]
    assert got.json()["candidate_total"] == 2

    unauthorized = client.get(f"/v1/optimizations/{run_id}")
    assert unauthorized.status_code == 401
