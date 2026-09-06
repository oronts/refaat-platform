from __future__ import annotations

import io

import pytest

from refaat.application.worker import DurableWorker

pytestmark = pytest.mark.e2e


def bootstrap(client, slug: str = "api-acme") -> tuple[str, str]:
    response = client.post(
        "/v1/bootstrap/workspaces",
        headers={"X-Bootstrap-Token": "bootstrap-test-token"},
        json={"slug": slug, "name": slug.title()},
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    return payload["workspace_id"], payload["api_key"]


def auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def test_full_http_flow_and_workspace_isolation(client) -> None:
    _workspace_id, key = bootstrap(client)
    headers = auth(key)

    unauthorized = client.get("/v1/corpora")
    assert unauthorized.status_code == 401
    assert unauthorized.json()["error"]["code"] == "authentication_failed"

    corpus_response = client.post(
        "/v1/corpora",
        headers=headers,
        json={"name": "engineering", "settings": {"language": "en"}},
    )
    assert corpus_response.status_code == 201, corpus_response.text
    corpus_id = corpus_response.json()["id"]

    ingest = client.post(
        f"/v1/corpora/{corpus_id}/documents/text",
        headers=headers,
        json={
            "source_id": "release.md",
            "title": "Release policy",
            "content": "Production releases require peer review and a rollback plan.",
            "media_type": "text/markdown",
        },
    )
    assert ingest.status_code == 201, ingest.text
    document_id = ingest.json()["document_id"]

    retrieval = client.post(
        f"/v1/corpora/{corpus_id}/retrieve",
        headers=headers,
        json={"query": "What does a production release require?"},
    )
    assert retrieval.status_code == 200, retrieval.text
    assert retrieval.json()["hits"][0]["document_id"] == document_id

    first = client.post(
        f"/v1/corpora/{corpus_id}/query",
        headers=headers,
        json={"query": "What does a production release require?"},
    )
    second = client.post(
        f"/v1/corpora/{corpus_id}/query",
        headers=headers,
        json={"query": "What does a production release require?"},
    )
    assert first.status_code == 200, first.text
    assert first.json()["citations"]
    assert not first.json()["cached"]
    assert second.json()["cached"]
    assert second.json()["routes"][0] == "cag:exact"
    assert first.headers["x-request-id"]

    skill = client.post(
        "/v1/skills/corpus_health/run",
        headers=headers,
        json={"corpus_id": corpus_id, "query": "production release"},
    )
    assert skill.status_code == 200
    assert skill.json()["retrieved_chunks"] == 1

    _, other_key = bootstrap(client, "api-other")
    forbidden_by_scope = client.get(f"/v1/corpora/{corpus_id}", headers=auth(other_key))
    assert forbidden_by_scope.status_code == 404


def test_run_skill_requires_admin_scope(client) -> None:
    _, key = bootstrap(client, "api-skill-scope")
    read_only = client.post("/v1/api-keys", headers=auth(key), json={"name": "read-only", "scopes": ["read"]})
    assert read_only.status_code == 201, read_only.text
    forbidden = client.post(
        "/v1/skills/corpus_health/run",
        headers=auth(read_only.json()["api_key"]),
        json={"corpus_id": "any", "query": "x"},
    )
    assert forbidden.status_code == 403


def test_upload_sessions_agents_and_jobs(client) -> None:
    _, key = bootstrap(client, "api-workflows")
    headers = auth(key)
    corpus_id = client.post("/v1/corpora", headers=headers, json={"name": "ops"}).json()["id"]

    upload = client.post(
        f"/v1/corpora/{corpus_id}/documents/upload",
        headers=headers,
        data={"source_id": "ops.txt", "title": "Operations", "metadata_json": '{"team":"platform"}'},
        files={"file": ("ops.txt", io.BytesIO(b"Operations incidents require a commander."), "text/plain")},
    )
    assert upload.status_code == 201, upload.text

    session = client.post(
        "/v1/sessions",
        headers=headers,
        json={"corpus_id": corpus_id, "user_id": "alice", "goal": "Learn operations"},
    )
    assert session.status_code == 201
    session_id = session.json()["id"]
    answer = client.post(
        f"/v1/corpora/{corpus_id}/query",
        headers=headers,
        json={"query": "Who is required for incidents?", "session_id": session_id, "user_id": "alice"},
    )
    assert answer.status_code == 200
    stored_session = client.get(f"/v1/sessions/{session_id}", headers=headers)
    assert len(stored_session.json()["turns"]) == 2

    planner = client.post(
        "/v1/agents",
        headers=headers,
        json={"name": "planner", "kind": "planner", "goal": "Plan"},
    ).json()["id"]
    worker = client.post(
        "/v1/agents",
        headers=headers,
        json={"name": "worker", "kind": "worker", "goal": "Execute", "parent_agent_id": planner},
    ).json()["id"]
    sent = client.post(
        f"/v1/agents/{planner}/messages",
        headers=headers,
        json={"recipient_agent_id": worker, "kind": "task", "payload": {"action": "review"}},
    )
    assert sent.status_code == 201
    mailbox = client.get(f"/v1/agents/{worker}/messages", headers=headers).json()
    assert mailbox[0]["payload"]["action"] == "review"
    ack = client.post(f"/v1/agent-messages/{mailbox[0]['id']}/acknowledge", headers=headers)
    assert ack.status_code == 204

    job = client.post(
        "/v1/jobs",
        headers=headers,
        json={"kind": "session.compact", "payload": {"session_id": session_id}},
    )
    assert job.status_code == 202
    assert client.get(f"/v1/jobs/{job.json()['id']}", headers=headers).json()["status"] == "pending"


def test_validation_and_bootstrap_guards(client) -> None:
    invalid_bootstrap = client.post(
        "/v1/bootstrap/workspaces",
        headers={"X-Bootstrap-Token": "wrong"},
        json={"slug": "invalid", "name": "Invalid"},
    )
    assert invalid_bootstrap.status_code == 401

    _, key = bootstrap(client, "api-errors")
    invalid_corpus = client.post("/v1/corpora", headers=auth(key), json={"name": ""})
    assert invalid_corpus.status_code == 422
    payload = invalid_corpus.json()["error"]
    assert payload["code"] == "request_validation_error"
    assert payload["trace_id"]


def test_administrative_lifecycle_and_background_rlm(client, runtime, settings) -> None:
    _, key = bootstrap(client, "api-admin")
    headers = auth(key)

    delegated = client.post(
        "/v1/api-keys",
        headers=headers,
        json={"name": "read-only", "scopes": ["read"]},
    )
    assert delegated.status_code == 201, delegated.text
    delegated_key_id = delegated.json()["id"]
    delegated_key = delegated.json()["api_key"]

    corpus = client.post(
        "/v1/corpora",
        headers=headers,
        json={"name": "administration", "settings": {"language": "en"}},
    )
    assert corpus.status_code == 201, corpus.text
    corpus_id = corpus.json()["id"]
    assert client.get("/v1/corpora", headers=auth(delegated_key)).status_code == 200
    assert client.get(f"/v1/corpora/{corpus_id}", headers=headers).json()["name"] == "administration"

    settings.connector_file_root.mkdir(parents=True, exist_ok=True)
    (settings.connector_file_root / "policy.md").write_text(
        "Refaat production changes require evidence, peer review, and rollback readiness.",
        encoding="utf-8",
    )
    ingested = client.post(
        f"/v1/corpora/{corpus_id}/documents/source",
        headers=headers,
        json={"source": "file://policy.md", "metadata": {"team": "platform"}},
    )
    assert ingested.status_code == 201, ingested.text
    document_id = ingested.json()["document_id"]

    documents = client.get(f"/v1/corpora/{corpus_id}/documents", headers=headers)
    assert documents.status_code == 200
    assert documents.json()[0]["id"] == document_id
    stats = client.get(f"/v1/corpora/{corpus_id}/stats", headers=headers)
    assert stats.status_code == 200
    assert stats.json()["documents"] == 1

    sync_rlm = client.post(
        f"/v1/corpora/{corpus_id}/rlm",
        headers=headers,
        json={
            "query": "What is required for production changes?",
            "limits": {"max_depth": 1, "max_nodes": 4, "branch_factor": 2, "leaf_token_budget": 256},
        },
    )
    assert sync_rlm.status_code == 200, sync_rlm.text
    assert sync_rlm.json()["citations"]

    queued_rlm = client.post(
        f"/v1/corpora/{corpus_id}/rlm/jobs",
        headers=headers,
        json={
            "query": "Summarize the production change policy.",
            "limits": {"max_depth": 1, "max_nodes": 4, "branch_factor": 2, "leaf_token_budget": 256},
        },
    )
    assert queued_rlm.status_code == 202, queued_rlm.text
    rlm_job_id = queued_rlm.json()["id"]
    assert DurableWorker(runtime).run_once()
    rlm_job = client.get(f"/v1/jobs/{rlm_job_id}", headers=headers)
    assert rlm_job.status_code == 200
    assert rlm_job.json()["status"] == "succeeded"
    assert rlm_job.json()["result"]["citations"]

    feedback = client.post(
        f"/v1/corpora/{corpus_id}/feedback",
        headers=headers,
        json={"rating": 1, "selected_chunk_ids": [sync_rlm.json()["citations"][0]["chunk_id"]]},
    )
    assert feedback.status_code == 204

    agent = client.post(
        "/v1/agents",
        headers=headers,
        json={"name": "release-agent", "kind": "reviewer", "goal": "Review evidence"},
    )
    assert agent.status_code == 201
    heartbeat = client.post(
        f"/v1/agents/{agent.json()['id']}/heartbeat",
        headers=headers,
        json={"state": {"phase": "reviewing"}},
    )
    assert heartbeat.status_code == 204

    schedule = client.post(
        "/v1/schedules",
        headers=headers,
        json={
            "name": "periodic-outbox",
            "job_kind": "outbox.publish",
            "payload": {},
            "interval_seconds": 3600,
        },
    )
    assert schedule.status_code == 201

    snapshot = client.post(
        "/v1/harness/snapshots",
        headers=headers,
        json={
            "supplemental": {"retrieval": {"top_k": 8}},
            "evidence": {"reason": "administrative E2E test"},
        },
    )
    assert snapshot.status_code == 201, snapshot.text
    activation = client.post(
        f"/v1/harness/snapshots/{snapshot.json()['id']}/activate",
        headers=headers,
    )
    assert activation.status_code == 200
    active = client.get("/v1/harness/active", headers=headers)
    assert active.status_code == 200
    assert active.json()["id"] == snapshot.json()["id"]

    cases = [
        {
            "query": "What is required for production changes?",
            "expected_terms": ["evidence", "rollback"],
            "expected_document_ids": [document_id],
        }
        for _ in range(10)
    ]
    suite = client.post(
        "/v1/evaluations/suites",
        headers=headers,
        json={"name": "release-policy", "cases": cases},
    )
    assert suite.status_code == 201, suite.text
    suite_id = suite.json()["id"]
    evaluation = client.post(
        f"/v1/evaluations/suites/{suite_id}/runs",
        headers=headers,
        json={"corpus_id": corpus_id, "asynchronous": False},
    )
    assert evaluation.status_code == 200, evaluation.text
    run_id = evaluation.json()["run_id"]
    run = client.get(f"/v1/evaluations/runs/{run_id}", headers=headers)
    assert run.status_code == 200
    assert run.json()["status"] == "succeeded"
    proposal = client.post(
        f"/v1/harness/proposals/from-evaluation/{run_id}",
        headers=headers,
    )
    assert proposal.status_code == 200, proposal.text
    assert proposal.json()["evidence"]["run_id"] == run_id

    async_evaluation = client.post(
        f"/v1/evaluations/suites/{suite_id}/runs",
        headers=headers,
        json={"corpus_id": corpus_id, "asynchronous": True},
    )
    assert async_evaluation.status_code == 200
    evaluation_job_id = async_evaluation.json()["job_id"]
    assert DurableWorker(runtime).run_once()
    evaluation_job = client.get(f"/v1/jobs/{evaluation_job_id}", headers=headers).json()
    assert evaluation_job["status"] == "succeeded"
    async_run_id = evaluation_job["result"]["run_id"]
    assert client.get(f"/v1/evaluations/runs/{async_run_id}", headers=headers).status_code == 200

    skills = client.get("/v1/skills", headers=headers)
    assert skills.status_code == 200
    assert any(skill["name"] == "corpus_health" for skill in skills.json())
    skill_project = client.post(
        "/v1/skills/projects",
        headers=headers,
        json={"name": "api_release_guard", "description": "Checks release evidence"},
    )
    assert skill_project.status_code == 201, skill_project.text

    deleted = client.delete(
        f"/v1/corpora/{corpus_id}/documents/{document_id}",
        headers=headers,
    )
    assert deleted.status_code == 200
    assert client.get(f"/v1/corpora/{corpus_id}/stats", headers=headers).json()["documents"] == 0

    revoked = client.delete(f"/v1/api-keys/{delegated_key_id}", headers=headers)
    assert revoked.status_code == 204
    assert client.get("/v1/corpora", headers=auth(delegated_key)).status_code == 401
