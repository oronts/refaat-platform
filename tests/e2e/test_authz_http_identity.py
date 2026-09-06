"""HTTP identity resolution: the principal is resolved server-side, never from the body.

Proves that with a resolver and an access policy installed, an authenticated request is
authorized against the identity carried in a trusted header, a request with no identity is
denied (fail closed), and evidence stays scoped to the resolved principal. The default app
with no resolver is covered by the rest of the API suite.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from refaat.api.app import create_app
from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.application.worker import DurableWorker
from refaat.config import Settings
from refaat.domain.audit import AuditEvent
from refaat.domain.authz import (
    CanonicalAccessPolicy,
    HeaderPrincipalResolver,
    Principal,
    durable_entitlement_digest,
)
from refaat.domain.models import DocumentInput

pytestmark = pytest.mark.e2e

DOCS = {
    "alpha.md": {"acl": {"classification": "confidential", "allow_groups": ["research-a"]}, "doc": "alpha"},
    "bravo.md": {"acl": {"classification": "confidential", "allow_groups": ["research-b"]}, "doc": "bravo"},
}


def _identity_header(groups: list[str]) -> dict[str, str]:
    return {"x-refaat-principal": json.dumps({"id": "alice", "tenant": "merck", "groups": groups})}


@pytest.fixture
def authz_client(settings: Settings) -> Iterator[tuple[TestClient, str, str]]:
    runtime = (
        RuntimeBuilder(settings)
        .with_access_policy(CanonicalAccessPolicy())
        .with_principal_resolver(HeaderPrincipalResolver())
        .build()
    )
    workspace_id = runtime.create_workspace("merck", "Merck")
    _, api_key = runtime.issue_api_key(workspace_id, "svc", scopes=("read", "write"))
    corpus_id = runtime.create_corpus(workspace_id, "research")
    for source_id, metadata in DOCS.items():
        content = f"The {metadata['doc']} checkout production release requires an approved rollback plan."
        runtime.ingest(
            workspace_id,
            corpus_id,
            DocumentInput(
                source_id=source_id,
                title=source_id,
                media_type="text/markdown",
                content=content.encode(),
                metadata=dict(metadata),
            ),
        )
    with TestClient(create_app(settings, runtime), raise_server_exceptions=False) as client:
        yield client, api_key, corpus_id
    runtime.close()


def _docs_in_response(payload: dict) -> set[str]:
    return {str(hit["metadata"].get("doc")) for hit in payload["hits"]}


def test_resolved_identity_scopes_retrieval(authz_client: tuple[TestClient, str, str]) -> None:
    client, api_key, corpus_id = authz_client
    headers = {"x-api-key": api_key, **_identity_header(["research-a"])}
    response = client.post(
        f"/v1/corpora/{corpus_id}/retrieve",
        headers=headers,
        json={"query": "checkout production release rollback plan", "top_k": 20},
    )
    assert response.status_code == 200
    docs = _docs_in_response(response.json())
    assert "alpha" in docs
    assert "bravo" not in docs


def test_a_client_cannot_widen_access_through_the_body(authz_client: tuple[TestClient, str, str]) -> None:
    client, api_key, corpus_id = authz_client
    # The body carries no identity field, and a metadata attempt to forge groups is ignored:
    # identity comes only from the trusted header, which grants research-a.
    headers = {"x-api-key": api_key, **_identity_header(["research-a"])}
    response = client.post(
        f"/v1/corpora/{corpus_id}/retrieve",
        headers=headers,
        json={
            "query": "checkout production release rollback plan",
            "top_k": 20,
            "metadata": {"groups": ["research-b"], "acl": {"allow_groups": ["research-b"]}},
        },
    )
    assert response.status_code == 200
    assert "bravo" not in _docs_in_response(response.json())


def test_missing_identity_is_denied_when_authorization_is_active(authz_client: tuple[TestClient, str, str]) -> None:
    client, api_key, corpus_id = authz_client
    response = client.post(
        f"/v1/corpora/{corpus_id}/retrieve",
        headers={"x-api-key": api_key},
        json={"query": "checkout production release rollback plan", "top_k": 20},
    )
    assert response.status_code == 403


def test_query_answer_only_cites_authorized_evidence(authz_client: tuple[TestClient, str, str]) -> None:
    client, api_key, corpus_id = authz_client
    headers = {"x-api-key": api_key, **_identity_header(["research-b"])}
    response = client.post(
        f"/v1/corpora/{corpus_id}/query",
        headers=headers,
        json={"query": "checkout production release rollback plan"},
    )
    assert response.status_code == 200
    quotes = " ".join(citation["quote"] for citation in response.json()["citations"])
    assert "alpha" not in quotes


def _program_cited_documents(client: TestClient, api_key: str, corpus_id: str, groups: list[str]) -> set[str]:
    response = client.post(
        f"/v1/corpora/{corpus_id}/programs/grounded-rag/run",
        headers={"x-api-key": api_key, **_identity_header(groups)},
        json={"question": "checkout production release rollback plan"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["citations"]
    return {str(citation["document_id"]) for citation in body["citations"]}


def test_program_run_scopes_evidence_to_the_resolved_principal(authz_client: tuple[TestClient, str, str]) -> None:
    client, api_key, corpus_id = authz_client
    # The program run enforces the resolved principal's ACL exactly as a direct query does:
    # research-a and research-b cite disjoint confidential evidence (alpha vs bravo), so a
    # program run as one principal never surfaces the other principal's document.
    alice_docs = _program_cited_documents(client, api_key, corpus_id, ["research-a"])
    bob_docs = _program_cited_documents(client, api_key, corpus_id, ["research-b"])
    assert alice_docs != bob_docs
    assert alice_docs - bob_docs
    assert bob_docs - alice_docs


def test_program_run_denies_a_missing_identity_under_an_active_policy(
    authz_client: tuple[TestClient, str, str],
) -> None:
    client, api_key, corpus_id = authz_client
    response = client.post(
        f"/v1/corpora/{corpus_id}/programs/grounded-rag/run",
        headers={"x-api-key": api_key},
        json={"question": "checkout production release rollback plan"},
    )
    assert response.status_code == 403


def test_skill_and_evaluate_surfaces_stay_fail_closed_under_an_active_policy(
    authz_client: tuple[TestClient, str, str],
) -> None:
    client, api_key, corpus_id = authz_client
    headers = {"x-api-key": api_key, **_identity_header(["research-a"])}
    run_skill = client.post(
        "/v1/skills/does-not-matter/run",
        headers=headers,
        json={"corpus_id": corpus_id, "query": "checkout production release rollback plan"},
    )
    assert run_skill.status_code == 403
    evaluate = client.post(
        f"/v1/corpora/{corpus_id}/programs/grounded-rag/evaluate",
        headers=headers,
        json={"cases": [{"query": "checkout production release rollback plan"}]},
    )
    assert evaluate.status_code == 403


def test_list_documents_filters_by_resolved_principal(authz_client: tuple[TestClient, str, str]) -> None:
    client, api_key, corpus_id = authz_client
    headers = {"x-api-key": api_key, **_identity_header(["research-a"])}
    response = client.get(f"/v1/corpora/{corpus_id}/documents", headers=headers)
    assert response.status_code == 200
    sources = {str(doc["source_id"]) for doc in response.json()}
    assert "alpha.md" in sources
    assert "bravo.md" not in sources


def test_session_stats_and_evaluation_read_surfaces_fail_closed(authz_client: tuple[TestClient, str, str]) -> None:
    client, api_key, corpus_id = authz_client
    headers = {"x-api-key": api_key, **_identity_header(["research-a"])}
    assert client.get(f"/v1/corpora/{corpus_id}/stats", headers=headers).status_code == 403
    assert client.get("/v1/sessions/does-not-matter", headers=headers).status_code == 403
    assert (
        client.post(
            "/v1/evaluations/suites/some-suite/runs",
            headers=headers,
            json={"corpus_id": corpus_id, "asynchronous": False},
        ).status_code
        == 403
    )


@pytest.fixture
def policy_only_client(settings: Settings) -> Iterator[tuple[TestClient, str, str]]:
    runtime = RuntimeBuilder(settings).with_access_policy(CanonicalAccessPolicy()).build()
    workspace_id = runtime.create_workspace("merck", "Merck")
    _, api_key = runtime.issue_api_key(workspace_id, "svc", scopes=("read", "write"))
    corpus_id = runtime.create_corpus(workspace_id, "research")
    for source_id, metadata in DOCS.items():
        content = f"The {metadata['doc']} checkout production release requires an approved rollback plan."
        runtime.ingest(
            workspace_id,
            corpus_id,
            DocumentInput(
                source_id=source_id,
                title=source_id,
                media_type="text/markdown",
                content=content.encode(),
                metadata=dict(metadata),
            ),
        )
    with TestClient(create_app(settings, runtime), raise_server_exceptions=False) as client:
        yield client, api_key, corpus_id
    runtime.close()


def test_policy_without_a_resolver_denies_the_http_surface(policy_only_client: tuple[TestClient, str, str]) -> None:
    client, api_key, corpus_id = policy_only_client
    response = client.post(
        f"/v1/corpora/{corpus_id}/retrieve",
        headers={"x-api-key": api_key},
        json={"query": "checkout production release rollback plan", "top_k": 20},
    )
    assert response.status_code == 403


def test_async_job_surfaces_fail_closed_under_an_active_policy(authz_client: tuple[TestClient, str, str]) -> None:
    client, api_key, corpus_id = authz_client
    headers = {"x-api-key": api_key, **_identity_header(["research-a"])}
    enqueue_rlm = client.post(
        f"/v1/corpora/{corpus_id}/rlm/jobs",
        headers=headers,
        json={"query": "checkout production release rollback plan"},
    )
    assert enqueue_rlm.status_code == 403
    create_job = client.post(
        "/v1/jobs",
        headers=headers,
        json={"kind": "rlm.run", "payload": {"corpus_id": corpus_id, "query": "checkout"}},
    )
    assert create_job.status_code == 403


def test_async_optimization_surface_fails_closed_under_an_active_policy(
    authz_client: tuple[TestClient, str, str],
) -> None:
    client, api_key, corpus_id = authz_client
    headers = {"x-api-key": api_key, **_identity_header(["research-a"])}
    enqueue = client.post(
        "/v1/programs/grounded-rag/optimizations",
        headers=headers,
        json={"corpus_id": corpus_id, "cases": [{"query": "checkout", "expected_terms": ["rollback"]}], "budget": 2},
    )
    assert enqueue.status_code == 403
    assert client.get("/v1/optimizations/does-not-matter", headers=headers).status_code == 403


def test_schedule_surface_fails_closed_under_an_active_policy(authz_client: tuple[TestClient, str, str]) -> None:
    client, api_key, corpus_id = authz_client
    headers = {"x-api-key": api_key, **_identity_header(["research-a"])}
    response = client.post(
        "/v1/schedules",
        headers=headers,
        json={
            "name": "shadow",
            "job_kind": "program.shadow_compare",
            "payload": {
                "corpus_id": corpus_id,
                "program": "grounded-rag",
                "environment": "staging",
                "questions": ["x"],
            },
            "interval_seconds": 1,
        },
    )
    assert response.status_code == 403


def test_run_and_job_read_backs_fail_closed_under_an_active_policy(authz_client: tuple[TestClient, str, str]) -> None:
    client, api_key, _ = authz_client
    headers = {"x-api-key": api_key, **_identity_header(["research-a"])}
    assert client.get("/v1/jobs/does-not-matter", headers=headers).status_code == 403
    assert client.get("/v1/evaluations/runs/does-not-matter", headers=headers).status_code == 403
    assert client.post("/v1/harness/proposals/from-evaluation/does-not-matter", headers=headers).status_code == 403


ALICE_PRINCIPAL = Principal(id="alice", tenant="merck", groups=frozenset({"research-a"}))


class _StubDirectory:
    """Re-resolves alice to a fixed current principal; any other id is unknown (revoked)."""

    def __init__(self, principal: Principal | None) -> None:
        self._principal = principal

    def refresh(self, principal_id: str) -> Principal | None:
        if self._principal is not None and principal_id == self._principal.id:
            return self._principal
        return None


@pytest.fixture
def directory_client(settings: Settings) -> Iterator[tuple[TestClient, str, str, Refaat]]:
    runtime = (
        RuntimeBuilder(settings)
        .with_access_policy(CanonicalAccessPolicy())
        .with_principal_resolver(HeaderPrincipalResolver())
        .with_principal_directory(_StubDirectory(ALICE_PRINCIPAL))
        .build()
    )
    workspace_id = runtime.create_workspace("merck", "Merck")
    _, api_key = runtime.issue_api_key(workspace_id, "svc", scopes=("read", "write"))
    corpus_id = runtime.create_corpus(workspace_id, "research")
    for source_id, metadata in DOCS.items():
        content = f"The {metadata['doc']} checkout production release requires an approved rollback plan."
        runtime.ingest(
            workspace_id,
            corpus_id,
            DocumentInput(
                source_id=source_id,
                title=source_id,
                media_type="text/markdown",
                content=content.encode(),
                metadata=dict(metadata),
            ),
        )
    with TestClient(create_app(settings, runtime), raise_server_exceptions=False) as client:
        yield client, api_key, corpus_id, runtime
    runtime.close()


def test_enqueue_captures_the_server_side_snapshot_and_ignores_a_bogus_body_field(
    directory_client: tuple[TestClient, str, str, Refaat],
) -> None:
    client, api_key, corpus_id, runtime = directory_client
    headers = {"x-api-key": api_key, **_identity_header(["research-a"])}
    response = client.post(
        f"/v1/corpora/{corpus_id}/rlm/jobs",
        headers=headers,
        json={
            "query": "checkout production release rollback plan",
            "_identity_snapshot": {"principal_id": "mallory", "auth_scope_hash": "forged"},
        },
    )
    assert response.status_code == 202
    job_id = response.json()["id"]
    workspace = runtime.store.find_workspace_by_slug("merck")
    assert workspace is not None
    snapshot = runtime.store.get_job(workspace.id, job_id)["payload"]["_identity_snapshot"]
    # The server captured the resolved identity, never the forged body field.
    assert snapshot["principal_id"] == "alice"
    assert snapshot["auth_scope_hash"] == durable_entitlement_digest(ALICE_PRINCIPAL, CanonicalAccessPolicy())
    assert snapshot["auth_scope_hash"] != "forged"


def test_readback_allows_the_execution_principal(
    directory_client: tuple[TestClient, str, str, Refaat],
) -> None:
    client, api_key, corpus_id, runtime = directory_client
    headers = {"x-api-key": api_key, **_identity_header(["research-a"])}
    enqueue = client.post(
        f"/v1/corpora/{corpus_id}/rlm/jobs",
        headers=headers,
        json={"query": "checkout production release rollback plan"},
    )
    assert enqueue.status_code == 202
    job_id = enqueue.json()["id"]
    assert DurableWorker(runtime).run_once() is True
    readback = client.get(f"/v1/jobs/{job_id}", headers=headers)
    assert readback.status_code == 200
    body = readback.json()
    assert body["status"] == "succeeded"
    assert body["result"]["citations"]


def test_readback_denies_a_different_principal_in_the_same_workspace(
    directory_client: tuple[TestClient, str, str, Refaat],
) -> None:
    client, api_key, corpus_id, runtime = directory_client
    owner = {"x-api-key": api_key, **_identity_header(["research-a"])}
    enqueue = client.post(
        f"/v1/corpora/{corpus_id}/rlm/jobs",
        headers=owner,
        json={"query": "checkout production release rollback plan"},
    )
    job_id = enqueue.json()["id"]
    assert DurableWorker(runtime).run_once() is True
    # Same id alice, different entitlements: the durable digest differs, so the reader is not the
    # execution-time principal. The denial is obscured to the same not-found a missing id returns, so
    # a same-workspace principal cannot use the job read as an existence oracle.
    other = {"x-api-key": api_key, **_identity_header(["research-b"])}
    denied = client.get(f"/v1/jobs/{job_id}", headers=other)
    missing = client.get("/v1/jobs/does-not-exist", headers=other)
    assert denied.status_code == missing.status_code == 404
    # The only distinguishing field an attacker could read is the error code and message; both are the
    # canonical not-found, so the two cases are indistinguishable (trace_id is per-request noise).
    assert denied.json()["error"]["code"] == missing.json()["error"]["code"] == "not_found"
    assert denied.json()["error"]["message"] == missing.json()["error"]["message"] == "Job not found"


def test_cross_workspace_readback_is_workspace_scoped(
    directory_client: tuple[TestClient, str, str, Refaat],
) -> None:
    client, api_key, corpus_id, runtime = directory_client
    owner = {"x-api-key": api_key, **_identity_header(["research-a"])}
    job_id = client.post(
        f"/v1/corpora/{corpus_id}/rlm/jobs",
        headers=owner,
        json={"query": "checkout production release rollback plan"},
    ).json()["id"]
    # A second workspace's key cannot see the job row at all: get_job is workspace-scoped, so it 404s
    # before the digest check ever runs. This is a scoping guarantee, not the digest guard test.
    other_ws = runtime.create_workspace("pfizer", "Pfizer")
    _, other_key = runtime.issue_api_key(other_ws, "svc", scopes=("read", "write"))
    response = client.get(f"/v1/jobs/{job_id}", headers={"x-api-key": other_key, **_identity_header(["research-a"])})
    assert response.status_code == 404


def test_readback_denies_a_non_rlm_job_under_a_policy(
    directory_client: tuple[TestClient, str, str, Refaat],
) -> None:
    client, api_key, _corpus_id, runtime = directory_client
    workspace = runtime.store.find_workspace_by_slug("merck")
    assert workspace is not None
    job_id = runtime.store.enqueue_job(workspace.id, "cache.purge", {})
    headers = {"x-api-key": api_key, **_identity_header(["research-a"])}
    # A non-rlm job carries no server-side snapshot, so it is not principal-readable; the denial is
    # obscured to the same not-found a missing id returns.
    assert client.get(f"/v1/jobs/{job_id}", headers=headers).status_code == 404


def test_create_job_with_a_forged_snapshot_is_denied_even_with_a_directory(
    directory_client: tuple[TestClient, str, str, Refaat],
) -> None:
    client, api_key, corpus_id, _runtime = directory_client
    headers = {"x-api-key": api_key, **_identity_header(["research-a"])}
    # The generic job surface stays fail-closed under a policy, so a client cannot smuggle a forged
    # snapshot onto an rlm.run job it enqueues directly; only the server-side enqueue_rlm writes one.
    response = client.post(
        "/v1/jobs",
        headers=headers,
        json={
            "kind": "rlm.run",
            "payload": {
                "corpus_id": corpus_id,
                "query": "checkout",
                "_identity_snapshot": {"principal_id": "alice", "auth_scope_hash": "forged"},
            },
        },
    )
    assert response.status_code == 403


class _RecordingSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        self.events.append(event)


@pytest.fixture
def audited_directory_client(
    settings: Settings,
) -> Iterator[tuple[TestClient, str, str, Refaat, _RecordingSink]]:
    sink = _RecordingSink()
    runtime = (
        RuntimeBuilder(settings)
        .with_access_policy(CanonicalAccessPolicy())
        .with_principal_resolver(HeaderPrincipalResolver())
        .with_principal_directory(_StubDirectory(ALICE_PRINCIPAL))
        .with_audit_sink(sink)
        .build()
    )
    workspace_id = runtime.create_workspace("merck", "Merck")
    _, api_key = runtime.issue_api_key(workspace_id, "svc", scopes=("read", "write"))
    corpus_id = runtime.create_corpus(workspace_id, "research")
    for source_id, metadata in DOCS.items():
        content = f"The {metadata['doc']} checkout production release requires an approved rollback plan."
        runtime.ingest(
            workspace_id,
            corpus_id,
            DocumentInput(
                source_id=source_id,
                title=source_id,
                media_type="text/markdown",
                content=content.encode(),
                metadata=dict(metadata),
            ),
        )
    with TestClient(create_app(settings, runtime), raise_server_exceptions=False) as client:
        yield client, api_key, corpus_id, runtime, sink
    runtime.close()


def test_an_obscured_job_read_audits_the_true_reason(
    audited_directory_client: tuple[TestClient, str, str, Refaat, _RecordingSink],
) -> None:
    client, api_key, corpus_id, runtime, sink = audited_directory_client
    owner = {"x-api-key": api_key, **_identity_header(["research-a"])}
    job_id = client.post(
        f"/v1/corpora/{corpus_id}/rlm/jobs",
        headers=owner,
        json={"query": "checkout production release rollback plan"},
    ).json()["id"]
    assert DurableWorker(runtime).run_once() is True
    sink.events.clear()
    other = {"x-api-key": api_key, **_identity_header(["research-b"])}
    denied = client.get(f"/v1/jobs/{job_id}", headers=other)
    assert denied.status_code == 404
    # The client sees a uniform not-found, but the true reason is recorded internally, so the obscured
    # denial is not an audit blind spot, and the event carries no job payload.
    reasons = [event.event_type for event in sink.events]
    assert "job.read.denied:not_authorized" in reasons


def test_job_read_with_unresolved_identity_is_obscured_not_403(
    directory_client: tuple[TestClient, str, str, Refaat],
) -> None:
    client, api_key, corpus_id, runtime = directory_client
    owner = {"x-api-key": api_key, **_identity_header(["research-a"])}
    job_id = client.post(
        f"/v1/corpora/{corpus_id}/rlm/jobs",
        headers=owner,
        json={"query": "checkout production release rollback plan"},
    ).json()["id"]
    assert DurableWorker(runtime).run_once() is True
    # No identity header: the resolver returns no principal. The read must still obscure to the same
    # not-found a missing id returns, never a distinguishable 403 from the identity boundary.
    no_identity = {"x-api-key": api_key}
    existing = client.get(f"/v1/jobs/{job_id}", headers=no_identity)
    missing = client.get("/v1/jobs/does-not-exist", headers=no_identity)
    assert existing.status_code == missing.status_code == 404
    assert existing.json()["error"]["code"] == missing.json()["error"]["code"] == "not_found"
    assert existing.json()["error"]["message"] == missing.json()["error"]["message"] == "Job not found"


def test_an_unresolved_identity_job_read_is_audited_as_system(
    audited_directory_client: tuple[TestClient, str, str, Refaat, _RecordingSink],
) -> None:
    client, api_key, corpus_id, runtime, sink = audited_directory_client
    owner = {"x-api-key": api_key, **_identity_header(["research-a"])}
    job_id = client.post(
        f"/v1/corpora/{corpus_id}/rlm/jobs",
        headers=owner,
        json={"query": "checkout production release rollback plan"},
    ).json()["id"]
    assert DurableWorker(runtime).run_once() is True
    sink.events.clear()
    assert client.get(f"/v1/jobs/{job_id}", headers={"x-api-key": api_key}).status_code == 404
    # The unresolved caller is attributed to the system sentinel, so an unresolved probe is not an
    # audit blind spot even though no principal resolved.
    denied = [event for event in sink.events if event.event_type == "job.read.denied:not_authorized"]
    assert denied
    assert denied[0].principal_id == "system:unresolved"
