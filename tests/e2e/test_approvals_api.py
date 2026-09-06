"""The HTTP approval surface: grant, list, and cancel durable single-use approvals.

The approver is the server-resolved identity, never a request-body field, so a caller can only
approve as itself. The surface is admin-scoped and denies a grant that carries no resolved identity.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from refaat.api.app import create_app
from refaat.api.dependencies import deny_ambient_approvals
from refaat.application.approvals import ApprovalContext, ApprovalService, approval_scope
from refaat.application.runtime import RuntimeBuilder
from refaat.config import Settings
from refaat.domain.audit import AuditEvent
from refaat.domain.authz import CanonicalAccessPolicy, HeaderPrincipalResolver
from refaat.domain.governance import ApprovalRecord
from refaat.errors import AuthorizationError
from refaat.infrastructure.database.schema import approval_requests

pytestmark = pytest.mark.e2e


def _fp(seed: str) -> str:
    # The grant surface accepts only a sha256 hex fingerprint, so tests use real digests.
    return hashlib.sha256(seed.encode()).hexdigest()


def _approver(principal_id: str = "bob", roles: tuple[str, ...] = ("reviewer",)) -> dict[str, str]:
    return {"x-refaat-principal": json.dumps({"id": principal_id, "tenant": "merck", "roles": list(roles)})}


@pytest.fixture
def approvals_client(settings: Settings) -> Iterator[tuple[TestClient, str, str]]:
    runtime = (
        RuntimeBuilder(settings)
        .with_access_policy(CanonicalAccessPolicy())
        .with_principal_resolver(HeaderPrincipalResolver())
        .build()
    )
    workspace_id = runtime.create_workspace("merck", "Merck")
    _, admin_key = runtime.issue_api_key(workspace_id, "admin", scopes=("read", "write", "admin"))
    _, member_key = runtime.issue_api_key(workspace_id, "member", scopes=("read", "write"))
    with TestClient(create_app(settings, runtime), raise_server_exceptions=False) as client:
        yield client, admin_key, member_key
    runtime.close()


def test_grant_records_the_resolved_identity_as_approver(approvals_client: tuple[TestClient, str, str]) -> None:
    client, admin_key, _ = approvals_client
    headers = {"x-api-key": admin_key, **_approver("bob", ("reviewer",))}
    granted = client.post("/v1/approvals", headers=headers, json={"fingerprint": _fp("1"), "capability_name": "deploy"})
    assert granted.status_code == 201
    assert granted.json()["approval_id"]

    listed = client.get("/v1/approvals", headers=headers, params={"fingerprint": _fp("1")})
    assert listed.status_code == 200
    rows = listed.json()
    assert len(rows) == 1
    assert rows[0]["approver_id"] == "bob"
    assert "reviewer" in rows[0]["approver_roles"]


def test_body_cannot_set_the_approver(approvals_client: tuple[TestClient, str, str]) -> None:
    client, admin_key, _ = approvals_client
    headers = {"x-api-key": admin_key, **_approver("bob", ("reviewer",))}
    granted = client.post(
        "/v1/approvals",
        headers=headers,
        json={"fingerprint": _fp("3"), "capability_name": "deploy", "approver_id": "alice"},
    )
    assert granted.status_code == 201
    rows = client.get("/v1/approvals", headers=headers, params={"fingerprint": _fp("3")}).json()
    assert rows[0]["approver_id"] == "bob"


def test_grant_requires_a_resolved_identity(approvals_client: tuple[TestClient, str, str]) -> None:
    client, admin_key, _ = approvals_client
    granted = client.post(
        "/v1/approvals",
        headers={"x-api-key": admin_key},
        json={"fingerprint": _fp("1"), "capability_name": "deploy"},
    )
    assert granted.status_code == 403


def test_grant_requires_the_admin_scope(approvals_client: tuple[TestClient, str, str]) -> None:
    client, _, member_key = approvals_client
    headers = {"x-api-key": member_key, **_approver("bob", ("reviewer",))}
    granted = client.post("/v1/approvals", headers=headers, json={"fingerprint": _fp("1"), "capability_name": "deploy"})
    assert granted.status_code == 403


def test_ambient_approvals_are_rejected_on_the_program_run_surface() -> None:
    # The in-memory ApprovalContext is a trusted in-process channel; an untrusted surface must
    # never honor one, so the program-run guard fails closed when it is non-empty.
    deny_ambient_approvals()
    with (
        approval_scope(ApprovalContext(approvals=(ApprovalRecord(fingerprint="fp", approver_id="bob"),))),
        pytest.raises(AuthorizationError, match="Ambient approvals"),
    ):
        deny_ambient_approvals()


def test_cancel_removes_the_active_approval(approvals_client: tuple[TestClient, str, str]) -> None:
    client, admin_key, _ = approvals_client
    headers = {"x-api-key": admin_key, **_approver("bob", ("reviewer",))}
    granted = client.post("/v1/approvals", headers=headers, json={"fingerprint": _fp("2"), "capability_name": "deploy"})
    approval_id = granted.json()["approval_id"]

    cancelled = client.delete(f"/v1/approvals/{approval_id}", headers={"x-api-key": admin_key, **_approver("bob")})
    assert cancelled.status_code == 200
    assert cancelled.json()["cancelled"] is True
    assert client.get("/v1/approvals", headers=headers, params={"fingerprint": _fp("2")}).json() == []


class _CapturingSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        self.events.append(event)


@pytest.fixture
def audited_approvals_client(settings: Settings) -> Iterator[tuple[TestClient, str, _CapturingSink]]:
    sink = _CapturingSink()
    runtime = (
        RuntimeBuilder(settings)
        .with_access_policy(CanonicalAccessPolicy())
        .with_principal_resolver(HeaderPrincipalResolver())
        .with_audit_sink(sink)
        .build()
    )
    workspace_id = runtime.create_workspace("merck", "Merck")
    _, admin_key = runtime.issue_api_key(workspace_id, "admin", scopes=("read", "write", "admin"))
    with TestClient(create_app(settings, runtime), raise_server_exceptions=False) as client:
        yield client, admin_key, sink
    runtime.close()


def test_grant_and_cancel_emit_redacted_lifecycle_events(
    audited_approvals_client: tuple[TestClient, str, _CapturingSink],
) -> None:
    client, admin_key, sink = audited_approvals_client
    headers = {"x-api-key": admin_key, **_approver("bob", ("reviewer",))}
    granted = client.post(
        "/v1/approvals", headers=headers, json={"fingerprint": _fp("secret"), "capability_name": "deploy"}
    )
    approval_id = granted.json()["approval_id"]

    grants = [event for event in sink.events if event.event_type == "approval.granted"]
    assert len(grants) == 1
    assert grants[0].capability_name == "deploy"
    assert grants[0].action_fingerprint == _fp("secret")
    assert grants[0].principal_id == "bob"

    client.delete(f"/v1/approvals/{approval_id}", headers=headers)
    cancels = [event for event in sink.events if event.event_type == "approval.cancelled"]
    assert len(cancels) == 1
    assert cancels[0].capability_name == "deploy"
    assert cancels[0].action_fingerprint == _fp("secret")
    assert cancels[0].principal_id == "bob"


def test_two_grants_of_one_fingerprint_get_distinct_dedup_ids(
    audited_approvals_client: tuple[TestClient, str, _CapturingSink],
) -> None:
    client, admin_key, sink = audited_approvals_client
    headers = {"x-api-key": admin_key, **_approver("bob", ("reviewer",))}
    body = {"fingerprint": _fp("dup"), "capability_name": "deploy"}
    client.post("/v1/approvals", headers=headers, json=body)
    client.post("/v1/approvals", headers=headers, json=body)

    grants = [event for event in sink.events if event.event_type == "approval.granted"]
    assert len(grants) == 2
    assert len({event.dedup_id for event in grants}) == 2


def test_lifecycle_emits_nothing_without_a_sink(approvals_client: tuple[TestClient, str, str]) -> None:
    client, admin_key, _ = approvals_client
    headers = {"x-api-key": admin_key, **_approver("bob", ("reviewer",))}
    granted = client.post("/v1/approvals", headers=headers, json={"fingerprint": _fp("x"), "capability_name": "deploy"})
    approval_id = granted.json()["approval_id"]
    cancelled = client.delete(f"/v1/approvals/{approval_id}", headers=headers)
    assert cancelled.json()["cancelled"] is True


def test_cancel_works_without_a_resolver_under_an_access_policy(settings: Settings) -> None:
    # Cancel is authorized by the admin api key, so an access policy with no resolver must not 403 it.
    runtime = RuntimeBuilder(settings).with_access_policy(CanonicalAccessPolicy()).build()
    try:
        workspace_id = runtime.create_workspace("merck", "Merck")
        _, admin_key = runtime.issue_api_key(workspace_id, "admin", scopes=("read", "write", "admin"))
        approval_id = ApprovalService(runtime.store, workspace_id).grant(
            fingerprint=_fp("noresolver"), capability_name="deploy", approver_id="bob"
        )
        with TestClient(create_app(settings, runtime), raise_server_exceptions=False) as client:
            cancelled = client.delete(f"/v1/approvals/{approval_id}", headers={"x-api-key": admin_key})
        assert cancelled.status_code == 200
        assert cancelled.json()["cancelled"] is True
    finally:
        runtime.close()


def test_grant_rejects_a_non_identifier_capability_name(approvals_client: tuple[TestClient, str, str]) -> None:
    client, admin_key, _ = approvals_client
    headers = {"x-api-key": admin_key, **_approver("bob", ("reviewer",))}
    granted = client.post(
        "/v1/approvals", headers=headers, json={"fingerprint": _fp("1"), "capability_name": "raw query SECRET-123"}
    )
    assert granted.status_code == 422


def test_a_policy_only_ambient_context_is_also_rejected() -> None:
    with (
        approval_scope(ApprovalContext(approvals=(), policy_id="custom", policy_version="2")),
        pytest.raises(AuthorizationError, match="Ambient approvals"),
    ):
        deny_ambient_approvals()


def test_break_glass_grant_stores_the_reason_and_emits_a_redacted_event(
    audited_approvals_client: tuple[TestClient, str, _CapturingSink],
) -> None:
    client, admin_key, sink = audited_approvals_client
    headers = {"x-api-key": admin_key, **_approver("bob", ("incident-commander",))}
    reason = "SECRET-INCIDENT-REASON-77"
    granted = client.post(
        "/v1/approvals",
        headers=headers,
        json={
            "fingerprint": _fp("bg"),
            "capability_name": "deploy",
            "break_glass": True,
            "reason": reason,
        },
    )
    assert granted.status_code == 201

    events = [event for event in sink.events if event.event_type == "approval.break_glass"]
    assert len(events) == 1
    assert events[0].capability_name == "deploy"
    assert events[0].action_fingerprint == _fp("bg")
    # The break-glass reason must never appear in any string field of any emitted event.
    for event in sink.events:
        payload = event.to_json()
        strings = [value for value in payload.values() if isinstance(value, str)]
        for value in payload.values():
            if isinstance(value, list):
                strings.extend(item for item in value if isinstance(item, str))
        assert all(reason not in value for value in strings)


def test_break_glass_reason_is_bounded(approvals_client: tuple[TestClient, str, str]) -> None:
    client, admin_key, _ = approvals_client
    headers = {"x-api-key": admin_key, **_approver("bob", ("incident-commander",))}
    granted = client.post(
        "/v1/approvals",
        headers=headers,
        json={
            "fingerprint": _fp("bg2"),
            "capability_name": "deploy",
            "break_glass": True,
            "reason": "x" * 501,
        },
    )
    assert granted.status_code == 422


class _RaisingResolver:
    def resolve(self, context: dict[str, object]) -> None:
        del context
        raise RuntimeError("resolver unavailable")


def test_request_records_the_resolved_identity_as_requester(settings: Settings) -> None:
    # The requester is the server-resolved identity, never a request-body field, so a body that tries
    # to set requested_by is ignored and the row is attributed to the caller.
    runtime = (
        RuntimeBuilder(settings)
        .with_access_policy(CanonicalAccessPolicy())
        .with_principal_resolver(HeaderPrincipalResolver())
        .build()
    )
    try:
        workspace_id = runtime.create_workspace("merck", "Merck")
        _, admin_key = runtime.issue_api_key(workspace_id, "admin", scopes=("read", "write", "admin"))
        headers = {"x-api-key": admin_key, **_approver("bob", ("reviewer",))}
        fingerprint = _fp("req")
        with TestClient(create_app(settings, runtime), raise_server_exceptions=False) as client:
            created = client.post(
                "/v1/approvals/requests",
                headers=headers,
                json={"fingerprint": fingerprint, "capability_name": "deploy", "requested_by": "mallory"},
            )
        assert created.status_code == 201
        body = created.json()
        assert body["request_id"]
        assert body["created_at"]
        with runtime.store.engine.connect() as conn:
            rows = (
                conn.execute(sa.select(approval_requests).where(approval_requests.c.fingerprint == fingerprint))
                .mappings()
                .all()
            )
        assert len(rows) == 1
        assert rows[0]["requested_by"] == "bob"
        assert rows[0]["status"] == "open"
    finally:
        runtime.close()


def test_request_requires_a_resolved_identity(approvals_client: tuple[TestClient, str, str]) -> None:
    client, admin_key, _ = approvals_client
    created = client.post(
        "/v1/approvals/requests",
        headers={"x-api-key": admin_key},
        json={"fingerprint": _fp("req"), "capability_name": "deploy"},
    )
    assert created.status_code == 403


def test_request_requires_the_admin_scope(approvals_client: tuple[TestClient, str, str]) -> None:
    client, _, member_key = approvals_client
    headers = {"x-api-key": member_key, **_approver("bob", ("reviewer",))}
    created = client.post(
        "/v1/approvals/requests", headers=headers, json={"fingerprint": _fp("req"), "capability_name": "deploy"}
    )
    assert created.status_code == 403


def test_request_rejects_a_non_hex_fingerprint(approvals_client: tuple[TestClient, str, str]) -> None:
    client, admin_key, _ = approvals_client
    headers = {"x-api-key": admin_key, **_approver("bob", ("reviewer",))}
    created = client.post(
        "/v1/approvals/requests", headers=headers, json={"fingerprint": "not-a-digest", "capability_name": "deploy"}
    )
    assert created.status_code == 422


def test_cancel_survives_a_raising_resolver(settings: Settings) -> None:
    runtime = (
        RuntimeBuilder(settings)
        .with_access_policy(CanonicalAccessPolicy())
        .with_principal_resolver(_RaisingResolver())
        .build()
    )
    try:
        workspace_id = runtime.create_workspace("merck", "Merck")
        _, admin_key = runtime.issue_api_key(workspace_id, "admin", scopes=("read", "write", "admin"))
        approval_id = ApprovalService(runtime.store, workspace_id).grant(
            fingerprint=_fp("raise"), capability_name="deploy", approver_id="bob"
        )
        with TestClient(create_app(settings, runtime), raise_server_exceptions=False) as client:
            cancelled = client.delete(f"/v1/approvals/{approval_id}", headers={"x-api-key": admin_key})
        assert cancelled.status_code == 200
        assert cancelled.json()["cancelled"] is True
    finally:
        runtime.close()
