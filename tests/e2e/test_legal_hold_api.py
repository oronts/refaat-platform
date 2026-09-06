"""The HTTP legal-hold surface: admin place, list, lift, and the two-shape refusal contract.

The actor is the authenticated admin identity, never a body field. Delete of a held document returns
409; erase of a held document returns 200 with status "held". On a non-registry runtime the place
route fails closed with 409 and records nothing.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from refaat.api.app import create_app
from refaat.application.runtime import RuntimeBuilder
from refaat.config import Settings
from refaat.infrastructure.database.schema import legal_holds

pytestmark = pytest.mark.e2e

_BODY = "Payment gateway timeout cascaded into a checkout outage; the rollback restored service."


@pytest.fixture
def holds_client(settings: Settings) -> Iterator[tuple[TestClient, str, str]]:
    runtime = RuntimeBuilder(settings).with_legal_hold_registry().build()
    workspace_id = runtime.create_workspace("acme", "Acme")
    _, admin_key = runtime.issue_api_key(workspace_id, "admin", scopes=("read", "write", "admin"))
    corpus_id = runtime.create_corpus(workspace_id, "handbook")
    with TestClient(create_app(settings, runtime), raise_server_exceptions=False) as client:
        yield client, admin_key, corpus_id
    runtime.close()


def _ingest(client: TestClient, admin_key: str, corpus_id: str) -> str:
    response = client.post(
        f"/v1/corpora/{corpus_id}/documents/text",
        headers={"x-api-key": admin_key},
        json={"source_id": "policy.txt", "title": "Policy", "content": _BODY},
    )
    assert response.status_code in (200, 201), response.text
    return response.json()["document_id"]


def test_place_list_and_lift_over_the_admin_surface(holds_client: tuple[TestClient, str, str]) -> None:
    client, admin_key, corpus_id = holds_client
    headers = {"x-api-key": admin_key}
    doc = _ingest(client, admin_key, corpus_id)

    placed = client.post(
        f"/v1/corpora/{corpus_id}/legal-holds",
        headers=headers,
        json={"document_id": doc, "reason": "matter 14"},
    )
    assert placed.status_code == 200, placed.text
    hold_id = placed.json()["hold_id"]

    listed = client.get(f"/v1/corpora/{corpus_id}/legal-holds", headers=headers)
    assert listed.status_code == 200
    holds = listed.json()["holds"]
    assert len(holds) == 1
    assert holds[0]["document_id"] == doc
    assert holds[0]["lifted_at"] is None

    lifted = client.post(f"/v1/legal-holds/{hold_id}/lift", headers=headers)
    assert lifted.status_code == 200
    assert lifted.json()["lifted"] is True
    assert client.get(f"/v1/corpora/{corpus_id}/legal-holds", headers=headers).json()["holds"] == []


def test_a_non_admin_scope_is_rejected(holds_client: tuple[TestClient, str, str]) -> None:
    client, _admin_key, corpus_id = holds_client
    # A read/write member without admin cannot place a hold.
    runtime = client.app.state.runtime
    _, member_key = runtime.issue_api_key(
        runtime.store.find_workspace_by_slug("acme").id, "member2", scopes=("read", "write")
    )
    placed = client.post(f"/v1/corpora/{corpus_id}/legal-holds", headers={"x-api-key": member_key}, json={})
    assert placed.status_code == 403


def test_delete_of_a_held_document_is_409_and_erase_is_200_held(holds_client: tuple[TestClient, str, str]) -> None:
    client, admin_key, corpus_id = holds_client
    headers = {"x-api-key": admin_key}
    doc = _ingest(client, admin_key, corpus_id)
    client.post(f"/v1/corpora/{corpus_id}/legal-holds", headers=headers, json={"document_id": doc})

    deleted = client.delete(f"/v1/corpora/{corpus_id}/documents/{doc}", headers=headers)
    assert deleted.status_code == 409

    erased = client.post(f"/v1/corpora/{corpus_id}/documents/{doc}/erase", headers=headers)
    assert erased.status_code == 200
    assert erased.json()["status"] == "held"


def test_the_body_cannot_set_the_actor(holds_client: tuple[TestClient, str, str]) -> None:
    client, admin_key, corpus_id = holds_client
    headers = {"x-api-key": admin_key}
    doc = _ingest(client, admin_key, corpus_id)
    # placed_by is not a body field, so an attempt to set it is ignored; the row is attributed to the
    # authenticated admin api key.
    placed = client.post(
        f"/v1/corpora/{corpus_id}/legal-holds",
        headers=headers,
        json={"document_id": doc, "placed_by": "mallory"},
    )
    assert placed.status_code == 200
    holds = client.get(f"/v1/corpora/{corpus_id}/legal-holds", headers=headers).json()["holds"]
    assert holds[0]["placed_by"] != "mallory"


def test_default_runtime_place_route_is_409_and_records_nothing(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings).build()
    workspace_id = runtime.create_workspace("acme", "Acme")
    _, admin_key = runtime.issue_api_key(workspace_id, "admin", scopes=("read", "write", "admin"))
    corpus_id = runtime.create_corpus(workspace_id, "handbook")
    try:
        with TestClient(create_app(settings, runtime), raise_server_exceptions=False) as client:
            placed = client.post(f"/v1/corpora/{corpus_id}/legal-holds", headers={"x-api-key": admin_key}, json={})
        assert placed.status_code == 409
        with runtime.store.engine.connect() as conn:
            rows = conn.execute(select(func.count()).select_from(legal_holds)).scalar_one()
        assert rows == 0
    finally:
        runtime.close()


def test_held_re_tag_over_http_records_the_api_key_actor(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings).with_legal_hold_registry().build()
    try:
        workspace_id = runtime.create_workspace("acme", "Acme")
        key_id, admin_key = runtime.issue_api_key(workspace_id, "admin", scopes=("read", "write", "admin"))
        corpus_id = runtime.create_corpus(workspace_id, "handbook")
        headers = {"x-api-key": admin_key}
        with TestClient(create_app(settings, runtime), raise_server_exceptions=False) as client:
            first = client.post(
                f"/v1/corpora/{corpus_id}/documents/text",
                headers=headers,
                json={
                    "source_id": "policy.txt",
                    "title": "Policy",
                    "content": _BODY,
                    "metadata": {"acl": {"classification": "INTERNAL"}},
                },
            )
            assert first.status_code in (200, 201), first.text
            doc = first.json()["document_id"]

            placed = client.post(
                f"/v1/corpora/{corpus_id}/legal-holds",
                headers=headers,
                json={"document_id": doc, "reason": "matter 21"},
            )
            assert placed.status_code == 200, placed.text

            retag = client.post(
                f"/v1/corpora/{corpus_id}/documents/text",
                headers=headers,
                json={
                    "source_id": "policy.txt",
                    "title": "Policy",
                    "content": _BODY,
                    "metadata": {"acl": {"classification": "CONFIDENTIAL"}},
                },
            )
            assert retag.status_code in (200, 201), retag.text

        history = runtime.store.list_hold_permission_history(workspace_id, corpus_id, doc)
        assert len(history) == 1
        assert history[0]["actor"] == key_id
        assert history[0]["old_acl"]["classification"] == "INTERNAL"
        assert history[0]["new_acl"]["classification"] == "CONFIDENTIAL"
    finally:
        runtime.close()
