"""The read-only derivation-manifest HTTP surface: list, get, verify, compare, and the disabled 404."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from refaat.api.app import create_app
from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.config import Settings
from refaat.domain.models import DocumentInput

pytestmark = pytest.mark.e2e

_BODY = b"The production release checklist requires reviewer approval before deployment."


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {
        "environment": "test",
        "database_url": f"sqlite:///{tmp_path / 'refaat.db'}",
        "auto_create_schema": True,
        "blob_root": tmp_path / "blobs",
        "connector_file_root": tmp_path / "connectors",
        "skill_root": tmp_path / "skills",
        "api_key_pepper": "test-pepper-with-enough-entropy",
        "bootstrap_token": "bootstrap-test-token",
        "trusted_hosts": ["testserver", "localhost", "127.0.0.1"],
        "rate_limit_requests": 10_000,
        "log_json": False,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _serve(settings: Settings, *, ingest: bool) -> tuple[Refaat, str, dict[str, str]]:
    runtime = RuntimeBuilder(settings).build()
    ws = runtime.create_workspace("acme", "Acme")
    corpus = runtime.create_corpus(ws, "handbook")
    _, raw = runtime.issue_api_key(ws, "admin", scopes=("read", "write", "admin"))
    if ingest:
        runtime.ingest(
            ws,
            corpus,
            DocumentInput(source_id="a", title="a", media_type="text/plain", content=_BODY, metadata={}),
        )
    return runtime, corpus, {"Authorization": f"Bearer {raw}"}


def test_derivation_routes_list_get_verify_compare(tmp_path: Path) -> None:
    settings = _settings(tmp_path, derivation_manifest_enabled=True)
    runtime, corpus, headers = _serve(settings, ingest=True)
    try:
        with TestClient(create_app(settings, runtime), raise_server_exceptions=False) as client:
            listed = client.get(f"/v1/corpora/{corpus}/derivations", headers=headers)
            assert listed.status_code == 200
            manifests = listed.json()
            assert len(manifests) == 1
            key = manifests[0]["derivation_key"]

            got = client.get(f"/v1/corpora/{corpus}/derivations/{key}", headers=headers)
            assert got.status_code == 200
            assert got.json()["derivation_key"] == key
            assert client.get(f"/v1/corpora/{corpus}/derivations/not-a-key", headers=headers).status_code == 404

            # verify and compare must not be shadowed by the /{derivation_key} catch-all route.
            verified = client.get(f"/v1/corpora/{corpus}/derivations/verify", headers=headers)
            assert verified.status_code == 200
            assert verified.json()["status"] == "match"

            compared = client.get(
                f"/v1/corpora/{corpus}/derivations/compare", params={"left": key, "right": key}, headers=headers
            )
            assert compared.status_code == 200
            assert compared.json()["identical"] is True
    finally:
        runtime.close()


def test_derivation_routes_404_when_disabled(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    runtime, corpus, headers = _serve(settings, ingest=False)
    try:
        with TestClient(create_app(settings, runtime), raise_server_exceptions=False) as client:
            assert client.get(f"/v1/corpora/{corpus}/derivations", headers=headers).status_code == 404
            assert client.get(f"/v1/corpora/{corpus}/derivations/verify", headers=headers).status_code == 404
            assert (
                client.get(f"/v1/corpora/{corpus}/documents/any/derivation/verify", headers=headers).status_code == 404
            )
    finally:
        runtime.close()


def test_document_derivation_verify_route(tmp_path: Path) -> None:
    settings = _settings(tmp_path, derivation_manifest_enabled=True)
    runtime = RuntimeBuilder(settings).build()
    ws = runtime.create_workspace("acme", "Acme")
    corpus = runtime.create_corpus(ws, "handbook")
    _, admin = runtime.issue_api_key(ws, "admin", scopes=("read", "write", "admin"))
    _, reader = runtime.issue_api_key(ws, "reader", scopes=("read",))
    doc = runtime.ingest(
        ws,
        corpus,
        DocumentInput(source_id="a", title="a", media_type="text/plain", content=_BODY, metadata={}),
    ).document_id
    try:
        with TestClient(create_app(settings, runtime), raise_server_exceptions=False) as client:
            path = f"/v1/corpora/{corpus}/documents/{doc}/derivation/verify"
            ok = client.get(path, headers={"Authorization": f"Bearer {admin}"})
            assert ok.status_code == 200
            assert ok.json() == {"document_id": doc, "status": "match"}
            # verify recomputes an O(document) fold, so it is admin-scoped: a read principal is denied.
            assert client.get(path, headers={"Authorization": f"Bearer {reader}"}).status_code == 403
    finally:
        runtime.close()
