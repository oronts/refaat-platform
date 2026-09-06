"""The kill-switch HTTP surface and the freeze darkening it.

An admin engages a workspace freeze over HTTP; the whole HTTP surface for that workspace then returns
503, including the disengage route, so a workspace credential cannot lift its own freeze. Only the
operator path (the management service, mirrored by the CLI) lifts it, after which HTTP serves again.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from refaat.api.app import create_app
from refaat.application.kill_switch import KillSwitchService
from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.config import Settings
from refaat.domain.kill_switch import KillSwitchScope

pytestmark = pytest.mark.e2e


def _settings(tmp_path: Path, *, backend: str = "database") -> Settings:
    return Settings(  # type: ignore[arg-type]
        environment="test",
        database_url=f"sqlite:///{tmp_path / 'refaat.db'}",
        auto_create_schema=True,
        blob_root=tmp_path / "blobs",
        connector_file_root=tmp_path / "connectors",
        skill_root=tmp_path / "skills",
        api_key_pepper="test-pepper-with-enough-entropy",
        bootstrap_token="bootstrap-test-token",
        trusted_hosts=["testserver", "localhost", "127.0.0.1"],
        rate_limit_requests=10_000,
        kill_switch_backend=backend,
        kill_switch_cache_ttl_seconds=0.0,
        log_json=False,
    )


@pytest.fixture
def kill_switch_client(tmp_path: Path) -> Iterator[tuple[TestClient, str, Refaat]]:
    runtime = RuntimeBuilder(_settings(tmp_path)).build()
    ws = runtime.create_workspace("acme", "Acme")
    _, api_key = runtime.issue_api_key(ws, "admin", scopes=("read", "write", "admin"))
    with TestClient(create_app(runtime.settings, runtime), raise_server_exceptions=False) as client:
        yield client, api_key, runtime
    runtime.close()


def test_freeze_engaged_over_http_darkens_the_surface_and_only_the_operator_lifts(
    kill_switch_client: tuple[TestClient, str, Refaat],
) -> None:
    client, api_key, runtime = kill_switch_client
    headers = {"x-api-key": api_key}

    assert client.get("/v1/kill-switches", headers=headers).status_code == 200

    engaged = client.post("/v1/kill-switches", headers=headers, json={"scope": "workspace_freeze"})
    assert engaged.status_code == 201
    assert engaged.json()["switch_id"]

    # The whole HTTP surface for this workspace is now dark, including the disengage route.
    assert client.get("/v1/kill-switches", headers=headers).status_code == 503
    assert client.post("/v1/kill-switches/workspace_freeze/disengage", headers=headers).status_code == 503

    # Only the operator path lifts a freeze; then HTTP serves again.
    ws = runtime.store.find_workspace_by_slug("acme")
    assert ws is not None
    lifted = KillSwitchService(runtime.store, ws.id).disengage(
        scope=KillSwitchScope.WORKSPACE_FREEZE, disengaged_by="operator"
    )
    assert lifted is True
    assert client.get("/v1/kill-switches", headers=headers).status_code == 200


def test_writes_engaged_over_http_denies_harness_routes(
    kill_switch_client: tuple[TestClient, str, Refaat],
) -> None:
    # A writes switch does not darken the HTTP edge (only a freeze does), so the request reaches the
    # route and the service guard must deny it. This locks the route-to-service delegation at the
    # boundary, so a later revert of a harness route to a direct store call fails the gate.
    client, api_key, _runtime = kill_switch_client
    headers = {"x-api-key": api_key}
    created = client.post(
        "/v1/harness/snapshots", headers=headers, json={"supplemental": {}, "evidence": {"reason": "seed"}}
    )
    assert created.status_code == 201
    snapshot_id = created.json()["id"]
    assert client.post("/v1/kill-switches", headers=headers, json={"scope": "writes"}).status_code == 201
    denied_create = client.post(
        "/v1/harness/snapshots", headers=headers, json={"supplemental": {}, "evidence": {"reason": "x"}}
    )
    assert denied_create.status_code == 503
    assert client.post(f"/v1/harness/snapshots/{snapshot_id}/activate", headers=headers).status_code == 503


def test_unknown_scope_is_rejected(kill_switch_client: tuple[TestClient, str, Refaat]) -> None:
    client, api_key, _runtime = kill_switch_client
    response = client.post("/v1/kill-switches", headers={"x-api-key": api_key}, json={"scope": "not_a_real_scope"})
    assert response.status_code == 422


def test_management_surface_is_unavailable_when_backend_off(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_settings(tmp_path, backend="none")).build()
    ws = runtime.create_workspace("acme", "Acme")
    _, api_key = runtime.issue_api_key(ws, "admin", scopes=("read", "write", "admin"))
    try:
        with TestClient(create_app(runtime.settings, runtime), raise_server_exceptions=False) as client:
            headers = {"x-api-key": api_key}
            assert runtime.kill_switches is None
            # No management route reads the table when the feature is off; each is a uniform 404.
            assert client.get("/v1/kill-switches", headers=headers).status_code == 404
            engage = client.post("/v1/kill-switches", headers=headers, json={"scope": "workspace_freeze"})
            assert engage.status_code == 404
    finally:
        runtime.close()
