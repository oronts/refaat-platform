"""The unified admin status plane: presence, health, degrade-to-unknown, redaction, and HTTP auth.

The plane is additive and read-only, so an unconfigured runtime reports every opt-in subsystem
NOT_CONFIGURED and the pinned artifact id and default context hash recompute unchanged. A configured
subsystem reports its real posture; a subsystem whose accessor raises degrades to UNKNOWN while the
rest still report; and no serialized status ever carries a secret, reason, raw principal, or a raw
quota scope key.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import insert

from refaat.api.app import create_app
from refaat.application.legal_hold import LegalHoldService
from refaat.application.runtime import RuntimeBuilder
from refaat.application.services import QueryService
from refaat.config import Settings
from refaat.core.retrieval import RetrievalPolicy
from refaat.domain.kill_switch import KillSwitchScope
from refaat.domain.status import SubsystemState
from refaat.infrastructure.database.schema import jobs, quota_counters, tool_approvals
from refaat.programs.program import GroundedRagProgram

pytestmark = pytest.mark.integration

PINNED_GROUNDED_RAG_ID = "sha256:b76d73f8d956bcd38ccc9d4c24372b07b387647a189f3a9e300cded186febbb4"
PINNED_CONTEXT_HASH = "7937b529c3dadd9ada1e5483bac13fcef58f71fcee4aa31da688b7d137df034c"


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


def _default_context_hash() -> str:
    policy = RetrievalPolicy(
        top_k=8,
        candidate_limit=32,
        token_budget=600,
        semantic_weight=0.55,
        lexical_weight=0.30,
        graph_weight=0.15,
        fusion="rrf",
        rrf_k=60,
        autocut_enabled=False,
        autocut_threshold=0.5,
        metadata_filter=None,
        bm25_k1=1.2,
        bm25_b=0.75,
        query_expansion=True,
        max_query_variants=4,
        compression_enabled=False,
        compression_min_salience=0.0,
        expansion_mode="flat",
        window_radius=1,
    )
    return QueryService._context_hash(
        policy=policy,
        profile={},
        supplemental={},
        session_summary=None,
        embedding_model="emb",
        generation_model="gen",
        reranker="none",
        query_transform="none",
        guidance=None,
        response_schema=None,
    )


def _by_name(plane: object) -> dict[str, object]:
    return {sub.name: sub for sub in plane.subsystems}  # type: ignore[attr-defined]


def test_unconfigured_all_not_configured_and_byte_identical(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_settings(tmp_path)).build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        plane = runtime.status(workspace_id=ws)
        subs = _by_name(plane)
        for name in (
            "kill_switches",
            "quotas",
            "audit",
            "degradation",
            "model_routing",
            "output_governance",
            "egress",
            "source_trust",
            "identity",
            "plugin_pins",
            "secrets",
        ):
            assert subs[name].state is SubsystemState.NOT_CONFIGURED, name  # type: ignore[attr-defined]
        assert subs["legal_holds"].detail["active_hold_count"] == 0  # type: ignore[attr-defined,index]
        assert plane.schema.state is SubsystemState.UNMANAGED

        assert GroundedRagProgram().artifact().artifact_id == PINNED_GROUNDED_RAG_ID
        assert _default_context_hash() == PINNED_CONTEXT_HASH
    finally:
        runtime.close()


def test_configured_subsystems_report_their_posture(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        kill_switch_backend="database",
        kill_switch_cache_ttl_seconds=0.0,
        quota_backend="database",
        quota_rules=[{"scope": "workspace", "dimension": "requests", "limit": 1000, "window_seconds": 3600}],
        degradation_policy="safe_defaults",
    )
    runtime = RuntimeBuilder(settings).build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        from refaat.application.kill_switch import KillSwitchService

        KillSwitchService(runtime.store, ws).engage(scope=KillSwitchScope.WORKSPACE_FREEZE, engaged_by="op")

        plane = runtime.status(workspace_id=ws, operator=True)
        subs = _by_name(plane)
        assert subs["kill_switches"].state is SubsystemState.ENGAGED  # type: ignore[attr-defined]
        assert subs["kill_switches"].detail["workspace_frozen"] is True  # type: ignore[attr-defined,index]
        assert subs["quotas"].configured is True  # type: ignore[attr-defined]
        assert subs["quotas"].detail["rule_count"] == 1  # type: ignore[attr-defined,index]
        assert subs["degradation"].detail["policy_id"] == "safe-defaults"  # type: ignore[attr-defined,index]
    finally:
        runtime.close()


def test_subsystem_read_failure_degrades_to_unknown_without_failing_the_plane(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_settings(tmp_path, kill_switch_backend="database")).build()
    try:
        ws = runtime.create_workspace("acme", "Acme")

        class _BrokenRegistry:
            def engaged(self) -> object:
                raise RuntimeError("registry down")

        runtime.kill_switches = _BrokenRegistry()  # type: ignore[assignment]
        plane = runtime.status(workspace_id=ws)
        subs = _by_name(plane)
        assert subs["kill_switches"].state is SubsystemState.UNKNOWN  # type: ignore[attr-defined]
        assert subs["kill_switches"].detail == {}  # type: ignore[attr-defined]
        assert subs["legal_holds"].state is SubsystemState.ACTIVE  # type: ignore[attr-defined]
        assert plane.build.state is SubsystemState.ACTIVE
    finally:
        runtime.close()


def test_configured_but_unreadable_reports_degraded_not_false_active(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        quota_backend="database",
        quota_rules=[{"scope": "workspace", "dimension": "requests", "limit": 1000, "window_seconds": 3600}],
    )
    runtime = RuntimeBuilder(settings).build()
    try:
        ws = runtime.create_workspace("acme", "Acme")

        def _raise(_ws: str) -> list[dict[str, object]]:
            raise RuntimeError("db down")

        runtime.store.quota_usage_for_workspace = _raise  # type: ignore[method-assign]
        plane = runtime.status(workspace_id=ws)
        quotas = _by_name(plane)["quotas"]
        assert quotas.configured is True  # type: ignore[attr-defined]
        assert quotas.state is SubsystemState.DEGRADED  # type: ignore[attr-defined]
    finally:
        runtime.close()


def test_status_redacts_secrets_reasons_principals_and_scope_keys(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        kill_switch_backend="database",
        kill_switch_cache_ttl_seconds=0.0,
        quota_backend="database",
        quota_rules=[{"scope": "workspace", "dimension": "requests", "limit": 1000, "window_seconds": 3600}],
    )
    runtime = RuntimeBuilder(settings).with_legal_hold_registry().build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "handbook")
        from refaat.application.kill_switch import KillSwitchService

        KillSwitchService(runtime.store, ws).engage(
            scope=KillSwitchScope.WORKSPACE_FREEZE,
            engaged_by="ENGAGED-BY-SECRET-9f",
            reason="KILL-REASON-SECRET-XZ",
        )
        LegalHoldService(runtime.store, ws).place(
            corpus_id=corpus,
            document_id="doc-1",
            placed_by="PLACED-BY-SECRET-7a",
            reason="HOLD-REASON-SECRET-QP",
        )
        now = datetime.now(UTC)
        with runtime.store.engine.begin() as conn:
            conn.execute(
                insert(quota_counters).values(
                    id="qc-secret-1",
                    workspace_id=ws,
                    scope="principal",
                    scope_key="PRINCIPAL-SCOPEKEY-SECRET-abc",
                    dimension="generation_tokens",
                    window_start=0,
                    window_seconds=3600,
                    spent=5,
                    reserved=0,
                    updated_at=now,
                )
            )
            conn.execute(
                insert(tool_approvals).values(
                    id="ta-secret-1",
                    workspace_id=ws,
                    fingerprint="a" * 64,
                    capability_name="cap.secret",
                    approver_id="APPROVER-ID-SECRET-33",
                    approver_roles=["role"],
                    granted_at=now,
                    status="granted",
                )
            )

        import json

        blob = json.dumps(runtime.status(workspace_id=ws, operator=True).to_json())
        for secret in (
            "ENGAGED-BY-SECRET-9f",
            "KILL-REASON-SECRET-XZ",
            "PLACED-BY-SECRET-7a",
            "HOLD-REASON-SECRET-QP",
            "PRINCIPAL-SCOPEKEY-SECRET-abc",
            "APPROVER-ID-SECRET-33",
        ):
            assert secret not in blob
        subs = _by_name(runtime.status(workspace_id=ws, operator=True))
        assert subs["legal_holds"].detail["active_hold_count"] == 1  # type: ignore[attr-defined,index]
        assert subs["approvals"].detail["active_grant_count"] == 1  # type: ignore[attr-defined,index]
    finally:
        runtime.close()


def test_stuck_job_lease_reports_degraded_not_active(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_settings(tmp_path)).build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        now = datetime.now(UTC)
        with runtime.store.engine.begin() as conn:
            conn.execute(
                insert(jobs).values(
                    id="job-stuck-1",
                    workspace_id=ws,
                    kind="cache.purge",
                    payload={},
                    status="running",
                    attempts=1,
                    max_attempts=3,
                    run_after=now,
                    lease_owner="STUCK-OWNER-SECRET",
                    leased_until=now.replace(year=now.year - 1),
                    heartbeat_at=now.replace(year=now.year - 1),
                    created_at=now,
                    updated_at=now,
                )
            )
        subs = _by_name(runtime.status(workspace_id=ws))
        jobs_status = subs["jobs"]
        assert jobs_status.state is SubsystemState.DEGRADED  # type: ignore[attr-defined]
        assert jobs_status.detail["stuck_lease"] == 1  # type: ignore[attr-defined,index]
        import json

        assert "STUCK-OWNER-SECRET" not in json.dumps(jobs_status.to_json())  # type: ignore[attr-defined]
    finally:
        runtime.close()


@pytest.fixture
def status_client(tmp_path: Path) -> Iterator[tuple[TestClient, str, str]]:
    runtime = RuntimeBuilder(_settings(tmp_path)).build()
    ws = runtime.create_workspace("acme", "Acme")
    _, admin_key = runtime.issue_api_key(ws, "admin", scopes=("read", "write", "admin"))
    _, read_key = runtime.issue_api_key(ws, "reader", scopes=("read",))
    with TestClient(create_app(runtime.settings, runtime), raise_server_exceptions=False) as client:
        yield client, admin_key, read_key
    runtime.close()


def test_http_denies_non_admin_and_serves_admin(status_client: tuple[TestClient, str, str]) -> None:
    client, admin_key, read_key = status_client
    assert client.get("/v1/status", headers={"x-api-key": read_key}).status_code == 403
    served = client.get("/v1/status", headers={"x-api-key": admin_key})
    assert served.status_code == 200
    body = served.json()
    assert body["scope"] == "workspace"
    assert "schema" in body
    names = {sub["name"] for sub in body["subsystems"]}
    assert "kill_switches" in names
    assert "audit" in names
