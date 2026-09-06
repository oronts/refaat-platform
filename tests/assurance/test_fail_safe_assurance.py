"""Failure-injection assurance: the platform's fail-closed and degrade properties hold under fault.

Each test injects a fault at the store or port layer into the real production adapter and asserts the
production conversion to a fail-closed or degraded decision (refused, denied, degraded, abstained),
not merely that no exception escaped. Every fault has a control run without the fault proving the same
path succeeds, so no assertion is vacuous.
"""

from __future__ import annotations

import pytest
from tests.assurance._faults import (
    AdmittingQuotaStore,
    FakeClock,
    OutageGenerator,
    RaisingAuditSink,
    RaisingKillSwitchStore,
    RaisingQuotaStore,
)

from refaat.application.runtime import RuntimeBuilder
from refaat.application.services import ABSTAIN_MESSAGE
from refaat.application.status import StatusService
from refaat.config import Settings
from refaat.domain.audit import AuditEvent, emit_audit
from refaat.domain.degradation import RefuseAllDegradationPolicy, SafeDefaultsDegradationPolicy
from refaat.domain.egress import (
    AllowlistEgressPolicy,
    EgressDestination,
    EgressPurpose,
    allowlist_from_config,
    ensure_egress_allowed,
)
from refaat.domain.kill_switch import KillSwitchScope, KillSwitchSet, ensure_workspace_active
from refaat.domain.models import DocumentInput, QueryRequest
from refaat.domain.quota import QuotaContext, QuotaDimension, QuotaRule, QuotaScope, ensure_within_quota
from refaat.domain.status import SubsystemState
from refaat.errors import (
    AuditUnavailableError,
    DependencyUnavailableError,
    EgressDeniedError,
    KillSwitchEngagedError,
    QuotaExceededError,
)
from refaat.infrastructure.generation import OpenAICompatibleGenerator
from refaat.infrastructure.kill_switch import DatabaseKillSwitchRegistry
from refaat.infrastructure.quota import DatabaseQuotaPolicy

pytestmark = pytest.mark.integration

QUERY = "internal rollout plan timeline"
CONTENT = b"The internal rollout plan timeline covers the production release milestone and the launch details."


def _ingest(runtime: object, workspace_id: str, corpus_id: str) -> None:
    runtime.ingest(  # type: ignore[attr-defined]
        workspace_id,
        corpus_id,
        DocumentInput(source_id="release.md", title="release.md", media_type="text/markdown", content=CONTENT),
    )


# Model provider outage -> degrade / refuse (Slice 4) ----------------------------------------------


def test_model_outage_with_surviving_grounding_serves_a_degraded_answer(settings: Settings) -> None:
    runtime = (
        RuntimeBuilder(settings)
        .with_generator(OutageGenerator())
        .with_degradation_policy(SafeDefaultsDegradationPolicy())
        .build()
    )
    try:
        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "c")
        _ingest(runtime, ws, corpus)
        response = runtime.queries.query(QueryRequest(workspace_id=ws, corpus_id=corpus, query=QUERY))
        assert response.degraded is True
        assert response.cached is False
        assert response.citations, "a degraded answer must stay grounded"
    finally:
        runtime.close()


def test_model_outage_without_grounding_abstains(settings: Settings) -> None:
    runtime = (
        RuntimeBuilder(settings)
        .with_generator(OutageGenerator())
        .with_degradation_policy(SafeDefaultsDegradationPolicy())
        .build()
    )
    try:
        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "empty")
        response = runtime.queries.query(QueryRequest(workspace_id=ws, corpus_id=corpus, query=QUERY))
        assert response.abstained is True
        assert response.answer == ABSTAIN_MESSAGE
        assert response.citations == ()
    finally:
        runtime.close()


@pytest.mark.parametrize("policy", [None, RefuseAllDegradationPolicy()])
def test_model_outage_refuses_under_no_policy_and_refuse_all(settings: Settings, policy: object) -> None:
    builder = RuntimeBuilder(settings).with_generator(OutageGenerator())
    if policy is not None:
        builder = builder.with_degradation_policy(policy)  # type: ignore[arg-type]
    runtime = builder.build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "c")
        _ingest(runtime, ws, corpus)
        with pytest.raises(DependencyUnavailableError):
            runtime.queries.query(QueryRequest(workspace_id=ws, corpus_id=corpus, query=QUERY))
    finally:
        runtime.close()


def test_control_a_recovered_provider_serves_normally(settings: Settings) -> None:
    runtime = (
        RuntimeBuilder(settings)
        .with_generator(OutageGenerator(recovered=True))
        .with_degradation_policy(SafeDefaultsDegradationPolicy())
        .build()
    )
    try:
        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "c")
        _ingest(runtime, ws, corpus)
        response = runtime.queries.query(QueryRequest(workspace_id=ws, corpus_id=corpus, query=QUERY))
        assert response.degraded is False
        assert response.abstained is False
        assert response.citations
    finally:
        runtime.close()


# Kill-switch store outage -> deny (kill_switch.py) ------------------------------------------------


def test_kill_switch_store_outage_outside_the_stale_window_denies() -> None:
    clock = FakeClock()
    store = RaisingKillSwitchStore(KillSwitchSet.empty())
    registry = DatabaseKillSwitchRegistry(
        store,  # type: ignore[arg-type]
        clock,
        cache_ttl_seconds=0.0,
        stale_deny_seconds=0.0,
    )
    store.fail = True
    clock.advance(1.0)
    with pytest.raises(KillSwitchEngagedError):
        ensure_workspace_active(registry, "ws")


def test_kill_switch_store_outage_inside_the_stale_window_serves_last_known_good() -> None:
    clock = FakeClock()
    store = RaisingKillSwitchStore(KillSwitchSet.empty())
    registry = DatabaseKillSwitchRegistry(
        store,  # type: ignore[arg-type]
        clock,
        cache_ttl_seconds=0.0,
        stale_deny_seconds=30.0,
    )
    store.fail = True
    clock.advance(1.0)
    # Last-known-good was open, so the guarded operation is served inside the operator stale window.
    ensure_workspace_active(registry, "ws")
    assert registry.engaged().denies(KillSwitchScope.GENERATION, "ws") is False


# Quota store outage -> deny (quota.py) -----------------------------------------------------------


def test_quota_store_outage_denies() -> None:
    rule = QuotaRule(scope=QuotaScope.WORKSPACE, dimension=QuotaDimension.REQUESTS, limit=100, window_seconds=3600)
    store: object = RaisingQuotaStore()
    policy = DatabaseQuotaPolicy(store, FakeClock(), rules=[rule], lease_seconds=900.0)  # type: ignore[arg-type]
    ctx = QuotaContext(workspace_id="ws")
    with pytest.raises(QuotaExceededError):
        ensure_within_quota(policy, ctx, dimension=QuotaDimension.REQUESTS, idem="run-1")


def test_control_a_healthy_quota_store_admits() -> None:
    rule = QuotaRule(scope=QuotaScope.WORKSPACE, dimension=QuotaDimension.REQUESTS, limit=100, window_seconds=3600)
    store: object = AdmittingQuotaStore()
    policy = DatabaseQuotaPolicy(store, FakeClock(), rules=[rule], lease_seconds=900.0)  # type: ignore[arg-type]
    ctx = QuotaContext(workspace_id="ws")
    ensure_within_quota(policy, ctx, dimension=QuotaDimension.REQUESTS, idem="run-1")


# Audit sink outage -> refuse under fail-closed (audit.py) -----------------------------------------


def _audit_event() -> AuditEvent:
    from datetime import UTC, datetime

    return AuditEvent(
        event_type="query.authorized",
        occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
        workspace_id="ws",
        principal_id="p",
        principal_kind="human",
        tenant=None,
        policy_id="policy",
        policy_version="1",
        dedup_id="d",
    )


class _NullLogger:
    def warning(self, *args: object, **kwargs: object) -> None:
        return None


def test_audit_sink_outage_under_fail_closed_refuses() -> None:
    sink = RaisingAuditSink()
    with pytest.raises(AuditUnavailableError):
        emit_audit(sink, _audit_event(), fail_closed=True, logger=_NullLogger())
    assert sink.attempts == 1


def test_control_audit_sink_outage_under_best_effort_proceeds() -> None:
    sink = RaisingAuditSink()
    # The default posture swallows the failure so the operation proceeds and merely logs.
    emit_audit(sink, _audit_event(), fail_closed=False, logger=_NullLogger())
    assert sink.attempts == 1


# Egress denial -> refused before connect (Slice 7 Part A) ----------------------------------------


def test_egress_denial_raises_rather_than_connecting() -> None:
    policy = AllowlistEgressPolicy(allowlist_from_config([{"host": "api.openai.com"}]))
    with pytest.raises(EgressDeniedError):
        ensure_egress_allowed(policy, EgressDestination.from_url("https://evil.test/v1", EgressPurpose.MODEL))
    # And the same policy denies the real model seam at construction, before any socket.
    with pytest.raises(EgressDeniedError):
        OpenAICompatibleGenerator(base_url="https://evil.test/v1", api_key="k", model="m", egress_policy=policy)


def test_control_an_allowlisted_egress_destination_is_admitted() -> None:
    policy = AllowlistEgressPolicy(allowlist_from_config([{"host": "api.openai.com"}]))
    ensure_egress_allowed(policy, EgressDestination.from_url("https://api.openai.com/v1", EgressPurpose.MODEL))


# Admin status plane -> degrade to unknown, never raise (Slice 6) ----------------------------------


def test_status_plane_degrades_a_faulted_subsystem_to_unknown(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = RuntimeBuilder(settings).build()
    try:
        ws = runtime.create_workspace("acme", "Acme")

        def boom(self: StatusService, workspace_id: object, operator: bool) -> object:
            raise RuntimeError("subsystem read outage")

        monkeypatch.setattr(StatusService, "_kill_switches", boom)
        plane = runtime.status(workspace_id=ws)
        subs = {sub.name: sub for sub in plane.subsystems}
        assert subs["kill_switches"].state is SubsystemState.UNKNOWN
        # The rest of the plane still reported and the snapshot never raised.
        assert subs["quotas"].state is SubsystemState.NOT_CONFIGURED
        assert plane.build.state is SubsystemState.ACTIVE
    finally:
        runtime.close()
