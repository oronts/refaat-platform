"""Identity-freshness decision matrix.

These prove the fail-closed re-resolution rules of the pure decision core. The module is inert,
so there is no enforcement-path suite yet; wiring it into the worker and approval service is a
separate slice.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, tzinfo

from refaat.domain.authz import CanonicalAccessPolicy, Principal, auth_scope_hash
from refaat.domain.freshness import (
    DEFAULT_FRESHNESS_POLICY,
    FreshnessPolicy,
    IdentitySnapshot,
    RiskClass,
    evaluate_reresolution,
    snapshot_from_payload,
    snapshot_from_principal,
    snapshot_to_payload,
)

NOW = datetime(2026, 8, 21, 12, 0, 0, tzinfo=UTC)
POLICY = CanonicalAccessPolicy()


class _OffsetlessTZ(tzinfo):
    """A tzinfo whose utcoffset is None, so an attached datetime is aware yet offsetless."""

    def utcoffset(self, dt: datetime | None) -> timedelta | None:
        return None

    def tzname(self, dt: datetime | None) -> str:
        return "OFFSETLESS"

    def dst(self, dt: datetime | None) -> timedelta | None:
        return None


ALICE = Principal(id="alice", tenant="merck", groups=frozenset({"research-a"}), roles=frozenset({"scientist"}))


def _snapshot(captured_at: datetime = NOW, *, tenant: str | None = "merck", scope: str = "scope-a") -> IdentitySnapshot:
    return IdentitySnapshot(
        principal_id="alice",
        tenant=tenant,
        auth_scope_hash=scope,
        policy_id="canonical",
        policy_version="1",
        captured_at=captured_at,
    )


def test_max_age_for_returns_configured_and_fails_closed_to_zero() -> None:
    assert DEFAULT_FRESHNESS_POLICY.max_age_for(RiskClass.READ) == 86_400
    assert DEFAULT_FRESHNESS_POLICY.max_age_for(RiskClass.IRREVERSIBLE) == 0
    partial = FreshnessPolicy(max_age_seconds={RiskClass.READ: 500}, require_approval=frozenset())
    # An unconfigured class gets no window, so any positive age denies.
    assert partial.max_age_for(RiskClass.SENSITIVE_WRITE) == 0
    assert partial.max_age_for(RiskClass.IRREVERSIBLE) == 0
    empty = FreshnessPolicy(max_age_seconds={}, require_approval=frozenset())
    assert empty.max_age_for(RiskClass.READ) == 0


def test_a_partial_policy_denies_a_stale_omitted_high_risk_class() -> None:
    partial = FreshnessPolicy(
        max_age_seconds={RiskClass.READ: 500}, require_approval=frozenset({RiskClass.IRREVERSIBLE})
    )
    snap = _snapshot(NOW - timedelta(seconds=1))
    decision = evaluate_reresolution(snap, ALICE, "scope-a", RiskClass.IRREVERSIBLE, partial, NOW)
    assert decision.allowed is False
    assert decision.reason == "snapshot_too_old"
    assert decision.require_approval is True


def test_an_indeterminate_current_scope_denies() -> None:
    snap = _snapshot(NOW - timedelta(seconds=10))
    decision = evaluate_reresolution(snap, ALICE, None, RiskClass.WRITE, DEFAULT_FRESHNESS_POLICY, NOW)
    assert decision.allowed is False
    assert decision.reason == "scope_unknown"


def test_fresh_within_max_age_allows() -> None:
    snap = _snapshot(NOW - timedelta(seconds=60))
    decision = evaluate_reresolution(snap, ALICE, "scope-a", RiskClass.WRITE, DEFAULT_FRESHNESS_POLICY, NOW)
    assert decision.allowed is True
    assert decision.reason == "fresh"
    assert decision.require_approval is False


def test_snapshot_too_old_denies_at_each_risk_class_boundary() -> None:
    for risk_class in (RiskClass.READ, RiskClass.WRITE, RiskClass.SENSITIVE_WRITE):
        max_age = DEFAULT_FRESHNESS_POLICY.max_age_for(risk_class)
        at_boundary = _snapshot(NOW - timedelta(seconds=max_age))
        ok = evaluate_reresolution(at_boundary, ALICE, "scope-a", risk_class, DEFAULT_FRESHNESS_POLICY, NOW)
        assert ok.allowed is True, risk_class
        assert ok.reason == "fresh"
        just_over = _snapshot(NOW - timedelta(seconds=max_age + 1))
        stale = evaluate_reresolution(just_over, ALICE, "scope-a", risk_class, DEFAULT_FRESHNESS_POLICY, NOW)
        assert stale.allowed is False, risk_class
        assert stale.reason == "snapshot_too_old"


def test_identity_revoked_when_refreshed_is_none() -> None:
    snap = _snapshot(NOW - timedelta(seconds=10))
    decision = evaluate_reresolution(snap, None, "scope-a", RiskClass.WRITE, DEFAULT_FRESHNESS_POLICY, NOW)
    assert decision.allowed is False
    assert decision.reason == "identity_revoked"


def test_tenant_changed_denies_on_mismatch() -> None:
    snap = _snapshot(NOW - timedelta(seconds=10), tenant="merck")
    moved = Principal(id="alice", tenant="pfizer")
    decision = evaluate_reresolution(snap, moved, "scope-a", RiskClass.WRITE, DEFAULT_FRESHNESS_POLICY, NOW)
    assert decision.allowed is False
    assert decision.reason == "tenant_changed"


def test_principal_mismatch_denies() -> None:
    snap = _snapshot(NOW - timedelta(seconds=10))
    other = Principal(id="mallory", tenant="merck")
    decision = evaluate_reresolution(snap, other, "scope-a", RiskClass.WRITE, DEFAULT_FRESHNESS_POLICY, NOW)
    assert decision.allowed is False
    assert decision.reason == "principal_mismatch"


def test_invalid_snapshot_time_for_naive_offsetless_and_future() -> None:
    naive = _snapshot(datetime(2026, 8, 21, 11, 0, 0))
    naive_decision = evaluate_reresolution(naive, ALICE, "scope-a", RiskClass.READ, DEFAULT_FRESHNESS_POLICY, NOW)
    assert naive_decision.reason == "invalid_snapshot_time"

    offsetless = _snapshot(datetime(2026, 8, 21, 11, 0, 0, tzinfo=_OffsetlessTZ()))
    offsetless_decision = evaluate_reresolution(
        offsetless, ALICE, "scope-a", RiskClass.READ, DEFAULT_FRESHNESS_POLICY, NOW
    )
    assert offsetless_decision.reason == "invalid_snapshot_time"

    future = _snapshot(NOW + timedelta(seconds=1))
    future_decision = evaluate_reresolution(future, ALICE, "scope-a", RiskClass.READ, DEFAULT_FRESHNESS_POLICY, NOW)
    assert future_decision.allowed is False
    assert future_decision.reason == "invalid_snapshot_time"


def test_invalid_now_for_a_naive_clock() -> None:
    snap = _snapshot(NOW - timedelta(seconds=10))
    naive_now = datetime(2026, 8, 21, 12, 0, 0)
    decision = evaluate_reresolution(snap, ALICE, "scope-a", RiskClass.READ, DEFAULT_FRESHNESS_POLICY, naive_now)
    assert decision.allowed is False
    assert decision.reason == "invalid_now"


def test_no_snapshot_denies() -> None:
    decision = evaluate_reresolution(None, ALICE, "scope-a", RiskClass.READ, DEFAULT_FRESHNESS_POLICY, NOW)
    assert decision.allowed is False
    assert decision.reason == "no_snapshot"


def test_scope_changed_still_allows() -> None:
    snap = _snapshot(NOW - timedelta(seconds=10), scope="scope-a")
    decision = evaluate_reresolution(snap, ALICE, "scope-b", RiskClass.WRITE, DEFAULT_FRESHNESS_POLICY, NOW)
    assert decision.allowed is True
    assert decision.reason == "scope_changed"
    assert decision.require_approval is False


def test_irreversible_with_positive_age_denies_and_requires_approval() -> None:
    snap = _snapshot(NOW - timedelta(seconds=1))
    decision = evaluate_reresolution(snap, ALICE, "scope-a", RiskClass.IRREVERSIBLE, DEFAULT_FRESHNESS_POLICY, NOW)
    assert decision.allowed is False
    assert decision.reason == "snapshot_too_old"
    assert decision.require_approval is True


def test_irreversible_at_zero_age_allows_and_requires_approval() -> None:
    snap = _snapshot(NOW)
    decision = evaluate_reresolution(snap, ALICE, "scope-a", RiskClass.IRREVERSIBLE, DEFAULT_FRESHNESS_POLICY, NOW)
    assert decision.allowed is True
    assert decision.reason == "fresh"
    assert decision.require_approval is True


def test_snapshot_from_principal_captures_redacted_scope() -> None:
    snap = snapshot_from_principal(ALICE, POLICY, NOW)
    assert snap.principal_id == "alice"
    assert snap.tenant == "merck"
    assert snap.auth_scope_hash == auth_scope_hash(ALICE, POLICY)
    assert snap.policy_id == "canonical"
    assert snap.policy_version == "1"
    assert snap.captured_at == NOW


def test_payload_round_trips_with_and_without_tenant() -> None:
    for tenant in ("merck", None):
        snap = _snapshot(NOW, tenant=tenant)
        restored = snapshot_from_payload(snapshot_to_payload(snap))
        assert restored == snap


def test_payload_missing_or_corrupt_field_parses_to_none() -> None:
    good = snapshot_to_payload(_snapshot(NOW))
    assert snapshot_from_payload(good) is not None

    for missing in ("principal_id", "tenant", "auth_scope_hash", "policy_id", "policy_version", "captured_at"):
        corrupt = dict(good)
        del corrupt[missing]
        assert snapshot_from_payload(corrupt) is None, missing

    wrong_type = dict(good)
    wrong_type["principal_id"] = 123
    assert snapshot_from_payload(wrong_type) is None

    naive_time = dict(good)
    naive_time["captured_at"] = "2026-08-21T11:00:00"
    assert snapshot_from_payload(naive_time) is None

    unparseable = dict(good)
    unparseable["captured_at"] = "not-a-timestamp"
    assert snapshot_from_payload(unparseable) is None

    assert snapshot_from_payload("not-a-mapping") is None  # type: ignore[arg-type]
