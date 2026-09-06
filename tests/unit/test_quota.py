"""The pure quota decision core and the atomic counter store.

scope_key derives a workspace-embedded key per scope, denies a principal-bearing seam with no
principal and an unvalidated corpus, and skips a rule that does not apply. The store reserves a
ceiling atomically, reconciles it down on commit, restores headroom on release, counts requests,
reaps a leaked reservation by charging its ceiling, and garbage-collects only fully-settled windows.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from refaat.domain.quota import (
    FailClosedQuotaPolicy,
    NullQuotaPolicy,
    QuotaContext,
    QuotaDimension,
    QuotaKey,
    QuotaRule,
    QuotaScope,
    ensure_within_quota,
    quota_rules_from_config,
    resolve_keys,
    scope_key,
    window_start,
)
from refaat.errors import ConfigurationError, QuotaExceededError
from refaat.infrastructure.database.store import SQLStore

_US = "\x1f"


class _Clock:
    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 1, 1, tzinfo=UTC)

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)


def _ctx(**kwargs: object) -> QuotaContext:
    base: dict[str, object] = {"workspace_id": "ws1"}
    base.update(kwargs)
    return QuotaContext(**base)  # type: ignore[arg-type]


def _key(
    *,
    scope: QuotaScope = QuotaScope.WORKSPACE,
    scope_key_value: str = "ws1",
    dimension: QuotaDimension = QuotaDimension.GENERATION_TOKENS,
    window: int = 0,
    limit: int = 100,
    workspace_id: str = "ws1",
) -> QuotaKey:
    return QuotaKey(
        scope=scope,
        scope_key=scope_key_value,
        dimension=dimension,
        window_start=window,
        window_seconds=60,
        limit=limit,
        workspace_id=workspace_id,
    )


def _store(tmp_path: Path) -> SQLStore:
    store = SQLStore(f"sqlite:///{tmp_path / 'quota.db'}", _Clock())
    store.initialize()
    return store


# --- pure decision core -------------------------------------------------------------------------


def test_scope_key_workspace_uses_the_bare_workspace() -> None:
    rule = QuotaRule(QuotaScope.WORKSPACE, QuotaDimension.REQUESTS, 10, 60)
    assert scope_key(rule, _ctx()) == ("ws1", "ok")


def test_scope_key_principal_is_workspace_prefixed_so_it_never_collides_cross_tenant() -> None:
    rule = QuotaRule(QuotaScope.PRINCIPAL, QuotaDimension.REQUESTS, 10, 60)
    a, code_a = scope_key(rule, _ctx(workspace_id="wsA", principal_id="alice"))
    b, code_b = scope_key(rule, _ctx(workspace_id="wsB", principal_id="alice"))
    assert code_a == "ok"
    assert code_b == "ok"
    assert a == f"wsA{_US}alice"
    assert b == f"wsB{_US}alice"
    assert a != b


def test_scope_key_missing_principal_at_a_principal_bearing_seam_denies() -> None:
    rule = QuotaRule(QuotaScope.PRINCIPAL, QuotaDimension.REQUESTS, 10, 60)
    key, code = scope_key(rule, _ctx(principal_id=None, principal_bearing=True))
    assert key is None
    assert code == "missing_scope"


def test_scope_key_missing_principal_at_a_non_bearing_seam_is_a_skip() -> None:
    rule = QuotaRule(QuotaScope.PRINCIPAL, QuotaDimension.REQUESTS, 10, 60)
    key, code = scope_key(rule, _ctx(principal_id=None, principal_bearing=False))
    assert key is None
    assert code == "not_applicable"


def test_scope_key_unvalidated_corpus_denies_so_a_client_id_cannot_mint_a_row() -> None:
    rule = QuotaRule(QuotaScope.CORPUS, QuotaDimension.REQUESTS, 10, 60)
    key, code = scope_key(rule, _ctx(corpus_id="c1", corpus_validated=False))
    assert key is None
    assert code == "missing_scope"
    key2, code2 = scope_key(rule, _ctx(corpus_id="c1", corpus_validated=True))
    assert key2 == f"ws1{_US}c1"
    assert code2 == "ok"


def test_scope_key_empty_workspace_is_a_valid_global_key_not_a_skip() -> None:
    rule = QuotaRule(QuotaScope.WORKSPACE, QuotaDimension.REQUESTS, 10, 60)
    key, code = scope_key(rule, _ctx(workspace_id=""))
    assert key == ""
    assert code == "ok"


def test_window_start_aligns_and_resets_on_a_boundary() -> None:
    assert window_start(125, 60) == 120
    assert window_start(179, 60) == 120
    assert window_start(180, 60) == 180


def test_resolve_keys_denies_on_a_single_missing_scope() -> None:
    rules = [
        QuotaRule(QuotaScope.WORKSPACE, QuotaDimension.REQUESTS, 10, 60),
        QuotaRule(QuotaScope.PRINCIPAL, QuotaDimension.REQUESTS, 5, 60),
    ]
    keys, code = resolve_keys(rules, _ctx(principal_id=None), QuotaDimension.REQUESTS, 0)
    assert code == "missing_scope"
    assert keys == ()


def test_resolve_keys_skips_other_dimensions_and_binds_the_window() -> None:
    rules = [
        QuotaRule(QuotaScope.WORKSPACE, QuotaDimension.REQUESTS, 10, 60),
        QuotaRule(QuotaScope.WORKSPACE, QuotaDimension.GENERATION_TOKENS, 1000, 60),
    ]
    keys, code = resolve_keys(rules, _ctx(), QuotaDimension.REQUESTS, 125)
    assert code == "ok"
    assert len(keys) == 1
    assert keys[0].dimension is QuotaDimension.REQUESTS
    assert keys[0].window_start == 120


def test_ensure_within_quota_none_policy_is_a_no_op() -> None:
    ensure_within_quota(None, _ctx(), dimension=QuotaDimension.REQUESTS, idem="x")
    ensure_within_quota(NullQuotaPolicy(), None, dimension=QuotaDimension.REQUESTS, idem="x")


def test_null_policy_admits_and_fail_closed_policy_denies() -> None:
    ensure_within_quota(NullQuotaPolicy(), _ctx(), dimension=QuotaDimension.REQUESTS, idem="x")
    with pytest.raises(QuotaExceededError) as excinfo:
        ensure_within_quota(FailClosedQuotaPolicy(), _ctx(), dimension=QuotaDimension.REQUESTS, idem="x")
    assert excinfo.value.status_code == 429


def test_rules_from_config_validates_shape() -> None:
    rules = quota_rules_from_config(
        [{"scope": "workspace", "dimension": "requests", "limit": 10, "window_seconds": 60}]
    )
    assert rules[0].scope is QuotaScope.WORKSPACE
    for bad in (
        [{"scope": "nope", "dimension": "requests", "limit": 10, "window_seconds": 60}],
        [{"scope": "workspace", "dimension": "requests", "limit": 0, "window_seconds": 60}],
        [{"scope": "workspace", "dimension": "requests", "limit": 10, "window_seconds": 0}],
        ["not-a-mapping"],
    ):
        with pytest.raises(ConfigurationError):
            quota_rules_from_config(bad)


# --- atomic counter store -----------------------------------------------------------------------


def test_reserve_admits_to_the_limit_then_denies(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        key = _key(dimension=QuotaDimension.GENERATION_TOKENS, limit=100)
        assert store.quota_reserve(key=key, reservation_id="r1", estimate=60, ttl_seconds=900) is True
        assert store.quota_reserve(key=key, reservation_id="r2", estimate=60, ttl_seconds=900) is False
        assert store.quota_reserve(key=key, reservation_id="r3", estimate=40, ttl_seconds=900) is True
    finally:
        store.close()


def test_reserve_is_idempotent_and_does_not_double_charge(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        key = _key(limit=100)
        assert store.quota_reserve(key=key, reservation_id="r1", estimate=60, ttl_seconds=900) is True
        # A re-run with the same id is admit-as-no-op, not a second 60-token charge.
        assert store.quota_reserve(key=key, reservation_id="r1", estimate=60, ttl_seconds=900) is True
        # 60 already reserved, so a fresh 60 is refused (a double charge would have left 0 headroom
        # under a different id, but the point is only 60 is held).
        assert store.quota_reserve(key=key, reservation_id="r2", estimate=41, ttl_seconds=900) is False
        assert store.quota_reserve(key=key, reservation_id="r2", estimate=40, ttl_seconds=900) is True
    finally:
        store.close()


def test_commit_moves_actual_into_spent_and_restores_reserved(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        key = _key(limit=100)
        store.quota_reserve(key=key, reservation_id="r1", estimate=80, ttl_seconds=900)
        store.quota_commit(reservation_id="r1", actual=20)
        # 20 spent, reserved back to 0, so 80 headroom remains.
        assert store.quota_reserve(key=key, reservation_id="r2", estimate=80, ttl_seconds=900) is True
        assert store.quota_reserve(key=key, reservation_id="r3", estimate=1, ttl_seconds=900) is False
        # A second commit is a no-op via the state gate.
        store.quota_commit(reservation_id="r1", actual=20)
        assert store.quota_reserve(key=key, reservation_id="r4", estimate=1, ttl_seconds=900) is False
    finally:
        store.close()


def test_commit_clamps_actual_to_the_reserved_ceiling(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        key = _key(limit=100)
        store.quota_reserve(key=key, reservation_id="r1", estimate=30, ttl_seconds=900)
        store.quota_commit(reservation_id="r1", actual=1000)
        # spent is clamped to 30, so 70 headroom remains, not a negative overflow.
        assert store.quota_reserve(key=key, reservation_id="r2", estimate=70, ttl_seconds=900) is True
        assert store.quota_reserve(key=key, reservation_id="r3", estimate=1, ttl_seconds=900) is False
    finally:
        store.close()


def test_release_restores_headroom_and_commit_after_release_is_a_no_op(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        key = _key(limit=100)
        store.quota_reserve(key=key, reservation_id="r1", estimate=100, ttl_seconds=900)
        assert store.quota_reserve(key=key, reservation_id="r2", estimate=1, ttl_seconds=900) is False
        store.quota_release(reservation_id="r1")
        assert store.quota_reserve(key=key, reservation_id="r2", estimate=100, ttl_seconds=900) is True
        # A commit after a release is a no-op, so nothing lands in spent.
        store.quota_commit(reservation_id="r1", actual=100)
    finally:
        store.close()


def test_reap_charges_a_leaked_reservations_ceiling_not_zero(tmp_path: Path) -> None:
    clock = _Clock()
    store = SQLStore(f"sqlite:///{tmp_path / 'quota.db'}", clock)
    store.initialize()
    try:
        key = _key(limit=100)
        store.quota_reserve(key=key, reservation_id="r1", estimate=70, ttl_seconds=1)
        clock.advance(5)
        reclaimed = store.quota_reap()
        assert reclaimed == 1
        # The ceiling is charged to spent (fail-closed), and reserved returned to baseline, so the
        # tenant is not permanently blocked: 30 headroom, not 0.
        assert store.quota_reserve(key=key, reservation_id="r2", estimate=30, ttl_seconds=900) is True
        assert store.quota_reserve(key=key, reservation_id="r3", estimate=1, ttl_seconds=900) is False
    finally:
        store.close()


def test_gc_removes_only_fully_settled_past_windows(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        settled = _key(scope_key_value="ws1", window=0, limit=100)
        store.quota_reserve(key=settled, reservation_id="r1", estimate=10, ttl_seconds=900)
        store.quota_commit(reservation_id="r1", actual=10)
        open_window = _key(scope_key_value="ws2", window=0, limit=100, workspace_id="ws2")
        store.quota_reserve(key=open_window, reservation_id="r2", estimate=10, ttl_seconds=900)
        removed = store.quota_gc(before_window_start=60)
        # The settled window is deleted; the one with an open reservation is skipped.
        assert removed == 1
        # The open window still admits against its held reservation.
        assert store.quota_reserve(key=open_window, reservation_id="r3", estimate=90, ttl_seconds=900) is True
    finally:
        store.close()


def test_two_workspaces_with_the_same_principal_id_get_distinct_rows(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        key_a = _key(scope=QuotaScope.PRINCIPAL, scope_key_value=f"wsA{_US}alice", workspace_id="wsA", limit=50)
        key_b = _key(scope=QuotaScope.PRINCIPAL, scope_key_value=f"wsB{_US}alice", workspace_id="wsB", limit=50)
        assert store.quota_reserve(key=key_a, reservation_id="a1", estimate=50, ttl_seconds=900) is True
        # wsB/alice is a distinct row, unaffected by wsA/alice being exhausted.
        assert store.quota_reserve(key=key_b, reservation_id="b1", estimate=50, ttl_seconds=900) is True
    finally:
        store.close()
