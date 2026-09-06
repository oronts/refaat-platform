"""Incident kill switches: the pure decision core and the store-backed registry cache.

The domain predicate denies a scope for a matching global or per-workspace switch, a workspace freeze
subsumes every scope, and an unavailable snapshot (a failed read) denies everything. The registry
caches within its TTL, fails closed to unavailable past the TTL by default, serves last-known-good
only inside an explicit stale window, and fails startup on a bad prime.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from refaat.domain.kill_switch import KillSwitchScope, KillSwitchSet, NullKillSwitchRegistry
from refaat.infrastructure.kill_switch import DatabaseKillSwitchRegistry

FREEZE = KillSwitchScope.WORKSPACE_FREEZE.value


def _set(*pairs: tuple[str, str]) -> KillSwitchSet:
    return KillSwitchSet(frozenset(pairs))


def test_denies_global_and_per_workspace() -> None:
    assert _set(("", "writes")).denies(KillSwitchScope.WRITES, "ws1") is True
    assert _set(("ws1", "writes")).denies(KillSwitchScope.WRITES, "ws1") is True
    assert _set(("ws2", "writes")).denies(KillSwitchScope.WRITES, "ws1") is False
    assert _set().denies(KillSwitchScope.WRITES, "ws1") is False


def test_workspace_freeze_subsumes_every_scope() -> None:
    frozen = _set(("ws1", FREEZE))
    assert frozen.freezes("ws1") is True
    for scope in KillSwitchScope:
        assert frozen.denies(scope, "ws1") is True
    assert frozen.freezes("ws2") is False


def test_global_freeze_subsumes_every_workspace() -> None:
    frozen = _set(("", FREEZE))
    assert frozen.freezes("anything") is True
    assert frozen.denies(KillSwitchScope.GENERATION, "anything") is True


def test_unavailable_denies_and_freezes_everything() -> None:
    unavailable = KillSwitchSet.unavailable()
    assert unavailable.available is False
    assert unavailable.freezes("ws1") is True
    for scope in KillSwitchScope:
        assert unavailable.denies(scope, "ws1") is True


def test_null_registry_is_always_open() -> None:
    assert NullKillSwitchRegistry().engaged() == KillSwitchSet.empty()


class _Clock:
    def __init__(self) -> None:
        self.now_value = datetime(2026, 1, 1, tzinfo=UTC)

    def now(self) -> datetime:
        return self.now_value

    def advance(self, seconds: float) -> None:
        self.now_value = self.now_value + timedelta(seconds=seconds)


class _Store:
    def __init__(self, snapshot: KillSwitchSet) -> None:
        self.snapshot = snapshot
        self.calls = 0
        self.fail = False

    def load_engaged_kill_switches(self) -> KillSwitchSet:
        self.calls += 1
        if self.fail:
            raise RuntimeError("store down")
        return self.snapshot


def test_registry_caches_within_ttl() -> None:
    clock = _Clock()
    store = _Store(_set(("ws1", FREEZE)))
    registry = DatabaseKillSwitchRegistry(store, clock, cache_ttl_seconds=2.0, stale_deny_seconds=0.0)
    assert store.calls == 1
    registry.engaged()
    registry.engaged()
    assert store.calls == 1
    clock.advance(3.0)
    assert registry.engaged().freezes("ws1") is True
    assert store.calls == 2


def test_registry_fails_closed_on_read_failure_by_default() -> None:
    clock = _Clock()
    store = _Store(_set())
    registry = DatabaseKillSwitchRegistry(store, clock, cache_ttl_seconds=0.0, stale_deny_seconds=0.0)
    store.fail = True
    clock.advance(1.0)
    snapshot = registry.engaged()
    assert snapshot.available is False
    assert snapshot.freezes("ws1") is True


def test_registry_serves_last_good_only_within_the_stale_window() -> None:
    clock = _Clock()
    store = _Store(_set(("ws1", FREEZE)))
    registry = DatabaseKillSwitchRegistry(store, clock, cache_ttl_seconds=0.0, stale_deny_seconds=10.0)
    store.fail = True
    clock.advance(1.0)
    served = registry.engaged()
    assert served.available is True
    assert served.freezes("ws1") is True
    clock.advance(20.0)
    assert registry.engaged().available is False


def test_registry_fails_startup_on_a_bad_prime() -> None:
    clock = _Clock()
    store = _Store(_set())
    store.fail = True
    with pytest.raises(RuntimeError):
        DatabaseKillSwitchRegistry(store, clock, cache_ttl_seconds=2.0, stale_deny_seconds=0.0)
