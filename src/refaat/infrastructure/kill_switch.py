"""The store-backed kill-switch registry with a short-TTL cache and fail-closed read semantics.

Engaging or clearing a switch is a database write, so a change reaches every worker within one cache
TTL with no restart. The registry primes once at construction, so a runtime never serves without a
known baseline, and a failed prime propagates out of build() as a startup failure. When a refresh
past the TTL fails, the snapshot becomes unavailable and every scope is denied, unless an operator
opted into a stale-serve window by setting stale_deny_seconds above zero, which relaxes the fail-closed
default for that bounded window before it too denies.
"""

from __future__ import annotations

import threading

from refaat.clock import Clock
from refaat.domain.kill_switch import KillSwitchSet
from refaat.infrastructure.database.store import SQLStore
from refaat.observability import get_logger

logger = get_logger(__name__)


class DatabaseKillSwitchRegistry:
    def __init__(
        self,
        store: SQLStore,
        clock: Clock,
        *,
        cache_ttl_seconds: float,
        stale_deny_seconds: float,
    ) -> None:
        self._store = store
        self._clock = clock
        self._ttl = cache_ttl_seconds
        self._stale_deny = stale_deny_seconds
        self._lock = threading.Lock()
        self._snapshot = store.load_engaged_kill_switches()
        now = clock.now()
        self._loaded_at = now
        self._last_good_at = now

    def engaged(self) -> KillSwitchSet:
        now = self._clock.now()
        if (now - self._loaded_at).total_seconds() < self._ttl:
            return self._snapshot
        with self._lock:
            now = self._clock.now()
            if (now - self._loaded_at).total_seconds() < self._ttl:
                return self._snapshot
            try:
                snapshot = self._store.load_engaged_kill_switches()
            except Exception:
                logger.warning("kill_switch_refresh_failed")
                # Serve last-known-good only inside an operator-opted stale window; otherwise deny
                # everything, since a store outage the enforcement cannot read must fail closed.
                if (now - self._last_good_at).total_seconds() < self._stale_deny:
                    return self._snapshot
                return KillSwitchSet.unavailable()
            self._snapshot = snapshot
            self._loaded_at = now
            self._last_good_at = now
            return snapshot


__all__ = ["DatabaseKillSwitchRegistry"]
