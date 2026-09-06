"""The store-backed quota policy: an immutable rule snapshot over the atomic counter store.

The rule set is primed once at construction from the operator config, so there is no rule read that
can fail open by returning an empty list; counter reads are always live and authoritative. Every
store error is converted to a deny, so a storage outage the enforcement cannot read fails closed
rather than serving. A failed ping at construction propagates out of build() as a startup failure,
mirroring the kill-switch registry.
"""

from __future__ import annotations

from collections.abc import Sequence

from refaat.clock import Clock
from refaat.core.text import stable_digest
from refaat.domain.quota import (
    QuotaContext,
    QuotaDecision,
    QuotaDimension,
    QuotaKey,
    QuotaReservation,
    QuotaRule,
    resolve_keys,
)
from refaat.errors import QuotaExceededError
from refaat.infrastructure.database.store import SQLStore
from refaat.observability import get_logger

logger = get_logger(__name__)


class DatabaseQuotaPolicy:
    cache_id: str | None = "database"

    def __init__(
        self,
        store: SQLStore,
        clock: Clock,
        *,
        rules: Sequence[QuotaRule],
        lease_seconds: float,
    ) -> None:
        self._store = store
        self._clock = clock
        self._rules = tuple(rules)
        self._lease = lease_seconds
        # Prime the store connection here so a broken database fails startup, not the first request.
        store.ping()

    @property
    def rules(self) -> tuple[QuotaRule, ...]:
        """The primed rule snapshot, read-only. Side-effect-free, for the admin status plane."""
        return self._rules

    def _now_epoch(self) -> int:
        return int(self._clock.now().timestamp())

    def reserve(
        self, ctx: QuotaContext, *, dimension: QuotaDimension, estimate: int | None, idem: str
    ) -> QuotaReservation:
        keys, code = resolve_keys(self._rules, ctx, dimension, self._now_epoch())
        if code == "missing_scope":
            raise QuotaExceededError("Quota scope is missing")
        if not keys:
            return QuotaReservation(dimension=dimension)
        if estimate is None:
            # A dimension governs this call but no finite ceiling was supplied to reserve against;
            # fail closed rather than admit an unbounded run.
            raise QuotaExceededError("Quota ceiling is missing")
        acquired: list[tuple[str, QuotaKey]] = []
        try:
            for key in keys:
                reservation_id = self._reservation_id(idem, dimension, key)
                admitted = self._store.quota_reserve(
                    key=key, reservation_id=reservation_id, estimate=int(estimate), ttl_seconds=self._lease
                )
                if not admitted:
                    raise QuotaExceededError("Quota exhausted")
                acquired.append((reservation_id, key))
        except QuotaExceededError:
            self._release_ids(acquired)
            raise
        except Exception as exc:
            self._release_ids(acquired)
            raise QuotaExceededError("Quota unavailable") from exc
        return QuotaReservation(dimension=dimension, entries=tuple(acquired))

    def commit(self, reservation: QuotaReservation, *, actual: int) -> None:
        for reservation_id, _ in reservation.entries:
            try:
                self._store.quota_commit(reservation_id=reservation_id, actual=actual)
            except Exception:
                logger.warning("quota_commit_failed")

    def release(self, reservation: QuotaReservation) -> None:
        self._release_ids(reservation.entries)

    def admit(self, ctx: QuotaContext, *, dimension: QuotaDimension, amount: int = 1, idem: str) -> QuotaDecision:
        try:
            keys, code = resolve_keys(self._rules, ctx, dimension, self._now_epoch())
        except Exception:
            return QuotaDecision(admitted=False, code="unavailable", dimension=dimension)
        if code == "missing_scope":
            return QuotaDecision(admitted=False, code="missing_scope", dimension=dimension)
        if not keys:
            return QuotaDecision(admitted=True, code="ok", dimension=dimension)
        # A fully-known unit reserves the amount on every key and commits only when all keys admit,
        # so a denial or error on any one key counts nothing on the others (all-or-nothing), and a
        # retry with the same idem is a no-op through the idempotent reservation id.
        acquired: list[tuple[str, QuotaKey]] = []
        try:
            for key in keys:
                reservation_id = self._reservation_id(idem, dimension, key)
                admitted = self._store.quota_reserve(
                    key=key, reservation_id=reservation_id, estimate=int(amount), ttl_seconds=self._lease
                )
                if not admitted:
                    self._release_ids(acquired)
                    return QuotaDecision(admitted=False, code="over_limit", dimension=dimension, scope=key.scope)
                acquired.append((reservation_id, key))
        except Exception:
            self._release_ids(acquired)
            return QuotaDecision(admitted=False, code="unavailable", dimension=dimension)
        for reservation_id, _ in acquired:
            try:
                self._store.quota_commit(reservation_id=reservation_id, actual=int(amount))
            except Exception:
                logger.warning("quota_admission_commit_failed")
        return QuotaDecision(admitted=True, code="ok", dimension=dimension)

    def _release_ids(self, entries: Sequence[tuple[str, QuotaKey]]) -> None:
        for reservation_id, _ in entries:
            try:
                self._store.quota_release(reservation_id=reservation_id)
            except Exception:
                logger.warning("quota_release_failed")

    @staticmethod
    def _reservation_id(idem: str, dimension: QuotaDimension, key: QuotaKey) -> str:
        return stable_digest((idem, dimension, key.scope_key, str(key.window_start)), length=64)


__all__ = ["DatabaseQuotaPolicy"]
