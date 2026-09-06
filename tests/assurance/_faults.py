"""Fault doubles for the failure-injection assurance suite.

These are injected at the store or port layer into the real production adapters, so each assurance
test exercises the genuine fail-closed conversion (a store outage becoming a deny, a provider outage
becoming a degrade or refuse), not a pre-cooked verdict. No runtime chaos capability ships; faults
enter only through the existing typed ports and the store-layer doubles here.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

from refaat.domain.audit import AuditEvent
from refaat.domain.kill_switch import KillSwitchSet
from refaat.domain.models import GenerationRequest, GenerationResult


class FakeClock:
    def __init__(self) -> None:
        self._now = datetime(2026, 1, 1, tzinfo=UTC)

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now = self._now + timedelta(seconds=seconds)


class RaisingKillSwitchStore:
    """A kill-switch store that primes cleanly, then fails every subsequent read once ``fail`` is
    set, so the real DatabaseKillSwitchRegistry exercises its fail-closed refresh path."""

    def __init__(self, snapshot: KillSwitchSet | None = None) -> None:
        self.snapshot = snapshot if snapshot is not None else KillSwitchSet.empty()
        self.fail = False

    def load_engaged_kill_switches(self) -> KillSwitchSet:
        if self.fail:
            raise RuntimeError("kill-switch store outage")
        return self.snapshot


class RaisingQuotaStore:
    """A quota store that pings cleanly (so construction succeeds) but fails every reservation, so
    the real DatabaseQuotaPolicy exercises its store-error-to-deny conversion."""

    def ping(self) -> None:
        return None

    def quota_reserve(self, *, key: object, reservation_id: str, estimate: int, ttl_seconds: float) -> bool:
        raise RuntimeError("quota store outage")

    def quota_commit(self, *, reservation_id: str, actual: int) -> None:
        return None

    def quota_release(self, *, reservation_id: str) -> None:
        return None


class AdmittingQuotaStore(RaisingQuotaStore):
    """The control: reservations always succeed, so the same admission path admits."""

    def quota_reserve(self, *, key: object, reservation_id: str, estimate: int, ttl_seconds: float) -> bool:
        return True


class RaisingAuditSink:
    def __init__(self) -> None:
        self.attempts = 0

    def record(self, event: AuditEvent) -> None:
        self.attempts += 1
        raise RuntimeError("audit sink outage")


class OutageGenerator:
    """A generation provider outage above the port: every call raises a dependency outage."""

    def __init__(self, *, recovered: bool = False) -> None:
        self.recovered = recovered

    @property
    def model_id(self) -> str:
        return "outage-v1"

    def _grounded(self, request: GenerationRequest) -> GenerationResult:
        top = request.context.hits[0]
        marker = f"[chunk:{top.chunk.id}]"
        return GenerationResult(
            answer=f"{top.chunk.text} {marker}", cited_chunk_ids=(top.chunk.id,), usage={"provider": "outage"}
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        from refaat.errors import DependencyUnavailableError

        if not self.recovered:
            raise DependencyUnavailableError("generation provider down")
        if not request.context.hits:
            return GenerationResult(answer="ok", cited_chunk_ids=(), usage={"provider": "outage"})
        return self._grounded(request)

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        from refaat.errors import DependencyUnavailableError

        if not self.recovered:
            raise DependencyUnavailableError("generation provider down")
        yield self._grounded(request).answer if request.context.hits else "ok"
