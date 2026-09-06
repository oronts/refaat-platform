from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from refaat.domain.quota import QuotaContext, QuotaDimension, QuotaPolicy, QuotaReservation
from refaat.errors import QuotaExceededError

_ACTIVE_METER: ContextVar[CostMeter | None] = ContextVar("refaat_active_meter", default=None)

# A ceiling far above any real cost reservation, used to charge a reservation its full remaining
# estimate: quota_commit clamps to min(actual, estimate), so any value above the estimate charges it.
_CHARGE_FULL_CEILING = 2**62


@dataclass
class CostMeter:
    """Run-scoped accrual of provider usage tokens and optional USD cost.

    ``charge`` sums ``prompt_tokens`` plus ``completion_tokens`` from a provider
    usage dict and accrues them; it never raises. Token and cost are known only
    after a call, so enforcement is a boundary check the multi-attempt loops run
    before starting the next unit of work: ``exhausted`` reports that a ceiling is
    already met, at which point the caller stops and selects the best result so far
    (always after at least one unit). USD cost accrues only when a per-model price
    map is supplied, so token-only budgeting is the clean default when no operator
    price table is configured.
    """

    token_ceiling: int | None = None
    cost_ceiling_usd: float | None = None
    prices: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    tokens_used: int = 0
    cost_used_usd: float = 0.0
    last_charge: int = 0
    # Serializes the accrual RMW against an abandoned tool worker still charging the shared meter.
    _lock: threading.Lock = field(default_factory=threading.Lock, compare=False, repr=False)

    def charge(self, usage: Mapping[str, object], *, model_id: str | None = None) -> int:
        prompt = _int(usage.get("prompt_tokens"))
        completion = _int(usage.get("completion_tokens"))
        total = prompt + completion
        price = self.prices.get(model_id) if model_id is not None else None
        with self._lock:
            self.tokens_used += total
            self.last_charge = total
            if price is not None:
                self.cost_used_usd += prompt / 1000.0 * float(price.get("input_per_1k", 0.0))
                self.cost_used_usd += completion / 1000.0 * float(price.get("output_per_1k", 0.0))
        return total

    def remaining_tokens(self) -> int | None:
        if self.token_ceiling is None:
            return None
        with self._lock:
            return self.token_ceiling - self.tokens_used

    def remaining_cost_usd(self) -> float | None:
        if self.cost_ceiling_usd is None:
            return None
        with self._lock:
            return self.cost_ceiling_usd - self.cost_used_usd

    def exhausted(self) -> bool:
        with self._lock:
            over_tokens = self.token_ceiling is not None and self.tokens_used >= self.token_ceiling
            over_cost = self.cost_ceiling_usd is not None and self.cost_used_usd >= self.cost_ceiling_usd
        return over_tokens or over_cost

    def snapshot(self) -> tuple[int, float]:
        # An atomic (tokens, cost) pair so settlement never reads a token count and a cost from either
        # side of a concurrent charge, which would falsely trip the unpriced-model cost guard.
        with self._lock:
            return self.tokens_used, self.cost_used_usd


def _int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def active_meter() -> CostMeter | None:
    return _ACTIVE_METER.get()


@contextmanager
def budget_scope(
    token_ceiling: int | None = None,
    cost_ceiling_usd: float | None = None,
    *,
    prices: Mapping[str, Mapping[str, float]] | None = None,
) -> Iterator[CostMeter | None]:
    """Open a run-scoped meter, sharing an already-active one, or a no-op scope.

    If a meter is already active (an outer scope, for example Best-of wrapping
    ReAct, or the shared query path nested inside a program), this scope shares it
    so all spend accrues against the one outermost ceiling instead of resetting the
    budget per inner unit. With no active meter and both ceilings None no meter is
    installed and the active meter is left untouched, so the deterministic offline
    stack runs the identical unmetered code path.
    """
    existing = _ACTIVE_METER.get()
    if existing is not None:
        yield existing
        return
    if token_ceiling is None and cost_ceiling_usd is None:
        yield None
        return
    meter = CostMeter(token_ceiling=token_ceiling, cost_ceiling_usd=cost_ceiling_usd, prices=prices or {})
    token = _ACTIVE_METER.set(meter)
    try:
        yield meter
    finally:
        _ACTIVE_METER.reset(token)


@contextmanager
def isolated_budget_scope(
    token_ceiling: int | None = None,
    cost_ceiling_usd: float | None = None,
    *,
    prices: Mapping[str, Mapping[str, float]] | None = None,
) -> Iterator[CostMeter | None]:
    """Open a fresh meter that never shares an outer one, or a detached no-charge scope.

    Unlike :func:`budget_scope`, this ALWAYS replaces the active meter for its
    duration, so an unserved shadow or canary run can never draw down the served
    run's meter: a fresh :class:`CostMeter` when a ceiling is set, or None (charging
    nothing, and detaching any outer meter) when neither ceiling is configured. The
    prior meter is restored on exit.
    """
    meter = (
        CostMeter(token_ceiling=token_ceiling, cost_ceiling_usd=cost_ceiling_usd, prices=prices or {})
        if token_ceiling is not None or cost_ceiling_usd is not None
        else None
    )
    token = _ACTIVE_METER.set(meter)
    try:
        yield meter
    finally:
        _ACTIVE_METER.reset(token)


def resolve_stream_meter(
    token_ceiling: int | None = None,
    cost_ceiling_usd: float | None = None,
    *,
    prices: Mapping[str, Mapping[str, float]] | None = None,
) -> CostMeter | None:
    """The one meter object a streamed run charges, or None for the unmetered path.

    A streamed generator is driven across per-iteration context copies (the ASGI
    threadpool copies the context on every ``next``), where a ``budget_scope`` whose
    ContextVar set and reset straddle a ``yield`` breaks: the reset token is invalid
    in a later copy and the meter set in one copy is invisible in the next. So a
    stream resolves ONE meter object here and re-binds it transiently with
    :func:`use_meter` around each synchronous unit, never across a yield. An already
    active outer meter is shared; with no active meter and both ceilings None no
    meter is created, so the deterministic offline stream runs the identical
    unmetered path.
    """
    existing = _ACTIVE_METER.get()
    if existing is not None:
        return existing
    if token_ceiling is None and cost_ceiling_usd is None:
        return None
    return CostMeter(token_ceiling=token_ceiling, cost_ceiling_usd=cost_ceiling_usd, prices=prices or {})


@contextmanager
def use_meter(meter: CostMeter | None) -> Iterator[None]:
    """Bind a caller-owned meter for one synchronous region, then reset it there.

    Unlike :func:`budget_scope` this binds an explicit meter object and resets it
    within the same region, so the set and reset never straddle a generator yield.
    A None meter is a no-op, so the unmetered path is untouched.
    """
    if meter is None:
        yield
        return
    token = _ACTIVE_METER.set(meter)
    try:
        yield
    finally:
        _ACTIVE_METER.reset(token)


def prices_from(mapping: Mapping[str, Mapping[str, float]] | None) -> dict[str, dict[str, float]]:
    if not mapping:
        return {}
    return {model: {key: float(value) for key, value in price.items()} for model, price in mapping.items()}


@dataclass(frozen=True, slots=True)
class QuotaHold:
    """The token and cost reservations acquired for one governed run, settled together."""

    token: QuotaReservation | None
    cost: QuotaReservation | None
    cost_active: bool


def _micro_usd(cost_ceiling_usd: float | None) -> int | None:
    if cost_ceiling_usd is None:
        return None
    return round(cost_ceiling_usd * 1_000_000)


def quota_reserve_all(
    policy: QuotaPolicy | None,
    ctx: QuotaContext | None,
    *,
    token_ceiling: int | None,
    cost_ceiling_usd: float | None,
    idem: str,
) -> QuotaHold | None:
    """Reserve the token and cost ceilings before the model spend, all-or-release-all.

    A no-op returning None when no policy or context governs the call. If either reservation raises
    (an exhausted counter or a store outage), the one already acquired is released, so a governed
    run never leaks a partial reservation.
    """
    if policy is None or ctx is None:
        return None
    token_res: QuotaReservation | None = None
    cost_res: QuotaReservation | None = None
    try:
        token_res = policy.reserve(
            ctx, dimension=QuotaDimension.GENERATION_TOKENS, estimate=token_ceiling, idem=f"{idem}:tok"
        )
        cost_res = policy.reserve(
            ctx, dimension=QuotaDimension.COST_USD, estimate=_micro_usd(cost_ceiling_usd), idem=f"{idem}:usd"
        )
    except BaseException:
        if token_res is not None:
            policy.release(token_res)
        if cost_res is not None:
            policy.release(cost_res)
        raise
    return QuotaHold(token=token_res, cost=cost_res, cost_active=cost_res.active)


def quota_commit_all(policy: QuotaPolicy | None, hold: QuotaHold | None, meter: CostMeter | None) -> None:
    """Reconcile the reservations down to the meter's measured spend.

    An unpriced model under an active cost quota (tokens spent at a recorded cost of zero) charges
    the full cost ceiling and fails closed, so the run consumes its worst case rather than nothing.
    """
    if policy is None or hold is None:
        return
    tokens_used, cost_used_usd = meter.snapshot() if meter is not None else (0, 0.0)
    if hold.token is not None:
        policy.commit(hold.token, actual=tokens_used)
    if hold.cost is None:
        return
    micro = round(cost_used_usd * 1_000_000)
    if hold.cost_active and tokens_used > 0 and micro == 0:
        policy.commit(hold.cost, actual=_CHARGE_FULL_CEILING)
        raise QuotaExceededError("Unpriced model under a cost quota")
    policy.commit(hold.cost, actual=micro)


def quota_release_all(policy: QuotaPolicy | None, hold: QuotaHold | None) -> None:
    """Release every reservation the hold acquired; a no-op on an already-settled reservation."""
    if policy is None or hold is None:
        return
    if hold.token is not None:
        policy.release(hold.token)
    if hold.cost is not None:
        policy.release(hold.cost)


@contextmanager
def quota_run(
    policy: QuotaPolicy | None,
    ctx: QuotaContext | None,
    *,
    meter: CostMeter | None,
    token_ceiling: int | None,
    cost_ceiling_usd: float | None,
    idem: str,
) -> Iterator[None]:
    """Reserve the run's ceiling, then reconcile to the actual measured spend on the way out.

    A pure no-op yield when no policy or context governs the call, so the runtime is byte-identical
    when unconfigured. On any exception inside the body the reservations are released; on a normal
    exit they are committed to the meter's actuals.
    """
    hold = quota_reserve_all(policy, ctx, token_ceiling=token_ceiling, cost_ceiling_usd=cost_ceiling_usd, idem=idem)
    committed = False
    try:
        yield
        quota_commit_all(policy, hold, meter)
        committed = True
    finally:
        if not committed:
            quota_release_all(policy, hold)


__all__ = [
    "CostMeter",
    "QuotaHold",
    "active_meter",
    "budget_scope",
    "isolated_budget_scope",
    "prices_from",
    "quota_commit_all",
    "quota_release_all",
    "quota_reserve_all",
    "quota_run",
    "resolve_stream_meter",
    "use_meter",
]
