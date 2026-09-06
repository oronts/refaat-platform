"""Persisted per-scope quotas: a pure decision core plus its policy port.

An operator configures rules that cap generation tokens, cost, or request count per scope over a
fixed window. The scope key is derived only from trusted server-side fields (the workspace, the
resolved principal, the validated corpus), never from client free text, and every key is
workspace-prefixed so a principal or corpus id that is only unique within a workspace can never
collide across tenants. A rule that does not apply to a context resolves to no key and is a skip, so
the runtime is byte-identical when no quota governs the call; a principal-scoped rule that meets a
principal-bearing seam with no principal, or a corpus rule over an unvalidated corpus, resolves to a
missing scope and denies. This module is a domain leaf: it imports only the standard library and
``refaat.errors``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from refaat.errors import ConfigurationError, QuotaExceededError

_UNIT_SEPARATOR = "\x1f"


class QuotaDimension(StrEnum):
    GENERATION_TOKENS = "generation_tokens"
    COST_USD = "cost_usd"
    REQUESTS = "requests"
    INGEST_REQUESTS = "ingest_requests"
    PROGRAM_REQUESTS = "program_requests"
    SKILL_REQUESTS = "skill_requests"
    OPTIMIZATION_REQUESTS = "optimization_requests"
    FINE_TUNE_REQUESTS = "fine_tune_requests"


class QuotaScope(StrEnum):
    WORKSPACE = "workspace"
    PRINCIPAL = "principal"
    WORKSPACE_PRINCIPAL = "workspace_principal"
    CORPUS = "corpus"


@dataclass(frozen=True, slots=True)
class QuotaRule:
    scope: QuotaScope
    dimension: QuotaDimension
    limit: int
    window_seconds: int


@dataclass(frozen=True, slots=True)
class QuotaContext:
    """The trusted derived inputs a scope key is built from, never client free text.

    ``workspace_id`` is server-derived; an empty string is the explicit global key, never a skip.
    ``tenant`` travels only into the audit event, never into a row predicate. ``corpus_validated``
    is True only after the corpus read succeeded, so an arbitrary client corpus id can never mint a
    fresh counter row.
    """

    workspace_id: str
    tenant: str | None = None
    principal_id: str | None = None
    principal_bearing: bool = True
    corpus_id: str | None = None
    corpus_validated: bool = False


@dataclass(frozen=True, slots=True)
class QuotaKey:
    """A rule bound to a resolved scope key and its aligned window, the row the store guards."""

    scope: QuotaScope
    scope_key: str
    dimension: QuotaDimension
    window_start: int
    window_seconds: int
    limit: int
    workspace_id: str


@dataclass(frozen=True, slots=True)
class QuotaDecision:
    admitted: bool
    code: str
    dimension: QuotaDimension | None = None
    scope: QuotaScope | None = None


@dataclass(frozen=True, slots=True)
class QuotaReservation:
    """An opaque handle over the durable, leased reservation rows a reserve acquired."""

    dimension: QuotaDimension
    entries: tuple[tuple[str, QuotaKey], ...] = ()

    @property
    def active(self) -> bool:
        return bool(self.entries)


def scope_key(rule: QuotaRule, ctx: QuotaContext) -> tuple[str | None, str]:
    """Resolve a rule to its scope key for a context.

    Returns ``(key, code)``. ``code`` is ``ok`` when the rule applies and the key is usable,
    ``missing_scope`` when the rule applies to this seam but the required field is absent (a deny),
    and ``not_applicable`` when the rule genuinely does not govern this context (a byte-identical
    skip). Every key embeds the workspace so a workspace-unique principal or corpus id never
    collides across tenants; the empty workspace is a valid global key.
    """
    ws = ctx.workspace_id
    if rule.scope is QuotaScope.WORKSPACE:
        return ws, "ok"
    if rule.scope in (QuotaScope.PRINCIPAL, QuotaScope.WORKSPACE_PRINCIPAL):
        if ctx.principal_id:
            return f"{ws}{_UNIT_SEPARATOR}{ctx.principal_id}", "ok"
        return None, ("missing_scope" if ctx.principal_bearing else "not_applicable")
    if rule.scope is QuotaScope.CORPUS:
        if ctx.corpus_id and ctx.corpus_validated:
            return f"{ws}{_UNIT_SEPARATOR}{ctx.corpus_id}", "ok"
        return None, "missing_scope"
    return None, "not_applicable"


def window_start(now_epoch: int, window_seconds: int) -> int:
    return now_epoch - (now_epoch % window_seconds)


def resolve_keys(
    rules: Sequence[QuotaRule], ctx: QuotaContext, dimension: QuotaDimension, now_epoch: int
) -> tuple[tuple[QuotaKey, ...], str]:
    """Bind every rule of one dimension to its window key for a context.

    Returns ``(keys, code)``. A single applicable rule with a missing scope short-circuits to
    ``("", "missing_scope")`` so the caller denies; otherwise the ``ok`` rules become keys and any
    ``not_applicable`` rule is skipped. An empty key tuple with ``ok`` means no rule governs the
    call (a byte-identical skip).
    """
    keys: list[QuotaKey] = []
    for rule in rules:
        if rule.dimension is not dimension:
            continue
        key, code = scope_key(rule, ctx)
        if code == "missing_scope":
            return (), "missing_scope"
        if code == "ok" and key is not None:
            start = window_start(now_epoch, rule.window_seconds)
            keys.append(
                QuotaKey(
                    scope=rule.scope,
                    scope_key=key,
                    dimension=dimension,
                    window_start=start,
                    window_seconds=rule.window_seconds,
                    limit=rule.limit,
                    workspace_id=ctx.workspace_id,
                )
            )
    return tuple(keys), "ok"


class QuotaPolicy(Protocol):
    """The persisted quota decision surface, resolved once per call from a rule snapshot."""

    cache_id: str | None

    def reserve(
        self, ctx: QuotaContext, *, dimension: QuotaDimension, estimate: int | None, idem: str
    ) -> QuotaReservation: ...

    def commit(self, reservation: QuotaReservation, *, actual: int) -> None: ...

    def release(self, reservation: QuotaReservation) -> None: ...

    def admit(self, ctx: QuotaContext, *, dimension: QuotaDimension, amount: int = 1, idem: str) -> QuotaDecision: ...


class NullQuotaPolicy:
    """An always-admitting policy: no rule ever governs. For tests that want an explicit no-op."""

    cache_id: str | None = None

    def reserve(
        self, ctx: QuotaContext, *, dimension: QuotaDimension, estimate: int | None, idem: str
    ) -> QuotaReservation:
        return QuotaReservation(dimension=dimension)

    def commit(self, reservation: QuotaReservation, *, actual: int) -> None:
        return None

    def release(self, reservation: QuotaReservation) -> None:
        return None

    def admit(self, ctx: QuotaContext, *, dimension: QuotaDimension, amount: int = 1, idem: str) -> QuotaDecision:
        return QuotaDecision(admitted=True, code="ok", dimension=dimension)


class FailClosedQuotaPolicy:
    """A deny-all reference policy: every reservation and admission is refused."""

    cache_id: str | None = "fail_closed"

    def reserve(
        self, ctx: QuotaContext, *, dimension: QuotaDimension, estimate: int | None, idem: str
    ) -> QuotaReservation:
        raise QuotaExceededError("Quota unavailable")

    def commit(self, reservation: QuotaReservation, *, actual: int) -> None:
        return None

    def release(self, reservation: QuotaReservation) -> None:
        return None

    def admit(self, ctx: QuotaContext, *, dimension: QuotaDimension, amount: int = 1, idem: str) -> QuotaDecision:
        return QuotaDecision(admitted=False, code="unavailable", dimension=dimension)


def ensure_within_quota(
    policy: QuotaPolicy | None,
    ctx: QuotaContext | None,
    *,
    dimension: QuotaDimension,
    amount: int = 1,
    idem: str,
) -> None:
    """Count one unit against a quota, raising when it is exhausted or unavailable.

    The shared count-only chokepoint the evidence-serving services call at admission. A no-op when
    no policy or context is installed, so the runtime is byte-identical when unconfigured; every
    non-admit decision denies, so a store outage or an exhausted counter both fail closed.
    """
    if policy is None or ctx is None:
        return
    decision = policy.admit(ctx, dimension=dimension, amount=amount, idem=idem)
    if not decision.admitted:
        raise QuotaExceededError("Quota exhausted")


def quota_rules_from_config(raw: Iterable[Any]) -> tuple[QuotaRule, ...]:
    """Parse and validate the operator rule list into an immutable snapshot.

    Raises ``ConfigurationError`` on any malformed rule so a bad quota configuration fails at
    startup rather than silently governing nothing.
    """
    rules: list[QuotaRule] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ConfigurationError("Each quota rule must be a mapping")
        try:
            scope = QuotaScope(str(entry["scope"]))
            dimension = QuotaDimension(str(entry["dimension"]))
            limit = int(entry["limit"])
            window = int(entry["window_seconds"])
        except (KeyError, ValueError) as exc:
            raise ConfigurationError(f"Malformed quota rule: {entry!r}") from exc
        if limit <= 0:
            raise ConfigurationError("A quota rule limit must be a positive integer")
        if window <= 0:
            raise ConfigurationError("A quota rule window_seconds must be a positive integer")
        rules.append(QuotaRule(scope=scope, dimension=dimension, limit=limit, window_seconds=window))
    return tuple(rules)


__all__ = [
    "FailClosedQuotaPolicy",
    "NullQuotaPolicy",
    "QuotaContext",
    "QuotaDecision",
    "QuotaDimension",
    "QuotaKey",
    "QuotaPolicy",
    "QuotaReservation",
    "QuotaRule",
    "QuotaScope",
    "ensure_within_quota",
    "quota_rules_from_config",
    "resolve_keys",
    "scope_key",
    "window_start",
]
