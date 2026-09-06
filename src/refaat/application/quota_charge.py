from __future__ import annotations

from refaat.domain.audit import AuditEvent, AuditSink, emit_audit
from refaat.domain.authz import Principal
from refaat.domain.kill_switch import KillSwitchRegistry, ensure_workspace_active
from refaat.domain.quota import QuotaContext, QuotaDimension, QuotaPolicy, ensure_within_quota
from refaat.errors import QuotaExceededError
from refaat.ids import new_id
from refaat.infrastructure.database.store import SQLStore
from refaat.observability import get_logger

logger = get_logger(__name__)


def charge_served_request(
    *,
    quota_policy: QuotaPolicy | None,
    kill_switches: KillSwitchRegistry | None,
    store: SQLStore,
    audit_sink: AuditSink | None,
    workspace_id: str,
    corpus_id: str | None,
    principal: Principal | None,
    dimension: QuotaDimension,
) -> None:
    """Count one served request at a program, skill, optimization, or fine-tune surface.

    The single charge body every served surface calls, so they meter consistently. A no-op when no
    quota policy is installed. The kill switch is checked before the charge so a frozen-workspace burst
    consumes no quota. A corpus-bound surface passes a corpus id, validated first so a corpus-scoped rule
    never mints a row for an unknown id; a corpus-less surface (fine-tune) passes None, so a corpus-scoped
    rule resolves to a missing scope and denies while a workspace or principal rule still governs it.
    """
    if quota_policy is None:
        return
    ensure_workspace_active(kill_switches, workspace_id)
    if corpus_id is not None:
        store.get_corpus(workspace_id, corpus_id)
    ctx = QuotaContext(
        workspace_id=workspace_id,
        tenant=principal.tenant if principal is not None else None,
        principal_id=principal.id if principal is not None else None,
        corpus_id=corpus_id,
        corpus_validated=corpus_id is not None,
    )
    try:
        ensure_within_quota(quota_policy, ctx, dimension=dimension, idem=new_id("qta"))
    except QuotaExceededError:
        _emit_denied(audit_sink, store, quota_policy, ctx, principal, dimension)
        raise


def _emit_denied(
    sink: AuditSink | None,
    store: SQLStore,
    quota_policy: QuotaPolicy,
    ctx: QuotaContext,
    principal: Principal | None,
    dimension: QuotaDimension,
) -> None:
    if sink is None:
        return
    from refaat.core.text import stable_digest

    name = dimension.value
    event = AuditEvent(
        event_type="quota.denied",
        occurred_at=store.clock.now(),
        workspace_id=ctx.workspace_id,
        corpus_id=ctx.corpus_id or "",
        principal_id=ctx.principal_id or "system",
        principal_kind=principal.kind.value if principal is not None else "system",
        tenant=ctx.tenant,
        policy_id=str(getattr(quota_policy, "cache_id", "quota") or "quota"),
        policy_version="",
        capability_name=name,
        action_fingerprint=stable_digest((ctx.workspace_id, name, ctx.principal_id or "")),
        dedup_id=stable_digest((ctx.workspace_id, ctx.corpus_id or "", name, "quota")),
    )
    emit_audit(sink, event, fail_closed=False, logger=logger)


__all__ = ["charge_served_request"]
