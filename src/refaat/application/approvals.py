"""Run-scoped human-approval inputs for governed tool execution.

An approval authorizes one run, not a program definition, so approvals are supplied at run time
through a context variable rather than pinned into the program artifact. The governed ReAct loop
reads the active context when a tool's capability requires approval. With nothing set the context
is empty, so a tool that requires approval is denied until an approval is supplied.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from refaat.core.text import stable_digest
from refaat.domain.audit import AuditEvent, AuditSink, emit_audit
from refaat.domain.authz import Principal
from refaat.domain.governance import ApprovalRecord
from refaat.domain.models import JsonObject
from refaat.errors import ValidationError
from refaat.observability import get_logger

if TYPE_CHECKING:
    from refaat.domain.authz import PrincipalDirectory
    from refaat.infrastructure.database.store import SQLStore

logger = get_logger(__name__)

# A fingerprint is a one-way sha256 hex digest and a capability name is a bounded identifier. The
# HTTP schema enforces these, but the in-process service is validated too so no client free text can
# reach the redacted audit event through the direct Python API.
_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")
_CAPABILITY_NAME = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")


def _validated_fingerprint(value: str) -> str:
    if not _FINGERPRINT.match(value):
        raise ValidationError("fingerprint must be a 64-character sha256 hex digest")
    return value


def _validated_capability_name(value: str) -> str:
    if not _CAPABILITY_NAME.match(value):
        raise ValidationError("capability_name must be a bounded identifier")
    return value


_ACTIVE_APPROVALS: ContextVar[ApprovalContext | None] = ContextVar("refaat_active_approvals", default=None)


@dataclass(frozen=True, slots=True)
class ApprovalContext:
    """The approvals in force for a run plus the policy they are bound to.

    ``policy_id`` and ``policy_version`` are fingerprint inputs the approval was granted against,
    so a change to either invalidates the approval and forces re-approval. The action version is
    not carried here: it is derived by the governed program from its content-addressed artifact id,
    so any change to the program definition, its policy, or its capabilities flips the fingerprint.
    The requester and the enforcement path share this context, so both compute the same fingerprint.
    """

    approvals: tuple[ApprovalRecord, ...] = ()
    policy_id: str = "react"
    policy_version: str = "1"


def active_approval_context() -> ApprovalContext:
    return _ACTIVE_APPROVALS.get() or ApprovalContext()


@contextmanager
def approval_scope(context: ApprovalContext) -> Iterator[None]:
    token = _ACTIVE_APPROVALS.set(context)
    try:
        yield
    finally:
        _ACTIVE_APPROVALS.reset(token)


def _record_from_row(row: dict[str, object]) -> ApprovalRecord:
    roles = row.get("approver_roles") or []
    granted = row.get("granted_at")
    if isinstance(granted, datetime) and granted.tzinfo is None:
        # SQLite returns a naive datetime; the store always writes clock.now() in UTC.
        granted = granted.replace(tzinfo=UTC)
    record_id = row.get("id")
    # A missing or None kind defaults to "normal": a legacy row written before the column existed
    # still satisfies the normal approval path (defense in depth alongside the server_default).
    kind = row.get("kind")
    return ApprovalRecord(
        fingerprint=str(row["fingerprint"]),
        approver_id=str(row["approver_id"]),
        approver_roles=frozenset(str(role) for role in roles) if isinstance(roles, (list, tuple, set)) else frozenset(),
        granted_at=granted if isinstance(granted, datetime) else None,
        record_id=str(record_id) if record_id is not None else None,
        kind=str(kind) if kind else "normal",
    )


def _reresolve(directory: PrincipalDirectory, record: ApprovalRecord) -> ApprovalRecord | None:
    # Approver-loses-permission: keep only the grant-time roles the approver STILL holds (intersect,
    # never replace, so a role gained after grant cannot retroactively validate an approval). An unknown
    # or revoked approver, an unavailable directory, or a malformed return all drop the record (fail-closed).
    try:
        current = directory.refresh(record.approver_id)
    except Exception:
        return None
    if not isinstance(current, Principal) or current.id != record.approver_id:
        return None
    # Coerce to a real frozenset so the intersection is genuine set reduction, never a role object's
    # __rand__; the directory is trusted to report roles honestly, this only bounds the operation.
    return replace(record, approver_roles=record.approver_roles & frozenset(current.roles))


class StoreApprovalBackend:
    """Durable, single-use approvals for the governed loop: load the granted approvals for an action
    fingerprint, and consume them on the execution they authorize so one approval is used once."""

    def __init__(
        self,
        store: SQLStore,
        workspace_id: str,
        audit_sink: AuditSink | None = None,
        *,
        audit_fail_closed: bool = False,
        directory: PrincipalDirectory | None = None,
    ) -> None:
        self._store = store
        self._workspace_id = workspace_id
        self._audit_sink = audit_sink
        self._audit_fail_closed = audit_fail_closed
        self._directory = directory

    def active(self, fingerprint: str) -> tuple[ApprovalRecord, ...]:
        records = (
            _record_from_row(row) for row in self._store.list_active_tool_approvals(self._workspace_id, fingerprint)
        )
        directory = self._directory
        if directory is None:
            return tuple(records)
        return tuple(fresh for record in records if (fresh := _reresolve(directory, record)) is not None)

    def request_anchor(self, fingerprint: str) -> datetime | None:
        return self._store.earliest_open_request_anchor(self._workspace_id, fingerprint)

    def record_consumed(
        self,
        fingerprint: str,
        *,
        principal_id: str,
        principal_kind: str,
        tenant: str | None,
        capability_name: str,
        policy_id: str,
        policy_version: str,
        consumed_ids: Sequence[str],
        occurred_at: datetime,
    ) -> None:
        """Emit one redacted approval.consumed audit event for a consumed durable approval.

        No-op unless a sink is installed. Best-effort by default; under fail-closed a sink failure
        raises so the governed run denies rather than proceeding on an unaudited consumption (the
        approval is already atomically consumed, so failing closed denies the action, never a
        double use). The consumed single-use approval ids discriminate the dedup_id, so two
        consumptions of the same fingerprint by the same principal stay distinct records on a
        dedup-on-dedup_id sink and only a redelivery collides.
        """
        if self._audit_sink is None:
            return
        event = self._consumed_event(
            fingerprint,
            principal_id=principal_id,
            principal_kind=principal_kind,
            tenant=tenant,
            capability_name=capability_name,
            policy_id=policy_id,
            policy_version=policy_version,
            consumed_ids=consumed_ids,
            occurred_at=occurred_at,
        )
        emit_audit(
            self._audit_sink,
            event,
            fail_closed=self._audit_fail_closed,
            logger=logger,
            log_key="approval_audit_sink_failed",
        )

    def _consumed_event(
        self,
        fingerprint: str,
        *,
        principal_id: str,
        principal_kind: str,
        tenant: str | None,
        capability_name: str,
        policy_id: str,
        policy_version: str,
        consumed_ids: Sequence[str],
        occurred_at: datetime,
    ) -> AuditEvent:
        return AuditEvent(
            event_type="approval.consumed",
            occurred_at=occurred_at,
            workspace_id=self._workspace_id,
            principal_id=principal_id,
            principal_kind=principal_kind,
            tenant=tenant,
            policy_id=policy_id,
            policy_version=policy_version,
            capability_name=capability_name,
            action_fingerprint=fingerprint,
            dedup_id=stable_digest(
                ("approval.consumed", self._workspace_id, fingerprint, principal_id, *sorted(consumed_ids))
            ),
        )

    def consume_with_audit(
        self,
        fingerprint: str,
        approval_ids: Sequence[str],
        *,
        principal_id: str,
        principal_kind: str,
        tenant: str | None,
        capability_name: str,
        policy_id: str,
        policy_version: str,
        occurred_at: datetime,
    ) -> bool:
        """Consume durable approvals and audit the consumption atomically under a durable sink.

        Under a DurableAuditSink the audit event is written in the same transaction as the consume, so an
        approval is never consumed without its audit record. For an inline or absent sink the consume and
        the best-effort emission stay the pre-existing two steps and are byte-identical, because an
        external sink cannot join the database transaction.
        """
        from refaat.infrastructure.audit import DurableAuditSink

        if isinstance(self._audit_sink, DurableAuditSink):
            event = self._consumed_event(
                fingerprint,
                principal_id=principal_id,
                principal_kind=principal_kind,
                tenant=tenant,
                capability_name=capability_name,
                policy_id=policy_id,
                policy_version=policy_version,
                consumed_ids=approval_ids,
                occurred_at=occurred_at,
            )
            return self._store.consume_tool_approvals(self._workspace_id, fingerprint, approval_ids, audit_event=event)
        if not self._store.consume_tool_approvals(self._workspace_id, fingerprint, approval_ids):
            return False
        self.record_consumed(
            fingerprint,
            principal_id=principal_id,
            principal_kind=principal_kind,
            tenant=tenant,
            capability_name=capability_name,
            policy_id=policy_id,
            policy_version=policy_version,
            consumed_ids=approval_ids,
            occurred_at=occurred_at,
        )
        return True


class ApprovalService:
    """Grant, list, and cancel durable tool-action approvals for a workspace."""

    def __init__(
        self,
        store: SQLStore,
        workspace_id: str,
        audit_sink: AuditSink | None = None,
        *,
        audit_fail_closed: bool = False,
    ) -> None:
        self._store = store
        self._workspace_id = workspace_id
        self._audit_sink = audit_sink
        self._audit_fail_closed = audit_fail_closed

    def grant(
        self,
        *,
        fingerprint: str,
        capability_name: str,
        approver_id: str,
        approver_roles: Sequence[str] = (),
        approver_kind: str = "human",
        granted_at: datetime | None = None,
        kind: str = "normal",
        reason: str | None = None,
    ) -> str:
        fingerprint = _validated_fingerprint(fingerprint)
        capability_name = _validated_capability_name(capability_name)
        timestamp = granted_at or self._store.clock.now()
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValidationError("granted_at must be timezone-aware")
        granted = timestamp.astimezone(UTC)
        # The break-glass justification lives on the durable row only; it is never passed to _emit, so
        # the redacted audit event carries no free-text reason (only ids, capability, and fingerprint).
        stored_reason = reason if kind == "break_glass" else None
        event_type = "approval.break_glass" if kind == "break_glass" else "approval.granted"
        # Emit-then-insert: audit under a reserved id before the row exists, so a fail-closed audit
        # failure raises before any consumable approval is created (closes the concurrent-consume race).
        approval_id = self._store.new_tool_approval_id()
        self._emit(
            event_type,
            approval_id=approval_id,
            principal_id=approver_id,
            principal_kind=approver_kind,
            capability_name=capability_name,
            fingerprint=fingerprint,
            occurred_at=granted,
        )
        self._store.record_tool_approval(
            self._workspace_id,
            fingerprint=fingerprint,
            capability_name=capability_name,
            approver_id=approver_id,
            approver_roles=approver_roles,
            granted_at=granted,
            approval_id=approval_id,
            kind=kind,
            reason=stored_reason,
        )
        return approval_id

    def list_active(self, fingerprint: str) -> list[JsonObject]:
        return self._store.list_active_tool_approvals(self._workspace_id, fingerprint)

    def cancel(self, approval_id: str, *, canceller_id: str | None = None, canceller_kind: str = "human") -> bool:
        from refaat.infrastructure.audit import DurableAuditSink

        # A cancel always audits (system sentinel when unattributed) so fail-closed also covers it. Under a
        # durable sink the cancel and its audit are one transaction, so a cancelled approval is never left
        # unaudited; an inline or absent sink keeps the pre-existing cancel-then-emit and is byte-identical.
        principal_id = canceller_id or "system"
        principal_kind = canceller_kind if canceller_id is not None else "system"
        if isinstance(self._audit_sink, DurableAuditSink):
            row = self._store.cancel_tool_approval(
                self._workspace_id,
                approval_id,
                audit_event_factory=lambda fingerprint, capability_name: self._lifecycle_event(
                    "approval.cancelled",
                    approval_id=approval_id,
                    principal_id=principal_id,
                    principal_kind=principal_kind,
                    capability_name=capability_name,
                    fingerprint=fingerprint,
                ),
            )
            return row is not None
        row = self._store.cancel_tool_approval(self._workspace_id, approval_id)
        if row is None:
            return False
        self._emit(
            "approval.cancelled",
            approval_id=approval_id,
            principal_id=principal_id,
            principal_kind=principal_kind,
            capability_name=row["capability_name"],
            fingerprint=row["fingerprint"],
        )
        return True

    def _lifecycle_event(
        self,
        event_type: str,
        *,
        approval_id: str,
        principal_id: str,
        principal_kind: str,
        capability_name: str,
        fingerprint: str,
        occurred_at: datetime | None = None,
    ) -> AuditEvent:
        # policy_id and policy_version are the standing default approval context, which the consume event
        # also carries in production (an ambient ApprovalContext is rejected on the run surface); a custom
        # approval-policy id is not plumbed to these lifecycle events.
        return AuditEvent(
            event_type=event_type,
            occurred_at=occurred_at or self._store.clock.now(),
            workspace_id=self._workspace_id,
            principal_id=principal_id,
            principal_kind=principal_kind,
            tenant=None,
            policy_id="react",
            policy_version="1",
            capability_name=capability_name,
            action_fingerprint=fingerprint,
            dedup_id=stable_digest((event_type, self._workspace_id, approval_id)),
        )

    def _emit(
        self,
        event_type: str,
        *,
        approval_id: str,
        principal_id: str,
        principal_kind: str,
        capability_name: str,
        fingerprint: str,
        occurred_at: datetime | None = None,
    ) -> None:
        if self._audit_sink is None:
            return
        event = self._lifecycle_event(
            event_type,
            approval_id=approval_id,
            principal_id=principal_id,
            principal_kind=principal_kind,
            capability_name=capability_name,
            fingerprint=fingerprint,
            occurred_at=occurred_at,
        )
        emit_audit(
            self._audit_sink,
            event,
            fail_closed=self._audit_fail_closed,
            logger=logger,
            log_key="approval_audit_sink_failed",
        )
