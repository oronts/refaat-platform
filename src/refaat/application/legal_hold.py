"""The workspace-scoped legal-hold management service.

Coverage is decided entirely in SQL by store.is_document_held and every destructive path enforces
an existing hold unconditionally. LegalHoldService places, lifts, and lists durable holds. A write
(place/lift) fails closed when the management surface is not enabled on this runtime, so a hold is
only ever recorded by a runtime that deliberately opted into managing holds.
"""

from __future__ import annotations

from collections.abc import Callable

from refaat.core.text import stable_digest
from refaat.domain.audit import AuditEvent, AuditSink, emit_audit
from refaat.domain.legal_hold import (
    LegalHoldError,
    LegalHoldRecord,
    validate_actor_id,
    validate_corpus_ref,
    validate_document_ref,
    validate_hold_reason,
)
from refaat.infrastructure.database.store import SQLStore
from refaat.observability import get_logger

logger = get_logger(__name__)


class LegalHoldService:
    """Place, lift, and list durable legal holds for a workspace.

    A write (place/lift) fails closed when the management surface is not enabled on this runtime
    (store.legal_hold_management_enabled is False). list_active is read-only and does not gate. An
    existing hold is enforced on every runtime regardless of this surface.
    """

    def __init__(
        self,
        store: SQLStore,
        workspace_id: str,
        audit_sink: AuditSink | None = None,
        *,
        audit_fail_closed: bool = False,
        deferred_processor: Callable[[str, str], object] | None = None,
    ) -> None:
        self._store = store
        self._workspace_id = workspace_id
        self._audit_sink = audit_sink
        self._audit_fail_closed = audit_fail_closed
        self._deferred_processor = deferred_processor

    def place(
        self,
        *,
        corpus_id: str,
        document_id: str | None = None,
        placed_by: str,
        reason: str | None = None,
    ) -> str:
        if not self._store.legal_hold_management_enabled:
            # Refuse to place a hold from a runtime that did not opt into the management surface.
            raise LegalHoldError(
                "the legal-hold management surface is not enabled on this runtime; "
                "wire with_legal_hold_registry() to place a hold"
            )
        actor = validate_actor_id(placed_by)
        # Bound the ids to an id shape BEFORE they reach an audit event, so client free text can
        # never enter cited_document_ids or the event corpus id. The reason stays row-only.
        corpus_ref = validate_corpus_ref(corpus_id)
        document_ref = validate_document_ref(document_id)
        stored_reason = validate_hold_reason(reason)
        # Insert-then-emit. Placing a hold is protection-increasing, so the protective row must land
        # even under a degraded sink; the audit follows best-effort. place_legal_hold holds the corpus
        # destroy lock across its insert, so it serializes against an in-flight destroy.
        hold_id = self._store.place_legal_hold(
            self._workspace_id,
            corpus_id=corpus_ref,
            document_id=document_ref,
            placed_by=actor,
            reason=stored_reason,
            placed_at=self._store.clock.now(),
        )
        self._emit(
            "hold.placed",
            hold_id=hold_id,
            principal_id=actor,
            corpus_id=corpus_ref,
            document_id=document_ref,
            capability_name="legal.hold.place",
            fail_closed=False,
        )
        return hold_id

    def lift(self, hold_id: str, *, lifted_by: str) -> bool:
        if not self._store.legal_hold_management_enabled:
            raise LegalHoldError(
                "the legal-hold management surface is not enabled on this runtime; "
                "wire with_legal_hold_registry() to lift a hold"
            )
        actor = validate_actor_id(lifted_by)
        row = self._store.get_legal_hold(self._workspace_id, hold_id)
        if row is None or row["lifted_at"] is not None:
            return False
        # Emit-then-lift under fail_closed. Lifting is protection-decreasing (it re-enables
        # spoliation), so under audit_fail_closed the audit must succeed BEFORE the hold is removed;
        # a sink outage then freezes the hold in place rather than lifting it unaudited.
        self._emit(
            "hold.lifted",
            hold_id=hold_id,
            principal_id=actor,
            corpus_id=str(row["corpus_id"]),
            document_id=(str(row["document_id"]) if row["document_id"] is not None else None),
            capability_name="legal.hold.lift",
            fail_closed=self._audit_fail_closed,
        )
        # Conditional UPDATE (WHERE lifted_at IS NULL): returns True only when it actually lifted a
        # row, so a lost double-lift returns False. A redundant already-emitted hold.lifted from the
        # loser dedup-collapses on a dedup_id-keyed sink (dedup_id folds hold_id).
        lifted = self._store.lift_legal_hold(
            self._workspace_id, hold_id, lifted_by=actor, lifted_at=self._store.clock.now()
        )
        if lifted and self._deferred_processor is not None:
            # Guarded so a deferred-processing error never masks the true lift result the caller sees.
            try:
                self._deferred_processor(self._workspace_id, str(row["corpus_id"]))
            except Exception:
                logger.warning("deferred_erasure_processing_failed", hold_id=hold_id)
        return lifted

    def list_active(self, corpus_id: str | None = None) -> list[LegalHoldRecord]:
        rows = self._store.list_legal_holds(self._workspace_id, corpus_id, active_only=True)
        return [LegalHoldRecord.from_row(row) for row in rows]

    def _emit(
        self,
        event_type: str,
        *,
        hold_id: str,
        principal_id: str,
        corpus_id: str,
        document_id: str | None,
        capability_name: str,
        fail_closed: bool,
    ) -> None:
        if self._audit_sink is None:
            return
        # Redacted by construction: ids, the capability, timestamps. The human reason is never passed
        # here (row-only), mirroring ApprovalService.grant's break-glass reason handling.
        event = AuditEvent(
            event_type=event_type,
            occurred_at=self._store.clock.now(),
            workspace_id=self._workspace_id,
            corpus_id=corpus_id,
            principal_id=principal_id,
            principal_kind="human",
            tenant=None,
            policy_id="legal_hold",
            policy_version="1",
            cited_document_ids=((document_id,) if document_id is not None else ()),
            capability_name=capability_name,
            dedup_id=stable_digest((event_type, self._workspace_id, hold_id)),
        )
        emit_audit(
            self._audit_sink,
            event,
            fail_closed=fail_closed,
            logger=logger,
            log_key="legal_hold_audit_sink_failed",
        )


__all__ = ["LegalHoldService"]
