"""The workspace-scoped kill-switch management service.

Enforcement of an engaged switch is state-driven: a runtime with a kill-switch registry installed
denies the guarded operations regardless of this service. KillSwitchService engages, disengages, and
lists switches, and a write fails closed when the management surface is not enabled on this runtime,
so a switch is only ever recorded by a runtime that deliberately opted into managing them.
"""

from __future__ import annotations

from refaat.core.text import stable_digest
from refaat.domain.audit import AuditEvent, AuditSink, emit_audit
from refaat.domain.kill_switch import KillSwitch, KillSwitchScope
from refaat.domain.legal_hold import validate_actor_id, validate_hold_reason
from refaat.errors import ConfigurationError
from refaat.infrastructure.database.store import SQLStore
from refaat.observability import get_logger

logger = get_logger(__name__)

_NOT_ENABLED = (
    "the kill-switch management surface is not enabled on this runtime; "
    "wire with_kill_switch_registry() to manage switches"
)

# The scopes whose enforcement seams are wired. Engaging a scope with no seam would be a silent
# fail-open, so the management surface refuses it until its seam lands. EXTERNAL_EGRESS remains gated
# until its seam (the connector and remote-tool layer) lands.
_ENFORCED_SCOPES = frozenset(
    {
        KillSwitchScope.WORKSPACE_FREEZE,
        KillSwitchScope.GENERATION,
        KillSwitchScope.WRITES,
        KillSwitchScope.TOOL_EXECUTION,
    }
)


class KillSwitchService:
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

    def engage(self, *, scope: KillSwitchScope, engaged_by: str, reason: str | None = None) -> str:
        if not self._store.kill_switch_management_enabled:
            raise ConfigurationError(_NOT_ENABLED)
        if scope not in _ENFORCED_SCOPES:
            raise ConfigurationError(f"kill-switch scope not enforced in this build: {scope.value}")
        actor = validate_actor_id(engaged_by)
        stored_reason = validate_hold_reason(reason)
        # Insert-then-emit. Engaging is protection-increasing, so the protective row must land even
        # under a degraded sink; the audit follows best-effort.
        switch_id = self._store.engage_kill_switch(
            self._workspace_id,
            scope=scope.value,
            engaged_by=actor,
            reason=stored_reason,
            engaged_at=self._store.clock.now(),
        )
        self._emit("killswitch.engaged", switch_id=switch_id, scope=scope, actor=actor, fail_closed=False)
        return switch_id

    def disengage(self, *, scope: KillSwitchScope, disengaged_by: str) -> bool:
        if not self._store.kill_switch_management_enabled:
            raise ConfigurationError(_NOT_ENABLED)
        actor = validate_actor_id(disengaged_by)
        row = self._store.get_active_kill_switch(self._workspace_id, scope.value)
        if row is None:
            return False
        # Emit-then-disengage under fail_closed. Disengaging is protection-decreasing, so under
        # audit_fail_closed the audit must land BEFORE the switch clears; a sink outage then freezes
        # the switch on rather than lifting it unaudited.
        self._emit(
            "killswitch.disengaged",
            switch_id=str(row["id"]),
            scope=scope,
            actor=actor,
            fail_closed=self._audit_fail_closed,
        )
        return self._store.disengage_kill_switch(
            self._workspace_id, scope=scope.value, disengaged_by=actor, disengaged_at=self._store.clock.now()
        )

    def list_engaged(self) -> list[KillSwitch]:
        rows = self._store.list_kill_switches(self._workspace_id, active_only=True)
        return [KillSwitch.from_row(row) for row in rows]

    def _emit(
        self,
        event_type: str,
        *,
        switch_id: str,
        scope: KillSwitchScope,
        actor: str,
        fail_closed: bool,
    ) -> None:
        if self._audit_sink is None:
            return
        # Redacted by construction: ids, the scope as a fixed enum value, the actor, timestamps. The
        # human reason is never passed here (row-only), mirroring the legal-hold and break-glass reason.
        event = AuditEvent(
            event_type=event_type,
            occurred_at=self._store.clock.now(),
            workspace_id=self._workspace_id,
            principal_id=actor,
            principal_kind="human",
            tenant=None,
            policy_id="kill_switch",
            policy_version="1",
            capability_name=f"killswitch.{scope.value}",
            action_fingerprint=stable_digest((self._workspace_id, scope.value, switch_id)),
            dedup_id=stable_digest((event_type, self._workspace_id, scope.value, switch_id)),
        )
        emit_audit(
            self._audit_sink,
            event,
            fail_closed=fail_closed,
            logger=logger,
            log_key="kill_switch_audit_sink_failed",
        )


__all__ = ["KillSwitchService"]
