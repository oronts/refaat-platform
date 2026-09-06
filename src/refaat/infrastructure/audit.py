"""Durable at-least-once delivery for governance audit events.

When durable audit is enabled the runtime installs a DurableAuditSink as the emission sink, so a
query, RLM, or approval emission persists the event to the audit_outbox instead of delivering it
inline. A worker runs the AuditPublisher, which claims persisted events, reconstructs each one, and
delivers it to the real configured sink with a bounded retry backoff and a per-sink ledger, so a
transient DELIVERY failure never loses an enqueued event and a redelivery never doubles a succeeded
sink. The delivery guarantee is scoped to enqueued events: by default emission persists best-effort
(a failure to insert the audit_outbox row is logged and swallowed like the inline path). Enabling
audit_fail_closed alongside audit_durable closes the last gap: a failed enqueue then raises on the
synchronous path so the operation refuses rather than proceeding un-enqueued, giving the strict
never-lost-before-enqueue guarantee for the query, RLM, and approval paths.

For the durable-approval consume and cancel paths the enqueue is written in the SAME transaction as
the state change, through the store's consume-with-audit and cancel factory seams, so a consumed or
cancelled approval and its audit record commit together or not at all. That closes the
persist-before-audit window outright for those paths rather than only raising after the state already
committed, which is why an approval cannot be consumed without its audit record under a durable sink.
"""

from __future__ import annotations

from collections.abc import Iterable

from refaat.domain.audit import AuditEvent, AuditSink, audit_event_from_json
from refaat.infrastructure.database.store import SQLStore
from refaat.observability import get_logger

logger = get_logger(__name__)

AUDIT_OUTBOX_LEASE_SECONDS = 300

# Matches the audit_outbox_deliveries.sink_name column width, so a ledger write never truncates.
LEDGER_SINK_NAME_MAX = 200


class DurableAuditSink:
    """An AuditSink that persists each event to the audit_outbox rather than delivering it.

    The AuditPublisher does the actual delivery to the real sink. record() only persists, so an
    emission on the query, RLM, or approval path is durable before the request returns.
    """

    def __init__(self, store: SQLStore) -> None:
        self._store = store

    def record(self, event: AuditEvent) -> None:
        self._store.enqueue_audit_event(event.workspace_id, event)


class AuditPublisher:
    """Delivers persisted audit events to the real configured sinks with retries and a ledger."""

    def __init__(
        self,
        store: SQLStore,
        sinks: Iterable[AuditSink],
        *,
        worker_id: str = "audit-publisher",
        lease_seconds: int = AUDIT_OUTBOX_LEASE_SECONDS,
    ) -> None:
        self.store = store
        self.sinks = tuple(sinks)
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        # Ledger key per sink, from its sink_name attribute or its class name. The keys must be
        # distinct so the mapping is stable across a restart or a config reorder (a positional
        # suffix would shift when sinks are reordered); a collision is an ambiguous configuration
        # and fails fast rather than risk re-delivering or skipping a sink after a reorder. A single
        # delivery sink, the shipped configuration, never collides.
        self._sink_names = self._stable_sink_names(self.sinks)

    @staticmethod
    def _stable_sink_names(sinks: tuple[AuditSink, ...]) -> tuple[str, ...]:
        names = tuple(str(getattr(sink, "sink_name", None) or type(sink).__name__) for sink in sinks)
        if any(len(name) > LEDGER_SINK_NAME_MAX for name in names):
            raise ValueError(f"A durable audit delivery sink_name exceeds {LEDGER_SINK_NAME_MAX} characters")
        if len(set(names)) != len(names):
            raise ValueError("Durable audit delivery sinks need distinct sink_name values; the ledger keys on them")
        return names

    def publish(self, limit: int = 100) -> int:
        delivered = 0
        for event in self.store.claim_audit_outbox(self.worker_id, self.lease_seconds, limit):
            event_id = str(event["id"])
            audit_event = audit_event_from_json(dict(event["payload"]))
            done = self.store.delivered_audit_sinks(event_id)
            last_error: Exception | None = None
            for sink, name in zip(self.sinks, self._sink_names, strict=True):
                if name in done:
                    continue
                try:
                    sink.record(audit_event)
                except Exception as exc:
                    # Isolate each sink: one failure must not skip the later sinks, and a
                    # succeeded sink is recorded in the durable ledger so a retry never
                    # re-invokes it. The row stays undelivered and is retried.
                    last_error = exc
                    continue
                self.store.record_audit_delivery(event_id, name)
                done.add(name)
            if not set(self._sink_names) <= done:
                delay = min(3600, 2 ** min(int(event["attempts"]), 10))
                self.store.mark_audit_failed(event_id, self.worker_id, str(last_error), delay)
                continue
            if self.store.mark_audit_delivered(event_id, self.worker_id):
                delivered += 1
        return delivered
