from __future__ import annotations

import hashlib
import inspect
from collections.abc import Iterable

import httpx

from refaat.domain.egress import (
    EgressDestination,
    EgressPolicy,
    EgressPurpose,
    ensure_egress_allowed,
)
from refaat.domain.models import JsonObject
from refaat.errors import DependencyUnavailableError
from refaat.infrastructure.database.store import SQLStore
from refaat.infrastructure.egress_client import build_http_client
from refaat.observability import get_logger
from refaat.ports import EventSink
from refaat.security import validate_outbound_url

logger = get_logger(__name__)

OUTBOX_LEASE_SECONDS = 300


class LoggingEventSink:
    sink_name = "logging"

    def emit(self, topic: str, payload: JsonObject, *, event_id: str | None = None) -> None:
        # The raw payload may carry ids or free text, so log only the topic, the event id, and the
        # field count; the payload dict itself never reaches the log.
        logger.info("outbox_event", topic=topic, event_id=event_id, payload_field_count=len(payload))


class WebhookEventSink:
    def __init__(
        self,
        url: str,
        secret: str,
        *,
        allow_private_hosts: bool = False,
        client: httpx.Client | None = None,
        egress_policy: EgressPolicy | None = None,
    ) -> None:
        validate_outbound_url(url, allow_private=allow_private_hosts)
        if egress_policy is not None:
            ensure_egress_allowed(egress_policy, EgressDestination.from_url(url, EgressPurpose.WEBHOOK))
        self.url = url
        # A stable per-sink identity derived from the destination, so two webhook sinks
        # to different URLs keep distinct ledger keys and reordering them across a
        # restart never shifts which sink a delivery record belongs to.
        self.sink_name = f"webhook:{hashlib.sha256(url.encode()).hexdigest()[:16]}"
        self.secret = secret
        self._owns_client = client is None
        self.client = build_http_client(
            timeout=15.0, egress_policy=egress_policy, purpose=EgressPurpose.WEBHOOK, client=client
        )

    def emit(self, topic: str, payload: JsonObject, *, event_id: str | None = None) -> None:
        try:
            response = self.client.post(
                self.url,
                headers={"Authorization": f"Bearer {self.secret}"},
                json={"id": event_id, "topic": topic, "payload": payload},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise DependencyUnavailableError("Webhook event delivery failed") from exc

    def close(self) -> None:
        if self._owns_client:
            self.client.close()


class OutboxPublisher:
    def __init__(
        self,
        store: SQLStore,
        sinks: Iterable[EventSink],
        *,
        worker_id: str = "outbox-publisher",
        lease_seconds: int = OUTBOX_LEASE_SECONDS,
    ) -> None:
        self.store = store
        self.sinks = tuple(sinks)
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self._accepts_event_id = tuple(self._sink_accepts_event_id(sink) for sink in self.sinks)
        # Stable per-sink names for the persistent delivery ledger, keyed on each
        # sink's own identity rather than its position in the list, so inserting or
        # reordering a configured sink across a restart never shifts the ledger
        # mapping and a redelivery still skips a sink already recorded as delivered.
        self._sink_names = self._stable_sink_names(self.sinks)

    @staticmethod
    def _stable_sink_names(sinks: tuple[EventSink, ...]) -> tuple[str, ...]:
        names: list[str] = []
        occurrences: dict[str, int] = {}
        for sink in sinks:
            base = str(getattr(sink, "sink_name", None) or type(sink).__name__)
            seen = occurrences.get(base, 0)
            occurrences[base] = seen + 1
            # Distinct-identity sinks are already stable; only indistinguishable
            # same-identity duplicates need a deterministic suffix, and those are
            # interchangeable so the occurrence order is a safe tie-break.
            names.append(base if seen == 0 else f"{base}#{seen}")
        return tuple(names)

    @staticmethod
    def _sink_accepts_event_id(sink: EventSink) -> bool:
        try:
            signature = inspect.signature(sink.emit)
        except (TypeError, ValueError):
            return False
        for name, parameter in signature.parameters.items():
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                return True
            # A positional-only event_id cannot be passed by keyword: the keyword call
            # would raise TypeError, so accept the parameter only when it is
            # keyword-compatible.
            if name == "event_id" and parameter.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                return True
        return False

    def publish(self, limit: int = 100) -> int:
        delivered = 0
        for event in self.store.claim_outbox(self.worker_id, self.lease_seconds, limit):
            event_id = str(event["id"])
            topic = str(event["topic"])
            payload = event["payload"]
            done = self.store.delivered_outbox_sinks(event_id)
            last_error: Exception | None = None
            for sink, accepts_event_id, name in zip(self.sinks, self._accepts_event_id, self._sink_names, strict=True):
                if name in done:
                    continue
                try:
                    if accepts_event_id:
                        sink.emit(topic, payload, event_id=event_id)
                    else:
                        sink.emit(topic, payload)
                except Exception as exc:
                    # Isolate each sink: one failure must not skip the later sinks (a
                    # failing custom sink must never starve the reconcile sink). A
                    # succeeded sink is recorded in the durable ledger, so a retry
                    # after a restart or lost lease never re-invokes it.
                    last_error = exc
                    continue
                self.store.record_outbox_delivery(event_id, name)
                done.add(name)
            if not set(self._sink_names) <= done:
                delay = min(3600, 2 ** min(int(event["attempts"]), 10))
                self.store.mark_outbox_failed(event_id, self.worker_id, str(last_error), delay)
                continue
            if self.store.mark_outbox_delivered(event_id, self.worker_id):
                delivered += 1
        return delivered
