from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from _shared import demo_settings, emit
from sqlalchemy import select

from refaat.application.runtime import Refaat, RefaatProject, RuntimeBuilder
from refaat.domain.audit import AuditEvent
from refaat.domain.authz import CanonicalAccessPolicy, Principal
from refaat.domain.models import DocumentInput
from refaat.infrastructure.audit import AuditPublisher
from refaat.infrastructure.database.schema import audit_outbox

QUERY = "how is the production release rollback plan approved"

# A distinctive token that appears only in the document body, so its absence from every audit
# event field proves no evidence text leaked into the redacted event.
BODY_MARKER = "peridot"

DOCUMENT = DocumentInput(
    source_id="runbook.md",
    title="Production runbook",
    media_type="text/markdown",
    content=(
        b"The production release rollback plan requires an approved sign-off. "
        b"The rollback control is codenamed peridot and gates every deploy."
    ),
    metadata={"acl": {"classification": "public"}},
)

ALICE = Principal(id="alice", tenant="acme")
APPROVER = "bob"
SECRETS = (QUERY, BODY_MARKER)


class CapturingSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        self.events.append(event)


def _event_strings(event: AuditEvent) -> list[str]:
    strings: list[str] = []
    for value in event.to_json().values():
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, list):
            strings.extend(str(item) for item in value)
    return strings


def _no_secret(events: list[AuditEvent]) -> bool:
    return all(secret not in text for event in events for text in _event_strings(event) for secret in SECRETS)


def _seed(runtime: Refaat) -> RefaatProject:
    project = runtime.project(workspace="acme", corpus="ops")
    runtime.ingest(project.workspace_id, project.corpus_id, DOCUMENT)
    return project


def _audit_rows(runtime: Refaat) -> list[dict[str, object]]:
    with runtime.store.engine.connect() as conn:
        return [dict(row) for row in conn.execute(select(audit_outbox)).mappings().all()]


def main() -> None:
    with TemporaryDirectory(prefix="refaat-audit-") as directory:
        root = Path(directory)
        inline_root = root / "inline"
        durable_root = root / "durable"
        inline_root.mkdir()
        durable_root.mkdir()

        # A policy plus a principal make the served query emit a redacted query.authorized event,
        # and the approval lifecycle emits its own granted and cancelled events to the same sink.
        sink = CapturingSink()
        inline_runtime = (
            RuntimeBuilder(demo_settings(inline_root))
            .with_access_policy(CanonicalAccessPolicy())
            .with_audit_sink(sink)
            .build()
        )
        try:
            project = _seed(inline_runtime)
            project.ask(QUERY, principal=ALICE)

            fingerprint = hashlib.sha256(b"deploy production release").hexdigest()
            approval_id = project.approvals.grant(
                fingerprint=fingerprint, capability_name="deploy", approver_id=APPROVER
            )
            project.approvals.cancel(approval_id, canceller_id=APPROVER)

            query_events = [event for event in sink.events if event.event_type == "query.authorized"]
            grant_events = [event for event in sink.events if event.event_type == "approval.granted"]
            cancel_events = [event for event in sink.events if event.event_type == "approval.cancelled"]

            query_emitted = len(query_events) == 1
            query_redacted = query_emitted and _no_secret(query_events)
            granted_ok = len(grant_events) == 1 and grant_events[0].capability_name == "deploy"
            cancelled_ok = len(cancel_events) == 1 and cancel_events[0].capability_name == "deploy"
            inline_no_secret = _no_secret(sink.events)
        finally:
            inline_runtime.close()

        # With durable audit on, a served query persists the event to the audit_outbox instead of
        # delivering it inline; the worker publisher delivers it to the real sink exactly once.
        real_sink = CapturingSink()
        durable_settings = demo_settings(durable_root).model_copy(update={"audit_durable": True})
        durable_runtime = (
            RuntimeBuilder(durable_settings)
            .with_access_policy(CanonicalAccessPolicy())
            .with_audit_sink(real_sink)
            .build()
        )
        try:
            durable_project = _seed(durable_runtime)
            durable_project.ask(QUERY, principal=ALICE)

            rows_before_delivery = _audit_rows(durable_runtime)
            persisted_not_inline = real_sink.events == [] and len(rows_before_delivery) == 1

            publisher = AuditPublisher(durable_runtime.store, durable_runtime.audit_delivery_sinks)
            first_delivered = publisher.publish()
            after_first = len(real_sink.events)
            second_delivered = publisher.publish()
            after_second = len(real_sink.events)

            delivered_row = _audit_rows(durable_runtime)[0]
            delivered_once = (
                first_delivered == 1
                and second_delivered == 0
                and after_first == after_second == 1
                and delivered_row["delivered_at"] is not None
            )
            durable_no_secret = _no_secret(real_sink.events)
        finally:
            durable_runtime.close()

        report: dict[str, Any] = {
            "mode": "deterministic",
            "query_event_emitted": query_emitted,
            "query_event_is_redacted": query_redacted,
            "approval_granted_event": granted_ok,
            "approval_cancelled_event": cancelled_ok,
            "durable_persists_not_inline": persisted_not_inline,
            "durable_delivers_once": delivered_once,
            "no_secret_in_any_event": inline_no_secret and durable_no_secret,
        }
        assert all(value for key, value in report.items() if key != "mode"), report

        emit(report)


if __name__ == "__main__":
    main()
