"""Durable at-least-once audit delivery: persist on emission, deliver with retries and a ledger."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta

import pytest
from sqlalchemy import select, update

from refaat.application.runtime import Refaat, RefaatProject, RuntimeBuilder
from refaat.config import Settings
from refaat.domain.audit import AuditEvent
from refaat.domain.authz import CanonicalAccessPolicy, Principal
from refaat.domain.models import DocumentInput
from refaat.infrastructure.audit import AuditPublisher
from refaat.infrastructure.database.schema import audit_outbox

pytestmark = pytest.mark.integration

ALICE = Principal(id="alice", tenant="merck", groups=frozenset({"research-a"}))
QUERY = "checkout production release rollback plan"


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        self.events.append(event)


class FailOnceSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []
        self._failed = False

    def record(self, event: AuditEvent) -> None:
        if not self._failed:
            self._failed = True
            raise RuntimeError("sink down")
        self.events.append(event)


def _ingest(project: RefaatProject) -> None:
    project.runtime.ingest(
        project.workspace_id,
        project.corpus_id,
        DocumentInput(
            source_id="alpha.md",
            title="alpha.md",
            media_type="text/markdown",
            content=b"The alpha checkout production release requires an approved rollback plan.",
            metadata={"acl": {"classification": "confidential", "allow_groups": ["research-a"]}},
        ),
    )


def _durable_settings(settings: Settings) -> Settings:
    return settings.model_copy(update={"audit_durable": True})


def _audit_rows(runtime: Refaat) -> list[dict[str, object]]:
    with runtime.store.engine.connect() as conn:
        return [dict(row) for row in conn.execute(select(audit_outbox)).mappings().all()]


def _force_audit_available(runtime: Refaat) -> None:
    past = runtime.store.clock.now() - timedelta(hours=1)
    with runtime.store.engine.begin() as conn:
        conn.execute(
            update(audit_outbox)
            .where(audit_outbox.c.delivered_at.is_(None))
            .values(available_at=past, claimed_by=None, claimed_until=None)
        )


@pytest.fixture
def durable(settings: Settings) -> Iterator[tuple[RefaatProject, RecordingSink, AuditPublisher]]:
    sink = RecordingSink()
    runtime = (
        RuntimeBuilder(_durable_settings(settings))
        .with_access_policy(CanonicalAccessPolicy())
        .with_audit_sink(sink)
        .build()
    )
    try:
        project = runtime.project(workspace="merck", corpus="research")
        _ingest(project)
        publisher = AuditPublisher(runtime.store, runtime.audit_delivery_sinks, worker_id="w1")
        yield project, sink, publisher
    finally:
        runtime.close()


def test_durable_emission_persists_and_delivers_exactly_once(
    durable: tuple[RefaatProject, RecordingSink, AuditPublisher],
) -> None:
    project, sink, publisher = durable
    project.ask(QUERY, principal=ALICE)

    # Emission persists the event but does not deliver it inline.
    assert sink.events == []
    rows = _audit_rows(project.runtime)
    assert len(rows) == 1
    assert rows[0]["delivered_at"] is None

    assert publisher.publish() == 1
    assert len(sink.events) == 1
    assert sink.events[0].event_type == "query.authorized"

    # A second publish must not re-deliver: the per-sink ledger records the success.
    assert publisher.publish() == 0
    assert len(sink.events) == 1
    assert _audit_rows(project.runtime)[0]["delivered_at"] is not None


def test_a_raising_sink_leaves_the_event_undelivered_and_retried(settings: Settings) -> None:
    sink = FailOnceSink()
    runtime = (
        RuntimeBuilder(_durable_settings(settings))
        .with_access_policy(CanonicalAccessPolicy())
        .with_audit_sink(sink)
        .build()
    )
    try:
        project = runtime.project(workspace="merck", corpus="research")
        _ingest(project)
        publisher = AuditPublisher(runtime.store, runtime.audit_delivery_sinks, worker_id="w1")
        project.ask(QUERY, principal=ALICE)

        # First publish: the sink raises, so the event is not lost and stays undelivered.
        assert publisher.publish() == 0
        assert sink.events == []
        row = _audit_rows(runtime)[0]
        assert row["delivered_at"] is None
        assert int(row["attempts"]) == 1

        # It is retried: once available again the working sink delivers it exactly once.
        _force_audit_available(runtime)
        assert publisher.publish() == 1
        assert len(sink.events) == 1
        assert _audit_rows(runtime)[0]["delivered_at"] is not None
    finally:
        runtime.close()


def test_audit_durable_off_delivers_directly_and_writes_no_outbox_row(settings: Settings) -> None:
    sink = RecordingSink()
    runtime = RuntimeBuilder(settings).with_access_policy(CanonicalAccessPolicy()).with_audit_sink(sink).build()
    try:
        project = runtime.project(workspace="merck", corpus="research")
        _ingest(project)
        assert runtime.audit_delivery_sinks == ()
        project.ask(QUERY, principal=ALICE)

        assert len(sink.events) == 1
        assert sink.events[0].event_type == "query.authorized"
        assert _audit_rows(runtime) == []
    finally:
        runtime.close()


def test_colliding_delivery_sink_names_fail_fast() -> None:
    class _Sink:
        def record(self, event: AuditEvent) -> None:
            del event

    with pytest.raises(ValueError, match="distinct sink_name"):
        AuditPublisher(None, (_Sink(), _Sink()))  # type: ignore[arg-type]

    named_a, named_b = _Sink(), _Sink()
    named_a.sink_name = "prod"  # type: ignore[attr-defined]
    named_b.sink_name = "backup"  # type: ignore[attr-defined]
    publisher = AuditPublisher(None, (named_a, named_b))  # type: ignore[arg-type]
    assert publisher._sink_names == ("prod", "backup")

    long_sink = _Sink()
    long_sink.sink_name = "x" * 201  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="exceeds"):
        AuditPublisher(None, (long_sink,))  # type: ignore[arg-type]
