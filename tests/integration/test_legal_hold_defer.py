"""The opt-in defer conflict policy: an erasure that meets a hold is recorded and executed on lift.

The default refuse policy is byte-identical (a held erase returns HELD and the deferred table stays
empty). Under defer, a held erase records a durable pending request and returns DEFERRED, erasing
nothing; lifting the last hold covering the document executes the request; a document still covered by
another hold is never erased (fail-closed re-check under the destroy lock).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import func, select

from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.config import Settings
from refaat.domain.audit import AuditEvent
from refaat.domain.erasure import ErasureStatus
from refaat.domain.models import DocumentInput
from refaat.infrastructure.database.schema import deferred_erasures, erasure_ledger

pytestmark = pytest.mark.integration

_BODY = b"Payment gateway timeout cascaded into a checkout outage; the rollback restored service."


class _RecordingSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        self.events.append(event)


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {
        "environment": "test",
        "database_url": f"sqlite:///{tmp_path / 'refaat.db'}",
        "auto_create_schema": True,
        "blob_root": tmp_path / "blobs",
        "connector_file_root": tmp_path / "connectors",
        "skill_root": tmp_path / "skills",
        "api_key_pepper": "test-pepper-with-enough-entropy",
        "bootstrap_token": "bootstrap-test-token",
        "default_chunk_tokens": 48,
        "chunk_overlap_tokens": 8,
        "log_json": False,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _ingest(runtime: Refaat, ws: str, corpus: str, source_id: str, body: bytes) -> str:
    result = runtime.ingest(
        ws, corpus, DocumentInput(source_id=source_id, title=source_id, media_type="text/plain", content=body)
    )
    return result.document_id


def _pending_count(runtime: Refaat) -> int:
    with runtime.store.engine.connect() as conn:
        return int(
            conn.execute(
                select(func.count()).select_from(deferred_erasures).where(deferred_erasures.c.status == "pending")
            ).scalar_one()
        )


def _ledger_status(runtime: Refaat, document_id: str) -> list[str]:
    with runtime.store.engine.connect() as conn:
        return [
            str(row[0])
            for row in conn.execute(
                select(erasure_ledger.c.status).where(erasure_ledger.c.document_id == document_id)
            ).all()
        ]


@pytest.fixture
def defer_runtime(tmp_path: Path) -> Iterator[tuple[Refaat, _RecordingSink]]:
    sink = _RecordingSink()
    runtime = (
        RuntimeBuilder(_settings(tmp_path, legal_hold_on_conflict="defer"))
        .with_legal_hold_registry()
        .with_audit_sink(sink)
        .build()
    )
    try:
        yield runtime, sink
    finally:
        runtime.close()


def test_refuse_default_is_byte_identical_and_leaves_the_deferred_table_empty(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_settings(tmp_path)).with_legal_hold_registry().build()
    try:
        assert runtime.erasure.on_conflict == "refuse"
        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "handbook")
        doc = _ingest(runtime, ws, corpus, "target", _BODY)
        project = runtime.project(workspace="acme", corpus="handbook")
        project.legal_holds.place(corpus_id=corpus, document_id=doc, placed_by="counsel")

        assert project.erase(doc).status is ErasureStatus.HELD
        assert runtime.store.list_document_chunks(ws, corpus, doc)
        assert _pending_count(runtime) == 0
        assert runtime.erasure.deferred_processor is None
    finally:
        runtime.close()


def test_defer_records_a_pending_request_and_erases_nothing(defer_runtime: tuple[Refaat, _RecordingSink]) -> None:
    runtime, sink = defer_runtime
    ws = runtime.create_workspace("acme", "Acme")
    corpus = runtime.create_corpus(ws, "handbook")
    doc = _ingest(runtime, ws, corpus, "target", _BODY)
    project = runtime.project(workspace="acme", corpus="handbook")
    project.legal_holds.place(corpus_id=corpus, document_id=doc, placed_by="counsel", reason="SECRETMATTER99")

    receipt = project.erase(doc)
    assert receipt.status is ErasureStatus.DEFERRED
    assert receipt.chunks == 0
    assert receipt.blob_deleted is False
    # The document is intact under the hold.
    assert runtime.store.list_document_chunks(ws, corpus, doc)
    assert _pending_count(runtime) == 1
    assert "deferred" in _ledger_status(runtime, doc)

    deferred = [event for event in sink.events if event.event_type == "erasure.deferred"]
    assert len(deferred) == 1
    assert deferred[0].cited_document_ids == (doc,)
    # The human reason and evidence text never appear in any emitted event.
    for event in sink.events:
        for value in event.to_json().values():
            assert "SECRETMATTER99" not in str(value)
            assert "Payment gateway" not in str(value)


def test_defer_is_idempotent_while_held(defer_runtime: tuple[Refaat, _RecordingSink]) -> None:
    runtime, _sink = defer_runtime
    ws = runtime.create_workspace("acme", "Acme")
    corpus = runtime.create_corpus(ws, "handbook")
    doc = _ingest(runtime, ws, corpus, "target", _BODY)
    project = runtime.project(workspace="acme", corpus="handbook")
    project.legal_holds.place(corpus_id=corpus, document_id=doc, placed_by="counsel")

    assert project.erase(doc).status is ErasureStatus.DEFERRED
    assert project.erase(doc).status is ErasureStatus.DEFERRED
    # A repeated erase of the same held document converges on one pending row.
    assert _pending_count(runtime) == 1


def test_lifting_the_hold_executes_the_deferred_erasure(defer_runtime: tuple[Refaat, _RecordingSink]) -> None:
    runtime, _sink = defer_runtime
    ws = runtime.create_workspace("acme", "Acme")
    corpus = runtime.create_corpus(ws, "handbook")
    doc = _ingest(runtime, ws, corpus, "target", _BODY)
    project = runtime.project(workspace="acme", corpus="handbook")
    hold_id = project.legal_holds.place(corpus_id=corpus, document_id=doc, placed_by="counsel")

    assert project.erase(doc).status is ErasureStatus.DEFERRED
    assert runtime.store.list_document_chunks(ws, corpus, doc)

    assert project.legal_holds.lift(hold_id, lifted_by="counsel") is True
    # The lift executed the deferred erasure: the document is gone and the request is marked executed.
    assert runtime.store.list_document_chunks(ws, corpus, doc) == []
    assert _pending_count(runtime) == 0
    assert "erased" in _ledger_status(runtime, doc)


def test_a_document_still_covered_by_another_hold_is_not_erased_on_lift(
    defer_runtime: tuple[Refaat, _RecordingSink],
) -> None:
    runtime, _sink = defer_runtime
    ws = runtime.create_workspace("acme", "Acme")
    corpus = runtime.create_corpus(ws, "handbook")
    doc = _ingest(runtime, ws, corpus, "target", _BODY)
    project = runtime.project(workspace="acme", corpus="handbook")
    doc_hold = project.legal_holds.place(corpus_id=corpus, document_id=doc, placed_by="counsel")
    corpus_hold = project.legal_holds.place(corpus_id=corpus, placed_by="counsel")

    assert project.erase(doc).status is ErasureStatus.DEFERRED

    # Lifting one of two covering holds must not erase: the corpus-wide hold still covers the document.
    project.legal_holds.lift(doc_hold, lifted_by="counsel")
    assert runtime.store.list_document_chunks(ws, corpus, doc)
    assert _pending_count(runtime) == 1

    # Lifting the last covering hold executes the deferred erasure.
    project.legal_holds.lift(corpus_hold, lifted_by="counsel")
    assert runtime.store.list_document_chunks(ws, corpus, doc) == []
    assert _pending_count(runtime) == 0


def test_process_deferred_is_directly_callable_and_fail_closed_while_held(
    defer_runtime: tuple[Refaat, _RecordingSink],
) -> None:
    runtime, _sink = defer_runtime
    ws = runtime.create_workspace("acme", "Acme")
    corpus = runtime.create_corpus(ws, "handbook")
    doc = _ingest(runtime, ws, corpus, "target", _BODY)
    project = runtime.project(workspace="acme", corpus="handbook")
    project.legal_holds.place(corpus_id=corpus, document_id=doc, placed_by="counsel")
    assert project.erase(doc).status is ErasureStatus.DEFERRED

    # Calling the processor directly while the hold is still active erases nothing and leaves it pending.
    receipts = runtime.erasure.process_deferred(ws, corpus)
    assert all(receipt.status is ErasureStatus.DEFERRED for receipt in receipts)
    assert runtime.store.list_document_chunks(ws, corpus, doc)
    assert _pending_count(runtime) == 1


def test_lift_through_the_http_and_cli_factories_executes_the_deferred_erasure(
    defer_runtime: tuple[Refaat, _RecordingSink],
) -> None:
    # The HTTP route and the CLI build the service through their own factories, so those must wire the
    # deferred processor too, or a hold lifted through the primary operator surfaces strands the request.
    from refaat.api.routes import _legal_hold_service as api_factory
    from refaat.cli import _legal_hold_service as cli_factory

    runtime, _sink = defer_runtime
    ws = runtime.create_workspace("acme", "Acme")
    corpus = runtime.create_corpus(ws, "handbook")
    project = runtime.project(workspace="acme", corpus="handbook")
    for factory, source in ((api_factory, "api"), (cli_factory, "cli")):
        doc = _ingest(runtime, ws, corpus, f"target-{source}", _BODY)
        service = factory(runtime, ws)
        hold_id = service.place(corpus_id=corpus, document_id=doc, placed_by="counsel")
        assert project.erase(doc).status is ErasureStatus.DEFERRED
        assert service.lift(hold_id, lifted_by="counsel") is True
        assert runtime.store.list_document_chunks(ws, corpus, doc) == [], source


def test_the_sweep_re_drives_a_stranded_pending_erasure_but_respects_active_holds(
    defer_runtime: tuple[Refaat, _RecordingSink],
) -> None:
    from refaat.application.worker import DurableWorker

    runtime, _sink = defer_runtime
    ws = runtime.create_workspace("acme", "Acme")
    corpus = runtime.create_corpus(ws, "handbook")
    project = runtime.project(workspace="acme", corpus="handbook")

    stranded = _ingest(runtime, ws, corpus, "stranded", _BODY)
    held = _ingest(runtime, ws, corpus, "held", _BODY)
    stranded_hold = project.legal_holds.place(corpus_id=corpus, document_id=stranded, placed_by="counsel")
    project.legal_holds.place(corpus_id=corpus, document_id=held, placed_by="counsel")
    assert project.erase(stranded).status is ErasureStatus.DEFERRED
    assert project.erase(held).status is ErasureStatus.DEFERRED

    # Strand the first request: lift its hold at the store level, bypassing the synchronous processor,
    # so it is pending with no active hold and no future lift to complete it.
    runtime.store.lift_legal_hold(ws, stranded_hold, lifted_by="counsel", lifted_at=runtime.store.clock.now())
    assert _pending_count(runtime) == 2

    worker = DurableWorker(runtime)
    assert "deferred_erasure.sweep" in worker.handlers
    result = worker._sweep_deferred_erasures("", {})

    assert result["deferred_erasures_executed"] == 1
    assert runtime.store.list_document_chunks(ws, corpus, stranded) == []
    assert runtime.store.list_document_chunks(ws, corpus, held)
    assert _pending_count(runtime) == 1
