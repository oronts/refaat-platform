"""Byte-identity: with no hold rows the legal-hold slice leaves outcomes unchanged.

A default runtime keeps the management surface off, opens no Postgres lock engine on SQLite, and
leaves the pinned artifact id unchanged. The R3 wiring order is asserted here too: IngestionService
receives the wrapped local audit_sink under durable audit.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import func, select

from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.config import Settings
from refaat.domain.audit import AuditEvent
from refaat.domain.legal_hold import LegalHoldError
from refaat.domain.models import DocumentInput
from refaat.infrastructure.audit import AuditPublisher, DurableAuditSink
from refaat.infrastructure.database.schema import legal_holds
from refaat.programs.program import GroundedRagProgram

pytestmark = pytest.mark.integration

PINNED_GROUNDED_RAG_ID = "sha256:b76d73f8d956bcd38ccc9d4c24372b07b387647a189f3a9e300cded186febbb4"
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
    return runtime.ingest(
        ws, corpus, DocumentInput(source_id=source_id, title=source_id, media_type="text/plain", content=body)
    ).document_id


def test_default_runtime_leaves_management_and_lock_engine_off(tmp_path: Path) -> None:
    sink = _RecordingSink()
    runtime = RuntimeBuilder(_settings(tmp_path)).with_audit_sink(sink).build()
    try:
        # The management surface is off and no Postgres lock engine exists on SQLite. Enforcement is
        # unconditional, but with no hold rows every destructive outcome is unchanged.
        assert runtime.store.legal_hold_management_enabled is False
        assert runtime.store._lock_engine is None

        # The destroy lock opens no connection on SQLite (a process-local reentrant lock).
        with runtime.store.document_destroy_lock("ws", "corpus"):
            assert runtime.store._lock_engine is None

        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "handbook")
        doc = _ingest(runtime, ws, corpus, "target", _BODY)
        runtime.ingestion.delete(ws, corpus, doc)
        other = _ingest(runtime, ws, corpus, "other", _BODY + b" other content.")
        runtime.project(workspace="acme", corpus="handbook").erase(other)

        with runtime.store.engine.connect() as conn:
            rows = conn.execute(select(func.count()).select_from(legal_holds)).scalar_one()
        assert rows == 0
        assert [event for event in sink.events if event.event_type == "hold.enforced"] == []
    finally:
        runtime.close()


def test_pinned_grounded_rag_artifact_id_is_unchanged() -> None:
    assert GroundedRagProgram().artifact().artifact_id == PINNED_GROUNDED_RAG_ID


def test_durable_audit_wires_the_wrapped_sink_into_ingestion_and_delivers_hold_enforced(tmp_path: Path) -> None:
    sink = _RecordingSink()
    runtime = (
        RuntimeBuilder(_settings(tmp_path, audit_durable=True)).with_legal_hold_registry().with_audit_sink(sink).build()
    )
    try:
        # R3: IngestionService received the wrapped local audit_sink (the DurableAuditSink), the same
        # object runtime.audit_sink exposes, not the un-wrapped self._audit_sink.
        assert runtime.ingestion.audit_sink is runtime.audit_sink
        assert isinstance(runtime.ingestion.audit_sink, DurableAuditSink)

        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "handbook")
        doc = _ingest(runtime, ws, corpus, "target", _BODY)
        runtime.project(workspace="acme", corpus="handbook").legal_holds.place(
            corpus_id=corpus, document_id=doc, placed_by="counsel"
        )
        with pytest.raises(LegalHoldError):
            runtime.ingestion.delete(ws, corpus, doc)

        # The refusal event persisted to the outbox and delivers through the durable path.
        AuditPublisher(runtime.store, runtime.audit_delivery_sinks, worker_id="w1").publish()
        assert any(event.event_type == "hold.enforced" for event in sink.events)
    finally:
        runtime.close()
