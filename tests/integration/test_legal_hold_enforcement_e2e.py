"""Legal-hold enforcement over the three destructive paths, with the built-in register active.

The register gates erase (HELD receipt, HTTP 200), delete (raises LegalHoldError, 409), and a
content overwrite (raises, 409). The write path fails closed when the register is not enforced.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import func, select

from refaat.application.legal_hold import LegalHoldService
from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.config import Settings
from refaat.domain.audit import AuditEvent
from refaat.domain.erasure import ErasureStatus
from refaat.domain.legal_hold import LegalHoldError
from refaat.domain.models import DocumentInput
from refaat.errors import AuditUnavailableError, ConfigurationError, NotFoundError, ValidationError
from refaat.infrastructure.database.schema import erasure_ledger, legal_holds

pytestmark = pytest.mark.integration

_BODY = b"Payment gateway timeout cascaded into a checkout outage; the rollback restored service."
_OTHER_BODY = b"Warehouse throughput and carrier routing options are covered by the logistics report."


class _RecordingSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        self.events.append(event)


class _RaisingSink:
    def record(self, event: AuditEvent) -> None:
        del event
        raise RuntimeError("audit sink down")


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


def _ingest(runtime: Refaat, ws: str, corpus: str, source_id: str, body: bytes, metadata: object = None) -> str:
    result = runtime.ingest(
        ws,
        corpus,
        DocumentInput(
            source_id=source_id,
            title=source_id,
            media_type="text/plain",
            content=body,
            metadata=metadata or {},
        ),
    )
    return result.document_id


@pytest.fixture
def registry(tmp_path: Path) -> Iterator[tuple[Refaat, _RecordingSink]]:
    sink = _RecordingSink()
    runtime = RuntimeBuilder(_settings(tmp_path)).with_legal_hold_registry().with_audit_sink(sink).build()
    try:
        yield runtime, sink
    finally:
        runtime.close()


def test_three_destructive_paths_are_gated_with_the_two_shape_contract(
    registry: tuple[Refaat, _RecordingSink],
) -> None:
    runtime, _sink = registry
    ws = runtime.create_workspace("acme", "Acme")
    corpus = runtime.create_corpus(ws, "handbook")
    doc = _ingest(runtime, ws, corpus, "target", _BODY)
    project = runtime.project(workspace="acme", corpus="handbook")
    project.legal_holds.place(corpus_id=corpus, document_id=doc, placed_by="counsel", reason="matter 14")

    # Erase: HELD receipt (HTTP 200 shape), nothing erased, HELD ledger row.
    receipt = project.erase(doc)
    assert receipt.status is ErasureStatus.HELD
    assert receipt.chunks == 0
    assert receipt.blob_deleted is False
    assert runtime.store.list_document_chunks(ws, corpus, doc)

    # Delete: raises (HTTP 409 shape), index restored (still queryable).
    with pytest.raises(LegalHoldError):
        runtime.ingestion.delete(ws, corpus, doc)
    assert runtime.store.list_document_chunks(ws, corpus, doc)

    # Content overwrite: raises.
    with pytest.raises(LegalHoldError):
        _ingest(runtime, ws, corpus, "target", _BODY + b" NEW CONTENT to force a checksum change.")

    with runtime.store.engine.connect() as conn:
        held_rows = conn.execute(
            select(func.count()).select_from(erasure_ledger).where(erasure_ledger.c.status == ErasureStatus.HELD.value)
        ).scalar_one()
    assert held_rows == 1


def test_same_checksum_retag_is_allowed_but_a_content_change_is_refused(
    registry: tuple[Refaat, _RecordingSink],
) -> None:
    runtime, _sink = registry
    ws = runtime.create_workspace("acme", "Acme")
    corpus = runtime.create_corpus(ws, "handbook")
    doc = _ingest(runtime, ws, corpus, "target", _BODY, metadata={"acl": {"classification": "INTERNAL"}})
    project = runtime.project(workspace="acme", corpus="handbook")
    project.legal_holds.place(corpus_id=corpus, document_id=doc, placed_by="counsel")

    # Same bytes, changed ACL: the no-op fast path is skipped (permission fingerprint changed) and the
    # re-tag is allowed because the checksum is unchanged.
    result = _ingest(runtime, ws, corpus, "target", _BODY, metadata={"acl": {"classification": "CONFIDENTIAL"}})
    assert result == doc

    with pytest.raises(LegalHoldError):
        _ingest(runtime, ws, corpus, "target", _BODY + b" genuinely new content body.")


def test_held_permission_change_is_ledgered_append_only(registry: tuple[Refaat, _RecordingSink]) -> None:
    runtime, _sink = registry
    ws = runtime.create_workspace("acme", "Acme")
    corpus = runtime.create_corpus(ws, "handbook")
    doc = _ingest(runtime, ws, corpus, "target", _BODY, metadata={"acl": {"classification": "INTERNAL"}})
    project = runtime.project(workspace="acme", corpus="handbook")
    hold_id = project.legal_holds.place(corpus_id=corpus, document_id=doc, placed_by="counsel")

    # A held re-tag that changes the ACL is allowed and ledgered append-only with the old and new state.
    _ingest(runtime, ws, corpus, "target", _BODY, metadata={"acl": {"classification": "CONFIDENTIAL"}})
    history = runtime.store.list_hold_permission_history(ws, corpus, doc)
    assert len(history) == 1
    record = history[0]
    assert record["old_acl"]["classification"] == "INTERNAL"
    assert record["new_acl"]["classification"] == "CONFIDENTIAL"
    assert record["old_fingerprint"] != record["new_fingerprint"]
    assert record["actor"] == "system"
    assert hold_id in record["covering_hold_ids"]

    # Append-only: a second permission change adds a second record, never overwriting the first.
    _ingest(runtime, ws, corpus, "target", _BODY, metadata={"acl": {"classification": "RESTRICTED"}})
    assert len(runtime.store.list_hold_permission_history(ws, corpus, doc)) == 2

    # A re-ingest with the same content and ACL takes the idempotency no-op and ledgers nothing.
    _ingest(runtime, ws, corpus, "target", _BODY, metadata={"acl": {"classification": "RESTRICTED"}})
    assert len(runtime.store.list_hold_permission_history(ws, corpus, doc)) == 2


def test_no_permission_history_without_a_hold(registry: tuple[Refaat, _RecordingSink]) -> None:
    runtime, _sink = registry
    ws = runtime.create_workspace("acme", "Acme")
    corpus = runtime.create_corpus(ws, "handbook")
    doc = _ingest(runtime, ws, corpus, "target", _BODY, metadata={"acl": {"classification": "INTERNAL"}})
    _ingest(runtime, ws, corpus, "target", _BODY, metadata={"acl": {"classification": "CONFIDENTIAL"}})
    assert runtime.store.list_hold_permission_history(ws, corpus, doc) == []


def test_held_permission_change_records_the_ingest_actor(registry: tuple[Refaat, _RecordingSink]) -> None:
    runtime, _sink = registry
    ws = runtime.create_workspace("acme", "Acme")
    corpus = runtime.create_corpus(ws, "handbook")
    doc = _ingest(runtime, ws, corpus, "target", _BODY, metadata={"acl": {"classification": "INTERNAL"}})
    runtime.project(workspace="acme", corpus="handbook").legal_holds.place(
        corpus_id=corpus, document_id=doc, placed_by="counsel"
    )
    runtime.ingest(
        ws,
        corpus,
        DocumentInput(
            source_id="target",
            title="target",
            media_type="text/plain",
            content=_BODY,
            metadata={"acl": {"classification": "CONFIDENTIAL"}},
        ),
        actor="key-42",
    )
    history = runtime.store.list_hold_permission_history(ws, corpus, doc)
    assert len(history) == 1
    assert history[0]["actor"] == "key-42"


def test_held_reorder_writes_no_ledger_row(registry: tuple[Refaat, _RecordingSink]) -> None:
    runtime, _sink = registry
    ws = runtime.create_workspace("acme", "Acme")
    corpus = runtime.create_corpus(ws, "handbook")
    acl = {"acl": {"classification": "INTERNAL", "allow_groups": ["a", "b"]}}
    doc = _ingest(runtime, ws, corpus, "target", _BODY, metadata=acl)
    runtime.project(workspace="acme", corpus="handbook").legal_holds.place(
        corpus_id=corpus, document_id=doc, placed_by="counsel"
    )
    reordered = {"acl": {"classification": "INTERNAL", "allow_groups": ["b", "a"]}}
    _ingest(runtime, ws, corpus, "target", _BODY, metadata=reordered)
    assert runtime.store.list_hold_permission_history(ws, corpus, doc) == []


def test_lifted_hold_is_excluded_from_covering_hold_ids(registry: tuple[Refaat, _RecordingSink]) -> None:
    runtime, _sink = registry
    ws = runtime.create_workspace("acme", "Acme")
    corpus = runtime.create_corpus(ws, "handbook")
    doc = _ingest(runtime, ws, corpus, "target", _BODY, metadata={"acl": {"classification": "INTERNAL"}})
    project = runtime.project(workspace="acme", corpus="handbook")
    old_hold = project.legal_holds.place(corpus_id=corpus, document_id=doc, placed_by="counsel")
    active_hold = project.legal_holds.place(corpus_id=corpus, document_id=doc, placed_by="counsel")
    project.legal_holds.lift(old_hold, lifted_by="counsel")
    _ingest(runtime, ws, corpus, "target", _BODY, metadata={"acl": {"classification": "CONFIDENTIAL"}})
    history = runtime.store.list_hold_permission_history(ws, corpus, doc)
    assert len(history) == 1
    assert active_hold in history[0]["covering_hold_ids"]
    assert old_hold not in history[0]["covering_hold_ids"]


def test_a_re_chunk_of_held_evidence_at_the_same_content_checksum_is_refused(tmp_path: Path) -> None:
    # Re-chunking a held document alters its evidence even though the raw content, and so the document
    # checksum, is unchanged. The content-overwrite gate keys on the document checksum, so it does not
    # fire; the hold must still refuse the re-tag because the stored chunk set would change.
    body = " ".join(
        f"Section {i} of the release policy defines control {i} and names its accountable owner." for i in range(40)
    ).encode()
    manager = (
        RuntimeBuilder(_settings(tmp_path, default_chunk_tokens=32, chunk_overlap_tokens=4))
        .with_legal_hold_registry()
        .build()
    )
    try:
        ws = manager.create_workspace("acme", "Acme")
        corpus = manager.create_corpus(ws, "handbook")
        doc = _ingest(manager, ws, corpus, "target", body, metadata={"acl": {"classification": "INTERNAL"}})
        manager.project(workspace="acme", corpus="handbook").legal_holds.place(
            corpus_id=corpus, document_id=doc, placed_by="counsel"
        )
    finally:
        manager.close()
    rechunker = (
        RuntimeBuilder(_settings(tmp_path, default_chunk_tokens=512, chunk_overlap_tokens=0))
        .with_legal_hold_registry()
        .build()
    )
    try:
        with pytest.raises(LegalHoldError, match="may not alter the held evidence"):
            _ingest(rechunker, ws, corpus, "target", body, metadata={"acl": {"classification": "CONFIDENTIAL"}})
    finally:
        rechunker.close()


def test_corpus_wide_hold_gates_existing_documents_but_allows_a_new_one(
    registry: tuple[Refaat, _RecordingSink],
) -> None:
    runtime, _sink = registry
    ws = runtime.create_workspace("acme", "Acme")
    corpus = runtime.create_corpus(ws, "handbook")
    doc = _ingest(runtime, ws, corpus, "existing", _BODY)
    project = runtime.project(workspace="acme", corpus="handbook")
    project.legal_holds.place(corpus_id=corpus, placed_by="counsel")  # document_id None -> corpus-wide

    assert project.erase(doc).status is ErasureStatus.HELD
    with pytest.raises(LegalHoldError):
        runtime.ingestion.delete(ws, corpus, doc)
    with pytest.raises(LegalHoldError):
        _ingest(runtime, ws, corpus, "existing", _BODY + b" changed content.")

    # A brand-new document is not spoliation of held evidence, so ingesting one is allowed.
    new_doc = _ingest(runtime, ws, corpus, "fresh", _OTHER_BODY)
    assert new_doc
    assert runtime.store.list_document_chunks(ws, corpus, new_doc)


def test_all_overlapping_holds_must_lift_before_a_destroy_is_permitted(
    registry: tuple[Refaat, _RecordingSink],
) -> None:
    runtime, _sink = registry
    ws = runtime.create_workspace("acme", "Acme")
    corpus = runtime.create_corpus(ws, "handbook")
    doc = _ingest(runtime, ws, corpus, "target", _BODY)
    project = runtime.project(workspace="acme", corpus="handbook")
    doc_hold = project.legal_holds.place(corpus_id=corpus, document_id=doc, placed_by="counsel")
    corpus_hold = project.legal_holds.place(corpus_id=corpus, placed_by="counsel")

    assert runtime.store.is_document_held(ws, corpus, doc) is True
    project.legal_holds.lift(doc_hold, lifted_by="counsel")
    assert runtime.store.is_document_held(ws, corpus, doc) is True  # corpus-wide still covers

    project.legal_holds.lift(corpus_hold, lifted_by="counsel")
    assert runtime.store.is_document_held(ws, corpus, doc) is False
    assert runtime.ingestion.delete(ws, corpus, doc) >= 0


def test_overwrite_refusal_is_not_swallowed_by_the_ingest_conflict_retry(
    registry: tuple[Refaat, _RecordingSink],
) -> None:
    runtime, _sink = registry
    ws = runtime.create_workspace("acme", "Acme")
    corpus = runtime.create_corpus(ws, "handbook")
    doc = _ingest(runtime, ws, corpus, "target", _BODY)
    project = runtime.project(workspace="acme", corpus="handbook")
    project.legal_holds.place(corpus_id=corpus, document_id=doc, placed_by="counsel")

    calls = {"n": 0}
    original = runtime.ingestion._ingest_once

    def counting(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    runtime.ingestion._ingest_once = counting  # type: ignore[assignment]
    with pytest.raises(LegalHoldError):
        _ingest(runtime, ws, corpus, "target", _BODY + b" changed content to overwrite.")
    # LegalHoldError is not a ConflictError, so the retry never fires: exactly one invocation.
    assert calls["n"] == 1


def test_refusal_emits_exactly_one_redacted_hold_enforced_event_per_path(
    registry: tuple[Refaat, _RecordingSink],
) -> None:
    runtime, sink = registry
    ws = runtime.create_workspace("acme", "Acme")
    corpus = runtime.create_corpus(ws, "handbook")
    doc = _ingest(runtime, ws, corpus, "target", _BODY)
    project = runtime.project(workspace="acme", corpus="handbook")
    project.legal_holds.place(corpus_id=corpus, document_id=doc, placed_by="counsel", reason="SECRETMATTER99")

    with pytest.raises(LegalHoldError):
        runtime.ingestion.delete(ws, corpus, doc)
    with pytest.raises(LegalHoldError):
        _ingest(runtime, ws, corpus, "target", _BODY + b" changed content.")

    enforced = [event for event in sink.events if event.event_type == "hold.enforced"]
    caps = sorted(event.capability_name for event in enforced)
    assert caps == ["document.delete", "ingest.overwrite"]
    for event in enforced:
        assert event.cited_document_ids == (doc,)
        assert event.principal_kind == "system"
    # The human reason and evidence text never appear in any emitted event.
    for event in sink.events:
        payload = event.to_json()
        for value in payload.values():
            assert "SECRETMATTER99" not in str(value)
            assert "Payment gateway" not in str(value)


def test_a_raising_sink_does_not_break_the_refusal(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_settings(tmp_path)).with_legal_hold_registry().with_audit_sink(_RaisingSink()).build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "handbook")
        doc = _ingest(runtime, ws, corpus, "target", _BODY)
        runtime.project(workspace="acme", corpus="handbook").legal_holds.place(
            corpus_id=corpus, document_id=doc, placed_by="counsel"
        )
        # The refusal audit is best-effort, so a raising sink still surfaces the refusal, not a 503.
        with pytest.raises(LegalHoldError):
            runtime.ingestion.delete(ws, corpus, doc)
        assert runtime.store.list_document_chunks(ws, corpus, doc)
    finally:
        runtime.close()


def test_lift_is_idempotent_and_conditional(registry: tuple[Refaat, _RecordingSink]) -> None:
    runtime, _sink = registry
    ws = runtime.create_workspace("acme", "Acme")
    corpus = runtime.create_corpus(ws, "handbook")
    doc = _ingest(runtime, ws, corpus, "target", _BODY)
    project = runtime.project(workspace="acme", corpus="handbook")
    hold_id = project.legal_holds.place(corpus_id=corpus, document_id=doc, placed_by="counsel")

    assert project.legal_holds.lift(hold_id, lifted_by="counsel") is True
    assert project.legal_holds.lift(hold_id, lifted_by="counsel") is False


def test_under_fail_closed_a_raising_sink_freezes_the_hold_on_lift(tmp_path: Path) -> None:
    runtime = (
        RuntimeBuilder(_settings(tmp_path, audit_fail_closed=True))
        .with_legal_hold_registry()
        .with_audit_sink(_RaisingSink())
        .build()
    )
    try:
        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "handbook")
        doc = _ingest(runtime, ws, corpus, "target", _BODY)
        project = runtime.project(workspace="acme", corpus="handbook")
        hold_id = project.legal_holds.place(corpus_id=corpus, document_id=doc, placed_by="counsel")
        # Emit-then-lift under fail_closed: the failing sink raises before the row is removed, so the
        # hold stays active rather than lifting unaudited.
        with pytest.raises(AuditUnavailableError):
            project.legal_holds.lift(hold_id, lifted_by="counsel")
        assert runtime.store.is_document_held(ws, corpus, doc) is True
    finally:
        runtime.close()


def test_default_runtime_refuses_to_record_a_hold_and_the_table_stays_empty(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_settings(tmp_path)).build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "handbook")
        with pytest.raises(LegalHoldError):
            LegalHoldService(runtime.store, ws).place(corpus_id=corpus, placed_by="counsel")
        with runtime.store.engine.connect() as conn:
            rows = conn.execute(select(func.count()).select_from(legal_holds)).scalar_one()
        assert rows == 0
    finally:
        runtime.close()


def test_default_runtime_accessor_raises_configuration_error(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_settings(tmp_path)).build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        runtime.create_corpus(ws, "handbook")
        with pytest.raises(ConfigurationError):
            _ = runtime.project(workspace="acme", corpus="handbook").legal_holds
    finally:
        runtime.close()


def test_a_non_management_runtime_still_enforces_a_hold_placed_elsewhere(tmp_path: Path) -> None:
    # Enforcement is state-driven and unconditional: a runtime that never wired the management surface
    # still refuses to erase, delete, or overwrite a document a management runtime placed a hold on.
    # This is the closed temporal fail-open.
    settings = _settings(tmp_path)
    manager = RuntimeBuilder(settings).with_legal_hold_registry().build()
    ws = manager.create_workspace("acme", "Acme")
    corpus = manager.create_corpus(ws, "handbook")
    doc = _ingest(manager, ws, corpus, "target", _BODY)
    manager.project(workspace="acme", corpus="handbook").legal_holds.place(
        corpus_id=corpus, document_id=doc, placed_by="counsel"
    )
    manager.close()

    plain = RuntimeBuilder(settings).build()
    try:
        assert plain.store.legal_hold_management_enabled is False
        assert plain.project(workspace="acme", corpus="handbook").erase(doc).status is ErasureStatus.HELD
        assert plain.store.list_document_chunks(ws, corpus, doc)
        with pytest.raises(LegalHoldError):
            plain.ingestion.delete(ws, corpus, doc)
        with pytest.raises(LegalHoldError):
            _ingest(plain, ws, corpus, "target", _BODY + b" changed content to overwrite.")
        # A same-checksum re-tag is still evidence-preserving and allowed on the plain runtime.
        assert _ingest(plain, ws, corpus, "target", _BODY, metadata={"acl": {"classification": "CONFIDENTIAL"}}) == doc
    finally:
        plain.close()


def test_a_hold_placed_after_a_runtime_started_is_enforced(tmp_path: Path) -> None:
    # Enforcement is not a build-time probe: a plain runtime that started before any hold existed
    # still enforces a hold a second runtime places while it is running.
    settings = _settings(tmp_path)
    plain = RuntimeBuilder(settings).build()
    try:
        ws = plain.create_workspace("acme", "Acme")
        corpus = plain.create_corpus(ws, "handbook")
        doc = _ingest(plain, ws, corpus, "target", _BODY)

        manager = RuntimeBuilder(settings).with_legal_hold_registry().build()
        manager.project(workspace="acme", corpus="handbook").legal_holds.place(
            corpus_id=corpus, document_id=doc, placed_by="counsel"
        )
        manager.close()

        assert plain.project(workspace="acme", corpus="handbook").erase(doc).status is ErasureStatus.HELD
        with pytest.raises(LegalHoldError):
            plain.ingestion.delete(ws, corpus, doc)
    finally:
        plain.close()


def test_sqlite_single_process_enforcement_serializes_a_destroy(tmp_path: Path) -> None:
    # On single-process SQLite the destroy lock is a real reentrant lock, so a full registry erase
    # completes without deadlock and the hold is enforced end to end.
    runtime = RuntimeBuilder(_settings(tmp_path)).with_legal_hold_registry().build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "handbook")
        doc = _ingest(runtime, ws, corpus, "target", _BODY)
        project = runtime.project(workspace="acme", corpus="handbook")
        hold_id = project.legal_holds.place(corpus_id=corpus, document_id=doc, placed_by="counsel")
        assert project.erase(doc).status is ErasureStatus.HELD
        # Lifting the hold lets a full erase run to completion under the reentrant destroy lock.
        project.legal_holds.lift(hold_id, lifted_by="counsel")
        assert project.erase(doc).status is ErasureStatus.ERASED
        assert runtime.store.list_document_chunks(ws, corpus, doc) == []
    finally:
        runtime.close()


def test_free_text_document_id_is_rejected_on_place(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_settings(tmp_path)).with_legal_hold_registry().build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "handbook")
        project = runtime.project(workspace="acme", corpus="handbook")
        # A free-text document id can never reach an audit event's cited ids, so place rejects it.
        with pytest.raises(ValidationError):
            project.legal_holds.place(corpus_id=corpus, document_id="SECRETMATTER99 free text", placed_by="counsel")
        with runtime.store.engine.connect() as conn:
            rows = conn.execute(select(func.count()).select_from(legal_holds)).scalar_one()
        assert rows == 0
    finally:
        runtime.close()


def test_an_interleaved_erase_during_an_overwrite_embed_is_not_resurrected(tmp_path: Path) -> None:
    # The resurrection race: an overwrite reads the document, then an erase removes it
    # before the overwrite takes the destroy lock. The overwrite must not recreate the erased id; the
    # bounded ingest retry creates a genuinely new document instead.
    runtime = RuntimeBuilder(_settings(tmp_path)).with_legal_hold_registry().build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "handbook")
        doc = _ingest(runtime, ws, corpus, "target", _BODY)

        real_embed = runtime.ingestion.embedder.embed
        fired = {"n": 0}

        def interleaving_embed(texts: object) -> object:
            if fired["n"] == 0:
                fired["n"] = 1
                runtime.erasure.erase(ws, corpus, doc)
            return real_embed(texts)  # type: ignore[arg-type]

        runtime.ingestion.embedder.embed = interleaving_embed  # type: ignore[assignment,method-assign]
        try:
            result = _ingest(runtime, ws, corpus, "target", _BODY + b" NEW content to overwrite.")
        finally:
            runtime.ingestion.embedder.embed = real_embed  # type: ignore[method-assign]

        assert result != doc
        with pytest.raises(NotFoundError):
            runtime.store.get_document(ws, corpus, doc)
        assert runtime.store.list_document_chunks(ws, corpus, result)
    finally:
        runtime.close()


def test_free_text_erase_document_id_is_rejected_before_any_hold_check_or_audit(
    registry: tuple[Refaat, _RecordingSink],
) -> None:
    # A corpus-wide hold makes any supplied id read as held, so an unvalidated free-text erase id
    # would land in the HELD receipt and the hold.enforced audit event. erase validates the id first.
    runtime, sink = registry
    ws = runtime.create_workspace("acme", "Acme")
    corpus = runtime.create_corpus(ws, "handbook")
    _ingest(runtime, ws, corpus, "target", _BODY)
    project = runtime.project(workspace="acme", corpus="handbook")
    project.legal_holds.place(corpus_id=corpus, placed_by="counsel")  # corpus-wide

    with pytest.raises(ValidationError):
        project.erase("SECRETMATTER99 free text\ncontrol")

    for event in sink.events:
        payload = event.to_json()
        for value in payload.values():
            assert "SECRETMATTER99" not in str(value)


def test_sqlite_two_runtimes_one_process_share_the_destroy_lock(tmp_path: Path) -> None:
    # Two SQLStore instances over one SQLite database in one process must share the destroy lock, or
    # a place on runtime B could slip into runtime A's mid-erase window and resurrect a now-held
    # document. The lock is process-wide, so B's place is serialized behind A's in-flight erase.
    settings = _settings(tmp_path, database_url=f"sqlite:///{tmp_path / 'shared.db'}")
    runtime_a = RuntimeBuilder(settings).with_legal_hold_registry().build()
    runtime_b = RuntimeBuilder(settings).with_legal_hold_registry().build()
    real_delete = runtime_a.store.delete_document
    try:
        ws = runtime_a.create_workspace("acme", "Acme")
        corpus = runtime_a.create_corpus(ws, "handbook")
        doc = _ingest(runtime_a, ws, corpus, "target", _BODY)

        assert runtime_a.store._sqlite_destroy_lock((ws, corpus)) is runtime_b.store._sqlite_destroy_lock((ws, corpus))

        order: list[str] = []
        inside = threading.Event()
        release = threading.Event()

        def paused_delete(w: str, c: str, d: str) -> int:
            inside.set()
            release.wait(5)
            return real_delete(w, c, d)

        runtime_a.store.delete_document = paused_delete  # type: ignore[method-assign]

        def run_erase() -> None:
            runtime_a.project(workspace="acme", corpus="handbook").erase(doc)
            order.append("erased")

        def run_place() -> None:
            inside.wait(5)
            runtime_b.project(workspace="acme", corpus="handbook").legal_holds.place(
                corpus_id=corpus, document_id=doc, placed_by="counsel"
            )
            order.append("placed")

        erase_thread = threading.Thread(target=run_erase)
        place_thread = threading.Thread(target=run_place)
        erase_thread.start()
        place_thread.start()
        assert inside.wait(5)  # A holds the shared destroy lock, paused before the row delete.
        time.sleep(0.2)  # give B's place a window to (wrongly) commit mid-erase.
        assert order == []  # B is blocked behind the shared lock; nothing committed during the erase.
        release.set()
        erase_thread.join(5)
        place_thread.join(5)

        assert set(order) == {"erased", "placed"}
        with pytest.raises(NotFoundError):
            runtime_a.store.get_document(ws, corpus, doc)
        assert runtime_b.store.is_document_held(ws, corpus, doc) is True
    finally:
        runtime_a.store.delete_document = real_delete  # type: ignore[method-assign]
        runtime_a.close()
        runtime_b.close()
