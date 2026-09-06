"""Per-document derivation digest: maintained in-transaction, byte-identical off, localizes drift."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from sqlalchemy import select, update

from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.config import Settings
from refaat.core.text import sha256_text
from refaat.domain.models import DocumentInput
from refaat.infrastructure.database.schema import chunks, documents

pytestmark = pytest.mark.integration

_A = b"The production release checklist requires reviewer approval before any deployment proceeds onward."
_B = b"Warehouse throughput and carrier routing options are covered by the separate logistics operations report."


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
        "log_json": False,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _ingest(runtime: Refaat, ws: str, corpus: str, source_id: str, body: bytes) -> str:
    return runtime.ingest(
        ws,
        corpus,
        DocumentInput(source_id=source_id, title=source_id, media_type="text/plain", content=body, metadata={}),
    ).document_id


def _stored_digest(runtime: Refaat, doc: str) -> str | None:
    with runtime.store.engine.connect() as conn:
        return conn.execute(select(documents.c.derivation_digest).where(documents.c.id == doc)).scalar_one()


def test_digest_written_and_verifies_when_enabled(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_settings(tmp_path, derivation_manifest_enabled=True)).build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "handbook")
        doc = _ingest(runtime, ws, corpus, "a", _A)
        stored = _stored_digest(runtime, doc)
        assert stored is not None
        assert len(stored) == 64
        assert stored == runtime.store.document_derivation_digest(ws, corpus, doc)
        assert runtime.derivation is not None
        assert runtime.derivation.verify_document(ws, corpus, doc) == "match"
    finally:
        runtime.close()


def test_digest_null_when_disabled(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_settings(tmp_path)).build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "handbook")
        doc = _ingest(runtime, ws, corpus, "a", _A)
        assert _stored_digest(runtime, doc) is None
        assert runtime.derivation is None
    finally:
        runtime.close()


def test_corpus_checksum_matches_the_pre_refactor_fold(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_settings(tmp_path, derivation_manifest_enabled=True)).build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "handbook")
        _ingest(runtime, ws, corpus, "a", _A)
        _ingest(runtime, ws, corpus, "b", _B)
        counts, checksum = runtime.store.derivation_rollup(ws, corpus)
        # Recompute content_checksum the exact pre-refactor way and assert derivation_rollup (which now
        # calls _fold_chunk_digest) produces byte-identical output.
        acc = bytearray(32)
        with runtime.store.engine.connect() as conn:
            for chunk_id, chunk_checksum in conn.execute(
                select(chunks.c.id, chunks.c.checksum).where(chunks.c.workspace_id == ws, chunks.c.corpus_id == corpus)
            ):
                digest = hashlib.sha256(f"{chunk_id}\x00{chunk_checksum}".encode()).digest()
                for i in range(32):
                    acc[i] ^= digest[i]
        expected = sha256_text("|".join([acc.hex(), *(f"{name}={counts[name]}" for name in sorted(counts))]))
        assert checksum == expected
    finally:
        runtime.close()


def test_digest_localizes_chunk_drift(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_settings(tmp_path, derivation_manifest_enabled=True)).build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "handbook")
        doc_a = _ingest(runtime, ws, corpus, "a", _A)
        doc_b = _ingest(runtime, ws, corpus, "b", _B)
        with runtime.store.engine.begin() as conn:
            conn.execute(update(chunks).where(chunks.c.document_id == doc_a).values(checksum="tampered"))
        assert runtime.derivation is not None
        assert runtime.derivation.verify_document(ws, corpus, doc_a) == "drift"
        assert runtime.derivation.verify_document(ws, corpus, doc_b) == "match"
        assert runtime.derivation.drifted_documents(ws, corpus) == [doc_a]
        assert runtime.derivation.verify_detail(ws, corpus) == "drift"
    finally:
        runtime.close()


def test_digest_removed_on_delete_and_erase(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_settings(tmp_path, derivation_manifest_enabled=True)).build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "handbook")
        doc_a = _ingest(runtime, ws, corpus, "a", _A)
        doc_b = _ingest(runtime, ws, corpus, "b", _B)
        assert runtime.derivation is not None
        runtime.project(workspace="acme", corpus="handbook").erase(doc_a)
        assert runtime.derivation.verify_document(ws, corpus, doc_a) == "no_document"
        assert runtime.derivation.verify_document(ws, corpus, doc_b) == "match"
    finally:
        runtime.close()


def test_off_lock_refresh_never_writes_digest(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_settings(tmp_path, derivation_manifest_enabled=True)).build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "handbook")
        doc = _ingest(runtime, ws, corpus, "a", _A)
        with runtime.store.engine.begin() as conn:
            conn.execute(update(documents).where(documents.c.id == doc).values(derivation_digest=None))
        assert runtime.derivation is not None
        runtime.derivation.refresh_quietly(ws, corpus)
        runtime.store.derivation_rollup(ws, corpus)
        # The off-lock refresh/rollup paths must never maintain the per-document digest.
        assert _stored_digest(runtime, doc) is None
        assert runtime.derivation.verify_document(ws, corpus, doc) == "no_digest"
    finally:
        runtime.close()


def test_per_document_digests_xor_to_corpus_body(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_settings(tmp_path, derivation_manifest_enabled=True)).build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "handbook")
        _ingest(runtime, ws, corpus, "a", _A)
        _ingest(runtime, ws, corpus, "b", _B)
        stored = runtime.store.list_document_derivation_digests(ws, corpus)
        folded = bytearray(32)
        for digest in stored.values():
            assert digest is not None
            for i, byte in enumerate(bytes.fromhex(digest)):
                folded[i] ^= byte
        with runtime.store.engine.connect() as conn:
            pairs = [(str(cid), str(cs)) for cid, cs in conn.execute(select(chunks.c.id, chunks.c.checksum))]
        assert folded.hex() == runtime.store._fold_chunk_digest(pairs)
    finally:
        runtime.close()
