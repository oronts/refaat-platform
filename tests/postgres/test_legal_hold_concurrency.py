"""Real cross-connection legal-hold concurrency on Postgres.

These exercise the ACTUAL advisory lock with real connections, not a stub that flips is_held. The
place-during-destroy race is closed by a session-level advisory lock held across the whole erase on
a dedicated NullPool connection, so a same-corpus place blocks while an erase holds the lock, and a
tiny QueuePool proves the lock waiters never drain the shared pool.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.config import Settings
from refaat.domain.erasure import ErasureStatus
from refaat.domain.models import DocumentInput
from refaat.errors import NotFoundError

pytestmark = [pytest.mark.postgres, pytest.mark.integration]

_PG = os.getenv("REFAAT_TEST_POSTGRES_URL")
_BODY = b"Payment gateway timeout cascaded into a checkout outage; the rollback restored service."


def _registry_runtime(tmp_path: Path, suffix: str) -> Refaat:
    return (
        RuntimeBuilder(
            Settings(
                environment="test",
                database_url=os.environ["REFAAT_TEST_POSTGRES_URL"],
                auto_create_schema=True,
                # A shared DB means a shared blob store, so a second runtime can erase a document the
                # first ingested; separate blob roots would reject the foreign blob uri.
                blob_root=tmp_path / "blobs",
                skill_root=tmp_path / f"skills-{suffix}",
                api_key_pepper="postgres-test-pepper",
                embedding_dimensions=64,
                default_chunk_tokens=48,
                chunk_overlap_tokens=8,
                log_json=False,
            )
        )
        .with_legal_hold_registry()
        .build()
    )


def _ingest(runtime: Refaat, ws: str, corpus: str, source_id: str, body: bytes) -> str:
    return runtime.ingest(
        ws, corpus, DocumentInput(source_id=source_id, title=source_id, media_type="text/plain", content=body)
    ).document_id


@pytest.mark.skipif(not _PG, reason="PostgreSQL test URL not configured")
def test_under_lock_recheck_sees_a_hold_committed_by_a_second_connection(tmp_path: Path) -> None:
    a = _registry_runtime(tmp_path, "a")
    b = _registry_runtime(tmp_path, "b")
    try:
        ws = a.create_workspace("acme-recheck", "Acme")
        corpus = a.create_corpus(ws, "handbook")
        doc = _ingest(a, ws, corpus, "target", _BODY)

        # A second, independent connection (runtime b) commits a hold.
        b.project(workspace="acme-recheck", corpus="handbook").legal_holds.place(
            corpus_id=corpus, document_id=doc, placed_by="counsel"
        )
        # Runtime a's erase acquires the destroy lock, rechecks, and sees the hold: nothing destroyed.
        receipt = a.project(workspace="acme-recheck", corpus="handbook").erase(doc)
        assert receipt.status is ErasureStatus.HELD
        assert a.store.list_document_chunks(ws, corpus, doc)
    finally:
        a.close()
        b.close()


@pytest.mark.skipif(not _PG, reason="PostgreSQL test URL not configured")
def test_the_session_lock_is_held_across_the_whole_erase_and_blocks_a_place(tmp_path: Path) -> None:
    a = _registry_runtime(tmp_path, "a")
    b = _registry_runtime(tmp_path, "b")
    try:
        ws = a.create_workspace("acme-block", "Acme")
        corpus = a.create_corpus(ws, "handbook")
        doc = _ingest(a, ws, corpus, "target", _BODY)

        reached = threading.Event()
        release = threading.Event()
        original_delete = a.erasure.blob_store.delete

        def blocking_delete(uri: str) -> None:
            # Pause mid-erase while the session destroy lock is held, then finish.
            reached.set()
            assert release.wait(timeout=30)
            original_delete(uri)

        a.erasure.blob_store.delete = blocking_delete  # type: ignore[method-assign]

        erase_result: dict[str, object] = {}

        def run_erase() -> None:
            erase_result["receipt"] = a.project(workspace="acme-block", corpus="handbook").erase(doc)

        place_result: dict[str, object] = {}

        def run_place() -> None:
            place_result["hold_id"] = b.project(workspace="acme-block", corpus="handbook").legal_holds.place(
                corpus_id=corpus, document_id=doc, placed_by="counsel"
            )

        erase_thread = threading.Thread(target=run_erase)
        erase_thread.start()
        assert reached.wait(timeout=30)  # erase now holds the session destroy lock

        place_thread = threading.Thread(target=run_place)
        place_thread.start()
        time.sleep(1.0)
        # The place blocks on the corpus advisory lock while the erase holds it.
        assert "hold_id" not in place_result
        assert place_thread.is_alive()

        release.set()
        erase_thread.join(timeout=30)
        place_thread.join(timeout=30)
        assert erase_result["receipt"].status is ErasureStatus.ERASED  # type: ignore[union-attr]
        assert "hold_id" in place_result  # the place proceeded only after the erase released
    finally:
        a.close()
        b.close()


@pytest.mark.skipif(not _PG, reason="PostgreSQL test URL not configured")
def test_a_full_registry_erase_completes_without_a_self_deadlock(tmp_path: Path) -> None:
    a = _registry_runtime(tmp_path, "a")
    try:
        ws = a.create_workspace("acme-deadlock", "Acme")
        corpus = a.create_corpus(ws, "handbook")
        doc = _ingest(a, ws, corpus, "target", _BODY)

        done: dict[str, object] = {}

        def run() -> None:
            done["receipt"] = a.project(workspace="acme-deadlock", corpus="handbook").erase(doc)

        thread = threading.Thread(target=run)
        thread.start()
        thread.join(timeout=30)
        # A regression that re-acquired the corpus lock on a second connection would hang here.
        assert not thread.is_alive()
        assert done["receipt"].status is ErasureStatus.ERASED  # type: ignore[union-attr]
    finally:
        a.close()


def _use_tiny_queue_pool(runtime: Refaat) -> None:
    # Replace the shared store QueuePool with a deliberately tiny one. The NullPool lock engine is
    # untouched, so lock waiters park outside this pool. Under the v2 design (lock connections drawn
    # from this pool) the no-starvation tests below would deadlock or time out.
    url = runtime.store.engine.url
    runtime.store.engine.dispose()
    runtime.store.engine = create_engine(url, poolclass=QueuePool, pool_size=2, max_overflow=0, pool_timeout=15)


@pytest.mark.skipif(not _PG, reason="PostgreSQL test URL not configured")
def test_no_starvation_across_different_corpora_with_a_tiny_pool(tmp_path: Path) -> None:
    runtime = _registry_runtime(tmp_path, "a")
    try:
        ws = runtime.create_workspace("acme-diff", "Acme")
        docs: list[tuple[str, str]] = []
        for index in range(8):
            corpus = runtime.create_corpus(ws, f"c{index}")
            docs.append((corpus, _ingest(runtime, ws, corpus, "target", _BODY)))
        _use_tiny_queue_pool(runtime)

        peak = {"value": 0}
        stop = threading.Event()

        def monitor() -> None:
            while not stop.is_set():
                peak["value"] = max(peak["value"], runtime.store.engine.pool.checkedout())
                time.sleep(0.01)

        errors: list[BaseException] = []

        def erase(corpus: str, doc: str) -> None:
            try:
                runtime.erasure.erase(ws, corpus, doc)
            except BaseException as exc:
                errors.append(exc)

        monitor_thread = threading.Thread(target=monitor)
        monitor_thread.start()
        threads = [threading.Thread(target=erase, args=(corpus, doc)) for corpus, doc in docs]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        stop.set()
        monitor_thread.join(timeout=5)

        assert errors == []
        assert all(not thread.is_alive() for thread in threads)
        assert peak["value"] <= len(docs)
    finally:
        runtime.close()


@pytest.mark.skipif(not _PG, reason="PostgreSQL test URL not configured")
def test_no_starvation_same_corpus_with_place_delete_overwrite_waiters(tmp_path: Path) -> None:
    # A paused erase holds the session destroy lock while place, delete, and overwrite waiters block
    # on the SAME corpus lock. Every waiter parks on a NullPool connection, so the tiny QueuePool is
    # never drained and the erase's own inner transactions never starve.
    runtime = _registry_runtime(tmp_path, "a")
    manager = _registry_runtime(tmp_path, "m")
    try:
        ws = runtime.create_workspace("acme-same", "Acme")
        corpus = runtime.create_corpus(ws, "handbook")
        sources = [f"target{index}" for index in range(4)]
        docs = {source: _ingest(runtime, ws, corpus, source, _BODY + source.encode()) for source in sources}
        _use_tiny_queue_pool(runtime)

        reached = threading.Event()
        release = threading.Event()
        original_delete = runtime.erasure.blob_store.delete

        def blocking_delete(uri: str) -> None:
            reached.set()
            assert release.wait(timeout=30)
            original_delete(uri)

        runtime.erasure.blob_store.delete = blocking_delete  # type: ignore[method-assign]

        peak = {"value": 0}
        stop = threading.Event()

        def monitor() -> None:
            while not stop.is_set():
                peak["value"] = max(peak["value"], runtime.store.engine.pool.checkedout())
                time.sleep(0.01)

        errors: list[BaseException] = []

        def run(fn: object) -> None:
            try:
                fn()  # type: ignore[operator]
            except BaseException as exc:
                errors.append(exc)

        monitor_thread = threading.Thread(target=monitor)
        monitor_thread.start()

        # The erase pauses mid-run holding the session destroy lock on target0.
        erase_thread = threading.Thread(target=run, args=(lambda: runtime.erasure.erase(ws, corpus, docs["target0"]),))
        erase_thread.start()
        assert reached.wait(timeout=30)

        # Place (on a separate document so it does not refuse the others), delete, and overwrite all
        # block on the same corpus lock while the erase holds it.
        waiters = [
            threading.Thread(
                target=run,
                args=(
                    lambda: manager.project(workspace="acme-same", corpus="handbook").legal_holds.place(
                        corpus_id=corpus, document_id=docs["target3"], placed_by="counsel"
                    ),
                ),
            ),
            threading.Thread(target=run, args=(lambda: runtime.ingestion.delete(ws, corpus, docs["target1"]),)),
            threading.Thread(
                target=run,
                args=(
                    lambda: runtime.ingest(  # overwrite target2 with new content
                        ws,
                        corpus,
                        DocumentInput(
                            source_id="target2", title="target2", media_type="text/plain", content=_BODY + b" changed."
                        ),
                    ),
                ),
            ),
        ]
        for thread in waiters:
            thread.start()
        time.sleep(1.0)
        # The waiters are blocked on the corpus lock, not draining the shared pool.
        assert all(thread.is_alive() for thread in waiters)

        release.set()
        erase_thread.join(timeout=60)
        for thread in waiters:
            thread.join(timeout=60)
        stop.set()
        monitor_thread.join(timeout=5)

        assert errors == []
        assert not erase_thread.is_alive()
        assert all(not thread.is_alive() for thread in waiters)
        # Waiters parked on NullPool connections, so the QueuePool never exceeded its two slots.
        assert peak["value"] <= 2
    finally:
        runtime.close()
        manager.close()


@pytest.mark.skipif(not _PG, reason="PostgreSQL test URL not configured")
def test_an_interleaved_erase_is_not_resurrected_by_an_overwrite(tmp_path: Path) -> None:
    # An overwrite reads the document, then an erase on a second connection removes it before the
    # overwrite takes the destroy lock. The overwrite must not recreate the erased id: the bounded
    # ingest retry creates a genuinely new document instead.
    writer = _registry_runtime(tmp_path, "w")
    eraser = _registry_runtime(tmp_path, "e")
    try:
        ws = writer.create_workspace("acme-res", "Acme")
        corpus = writer.create_corpus(ws, "handbook")
        doc = _ingest(writer, ws, corpus, "target", _BODY)

        real_embed = writer.ingestion.embedder.embed
        fired = {"n": 0}

        def interleaving_embed(texts: object) -> object:
            if fired["n"] == 0:
                fired["n"] = 1
                eraser.erasure.erase(ws, corpus, doc)
            return real_embed(texts)  # type: ignore[arg-type]

        writer.ingestion.embedder.embed = interleaving_embed  # type: ignore[assignment,method-assign]
        try:
            result = writer.ingest(
                ws,
                corpus,
                DocumentInput(
                    source_id="target", title="target", media_type="text/plain", content=_BODY + b" NEW content."
                ),
            )
        finally:
            writer.ingestion.embedder.embed = real_embed  # type: ignore[method-assign]

        assert result.document_id != doc
        with pytest.raises(NotFoundError):
            writer.store.get_document(ws, corpus, doc)
        assert writer.store.list_document_chunks(ws, corpus, result.document_id)
    finally:
        writer.close()
        eraser.close()
