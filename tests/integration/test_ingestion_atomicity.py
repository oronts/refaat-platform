from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta

import pytest

from refaat.application.blob_gc import collect_unreferenced_blobs
from refaat.domain.models import Chunk, DocumentInput


class RecordingSearchIndex:
    def __init__(self, *, fail_replace: bool = False) -> None:
        self.fail_replace = fail_replace
        self.replacements: list[tuple[str, tuple[Chunk, ...]]] = []
        self.deletions: list[str] = []

    def replace_document(self, document_id: str, chunks: Sequence[Chunk]) -> None:
        values = tuple(chunks)
        self.replacements.append((document_id, values))
        if self.fail_replace:
            raise RuntimeError("search index unavailable")

    def delete_document(self, document_id: str) -> None:
        self.deletions.append(document_id)


def _document(content: bytes) -> DocumentInput:
    return DocumentInput(
        source_id="policy.txt",
        title="Policy",
        media_type="text/plain",
        content=content,
    )


def test_search_index_is_written_only_after_relational_commit(
    runtime,
    workspace_id: str,
    corpus_id: str,
) -> None:
    """Relational-first: the committed document row is visible before the vector write.

    Because ``replace_document`` (relational rows plus the document.indexed outbox
    event) commits first, a citation can never resolve against a vector for a chunk
    that is not durably stored.
    """
    observed: dict[str, bool] = {}

    class OrderingSearchIndex(RecordingSearchIndex):
        def replace_document(self, document_id, chunks):  # type: ignore[no-untyped-def]
            observed["doc_committed_at_write"] = (
                runtime.store.get_document_by_source(workspace_id, corpus_id, "policy.txt") is not None
            )
            super().replace_document(document_id, chunks)

    search = OrderingSearchIndex()
    runtime.ingestion.search_index = search

    runtime.ingest(workspace_id, corpus_id, _document(b"Index this durable policy."))

    assert observed["doc_committed_at_write"] is True
    assert len(search.replacements) == 1


def test_db_commit_failure_leaves_index_untouched_and_defers_blob(
    runtime,
    workspace_id: str,
    corpus_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relational-commit failure never mutates the vector index or the document row.

    The staged blob is not deleted inline: a concurrent identical re-ingest may have
    staged and be about to commit against the same content-addressed key, so the
    orphan is left for the deferred sweep to reclaim past its grace window.
    """
    search = RecordingSearchIndex()
    runtime.ingestion.search_index = search

    def fail_replace_document(**_: object) -> int:
        raise RuntimeError("database commit failed")

    monkeypatch.setattr(runtime.store, "replace_document", fail_replace_document)

    with pytest.raises(RuntimeError, match="database commit failed"):
        runtime.ingest(
            workspace_id,
            corpus_id,
            _document(b"A changed policy that must not become visible."),
        )

    assert search.replacements == []
    assert search.deletions == []
    assert runtime.store.get_document_by_source(workspace_id, corpus_id, "policy.txt") is None

    staged = [
        path
        for path in runtime.settings.blob_root.rglob("*")
        if path.is_file() and not path.name.startswith(".refaat-")
    ]
    assert staged, "the failed ingest leaves its orphan for the deferred sweep"

    reclaimed = collect_unreferenced_blobs(
        blob_store=runtime.ingestion.blob_store,
        store=runtime.store,
        now=runtime.store.clock.now() + timedelta(hours=1),
        grace_seconds=0,
    )
    assert reclaimed == len(staged)
    assert not any(
        path.is_file() and not path.name.startswith(".refaat-") for path in runtime.settings.blob_root.rglob("*")
    )


def test_failed_ingest_keeps_a_concurrently_referenced_blob(
    runtime,
    workspace_id: str,
    corpus_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed ingest never deletes a content-addressed blob a committed row references.

    Two concurrent identical-content re-ingests of the same document stage the same
    content-addressed key. When one loses the first-write race, its except-path must
    not strand the winner's blob. The loser leaves the key untouched, and because the
    winner's committed row references it, even a past-grace sweep keeps it.
    """
    search = RecordingSearchIndex()
    runtime.ingestion.search_index = search

    def fail_replace_document(**_: object) -> int:
        raise RuntimeError("lost the first-write race")

    monkeypatch.setattr(runtime.store, "replace_document", fail_replace_document)
    monkeypatch.setattr(runtime.store, "is_blob_referenced", lambda blob_uri: True)
    monkeypatch.setattr(runtime.store, "referenced_blob_uris", lambda: set())

    with pytest.raises(RuntimeError, match="lost the first-write race"):
        runtime.ingest(
            workspace_id,
            corpus_id,
            _document(b"Identical content two writers race on."),
        )

    assert search.replacements == []
    staged_blobs = [
        path
        for path in runtime.settings.blob_root.rglob("*")
        if path.is_file() and not path.name.startswith(".refaat-")
    ]
    assert staged_blobs, "the staged blob a committed row references must not be deleted"

    reclaimed = collect_unreferenced_blobs(
        blob_store=runtime.ingestion.blob_store,
        store=runtime.store,
        now=runtime.store.clock.now() + timedelta(hours=1),
        grace_seconds=0,
    )
    assert reclaimed == 0, "the winner's referenced blob survives even a past-grace sweep"
    assert all(path.exists() for path in staged_blobs)


def test_post_commit_vector_failure_leaves_document_committed(
    runtime,
    workspace_id: str,
    corpus_id: str,
) -> None:
    """A vector write that fails after the commit leaves the document durable.

    On pgvector the document.indexed outbox event plus the VectorReconcileSink heal
    the vector table asynchronously; the relational store is the source of truth.
    """
    search = RecordingSearchIndex(fail_replace=True)
    runtime.ingestion.search_index = search

    with pytest.raises(RuntimeError, match="search index unavailable"):
        runtime.ingest(workspace_id, corpus_id, _document(b"Durable policy content."))

    assert runtime.store.get_document_by_source(workspace_id, corpus_id, "policy.txt") is not None
    assert len(search.replacements) == 1


def test_delete_restores_index_when_database_delete_fails(
    runtime,
    workspace_id: str,
    corpus_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = runtime.ingest(
        workspace_id,
        corpus_id,
        _document(b"Keep this document when deletion cannot commit."),
    )
    old_chunks = runtime.store.list_document_chunks(workspace_id, corpus_id, result.document_id)
    search = RecordingSearchIndex()
    runtime.ingestion.search_index = search

    def fail_delete_document(*_: object) -> int:
        raise RuntimeError("database delete failed")

    monkeypatch.setattr(runtime.store, "delete_document", fail_delete_document)

    with pytest.raises(RuntimeError, match="database delete failed"):
        runtime.ingestion.delete(workspace_id, corpus_id, result.document_id)

    assert runtime.store.get_document(workspace_id, corpus_id, result.document_id)
    assert search.deletions == [result.document_id]
    assert search.replacements == [(result.document_id, tuple(old_chunks))]
