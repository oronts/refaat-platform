from __future__ import annotations

from datetime import datetime, timedelta

from refaat.infrastructure.database.store import SQLStore
from refaat.observability import get_logger
from refaat.ports import BlobStore

logger = get_logger(__name__)


def collect_unreferenced_blobs(
    *,
    blob_store: BlobStore,
    store: SQLStore,
    now: datetime,
    grace_seconds: int,
) -> int:
    """Delete blobs no current document references and older than the grace window.

    The grace window is the primary guard: ingestion writes the content-addressed
    blob before it commits the referencing row, so a blob that is mid-ingest or was
    just reverted to carries a recent modification time and is kept. The referenced
    snapshot skips obviously live keys, and liveness is re-checked against the
    database immediately before each delete. The modification time is also re-stat'd
    immediately before the unlink, so a key a concurrent ingest overwrote (identical
    content, refreshed mtime) after the listing is protected by the grace window even
    though the listed timestamp was stale. That narrows the concurrent-overwrite
    window to the tiny re-stat-to-unlink gap. Since keys are content-addressed and the
    bytes are deterministic, the worst case is a benign orphan or a miss that
    re-ingesting the source re-materializes, never a silent corruption. Deletion is
    idempotent.
    """
    referenced = store.referenced_blob_uris()
    cutoff = now - timedelta(seconds=grace_seconds)
    reclaimed = 0
    for entry in blob_store.list_blobs():
        if entry.uri in referenced or entry.modified_at > cutoff:
            continue
        if store.is_blob_referenced(entry.uri):
            continue
        fresh = blob_store.modified_at(entry.uri)
        if fresh is None or fresh > cutoff:
            continue
        try:
            blob_store.delete(entry.uri)
        except Exception:
            logger.warning("blob_gc_delete_failed", blob_uri=entry.uri)
            continue
        reclaimed += 1
    return reclaimed
