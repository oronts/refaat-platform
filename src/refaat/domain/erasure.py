"""The right-to-erasure receipt.

Complete deletion physically removes or re-derives every auto-derived evidence surface a
document produced, so no residue of its content survives in a queryable or model-reachable
auto-derived surface. This module holds the pure value objects: an :class:`ErasureReceipt`
that carries only ids, per-surface counts, and timestamps, never any chunk text, hypothetical
question, summary, or other content. A legal hold is enforced directly from the durable
legal_holds table (store.is_document_held), not through a pluggable seam.

Physical deletion is the safe default here because the store is plaintext at rest. Cryptographic
erasure through per-document keys is a separate at-rest-encryption concern and out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class ErasureStatus(StrEnum):
    HELD = "held"
    DEFERRED = "deferred"
    ERASED = "erased"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ErasureReceipt:
    """The outcome of one erasure request, redacted by construction.

    Every field is an id, a per-surface count, a status, or a timestamp. No chunk text,
    hypothetical question, summary, answer, or other content is ever carried, so a receipt
    is safe to persist in the ledger and to derive an audit event from.
    """

    request_id: str
    workspace_id: str
    corpus_id: str
    document_id: str
    status: ErasureStatus
    occurred_at: datetime
    corpus_version: int = 0
    chunks: int = 0
    hypotheticals: int = 0
    entities: int = 0
    relations: int = 0
    raptor_rebuilt: int = 0
    query_cache_rows: int = 0
    generation_cache_rows: int = 0
    embedding_cache_rows: int = 0
    session_summaries_cleared: int = 0
    session_turns_deleted: int = 0
    preferences: int = 0
    outbox_rows: int = 0
    blob_deleted: bool = False

    def counts(self) -> dict[str, int | bool]:
        """The per-surface counts as a plain mapping for durable persistence.

        Ids, status, and timestamps live in dedicated ledger columns; this is only the
        count payload, so nothing here can carry content.
        """
        return {
            "corpus_version": self.corpus_version,
            "chunks": self.chunks,
            "hypotheticals": self.hypotheticals,
            "entities": self.entities,
            "relations": self.relations,
            "raptor_rebuilt": self.raptor_rebuilt,
            "query_cache_rows": self.query_cache_rows,
            "generation_cache_rows": self.generation_cache_rows,
            "embedding_cache_rows": self.embedding_cache_rows,
            "session_summaries_cleared": self.session_summaries_cleared,
            "session_turns_deleted": self.session_turns_deleted,
            "preferences": self.preferences,
            "outbox_rows": self.outbox_rows,
            "blob_deleted": self.blob_deleted,
        }


__all__ = ["ErasureReceipt", "ErasureStatus"]
