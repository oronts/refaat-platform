"""Legal-hold domain value objects and the actor-id bound.

LegalHoldError is a DIRECT RefaatError(409), never a ConflictError subclass, so the ingest()
except-ConflictError retry can never swallow an overwrite refusal. validate_actor_id bounds
placed_by/lifted_by before they reach an audit principal_id.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from refaat.errors import RefaatError, ValidationError


class LegalHoldError(RefaatError):
    code = "legal_hold"
    status_code = 409


_ACTOR_ID_MAX = 200
_ACTOR_ID_RE = re.compile(r"[^A-Za-z0-9_.:@\-]")


def validate_actor_id(actor_id: str) -> str:
    value = (actor_id or "").strip()
    if not value:
        raise ValidationError("actor id is required")
    if len(value) > _ACTOR_ID_MAX:
        raise ValidationError("actor id too long")
    if _ACTOR_ID_RE.search(value):
        raise ValidationError("actor id contains invalid characters")
    return value


def validate_hold_reason(reason: str | None) -> str | None:
    """Bound the operator-supplied reason before it lands on the row.

    The reason lives on the legal_holds row only and never enters an audit event, but it is still
    bounded so a client cannot store an unbounded blob. None passes through unchanged.
    """
    if reason is None:
        return None
    value = reason.strip()
    if not value:
        return None
    if len(value) > _REASON_MAX:
        raise ValidationError("hold reason too long")
    return value


_REASON_MAX = 2000

_ID_REF_MAX = 64
_ID_REF_RE = re.compile(r"^[A-Za-z0-9_.:\-]+$")


def validate_corpus_ref(corpus_id: str) -> str:
    """Bound a corpus id to an id shape before it reaches an audit event.

    A hold event carries the corpus id, so a bounded id with no whitespace or control characters
    keeps free text out of the redacted audit trail.
    """
    value = (corpus_id or "").strip()
    if not value or len(value) > _ID_REF_MAX or not _ID_REF_RE.match(value):
        raise ValidationError("corpus id is not a valid identifier")
    return value


def validate_document_ref(document_id: str | None) -> str | None:
    """Bound a document id to an id shape before it can reach an audit event's cited ids.

    None passes through (a corpus-wide hold). A present value must be a bounded id with no
    whitespace or control characters, so free text can never enter a redacted audit event.
    """
    if document_id is None:
        return None
    value = document_id.strip()
    if not value or len(value) > _ID_REF_MAX or not _ID_REF_RE.match(value):
        raise ValidationError("document id is not a valid identifier")
    return value


@dataclass(frozen=True, slots=True)
class LegalHoldRecord:
    """The render type for a stored hold. Wired as list_active()'s return type (not dead code)."""

    id: str
    workspace_id: str
    corpus_id: str
    document_id: str | None
    status: str
    reason: str | None
    placed_by: str
    placed_at: datetime
    lifted_by: str | None
    lifted_at: datetime | None

    @property
    def active(self) -> bool:
        return self.lifted_at is None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> LegalHoldRecord:
        return cls(
            id=str(row["id"]),
            workspace_id=str(row["workspace_id"]),
            corpus_id=str(row["corpus_id"]),
            document_id=(str(row["document_id"]) if row["document_id"] is not None else None),
            status=str(row["status"]),
            reason=(str(row["reason"]) if row["reason"] is not None else None),
            placed_by=str(row["placed_by"]),
            placed_at=row["placed_at"],
            lifted_by=(str(row["lifted_by"]) if row["lifted_by"] is not None else None),
            lifted_at=row["lifted_at"],
        )


__all__ = [
    "LegalHoldError",
    "LegalHoldRecord",
    "validate_actor_id",
    "validate_corpus_ref",
    "validate_document_ref",
    "validate_hold_reason",
]
