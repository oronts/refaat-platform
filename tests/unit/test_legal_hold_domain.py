"""Legal-hold domain value objects: the actor-id bound, the error type, and the render DTO."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from refaat.domain.legal_hold import (
    LegalHoldError,
    LegalHoldRecord,
    validate_actor_id,
    validate_corpus_ref,
    validate_document_ref,
    validate_hold_reason,
)
from refaat.errors import ConflictError, RefaatError, ValidationError


def test_validate_actor_id_accepts_a_bounded_identifier() -> None:
    assert validate_actor_id("  counsel@acme.com  ") == "counsel@acme.com"
    assert validate_actor_id("api_key-123.ABC:role") == "api_key-123.ABC:role"


def test_validate_actor_id_rejects_empty() -> None:
    with pytest.raises(ValidationError):
        validate_actor_id("   ")


def test_validate_actor_id_rejects_over_length() -> None:
    with pytest.raises(ValidationError):
        validate_actor_id("x" * 201)


def test_validate_actor_id_rejects_disallowed_characters() -> None:
    with pytest.raises(ValidationError):
        validate_actor_id("bob smith")
    with pytest.raises(ValidationError):
        validate_actor_id("drop; delete")


def test_validate_hold_reason_bounds_and_normalizes() -> None:
    assert validate_hold_reason(None) is None
    assert validate_hold_reason("   ") is None
    assert validate_hold_reason("  matter 14  ") == "matter 14"
    with pytest.raises(ValidationError):
        validate_hold_reason("x" * 2001)


def test_validate_document_ref_passes_none_and_bounds_an_id() -> None:
    assert validate_document_ref(None) is None
    assert validate_document_ref("doc_9f3a.b:c-1") == "doc_9f3a.b:c-1"


def test_validate_document_ref_rejects_free_text() -> None:
    with pytest.raises(ValidationError):
        validate_document_ref("SECRETMATTER99 free text")
    with pytest.raises(ValidationError):
        validate_document_ref("x" * 65)
    with pytest.raises(ValidationError):
        validate_document_ref("")


def test_validate_corpus_ref_bounds_an_id_and_rejects_free_text() -> None:
    assert validate_corpus_ref("  corpus_1  ") == "corpus_1"
    with pytest.raises(ValidationError):
        validate_corpus_ref("corpus 1")
    with pytest.raises(ValidationError):
        validate_corpus_ref("")


def test_legal_hold_error_is_a_refaat_error_not_a_conflict_error() -> None:
    # A direct RefaatError(409), never a ConflictError subclass, so the ingest except-ConflictError
    # retry can never swallow an overwrite refusal.
    assert issubclass(LegalHoldError, RefaatError)
    assert not issubclass(LegalHoldError, ConflictError)
    assert LegalHoldError.status_code == 409


def _row(document_id: str | None) -> dict[str, object]:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return {
        "id": "hold_1",
        "workspace_id": "ws_1",
        "corpus_id": "corpus_1",
        "document_id": document_id,
        "status": "active",
        "reason": None,
        "placed_by": "counsel@acme.com",
        "placed_at": now,
        "lifted_by": None,
        "lifted_at": None,
    }


def test_record_from_row_round_trips_a_corpus_wide_hold() -> None:
    record = LegalHoldRecord.from_row(_row(document_id=None))
    assert record.document_id is None
    assert record.active is True


def test_record_from_row_round_trips_a_document_hold_and_reflects_lifted() -> None:
    row = _row(document_id="doc_9")
    record = LegalHoldRecord.from_row(row)
    assert record.document_id == "doc_9"
    assert record.active is True

    lifted_at = datetime(2026, 2, 2, tzinfo=UTC)
    row["lifted_at"] = lifted_at
    row["lifted_by"] = "counsel@acme.com"
    row["status"] = "lifted"
    lifted = LegalHoldRecord.from_row(row)
    assert lifted.active is False
    assert lifted.lifted_by == "counsel@acme.com"


def test_record_has_no_covers_method() -> None:
    # Coverage is the SQL-only authority (store.is_document_held); the DTO carries no covers().
    assert not hasattr(LegalHoldRecord, "covers")
