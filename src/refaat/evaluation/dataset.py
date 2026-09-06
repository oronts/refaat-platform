from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

from refaat.domain.models import JsonObject


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    id: str
    query: str
    expected_terms: tuple[str, ...] = ()
    expected_document_ids: tuple[str, ...] = ()
    reference_answer: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Dataset:
    """An immutable, content-addressed set of evaluation cases."""

    name: str
    cases: tuple[EvaluationCase, ...]

    @property
    def version(self) -> str:
        payload = [
            {
                "id": case.id,
                "query": case.query,
                "expected_terms": list(case.expected_terms),
                "expected_document_ids": list(case.expected_document_ids),
                "reference_answer": case.reference_answer,
                "tags": list(case.tags),
            }
            for case in self.cases
        ]
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"

    def to_records(self) -> list[JsonObject]:
        return [
            {
                "id": case.id,
                "query": case.query,
                "expected_terms": list(case.expected_terms),
                "expected_document_ids": list(case.expected_document_ids),
                "reference_answer": case.reference_answer,
                "tags": list(case.tags),
            }
            for case in self.cases
        ]

    @classmethod
    def from_records(cls, name: str, records: Sequence[JsonObject]) -> Dataset:
        cases = tuple(cls._case_from_record(index, record) for index, record in enumerate(records))
        if not cases:
            raise ValueError("Dataset must contain at least one case")
        return cls(name=name, cases=cases)

    @staticmethod
    def _case_from_record(index: int, record: JsonObject) -> EvaluationCase:
        # Faithful inverse of to_records: a present but empty id or reference is
        # preserved as-is (not collapsed to a synthetic id or None), so a
        # round-trip recomputes the same content-addressed dataset version and a
        # resumed durable run reassembles a byte-identical result. A missing id
        # falls back to a stable positional id.
        raw_id = record.get("id")
        raw_reference = record.get("reference_answer")
        return EvaluationCase(
            id=str(raw_id) if raw_id is not None else f"case-{index}",
            query=str(record["query"]),
            expected_terms=tuple(str(value) for value in record.get("expected_terms", [])),
            expected_document_ids=tuple(str(value) for value in record.get("expected_document_ids", [])),
            reference_answer=str(raw_reference) if raw_reference is not None else None,
            tags=tuple(str(value) for value in record.get("tags", [])),
        )
