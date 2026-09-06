from __future__ import annotations

from enum import StrEnum


class DocumentStatus(StrEnum):
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MessageStatus(StrEnum):
    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"


class EvaluationStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class SideEffectClass(StrEnum):
    READ_ONLY = "read_only"
    WRITE = "write"
    SENSITIVE_WRITE = "sensitive_write"
    IRREVERSIBLE = "irreversible"
    EXTERNAL = "external"

    @property
    def rank(self) -> int:
        return _SIDE_EFFECT_RANK[self]


# Ranked by governance blast radius: external egress is highest.
_SIDE_EFFECT_RANK: dict[SideEffectClass, int] = {
    SideEffectClass.READ_ONLY: 0,
    SideEffectClass.WRITE: 1,
    SideEffectClass.SENSITIVE_WRITE: 2,
    SideEffectClass.IRREVERSIBLE: 3,
    SideEffectClass.EXTERNAL: 4,
}


class AuditLevel(StrEnum):
    STANDARD = "standard"
    ENHANCED = "enhanced"

    @property
    def rank(self) -> int:
        return _AUDIT_LEVEL_RANK[self]


_AUDIT_LEVEL_RANK: dict[AuditLevel, int] = {
    AuditLevel.STANDARD: 0,
    AuditLevel.ENHANCED: 1,
}
