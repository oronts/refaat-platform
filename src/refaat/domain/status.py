"""The admin status plane value objects: a redacted, read-only posture snapshot.

Every value here carries only ids, counts, booleans, enum values, cache identifiers, and
timestamps, never a secret, evidence text, a raw principal, a hold or switch reason, or a raw
quota scope key. Redaction is enforced at the readers that source these objects (counts and
booleans only); the scalar check in ``SubsystemStatus`` is defense in depth, not the guarantee.
This module is a domain leaf: it imports only the standard library.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

JsonScalar = str | int | float | bool | None


class SubsystemState(StrEnum):
    NOT_CONFIGURED = "not_configured"
    ACTIVE = "active"
    ENGAGED = "engaged"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"
    UNMANAGED = "unmanaged"


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


@dataclass(frozen=True, slots=True)
class SubsystemStatus:
    name: str
    state: SubsystemState
    configured: bool
    detail: Mapping[str, JsonScalar | tuple[JsonScalar, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key, value in self.detail.items():
            if isinstance(value, tuple):
                if not all(_is_scalar(item) for item in value):
                    raise TypeError(f"Non-scalar element in status detail tuple {key!r}")
            elif not _is_scalar(value):
                raise TypeError(f"Non-scalar status detail value {key!r}")

    @classmethod
    def not_configured(cls, name: str) -> SubsystemStatus:
        return cls(name=name, state=SubsystemState.NOT_CONFIGURED, configured=False, detail={})

    @classmethod
    def unknown(cls, name: str) -> SubsystemStatus:
        return cls(name=name, state=SubsystemState.UNKNOWN, configured=False, detail={})

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "configured": self.configured,
            "detail": {key: (list(value) if isinstance(value, tuple) else value) for key, value in self.detail.items()},
        }


@dataclass(frozen=True, slots=True)
class BuildInfo:
    version: str
    environment: str
    features: Mapping[str, bool]
    state: SubsystemState

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "environment": self.environment,
            "features": dict(self.features),
            "state": self.state.value,
        }


@dataclass(frozen=True, slots=True)
class SchemaInfo:
    code_head: str | None
    db_revision: str | None
    at_head: bool
    managed: bool
    state: SubsystemState

    def to_json(self) -> dict[str, Any]:
        return {
            "code_head": self.code_head,
            "db_revision": self.db_revision,
            "at_head": self.at_head,
            "managed": self.managed,
            "state": self.state.value,
        }


@dataclass(frozen=True, slots=True)
class AdminStatus:
    generated_at: datetime
    scope: str
    workspace_id: str | None
    build: BuildInfo
    schema: SchemaInfo
    subsystems: tuple[SubsystemStatus, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "scope": self.scope,
            "workspace_id": self.workspace_id,
            "build": self.build.to_json(),
            "schema": self.schema.to_json(),
            "subsystems": [subsystem.to_json() for subsystem in self.subsystems],
        }


__all__ = [
    "AdminStatus",
    "BuildInfo",
    "JsonScalar",
    "SchemaInfo",
    "SubsystemState",
    "SubsystemStatus",
]
