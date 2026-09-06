"""Incident kill switches: a pure decision core plus its registry port.

An operator engages a switch to deny a class of operations fleet-wide during an incident without a
redeploy. A switch is scoped to a workspace, or global when its workspace is the empty sentinel. The
snapshot is a pure predicate with no I/O; when the storage read fails the snapshot is unavailable and
every scope and workspace is denied, so a read failure fails closed rather than serving. This module
is a domain leaf: it imports only the standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol

from refaat.errors import KillSwitchEngagedError


class KillSwitchScope(StrEnum):
    EXTERNAL_EGRESS = "external_egress"
    TOOL_EXECUTION = "tool_execution"
    WRITES = "writes"
    GENERATION = "generation"
    WORKSPACE_FREEZE = "workspace_freeze"


@dataclass(frozen=True, slots=True)
class KillSwitch:
    id: str
    workspace_id: str
    scope: str
    engaged_by: str
    engaged_at: datetime
    reason: str | None = None
    disengaged_by: str | None = None
    disengaged_at: datetime | None = None

    @property
    def active(self) -> bool:
        return self.disengaged_at is None

    @classmethod
    def from_row(cls, row: Any) -> KillSwitch:
        return cls(
            id=str(row["id"]),
            workspace_id=str(row["workspace_id"]),
            scope=str(row["scope"]),
            engaged_by=str(row["engaged_by"]),
            engaged_at=row["engaged_at"],
            reason=(str(row["reason"]) if row["reason"] is not None else None),
            disengaged_by=(str(row["disengaged_by"]) if row["disengaged_by"] is not None else None),
            disengaged_at=row["disengaged_at"],
        )


@dataclass(frozen=True, slots=True)
class KillSwitchSet:
    """An immutable snapshot of the engaged switches, keyed by (workspace_id, scope).

    An empty workspace in a key is the global sentinel that applies to every workspace. When the
    snapshot is unavailable (the storage read failed) every query denies, so a read failure never
    serves.
    """

    engaged: frozenset[tuple[str, str]]
    available: bool = True

    @classmethod
    def empty(cls) -> KillSwitchSet:
        return cls(frozenset())

    @classmethod
    def unavailable(cls) -> KillSwitchSet:
        return cls(frozenset(), available=False)

    def freezes(self, workspace_id: str) -> bool:
        if not self.available:
            return True
        scope = KillSwitchScope.WORKSPACE_FREEZE.value
        return ("", scope) in self.engaged or (workspace_id, scope) in self.engaged

    def denies(self, scope: KillSwitchScope, workspace_id: str) -> bool:
        if not self.available:
            return True
        if self.freezes(workspace_id):
            return True
        name = scope.value
        return ("", name) in self.engaged or (workspace_id, name) in self.engaged


class KillSwitchRegistry(Protocol):
    """The engaged-switch snapshot source, resolved per request behind a cache."""

    def engaged(self) -> KillSwitchSet: ...


class NullKillSwitchRegistry:
    """An always-open registry: nothing is ever engaged. For tests that want an explicit no-op."""

    def engaged(self) -> KillSwitchSet:
        return KillSwitchSet.empty()


def ensure_workspace_active(registry: KillSwitchRegistry | None, workspace_id: str) -> None:
    """Raise if an incident kill switch freezes the workspace; a no-op when no registry is installed.

    The shared freeze chokepoint the evidence-serving and mutation services call at entry, so a frozen
    workspace is denied regardless of which caller reached the service. Byte-identical when off: with no
    registry the check short-circuits before any snapshot read.
    """
    if registry is not None and registry.engaged().freezes(workspace_id):
        raise KillSwitchEngagedError("This workspace is frozen by an incident kill switch")


def ensure_scope_active(registry: KillSwitchRegistry | None, workspace_id: str, scope: KillSwitchScope) -> None:
    """Raise if a kill switch denies this scope in the workspace; a no-op when no registry is installed.

    Extends the freeze chokepoint to the per-scope switches: a workspace freeze denies every scope, and
    a scope switch denies just its operations. Byte-identical when off: with no registry, and with the
    scope neither engaged nor frozen, the check short-circuits without raising.
    """
    if registry is not None and registry.engaged().denies(scope, workspace_id):
        raise KillSwitchEngagedError(f"This operation is blocked by an incident kill switch: {scope.value}")


__all__ = [
    "KillSwitch",
    "KillSwitchRegistry",
    "KillSwitchScope",
    "KillSwitchSet",
    "NullKillSwitchRegistry",
    "ensure_scope_active",
    "ensure_workspace_active",
]
