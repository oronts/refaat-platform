"""Structural conformance: every member a port declares must be present on an adapter.

This is the universal contract for any port. It catches the most common authoring error, a missing or
misnamed member, before the adapter reaches the runtime. It does not exercise a port's behavioral
invariants (the citation contract, fail-closed egress); those are per-port contracts layered on top.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

from refaat.devkit.catalog import get_port


@dataclass(frozen=True, slots=True)
class ConformanceViolation:
    port: str
    member: str
    problem: str


def structural_conformance(adapter: object, port_name: str) -> list[ConformanceViolation]:
    """Return an adapter's violations of a port's declared members; an empty list means conformant.

    ``adapter`` may be a class or an instance. A member is satisfied when it is present, whether as a
    method, a property, or a plain value; presence is checked with ``inspect.getattr_static`` so a
    property getter is never executed during verification. An unknown port name is itself a single
    violation. This is a structural check only, not a check of a port's behavioral invariants.
    """
    spec = get_port(port_name)
    if spec is None:
        return [ConformanceViolation(port=port_name, member="", problem="unknown port")]
    violations: list[ConformanceViolation] = []
    for member in spec.members:
        try:
            inspect.getattr_static(adapter, member)
        except AttributeError:
            violations.append(ConformanceViolation(port=port_name, member=member, problem="missing"))
    return violations
