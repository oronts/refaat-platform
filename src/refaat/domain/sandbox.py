from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from refaat.domain.enums import SideEffectClass
from refaat.domain.models import JsonObject


@dataclass(frozen=True, slots=True)
class SandboxLimits:
    """Hard per-execution bounds a sandbox enforces on one code run.

    ``max_nodes`` caps the size of the evaluated program (its parsed node and
    operation count), ``max_output_bytes`` caps every intermediate and final
    value, and ``max_wall_seconds`` caps wall time. Every bound is a fail-closed
    ceiling: a program that reaches one is refused, never truncated silently.
    """

    max_nodes: int = 256
    max_output_bytes: int = 65_536
    max_wall_seconds: float = 2.0

    def __post_init__(self) -> None:
        if not 1 <= self.max_nodes <= 100_000:
            raise ValueError("max_nodes must be between 1 and 100000")
        if not 1 <= self.max_output_bytes <= 16 * 1024 * 1024:
            raise ValueError("max_output_bytes must be between 1 and 16777216")
        if not 0.0 < self.max_wall_seconds <= 60.0:
            raise ValueError("max_wall_seconds must be between 0 and 60")


@dataclass(frozen=True, slots=True)
class CodeRequest:
    """A caller-supplied computation the governed RLM may admit and run.

    ``inputs`` is the only data the code can read; a sandbox exposes nothing
    else. ``required_scopes`` and ``side_effect`` are the governance facts the
    :class:`~refaat.application.sandbox.CodeGovernor` checks against the policy,
    exactly as a tool grant is checked in a governed ReAct run.
    """

    code: str
    inputs: JsonObject = field(default_factory=dict)
    required_scopes: tuple[str, ...] = ()
    side_effect: SideEffectClass = SideEffectClass.READ_ONLY


@dataclass(frozen=True, slots=True)
class CodePolicy:
    """Permissions and bounds for a governed code phase.

    The caller holds ``granted_scopes`` and admits requests up to
    ``side_effect_ceiling``. ``max_steps`` hard-caps how many requests one phase
    runs, and ``limits`` bounds each execution. Passing a policy is the explicit
    opt-in: with no policy the RLM never enters a code phase and runs no code.
    """

    granted_scopes: tuple[str, ...] = ()
    side_effect_ceiling: SideEffectClass = SideEffectClass.READ_ONLY
    max_steps: int = 8
    limits: SandboxLimits = field(default_factory=SandboxLimits)

    def __post_init__(self) -> None:
        if not 1 <= self.max_steps <= 64:
            raise ValueError("max_steps must be between 1 and 64")


@dataclass(frozen=True, slots=True)
class SandboxResult:
    """The outcome of one sandbox execution.

    ``ok`` is true only when the code ran to completion within every bound.
    ``value`` is the computed result (JSON-friendly), ``error`` is a refusal or
    failure reason, and ``usage`` records the bounded resource counters.
    """

    ok: bool
    value: Any = None
    stdout: str = ""
    error: str | None = None
    usage: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CodeStep:
    """The audit record of one governed code request, admitted or denied.

    Both an admitted-and-executed step and a denied step are recorded, so every
    governance decision in an RLM run is inspectable after the fact.
    """

    code: str
    admitted: bool
    reason: str
    ok: bool
    sandbox: str
    value: Any = None
    error: str | None = None

    def as_json(self) -> JsonObject:
        return {
            "code": self.code,
            "admitted": self.admitted,
            "reason": self.reason,
            "ok": self.ok,
            "sandbox": self.sandbox,
            "value": self.value,
            "error": self.error,
        }
