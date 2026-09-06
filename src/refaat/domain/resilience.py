"""The per-provider circuit breaker vocabulary: a pure, model-independent decision surface.

The breaker keys on a provider endpoint (a normalized base URL authority plus path prefix) so an
embedder and a generator on the same endpoint share one breaker while a path-multiplexed gateway keeps
independent ones. A call reports one of three outcomes: SUCCESS closes the breaker, FAILURE (a genuine
transport or 5xx fault) trips it toward OPEN, and INCONCLUSIVE (a client-side or parse fault, or an
abandoned stream) neither closes nor trips a closed breaker but fails an open probe. Importing this
module changes nothing; the stateful in-memory implementation lives in infrastructure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CallOutcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class CircuitBreakerConfig:
    failure_threshold: int
    reset_timeout_seconds: float
    half_open_deadline_seconds: float
    count_429: bool = False

    def __post_init__(self) -> None:
        # Validate at the value object so an injected breaker is checked like the settings path.
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        for name, value in (
            ("reset_timeout_seconds", self.reset_timeout_seconds),
            ("half_open_deadline_seconds", self.half_open_deadline_seconds),
        ):
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be a positive finite number")


@runtime_checkable
class CircuitBreaker(Protocol):
    def acquire(self, endpoint: str) -> int | None:
        """A token to pass back to record, or None to fast-fail (the circuit is open)."""

    def record(self, endpoint: str, token: int, outcome: CallOutcome) -> None:
        """Report the call outcome under the token returned by acquire."""


__all__ = ["CallOutcome", "CircuitBreaker", "CircuitBreakerConfig", "CircuitState"]
