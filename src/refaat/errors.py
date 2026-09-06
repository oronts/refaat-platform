from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from refaat.domain.egress import EgressPurpose


def error_message(exc: BaseException) -> str:
    """A never-empty error string for a persisted failure.

    An exception carrying an empty message must not surface as an empty error, or a
    downstream ``if error:`` truthiness check would read the failure as a success.
    Falls back to the exception type name.
    """
    return str(exc) or exc.__class__.__name__


class RefaatError(Exception):
    """Base class for expected platform errors."""

    code = "refaat_error"
    status_code = 500


class NotFoundError(RefaatError):
    code = "not_found"
    status_code = 404


class ConflictError(RefaatError):
    code = "conflict"
    status_code = 409


class AuthenticationError(RefaatError):
    code = "authentication_failed"
    status_code = 401


class AuthorizationError(RefaatError):
    code = "forbidden"
    status_code = 403


class ValidationError(RefaatError):
    code = "validation_error"
    status_code = 422


class RateLimitError(RefaatError):
    code = "rate_limited"
    status_code = 429


class ConfigurationError(RefaatError):
    code = "configuration_error"
    status_code = 500


class DependencyUnavailableError(RefaatError):
    code = "dependency_unavailable"
    status_code = 503


class ProviderCircuitOpenError(DependencyUnavailableError):
    code = "provider_circuit_open"
    status_code = 503

    def __init__(self, message: str = "Provider circuit open", *, retry_after_seconds: float | None = None) -> None:
        self.retry_after_seconds = retry_after_seconds
        super().__init__(message)


class CapabilityNotSupportedError(RefaatError):
    code = "capability_not_supported"
    status_code = 501


class AuditUnavailableError(RefaatError):
    code = "audit_unavailable"
    status_code = 503


class KillSwitchEngagedError(RefaatError):
    code = "kill_switch_engaged"
    status_code = 503


class QuotaExceededError(RefaatError):
    code = "quota_exceeded"
    status_code = 429


# Per-tool timeout outcome vocabulary. A tool that provably never started (its worker slot was never
# acquired) is a CONFIRMED no-effect; a tool that started but was abandoned at the deadline is UNKNOWN,
# because a plain skill's I/O is not runtime-terminated and may have already reached an external system.
TOOL_OUTCOME_UNKNOWN = "unknown"
TOOL_OUTCOME_CONFIRMED_NO_EFFECT = "confirmed_no_effect"
TOOL_TIMEOUT_REASON_CONFIRMED = "tool_timeout:confirmed_never_started"
TOOL_TIMEOUT_REASON_UNKNOWN = "tool_timeout:unknown_abandoned"


class ToolTimeoutError(RefaatError):
    code = "tool_timeout"
    # 409, not 504: a gateway-timeout invites a generic mesh or SDK to auto-retry, but a write-capable
    # tool abandoned at the deadline may already have reached an external system, so the outcome is a
    # reconcile-required conflict, never a safe nothing-happened retry.
    status_code = 409

    def __init__(self, *, outcome: str, side_effect: str, tool: str, execution_terminated: bool = False) -> None:
        self.outcome = outcome
        self.side_effect = side_effect
        self.tool = tool
        self.execution_terminated = execution_terminated
        # Redacted by construction: only the machine outcome, the side-effect class, and the tool name,
        # never an argument value, an observation, or a URL.
        super().__init__(f"Tool timed out ({outcome}:{side_effect})")


class EgressDeniedError(RefaatError):
    code = "egress_denied"
    status_code = 502

    def __init__(self, reason_code: str, host: str, purpose: EgressPurpose | str) -> None:
        self.reason_code = reason_code
        self.host = host
        self.purpose = purpose
        # Redacted by construction: the message carries only the machine reason code, never the URL
        # path, query, payload, or a secret.
        super().__init__(f"Outbound destination denied ({reason_code})")
