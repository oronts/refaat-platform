"""Run-scoped resolution of capability-declared secrets for governed tool execution.

A :class:`SecretResolver` wraps an installed :class:`SecretProvider` for one workspace and is the
single object the governed ReAct loop threads in. It answers a cheap availability check for
fail-closed admission and materializes the declared secrets just before ``skill.run``, auditing by
name only. With no provider installed the governed loop never constructs one, so it is byte-identical.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from typing import Any

from refaat.clock import Clock
from refaat.core.text import stable_digest
from refaat.domain.audit import AuditEvent, AuditSink, emit_audit
from refaat.domain.governance import ToolCapability
from refaat.domain.secrets import SecretBundle, SecretScope, SecretValue
from refaat.observability import get_logger
from refaat.ports import SecretProvider

logger = get_logger(__name__)

_SECRET_MASK = "[secret]"
_SECRET_FAILURE_MESSAGE = "the governed tool call failed while a secret was in scope"


class _SanitizedSkillError(RuntimeError):
    """A skill failure whose message was scrubbed of revealed secret values inside the governed loop."""


def redact_secret_values(obj: Any, values: frozenset[str]) -> Any:
    """Recursively mask every revealed secret substring in a skill observation or error payload.

    Walks strings, dicts, lists, and tuples and replaces any substring equal to a resolved secret
    value with a fixed mask, so a revealed secret a skill copies into its return value or an
    exception can never reach the recorded step, the model, a log line, or an SSE error. Blank
    values are ignored, and longer values are masked first so no fragment of one survives inside
    another.
    """
    active = _active_values(values)
    return _redact(obj, active) if active else obj


def sanitize_secret_exception(exc: BaseException, values: frozenset[str]) -> BaseException:
    """Replace a skill exception raised while secrets were live with a fixed, secret-free error.

    A skill holding a revealed secret can encode it into any exception surface (message, args, notes,
    group members, even the type name, split or transformed), all of which traceback.format_exception
    walks through the object's own adversarial __str__ and getters. No masking of that text is
    complete, so the sanitizer never touches the object and returns a fresh error carrying only a fixed
    message; the caller raises it outside the except handler so the original is not retained in
    __context__, dropping the credential-bearing graph entirely.
    """
    if not _active_values(values):
        return exc
    return _SanitizedSkillError(_SECRET_FAILURE_MESSAGE)


def _active_values(values: frozenset[str]) -> tuple[str, ...]:
    return tuple(sorted((value for value in values if value.strip()), key=len, reverse=True))


def _redact(obj: Any, values: tuple[str, ...]) -> Any:
    if isinstance(obj, str):
        for value in values:
            obj = obj.replace(value, _SECRET_MASK)
        return obj
    if isinstance(obj, Mapping):
        return {_redact(key, values): _redact(item, values) for key, item in obj.items()}
    if isinstance(obj, list):
        return [_redact(item, values) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_redact(item, values) for item in obj)
    return obj


class SecretResolver:
    def __init__(
        self,
        provider: SecretProvider,
        workspace_id: str,
        *,
        clock: Clock,
        audit_sink: AuditSink | None = None,
        audit_fail_closed: bool = False,
    ) -> None:
        self._provider = provider
        self._workspace_id = workspace_id
        self._clock = clock
        self._audit_sink = audit_sink
        self._audit_fail_closed = audit_fail_closed

    def available(self, capability: ToolCapability, scope: SecretScope) -> frozenset[str]:
        return self._provider.available(capability.required_secrets, scope=scope)

    def materialize(self, capability: ToolCapability, scope: SecretScope) -> SecretBundle | None:
        """Resolve every declared name, or return None (fail-closed) if any is unresolvable.

        Audits by name only. Called before the approval consume, so an unresolved required secret
        denies the step without burning any single-use approval. On any resolution or audit failure
        every already-resolved value is scrubbed before the error propagates, so a partial resolution
        leaves no revealable secret in memory.
        """
        resolved: dict[str, SecretValue] = {}
        # The value being resolved is tracked here before its reveal() is called, so a reveal that
        # raises (a provider returning a value whose reveal misbehaves) still scrubs it below.
        pending: SecretValue | None = None
        try:
            for name in sorted(capability.required_secrets):
                pending = value = self._provider.resolve(name, scope=scope)
                if value is None or not value.reveal():
                    for got in resolved.values():
                        got.scrub()
                    self._emit("tool.secret_denied", capability, scope, names=capability.required_secrets)
                    return None
                resolved[name] = value
                pending = None
            self._emit("tool.secret_resolved", capability, scope, names=frozenset(resolved))
        except Exception:
            for got in resolved.values():
                got.scrub()
            if pending is not None:
                with contextlib.suppress(Exception):
                    pending.scrub()
            raise
        return SecretBundle(resolved)

    def _emit(self, event_type: str, capability: ToolCapability, scope: SecretScope, *, names: frozenset[str]) -> None:
        if self._audit_sink is None:
            return
        ordered = sorted(names)
        event = AuditEvent(
            event_type=event_type,
            occurred_at=self._clock.now(),
            workspace_id=self._workspace_id,
            principal_id=scope.principal_id or "system",
            principal_kind=scope.principal_kind or "system",
            tenant=scope.tenant,
            policy_id="react",
            policy_version="1",
            capability_name=capability.name,
            action_fingerprint=stable_digest(("tool.secret", self._provider.provider_id, *ordered)),
            dedup_id=stable_digest((event_type, self._workspace_id, capability.name, *ordered)),
        )
        emit_audit(
            self._audit_sink,
            event,
            fail_closed=self._audit_fail_closed,
            logger=logger,
            log_key="secret_audit_sink_failed",
        )


__all__ = ["SecretResolver", "redact_secret_values", "sanitize_secret_exception"]
