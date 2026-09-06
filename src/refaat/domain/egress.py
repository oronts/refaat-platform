"""Secure-by-default egress governance: a pure decision core plus its policy port.

An operator declares which outbound network destinations the runtime may reach. Every outbound
httpx seam routes through one shared chokepoint, ``ensure_egress_allowed``, so a destination that is
not on the allowlist is denied before the socket opens. The runtime is byte-identical when no policy
is installed (the chokepoint is a no-op), and fail-closed when one is: an un-allowlisted destination,
a malformed destination, and a policy whose ``decide`` itself raises all deny.

Matching semantics: a rule host is lowercased, so ``API.Example.com`` matches a lowercased
destination. A URL with no explicit port normalizes to the scheme default (443 for https, 80 for
http) before the port comparison, so a rule ``port: 443`` matches ``https://host/``. A leading-dot
suffix rule ``.example.com`` matches ``example.com`` and any subdomain, and is boundary-safe, so it
can never be satisfied by ``evilexample.com``. Scheme and purpose are optional per rule (empty means
any). An IDN host must be listed in the same form urllib returns; every mismatch fails closed
(over-deny), so a misconfiguration never silently widens access.

This module is a domain leaf: it imports only the standard library and ``refaat.errors``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol
from urllib.parse import urlparse

from refaat.errors import ConfigurationError, EgressDeniedError

_DEFAULT_PORTS = {"https": 443, "http": 80}
_ALLOWED_RULE_KEYS = frozenset({"host", "port", "schemes", "purposes"})


class EgressPurpose(StrEnum):
    MODEL = "model"
    CONNECTOR = "connector"
    WEBHOOK = "webhook"
    FINE_TUNING = "fine_tuning"
    OBSERVABILITY = "observability"
    MCP = "mcp"
    BLOB = "blob"
    PLUGIN = "plugin"


@dataclass(frozen=True, slots=True)
class EgressDestination:
    """A logical outbound destination: the real hostname, never a pinned IP."""

    scheme: str
    host: str
    port: int | None
    purpose: EgressPurpose

    @classmethod
    def from_url(cls, url: str, purpose: EgressPurpose) -> EgressDestination:
        parsed = urlparse(url)
        host = parsed.hostname
        try:
            port = parsed.port
        except ValueError as exc:
            # An out-of-range or non-numeric port is a malformed destination, not a raw error.
            raise EgressDeniedError("malformed_destination", "", purpose) from exc
        if parsed.scheme not in _DEFAULT_PORTS or not host:
            # A destination the runtime cannot parse into a scheme and host is denied, so a
            # malformed base_url or connector source can never egress unchecked.
            raise EgressDeniedError("malformed_destination", "", purpose)
        return cls(scheme=parsed.scheme, host=host.lower(), port=port, purpose=purpose)

    @property
    def effective_port(self) -> int:
        return self.port if self.port is not None else _DEFAULT_PORTS[self.scheme]


@dataclass(frozen=True, slots=True)
class EgressDecision:
    allowed: bool
    reason_code: str
    host: str


@dataclass(frozen=True, slots=True)
class EgressRule:
    host: str
    port: int | None = None
    schemes: frozenset[str] = field(default_factory=frozenset)
    purposes: frozenset[EgressPurpose] = field(default_factory=frozenset)

    def matches(self, dest: EgressDestination) -> bool:
        if not self._host_matches(dest.host):
            return False
        if self.port is not None and self.port != dest.effective_port:
            return False
        if self.schemes and dest.scheme not in self.schemes:
            return False
        return not (self.purposes and dest.purpose not in self.purposes)

    def _host_matches(self, host: str) -> bool:
        if self.host.startswith("."):
            bare = self.host[1:]
            return host == bare or host.endswith(self.host)
        return host == self.host

    def _identity(self) -> str:
        schemes = ",".join(sorted(self.schemes))
        purposes = ",".join(sorted(purpose.value for purpose in self.purposes))
        return f"{self.host}|{self.port}|{schemes}|{purposes}"


class EgressPolicy(Protocol):
    """The outbound-destination decision surface, resolved once per seam call."""

    cache_id: str | None

    def decide(self, dest: EgressDestination) -> EgressDecision: ...


class NullEgressPolicy:
    """An always-allowing policy: every destination is permitted. For tests that want an explicit
    no-op double. It is never installed by default; the default installed state is None."""

    cache_id: str | None = None

    def decide(self, dest: EgressDestination) -> EgressDecision:
        return EgressDecision(allowed=True, reason_code="ok", host=dest.host)


class AllowlistEgressPolicy:
    """A default-deny reference policy: a destination is allowed only when a rule matches it."""

    def __init__(self, rules: tuple[EgressRule, ...]) -> None:
        self._rules = rules
        digest = hashlib.sha256("\n".join(sorted(rule._identity() for rule in rules)).encode()).hexdigest()
        self.cache_id: str | None = f"allowlist:{digest}"

    @property
    def rules(self) -> tuple[EgressRule, ...]:
        return self._rules

    def decide(self, dest: EgressDestination) -> EgressDecision:
        for rule in self._rules:
            if rule.matches(dest):
                return EgressDecision(allowed=True, reason_code="ok", host=dest.host)
        return EgressDecision(allowed=False, reason_code="not_allowlisted", host=dest.host)


def allowlist_from_config(raw: Iterable[Any]) -> tuple[EgressRule, ...]:
    """Parse and validate the operator allowlist into an immutable rule snapshot.

    Raises ``ConfigurationError`` on any malformed entry so a bad egress configuration fails at
    startup rather than silently denying every destination (or, worse, allowing them).
    """
    rules: list[EgressRule] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ConfigurationError("Each egress allowlist entry must be a mapping")
        unknown = set(entry) - _ALLOWED_RULE_KEYS
        if unknown:
            raise ConfigurationError(f"Unknown egress allowlist keys: {sorted(unknown)}")
        host = entry.get("host")
        if not isinstance(host, str) or not host.strip():
            raise ConfigurationError(f"An egress allowlist entry requires a non-empty host: {entry!r}")
        port_raw = entry.get("port")
        try:
            port = None if port_raw is None else int(port_raw)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"Malformed egress allowlist port: {entry!r}") from exc
        if port is not None and not 0 < port < 65536:
            raise ConfigurationError(f"An egress allowlist port must be in 1..65535: {entry!r}")
        schemes = _parse_scheme_set(entry.get("schemes"), entry)
        purposes = _parse_purpose_set(entry.get("purposes"), entry)
        rules.append(EgressRule(host=host.strip().lower(), port=port, schemes=schemes, purposes=purposes))
    return tuple(rules)


def _parse_scheme_set(raw: Any, entry: dict[str, Any]) -> frozenset[str]:
    if raw is None:
        return frozenset()
    if not isinstance(raw, (list, tuple)) or not all(isinstance(item, str) for item in raw):
        raise ConfigurationError(f"Egress allowlist schemes must be a list of strings: {entry!r}")
    schemes = {item.strip().lower() for item in raw}
    if not schemes <= set(_DEFAULT_PORTS):
        raise ConfigurationError(f"Egress allowlist schemes must be http or https: {entry!r}")
    return frozenset(schemes)


def _parse_purpose_set(raw: Any, entry: dict[str, Any]) -> frozenset[EgressPurpose]:
    if raw is None:
        return frozenset()
    if not isinstance(raw, (list, tuple)):
        raise ConfigurationError(f"Egress allowlist purposes must be a list: {entry!r}")
    try:
        return frozenset(EgressPurpose(str(item)) for item in raw)
    except ValueError as exc:
        raise ConfigurationError(f"Malformed egress allowlist purpose: {entry!r}") from exc


def ensure_egress_allowed(policy: EgressPolicy | None, dest: EgressDestination) -> None:
    """Admit one outbound destination, raising when the policy denies it.

    The shared chokepoint every outbound httpx seam calls before it opens a socket. A no-op when no
    policy is installed, so the runtime is byte-identical when unconfigured. A policy whose ``decide``
    raises is converted to a ``policy_unavailable`` deny, so a policy-read failure denies rather than
    serving unchecked.
    """
    if policy is None:
        return
    try:
        decision = policy.decide(dest)
    except EgressDeniedError:
        raise
    except Exception as exc:
        raise EgressDeniedError("policy_unavailable", dest.host, dest.purpose) from exc
    if not decision.allowed:
        raise EgressDeniedError(decision.reason_code, dest.host, dest.purpose)


__all__ = [
    "AllowlistEgressPolicy",
    "EgressDecision",
    "EgressDestination",
    "EgressPolicy",
    "EgressPurpose",
    "EgressRule",
    "NullEgressPolicy",
    "allowlist_from_config",
    "ensure_egress_allowed",
]
