"""Execution-time secret values, redacted by construction.

A :class:`SecretValue` wraps one credential the governed loop injects into an authorized tool at
the moment of a side effect. It is deliberately not a dataclass, a str, or any JSON scalar, so an
accidental repr, log, canonical serialization, or pickle can never leak the value: it masks on
str/repr/format, raises on json and pickle, and exposes a single ``reveal()`` accessor plus a
``scrub()`` that zeroes it after use. A :class:`SecretBundle` holds only the names one admitted
capability declared, so a skill can never reach beyond its grant. A :class:`SecretScope` is the
least-privilege context handed to the provider for one admitted capability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn, final

from refaat.core.text import stable_digest
from refaat.domain.enums import SideEffectClass


@final
class SecretValue:
    """One execution-time credential, masked on every surface except an explicit reveal."""

    __slots__ = ("_name", "_value")
    _MASK = "[secret]"

    def __init__(self, value: str, *, name: str) -> None:
        self._value = value
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def reveal(self) -> str:
        """The single accessor. Read it only at the moment of the side effect, never store it."""
        return self._value

    def fingerprint(self) -> str:
        # Name plus value digest: proves which credential version was used without revealing it.
        return stable_digest(("secret", self._name, self._value))

    def scrub(self) -> None:
        self._value = ""

    def __repr__(self) -> str:
        return f"SecretValue(name={self._name!r}, value={self._MASK})"

    def __str__(self) -> str:
        return self._MASK

    def __format__(self, spec: str) -> str:
        return self._MASK

    def __reduce__(self) -> NoReturn:
        raise TypeError("SecretValue is not picklable")


@dataclass(frozen=True, slots=True)
class SecretScope:
    """Least-privilege context handed to the provider for one admitted capability.

    The reference providers key only on the secret name; a real vault or KMS provider uses
    workspace, principal, tenant, and side_effect for per-tenant isolation and least privilege.
    """

    workspace_id: str
    capability_name: str
    side_effect: SideEffectClass
    principal_id: str | None = None
    principal_kind: str | None = None
    tenant: str | None = None


@final
class SecretBundle:
    """The resolved secrets for exactly one admitted capability. Reveal at the side effect, scrub after."""

    __slots__ = ("_by_name",)

    def __init__(self, values: dict[str, SecretValue]) -> None:
        self._by_name = values

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._by_name)

    def reveal(self, name: str) -> str:
        return self._by_name[name].reveal()

    def get(self, name: str) -> SecretValue | None:
        return self._by_name.get(name)

    def scrub(self) -> None:
        for value in self._by_name.values():
            value.scrub()

    def __reduce__(self) -> NoReturn:
        raise TypeError("SecretBundle is not picklable")


__all__ = ["SecretBundle", "SecretScope", "SecretValue"]
