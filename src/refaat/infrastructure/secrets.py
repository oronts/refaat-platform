"""Reference secret providers for tests and simple deployments.

Both key only on the secret name and ignore the rest of the :class:`SecretScope`, so neither
enforces per-tenant isolation: that is a real vault or KMS provider's job. The default is no
provider installed, in which case the governed loop resolves no secret at all.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from refaat.domain.secrets import SecretScope, SecretValue


class StaticSecretProvider:
    """In-memory name to value map for tests and simple deployments. Ignores scope beyond the name."""

    def __init__(self, secrets: Mapping[str, str], *, provider_id: str = "static") -> None:
        self._secrets = dict(secrets)
        self._provider_id = provider_id

    @property
    def provider_id(self) -> str:
        return self._provider_id

    def available(self, names: frozenset[str], *, scope: SecretScope) -> frozenset[str]:
        del scope
        return frozenset(name for name in names if self._secrets.get(name))

    def resolve(self, name: str, *, scope: SecretScope) -> SecretValue | None:
        del scope
        raw = self._secrets.get(name)
        # An empty value is treated as unresolvable so a blank credential fails closed at the gate.
        return SecretValue(raw, name=name) if raw else None


class EnvSecretProvider:
    """Resolves a name to ``os.environ[prefix + name.upper()]``. Env is process-global visibility."""

    def __init__(self, *, prefix: str = "REFAAT_TOOL_SECRET_", provider_id: str = "env") -> None:
        self._prefix = prefix
        self._provider_id = provider_id

    @property
    def provider_id(self) -> str:
        return self._provider_id

    def _key(self, name: str) -> str:
        return self._prefix + name.upper().replace("-", "_")

    def available(self, names: frozenset[str], *, scope: SecretScope) -> frozenset[str]:
        del scope
        return frozenset(name for name in names if os.environ.get(self._key(name)))

    def resolve(self, name: str, *, scope: SecretScope) -> SecretValue | None:
        del scope
        raw = os.environ.get(self._key(name))
        return SecretValue(raw, name=name) if raw else None


__all__ = ["EnvSecretProvider", "StaticSecretProvider"]
