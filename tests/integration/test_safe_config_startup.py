"""Safe-config fail-startup security invariants: a resolver needs a policy, fail-closed audit needs a sink.

Both are misconfigurations that resolve identity or promise audit yet enforce nothing, so build refuses
rather than composing a runtime that serves unenforced. Byte-identical when unconfigured: a plain build,
a policy-only build (the library or worker deployment that passes the principal in code), and the default
audit path all build unchanged.
"""

from __future__ import annotations

from typing import Any

import pytest

from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.config import Settings
from refaat.domain.audit import AuditEvent
from refaat.domain.authz import CanonicalAccessPolicy
from refaat.errors import ConfigurationError

pytestmark = pytest.mark.integration


class _Resolver:
    def resolve(self, context: Any) -> Any:
        del context
        return None


class _Sink:
    def record(self, event: AuditEvent) -> None:
        del event


def _built(builder: RuntimeBuilder) -> None:
    runtime: Refaat = builder.build()
    runtime.close()


def test_resolver_without_access_policy_fails_startup(settings: Settings) -> None:
    builder = RuntimeBuilder(settings).with_principal_resolver(_Resolver())
    with pytest.raises(ConfigurationError, match="access policy"):
        builder.build()


def test_access_policy_without_resolver_builds(settings: Settings) -> None:
    # The library, worker, and Python-API deployments pass the principal in code, so a policy with no
    # resolver is legitimate and must never be rejected.
    _built(RuntimeBuilder(settings).with_access_policy(CanonicalAccessPolicy()))


def test_resolver_with_access_policy_builds(settings: Settings) -> None:
    _built(RuntimeBuilder(settings).with_principal_resolver(_Resolver()).with_access_policy(CanonicalAccessPolicy()))


def test_default_build_has_neither_resolver_nor_policy(settings: Settings) -> None:
    _built(RuntimeBuilder(settings))


def test_audit_fail_closed_without_sink_fails_startup(settings: Settings) -> None:
    builder = RuntimeBuilder(settings.model_copy(update={"audit_fail_closed": True}))
    with pytest.raises(ConfigurationError, match="audit sink"):
        builder.build()


def test_audit_fail_closed_with_sink_builds(settings: Settings) -> None:
    builder = RuntimeBuilder(settings.model_copy(update={"audit_fail_closed": True})).with_audit_sink(_Sink())
    _built(builder)


def test_default_audit_path_builds_without_sink(settings: Settings) -> None:
    _built(RuntimeBuilder(settings))
