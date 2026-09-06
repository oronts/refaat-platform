"""Run-scoped governed egress for a plugin skill's own outbound HTTP.

An :class:`OutboundGateway` wraps the deployment egress policy for one workspace and is the single
object the governed ReAct loop threads in, mirroring :class:`~refaat.application.secrets.SecretResolver`.
It hands a skill a per-capability :class:`CapabilityOutbound` view that audits the admission decision
before any byte leaves the socket, refuses during an unserved shadow or canary run, and emits only a
redacted event (host, machine reason code, and capability name, never a URL path, query, body, or
secret). With no capabilities the loop never constructs one, so it is byte-identical.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlparse

import httpx

from refaat.clock import Clock
from refaat.core.text import stable_digest
from refaat.domain.audit import AuditEvent, AuditSink, emit_audit
from refaat.domain.egress import EgressPurpose
from refaat.errors import EgressDeniedError, ValidationError
from refaat.ids import new_id
from refaat.infrastructure.egress_client import build_http_client
from refaat.infrastructure.outbound import OutboundClient, OutboundResponse
from refaat.observability import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from refaat.domain.authz import Principal
    from refaat.domain.egress import EgressPolicy
    from refaat.domain.governance import ToolCapability
    from refaat.ports import Resolver

logger = get_logger(__name__)

_UNSERVED_RUN_REASON = "unserved_run_denied"
_PIN_DENIED_REASON = "pin_or_redirect_denied"
_MALFORMED_REASON = "malformed_destination"


class CapabilityOutbound:
    """The outbound handle a governed skill receives, bound to one capability and run."""

    def __init__(
        self,
        client: OutboundClient,
        *,
        workspace_id: str,
        capability_name: str,
        principal: Principal | None,
        clock: Clock,
        audit_sink: AuditSink | None,
        audit_fail_closed: bool,
        sequence: Iterator[int],
        run_nonce: str,
    ) -> None:
        self._client = client
        self._workspace_id = workspace_id
        self._capability_name = capability_name
        self._principal = principal
        self._clock = clock
        self._audit_sink = audit_sink
        self._audit_fail_closed = audit_fail_closed
        self._sequence = sequence
        self._run_nonce = run_nonce

    def get(self, url: str, *, headers: Mapping[str, str] | None = None) -> OutboundResponse:
        return self.request(url, method="GET", headers=headers)

    def request(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: Mapping[str, str] | None = None,
        content: bytes | None = None,
    ) -> OutboundResponse:
        from refaat.application.services import in_unserved_run

        try:
            host = (urlparse(url).hostname or "").lower()
        except ValueError:
            host = ""
        if in_unserved_run():
            # A shadow or canary comparison run serves nothing, so no outbound byte may leave it.
            raise EgressDeniedError(_UNSERVED_RUN_REASON, host, EgressPurpose.PLUGIN)
        try:
            prepared = self._client.prepare(url)
        except EgressDeniedError as exc:
            self._emit("plugin.egress_denied", exc.host, exc.reason_code)
            raise
        except ValidationError:
            self._emit("plugin.egress_denied", host, _PIN_DENIED_REASON)
            raise
        except ValueError as exc:
            # A malformed URL or a resolver returning a non-IP address: fail closed and audit it.
            self._emit("plugin.egress_denied", host, _MALFORMED_REASON)
            raise EgressDeniedError(_MALFORMED_REASON, host, EgressPurpose.PLUGIN) from exc
        # Audit the admission BEFORE the send, so under a fail-closed audit policy a sink outage denies
        # the egress rather than recording it after the bytes have left, matching the secret path.
        self._emit("plugin.egress_admitted", prepared.host, "admitted")
        return self._client.send(prepared, method=method, headers=headers, content=content)

    def _emit(self, event_type: str, host: str, reason_code: str) -> None:
        if self._audit_sink is None:
            return
        principal = self._principal
        event = AuditEvent(
            event_type=event_type,
            occurred_at=self._clock.now(),
            workspace_id=self._workspace_id,
            principal_id="system" if principal is None else principal.id,
            principal_kind="system" if principal is None else principal.kind.value,
            tenant=None if principal is None else principal.tenant,
            policy_id="react",
            policy_version="1",
            capability_name=self._capability_name,
            action_fingerprint=stable_digest(("plugin.egress", self._capability_name, host, reason_code)),
            dedup_id=stable_digest(
                (
                    event_type,
                    self._workspace_id,
                    self._capability_name,
                    host,
                    reason_code,
                    self._run_nonce,
                    str(next(self._sequence)),
                )
            ),
        )
        emit_audit(
            self._audit_sink,
            event,
            fail_closed=self._audit_fail_closed,
            logger=logger,
            log_key="plugin_egress_audit_sink_failed",
        )


class OutboundGateway:
    def __init__(
        self,
        workspace_id: str,
        *,
        egress_policy: EgressPolicy | None,
        clock: Clock,
        audit_sink: AuditSink | None = None,
        audit_fail_closed: bool = False,
        timeout: float = 30.0,
        max_bytes: int,
        resolver: Resolver | None = None,
    ) -> None:
        self._workspace_id = workspace_id
        self._egress_policy = egress_policy
        self._clock = clock
        self._audit_sink = audit_sink
        self._audit_fail_closed = audit_fail_closed
        self._timeout = timeout
        self._max_bytes = max_bytes
        self._resolver = resolver
        self._counter = _counter()
        # A per-run nonce folded into every event's dedup_id, so two runs' first egress to the same host
        # under the same capability do not collapse to one event in a dedup-on-dedup_id consumer.
        self._run_nonce = new_id("preg")
        # One guarded httpx client per run, built on first use so a run that never egresses is
        # byte-identical (no client, no socket), shared across capability views, closed once.
        self._http: httpx.Client | None = None

    def for_capability(self, capability: ToolCapability, principal: Principal | None) -> CapabilityOutbound:
        if self._http is None:
            self._http = build_http_client(
                timeout=self._timeout,
                egress_policy=self._egress_policy,
                purpose=EgressPurpose.PLUGIN,
                follow_redirects=False,
            )
        client = OutboundClient(
            egress_policy=self._egress_policy,
            resolver=self._resolver,
            timeout=self._timeout,
            max_bytes=self._max_bytes,
            allowed_network=capability.allowed_network,
            client=self._http,
        )
        return CapabilityOutbound(
            client,
            workspace_id=self._workspace_id,
            capability_name=capability.name,
            principal=principal,
            clock=self._clock,
            audit_sink=self._audit_sink,
            audit_fail_closed=self._audit_fail_closed,
            sequence=self._counter,
            run_nonce=self._run_nonce,
        )

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None


def _counter() -> Iterator[int]:
    value = 0
    while True:
        yield value
        value += 1


__all__ = ["CapabilityOutbound", "OutboundGateway"]
