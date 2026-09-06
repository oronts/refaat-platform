"""The guarded httpx client factory and its egress-enforcing transport.

``build_http_client`` is the single constructor every outbound httpx seam uses to build an owned
client. With no policy installed it returns a plain ``httpx.Client(timeout=...)``, byte-identical to
the object the adapters built before, so the unconfigured runtime is unchanged. With a policy
installed it wraps the client in a transport that reconstructs the logical destination and calls the
shared ``ensure_egress_allowed`` chokepoint before delegating, so an un-allowlisted host is denied
before the socket opens. An injected client is returned untouched; its seam owns the explicit
``ensure_egress_allowed`` call, which is the load-bearing control regardless of client ownership.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

from refaat.domain.egress import (
    EgressDestination,
    EgressPolicy,
    EgressPurpose,
    ensure_egress_allowed,
)
from refaat.errors import ValidationError
from refaat.security import pin_outbound_url

if TYPE_CHECKING:
    from refaat.ports import Resolver


class _EgressGuardTransport(httpx.BaseTransport):
    def __init__(self, inner: httpx.BaseTransport, policy: EgressPolicy, purpose: EgressPurpose) -> None:
        self._inner = inner
        self._policy = policy
        self._purpose = purpose

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        # Prefer the SNI hostname extension (an IP-rewritten connector URL carries its real hostname
        # there), then the Host header, then the URL host, so the logical hostname is checked even
        # when the URL was pinned to an IP.
        sni = request.extensions.get("sni_hostname")
        header_host = request.headers.get("host")
        host = str(sni or header_host or request.url.host).split(":", 1)[0]
        dest = EgressDestination(
            scheme=request.url.scheme,
            host=host.lower(),
            port=request.url.port,
            purpose=self._purpose,
        )
        ensure_egress_allowed(self._policy, dest)
        return self._inner.handle_request(request)

    def close(self) -> None:
        self._inner.close()


def build_http_client(
    *,
    timeout: float,
    egress_policy: EgressPolicy | None,
    purpose: EgressPurpose,
    client: httpx.Client | None = None,
    **kwargs: Any,
) -> httpx.Client:
    if client is not None:
        return client
    if egress_policy is None:
        return httpx.Client(timeout=timeout, **kwargs)
    transport = _EgressGuardTransport(httpx.HTTPTransport(), egress_policy, purpose)
    return httpx.Client(timeout=timeout, transport=transport, **kwargs)


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    url: str
    headers: dict[str, str]
    extensions: dict[str, str]
    host: str


def prepare_egress_request(
    source: str,
    *,
    purpose: EgressPurpose,
    egress_policy: EgressPolicy | None,
    allow_private_hosts: bool,
    resolver: Resolver | None,
    user_agent: str,
) -> PreparedRequest:
    """Apply the shared egress guard to an outbound URL: allowlist, then SSRF IP-pin, then Host and SNI.

    The one chokepoint the connector and the governed plugin egress client both send through, so neither
    can reach a host the policy denies or a rebound IP. With no policy the allowlist step is skipped and
    the destination is never reconstructed, so the connector path stays byte-identical.
    """
    if egress_policy is not None:
        ensure_egress_allowed(egress_policy, EgressDestination.from_url(source, purpose))
    target = pin_outbound_url(source, allow_private=allow_private_hosts, resolver=resolver)
    headers = {"User-Agent": user_agent, "Host": target.host_header}
    extensions = {"sni_hostname": target.hostname} if target.pinned_ip is not None else {}
    return PreparedRequest(url=target.rewritten_url, headers=headers, extensions=extensions, host=target.hostname)


def read_capped(
    response: httpx.Response, max_bytes: int, *, message: str = "response exceeds the configured byte limit"
) -> bytes:
    """Read a streamed response body, refusing once it exceeds the cap by declared length or by bytes read."""
    declared = response.headers.get("content-length")
    if declared and int(declared) > max_bytes:
        raise ValidationError(message)
    content = bytearray()
    for block in response.iter_bytes():
        content.extend(block)
        if len(content) > max_bytes:
            raise ValidationError(message)
    return bytes(content)


__all__ = ["PreparedRequest", "build_http_client", "prepare_egress_request", "read_capped"]
