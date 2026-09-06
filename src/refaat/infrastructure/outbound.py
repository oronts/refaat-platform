"""A governed outbound HTTP client for a plugin skill's own external calls.

Every request flows through the same egress guard the connector uses: the per-capability
``allowed_network`` allow-list first, then the deployment egress allowlist, then the SSRF IP-pin, then a
disabled-redirect send under a response byte cap. The ``prepare`` and ``send`` steps are split so the
run-scoped gateway can audit the admission decision before any byte leaves the socket. A plugin never
reaches a private or link-local address: the pin always runs with ``allow_private=False``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import httpx

from refaat.domain.egress import EgressPolicy, EgressPurpose
from refaat.errors import DependencyUnavailableError, EgressDeniedError, ValidationError
from refaat.infrastructure.egress_client import PreparedRequest, build_http_client, prepare_egress_request, read_capped
from refaat.version import __version__

if TYPE_CHECKING:
    from collections.abc import Mapping

    from refaat.ports import Resolver


@dataclass(frozen=True, slots=True)
class OutboundResponse:
    status_code: int
    headers: dict[str, str]
    content: bytes
    host: str


class OutboundClient:
    def __init__(
        self,
        *,
        egress_policy: EgressPolicy | None,
        resolver: Resolver | None = None,
        timeout: float = 30.0,
        max_bytes: int,
        allowed_network: frozenset[str] = frozenset(),
        client: httpx.Client | None = None,
    ) -> None:
        self._egress_policy = egress_policy
        self._resolver = resolver
        self._max_bytes = max_bytes
        self._allowed_network = allowed_network
        self._owns_client = client is None
        self._client = build_http_client(
            timeout=timeout,
            egress_policy=egress_policy,
            purpose=EgressPurpose.PLUGIN,
            client=client,
            follow_redirects=False,
        )

    def prepare(self, url: str) -> PreparedRequest:
        """Apply the guard without sending: the capability allow-list, then the allowlist and SSRF pin.

        Raises before any socket opens, so the gateway can audit the admission decision first.
        """
        if self._allowed_network:
            host = (urlparse(url).hostname or "").lower()
            if host not in self._allowed_network:
                raise EgressDeniedError("network_destination_denied", host, EgressPurpose.PLUGIN)
        return prepare_egress_request(
            url,
            purpose=EgressPurpose.PLUGIN,
            egress_policy=self._egress_policy,
            allow_private_hosts=False,
            resolver=self._resolver,
            user_agent=f"RefaatPlugin/{__version__}",
        )

    def send(
        self,
        prepared: PreparedRequest,
        *,
        method: str = "GET",
        headers: Mapping[str, str] | None = None,
        content: bytes | None = None,
    ) -> OutboundResponse:
        # Strip a skill-supplied Host or User-Agent so it cannot override the pinned Host or forge a
        # second Host line: the runtime owns the destination and the agent string.
        safe = {key: value for key, value in dict(headers or {}).items() if key.lower() not in {"host", "user-agent"}}
        merged = {**prepared.headers, **safe}
        try:
            with self._client.stream(
                method.upper(),
                prepared.url,
                headers=merged,
                extensions=prepared.extensions,
                content=content,
                follow_redirects=False,
            ) as response:
                if 300 <= response.status_code < 400:
                    raise ValidationError("plugin outbound redirects are disabled")
                body = read_capped(response, self._max_bytes, message="plugin outbound response exceeds the byte limit")
                out_headers = dict(response.headers)
                status = response.status_code
        except ValidationError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise DependencyUnavailableError("plugin outbound request failed") from exc
        return OutboundResponse(status_code=status, headers=out_headers, content=body, host=prepared.host)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


__all__ = ["OutboundClient", "OutboundResponse"]
