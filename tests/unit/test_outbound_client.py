"""The pure governed outbound client: the capability allow-list and the SSRF pin both fail closed."""

from __future__ import annotations

import httpx
import pytest

from refaat.domain.egress import AllowlistEgressPolicy, allowlist_from_config
from refaat.errors import EgressDeniedError, ValidationError
from refaat.infrastructure.outbound import OutboundClient


class _StubResolver:
    def resolve(self, host: str, port: int) -> tuple[str, ...]:
        del host, port
        return ("93.184.216.34",)


def _chain_client(
    handler: object, *, allowed_network: frozenset[str] = frozenset(), max_bytes: int = 1000
) -> OutboundClient:
    http = httpx.Client(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]
    policy = AllowlistEgressPolicy(allowlist_from_config([{"host": "api.example.com"}]))
    return OutboundClient(
        egress_policy=policy,
        resolver=_StubResolver(),
        max_bytes=max_bytes,
        allowed_network=allowed_network,
        client=http,
    )


def test_full_chain_reaches_the_transport_for_an_allowed_host() -> None:
    client = _chain_client(
        lambda request: httpx.Response(200, content=b"ok"), allowed_network=frozenset({"api.example.com"})
    )
    try:
        response = client.send(client.prepare("https://api.example.com/data"))
        assert response.status_code == 200
        assert response.content == b"ok"
    finally:
        client.close()


def test_full_chain_denies_a_host_outside_the_deployment_allowlist() -> None:
    client = _chain_client(lambda request: httpx.Response(200))
    try:
        with pytest.raises(EgressDeniedError):
            client.prepare("https://not-allowed.example.org/data")
    finally:
        client.close()


def test_send_refuses_a_redirect() -> None:
    client = _chain_client(
        lambda request: httpx.Response(302, headers={"location": "https://evil.example/"}),
        allowed_network=frozenset({"api.example.com"}),
    )
    try:
        prepared = client.prepare("https://api.example.com/x")
        with pytest.raises(ValidationError):
            client.send(prepared)
    finally:
        client.close()


def test_send_enforces_the_byte_cap() -> None:
    client = _chain_client(
        lambda request: httpx.Response(200, content=b"x" * 5000),
        allowed_network=frozenset({"api.example.com"}),
        max_bytes=100,
    )
    try:
        prepared = client.prepare("https://api.example.com/x")
        with pytest.raises(ValidationError):
            client.send(prepared)
    finally:
        client.close()


def test_a_host_outside_the_capability_allow_list_is_denied_before_the_pin() -> None:
    client = OutboundClient(egress_policy=None, max_bytes=1000, allowed_network=frozenset({"api.example.com"}))
    try:
        with pytest.raises(EgressDeniedError):
            client.prepare("https://evil.example.net/data")
    finally:
        client.close()


def test_a_private_address_is_refused_by_the_pin() -> None:
    # A plugin egress always pins with allow_private=False, so a private or loopback host is rejected.
    client = OutboundClient(egress_policy=None, max_bytes=1000)
    try:
        with pytest.raises(ValidationError):
            client.prepare("http://127.0.0.1/metadata")
    finally:
        client.close()
