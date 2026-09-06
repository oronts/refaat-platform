"""Unit tests for the guarded httpx client factory.

Prove the guard denies before the inner transport dispatches (so the socket never opens), the
no-policy path returns a plain client, an injected client is returned untouched, and the connector
IP-rewrite case is checked by its real SNI hostname rather than the rewritten IP host.
"""

from __future__ import annotations

import httpx
import pytest

from refaat.domain.egress import (
    AllowlistEgressPolicy,
    EgressPurpose,
    allowlist_from_config,
)
from refaat.errors import EgressDeniedError
from refaat.infrastructure.egress_client import build_http_client


class _SpyTransport(httpx.BaseTransport):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(str(request.url))
        return httpx.Response(200, content=b"ok")


def _policy(*entries: dict[str, object]) -> AllowlistEgressPolicy:
    return AllowlistEgressPolicy(allowlist_from_config(entries))


def test_no_policy_returns_a_plain_client_and_active_wraps_the_guard() -> None:
    from refaat.infrastructure.egress_client import _EgressGuardTransport

    plain = build_http_client(timeout=15.0, egress_policy=None, purpose=EgressPurpose.MODEL)
    guarded = build_http_client(timeout=15.0, egress_policy=_policy({"host": "x.test"}), purpose=EgressPurpose.MODEL)
    try:
        assert isinstance(plain, httpx.Client)
        assert not isinstance(plain._transport, _EgressGuardTransport)
        assert isinstance(guarded._transport, _EgressGuardTransport)
    finally:
        plain.close()
        guarded.close()


def test_injected_client_is_returned_untouched() -> None:
    injected = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(204)))
    returned = build_http_client(
        timeout=15.0, egress_policy=_policy({"host": "x.test"}), purpose=EgressPurpose.MODEL, client=injected
    )
    assert returned is injected
    injected.close()


def test_guard_denies_before_the_inner_transport_dispatches() -> None:
    spy = _SpyTransport()
    from refaat.infrastructure.egress_client import _EgressGuardTransport

    guarded = httpx.Client(
        transport=_EgressGuardTransport(spy, _policy({"host": "allowed.test"}), EgressPurpose.CONNECTOR)
    )
    try:
        with pytest.raises(EgressDeniedError):
            guarded.get("https://blocked.test/data")
        assert spy.calls == []  # the socket never opened
        response = guarded.get("https://allowed.test/data")
        assert response.status_code == 200
        assert spy.calls == ["https://allowed.test/data"]
    finally:
        guarded.close()


def test_guard_reads_the_sni_hostname_for_an_ip_rewritten_request() -> None:
    spy = _SpyTransport()
    from refaat.infrastructure.egress_client import _EgressGuardTransport

    guarded = httpx.Client(
        transport=_EgressGuardTransport(spy, _policy({"host": "real.test"}), EgressPurpose.CONNECTOR)
    )
    try:
        # The URL host is a pinned IP, but the logical hostname travels in the sni_hostname extension
        # (and the Host header), so the guard checks the real hostname, not the IP literal.
        response = guarded.get(
            "https://93.184.216.34/data",
            headers={"host": "real.test"},
            extensions={"sni_hostname": "real.test"},
        )
        assert response.status_code == 200
        with pytest.raises(EgressDeniedError):
            guarded.get(
                "https://93.184.216.34/data",
                headers={"host": "blocked.test"},
                extensions={"sni_hostname": "blocked.test"},
            )
    finally:
        guarded.close()
