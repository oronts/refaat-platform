from __future__ import annotations

import httpx
import pytest

from refaat.errors import ValidationError
from refaat.infrastructure.connectors import HttpConnector


class FakeResolver:
    def __init__(self, *batches: tuple[str, ...]) -> None:
        self._batches = list(batches)
        self.calls = 0

    def resolve(self, host: str, port: int) -> tuple[str, ...]:
        self.calls += 1
        return self._batches[min(self.calls - 1, len(self._batches) - 1)]


def test_pins_to_validated_ip_and_preserves_host_and_sni() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url_host"] = request.url.host
        seen["host_header"] = request.headers.get("host")
        seen["sni"] = request.extensions.get("sni_hostname")
        return httpx.Response(200, content=b"ok", headers={"content-type": "text/plain"})

    connector = HttpConnector(
        max_bytes=100,
        allow_private_hosts=False,
        resolver=FakeResolver(("93.184.216.34",)),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    content, _, metadata = connector.fetch("https://example.test/data")

    assert content == b"ok"
    assert seen["url_host"] == "93.184.216.34"
    assert seen["host_header"] == "example.test"
    assert seen["sni"] == "example.test"
    assert metadata["source_url"] == "https://example.test/data"


def test_resolves_once_so_dns_rebinding_cannot_reach_a_private_ip() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url_host"] = request.url.host
        return httpx.Response(200, content=b"ok")

    resolver = FakeResolver(("93.184.216.34",), ("10.0.0.5",))
    connector = HttpConnector(
        max_bytes=100,
        allow_private_hosts=False,
        resolver=resolver,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    connector.fetch("https://rebind.test/data")

    assert resolver.calls == 1
    assert seen["url_host"] == "93.184.216.34"


def test_blocks_a_private_resolved_ip() -> None:
    connector = HttpConnector(
        max_bytes=100,
        allow_private_hosts=False,
        resolver=FakeResolver(("10.0.0.5",)),
        client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200))),
    )
    with pytest.raises(ValidationError, match="blocked"):
        connector.fetch("https://evil.test/data")


def test_blocks_shared_address_space_used_by_cloud_metadata_and_overlay_networks() -> None:
    # 100.64.0.0/10 is not is_private but is not is_global either; a non-global address is refused, so a
    # plugin or connector cannot reach carrier-grade NAT space (cloud metadata, a Tailscale peer).
    connector = HttpConnector(
        max_bytes=100,
        allow_private_hosts=False,
        resolver=FakeResolver(("100.100.100.200",)),
        client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200))),
    )
    with pytest.raises(ValidationError, match="blocked"):
        connector.fetch("https://metadata.test/latest")
