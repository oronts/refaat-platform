from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from refaat.errors import DependencyUnavailableError, ValidationError
from refaat.infrastructure.connectors import ConnectorRegistry, FileConnector, HttpConnector
from refaat.security import validate_outbound_url
from refaat.version import __version__


def test_file_connector_is_rooted_and_size_limited(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    (root / "policy.md").write_text("Evidence", encoding="utf-8")
    connector = FileConnector(root, max_bytes=16)

    content, media_type, metadata = connector.fetch("file:///policy.md")
    assert content == b"Evidence"
    assert media_type == "text/markdown"
    assert metadata["source_path"] == "policy.md"

    (root / "large.txt").write_bytes(b"x" * 17)
    with pytest.raises(ValidationError, match="byte limit"):
        connector.fetch("file:///large.txt")
    with pytest.raises(ValidationError, match="not a file"):
        connector.fetch("file:///missing.txt")
    with pytest.raises(ValidationError, match="escapes"):
        connector.fetch("file:///../outside.txt")


def test_http_connector_streams_content_and_preserves_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["user-agent"] == f"RefaatConnector/{__version__}"
        return httpx.Response(
            200,
            content=b"remote evidence",
            headers={"content-type": "text/plain; charset=utf-8", "etag": "v1"},
        )

    connector = HttpConnector(
        max_bytes=100,
        allow_private_hosts=True,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    content, media_type, metadata = connector.fetch("https://models.test/policy.txt")
    assert content == b"remote evidence"
    assert media_type == "text/plain"
    assert metadata["etag"] == "v1"


def test_http_connector_rejects_redirects_limits_and_transport_failures() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/redirect":
            return httpx.Response(302, headers={"location": "/other"})
        if request.url.path == "/declared":
            return httpx.Response(200, content=b"x", headers={"content-length": "500"})
        if request.url.path == "/streamed":
            return httpx.Response(200, content=b"too large")
        raise httpx.ConnectError("offline", request=request)

    connector = HttpConnector(
        max_bytes=4,
        allow_private_hosts=True,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ValidationError, match="redirects"):
        connector.fetch("https://example.test/redirect")
    with pytest.raises(ValidationError, match="byte limit"):
        connector.fetch("https://example.test/declared")
    with pytest.raises(ValidationError, match="byte limit"):
        connector.fetch("https://example.test/streamed")
    with pytest.raises(DependencyUnavailableError):
        connector.fetch("https://example.test/offline")


def test_connector_registry_and_outbound_url_guards(tmp_path: Path) -> None:
    registry = ConnectorRegistry((FileConnector(tmp_path, max_bytes=10),))
    assert registry.schemes == frozenset({"file"})
    with pytest.raises(ValueError, match="already registered"):
        registry.register(FileConnector(tmp_path, max_bytes=10))
    with pytest.raises(ValidationError, match="No connector"):
        registry.fetch("ftp://example.com/file")
    with pytest.raises(ValidationError, match="credentials"):
        validate_outbound_url("https://user:secret@example.com/data", allow_private=False)
    with pytest.raises(ValidationError, match="blocked"):
        validate_outbound_url("http://127.0.0.1/data", allow_private=False)
    validate_outbound_url("http://127.0.0.1/data", allow_private=True)
