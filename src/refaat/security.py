from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import secrets
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse, urlunparse

from refaat.errors import ValidationError

if TYPE_CHECKING:
    from refaat.ports import Resolver
from refaat.ids import new_secret


def issue_api_key() -> tuple[str, str]:
    raw = new_secret("refaat", 36)
    return raw, raw[:14]


def hash_api_key(raw: str, pepper: str, *, salt: bytes | None = None) -> str:
    actual_salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(
        raw.encode(),
        salt=actual_salt + pepper.encode(),
        n=2**14,
        r=8,
        p=1,
        dklen=32,
    )
    return f"{base64.urlsafe_b64encode(actual_salt).decode()}${base64.urlsafe_b64encode(digest).decode()}"


def verify_api_key(raw: str, encoded: str, pepper: str) -> bool:
    try:
        salt_part, digest_part = encoded.split("$", maxsplit=1)
        salt = base64.urlsafe_b64decode(salt_part.encode())
        expected = base64.urlsafe_b64decode(digest_part.encode())
    except (ValueError, TypeError):
        return False
    actual = hashlib.scrypt(
        raw.encode(),
        salt=salt + pepper.encode(),
        n=2**14,
        r=8,
        p=1,
        dklen=32,
    )
    return hmac.compare_digest(actual, expected)


def safe_join(root: Path, candidate: str) -> Path:
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValidationError("Path escapes the configured root")
    return resolved


@dataclass(frozen=True, slots=True)
class PinnedTarget:
    scheme: str
    hostname: str
    port: int | None
    pinned_ip: str | None
    rewritten_url: str
    host_header: str


class SystemResolver:
    def resolve(self, host: str, port: int) -> tuple[str, ...]:
        try:
            addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise ValidationError("Connector host cannot be resolved") from exc
        return tuple(str(address[4][0]) for address in addresses)


def _rewrite_host(url: str, ip: str) -> str:
    parsed = urlparse(url)
    host = f"[{ip}]" if ":" in ip else ip
    netloc = host if parsed.port is None else f"{host}:{parsed.port}"
    return urlunparse(parsed._replace(netloc=netloc))


def sanitize_source_url(url: str, *, drop_query: bool) -> str:
    """Strip userinfo credentials from a source URL always, and the query and fragment when drop_query
    is set; a non-URL source is returned unchanged."""
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return url
    netloc = parsed.netloc.rsplit("@", 1)[-1]
    query = "" if drop_query else parsed.query
    fragment = "" if drop_query else parsed.fragment
    return urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, query, fragment))


def pin_outbound_url(url: str, *, allow_private: bool, resolver: Resolver | None = None) -> PinnedTarget:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValidationError("Only absolute HTTP(S) connector URLs are supported")
    if parsed.username or parsed.password:
        raise ValidationError("Connector URLs cannot contain credentials")
    if allow_private:
        return PinnedTarget(parsed.scheme, parsed.hostname, parsed.port, None, url, parsed.netloc)
    default_port = 443 if parsed.scheme == "https" else 80
    addresses = (resolver or SystemResolver()).resolve(parsed.hostname, parsed.port or default_port)
    if not addresses:
        raise ValidationError("Connector host cannot be resolved")
    for address in addresses:
        ip = ipaddress.ip_address(address)
        # Refuse any non-globally-routable address, a strict superset of private, loopback, link-local,
        # multicast, and reserved that also covers shared address space (100.64.0.0/10, used for
        # carrier-grade NAT and cloud metadata) and other non-public ranges an SSRF could target.
        if not ip.is_global or ip.is_multicast:
            raise ValidationError("Private, shared, or reserved connector hosts are blocked")
    return PinnedTarget(
        parsed.scheme,
        parsed.hostname,
        parsed.port,
        addresses[0],
        _rewrite_host(url, addresses[0]),
        parsed.netloc,
    )


def validate_outbound_url(url: str, *, allow_private: bool) -> None:
    pin_outbound_url(url, allow_private=allow_private)
