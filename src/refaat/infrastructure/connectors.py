from __future__ import annotations

import mimetypes
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx

from refaat.domain.egress import EgressPolicy, EgressPurpose
from refaat.domain.models import JsonObject
from refaat.errors import DependencyUnavailableError, EgressDeniedError, ValidationError
from refaat.infrastructure.egress_client import build_http_client, prepare_egress_request, read_capped
from refaat.observability import get_logger
from refaat.ports import Connector, Resolver
from refaat.security import safe_join, sanitize_source_url
from refaat.version import __version__

logger = get_logger(__name__)


class FileConnector:
    scheme = "file"

    def __init__(self, root: Path, *, max_bytes: int) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes

    def fetch(self, source: str) -> tuple[bytes, str, JsonObject]:
        parsed = urlparse(source)
        candidate = unquote(f"{parsed.netloc}{parsed.path}") if parsed.scheme == "file" else source
        path = safe_join(self.root, candidate.lstrip("/"))
        if not path.is_file():
            raise ValidationError("Connector source is not a file")
        size = path.stat().st_size
        if size > self.max_bytes:
            raise ValidationError("Connector source exceeds the configured byte limit")
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return (
            path.read_bytes(),
            media_type,
            {
                "source_path": str(path.relative_to(self.root)),
                "source_size": size,
            },
        )


class HttpConnector:
    def __init__(
        self,
        *,
        scheme: str = "https",
        max_bytes: int,
        allow_private_hosts: bool = False,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
        resolver: Resolver | None = None,
        egress_policy: EgressPolicy | None = None,
    ) -> None:
        if scheme not in {"http", "https"}:
            raise ValueError("HTTP connector scheme must be http or https")
        self.scheme = scheme
        self.max_bytes = max_bytes
        self.allow_private_hosts = allow_private_hosts
        self.resolver = resolver
        self._egress_policy = egress_policy
        self._owns_client = client is None
        self.client = build_http_client(
            timeout=timeout,
            egress_policy=egress_policy,
            purpose=EgressPurpose.CONNECTOR,
            client=client,
            follow_redirects=False,
        )

    def fetch(self, source: str) -> tuple[bytes, str, JsonObject]:
        # The shared egress guard denies an un-allowlisted host before the DNS resolve and pins the
        # resolved IP against SSRF; with no policy it is skipped and the path stays byte-identical.
        try:
            prepared = prepare_egress_request(
                source,
                purpose=EgressPurpose.CONNECTOR,
                egress_policy=self._egress_policy,
                allow_private_hosts=self.allow_private_hosts,
                resolver=self.resolver,
                user_agent=f"RefaatConnector/{__version__}",
            )
        except EgressDeniedError as exc:
            # Redacted by construction: only the host, machine reason code, and purpose.
            logger.warning("egress_denied", host=exc.host, reason_code=exc.reason_code, purpose=str(exc.purpose))
            raise
        try:
            with self.client.stream(
                "GET",
                prepared.url,
                headers=prepared.headers,
                extensions=prepared.extensions,
            ) as response:
                if 300 <= response.status_code < 400:
                    raise ValidationError("HTTP connector redirects are disabled")
                response.raise_for_status()
                content = read_capped(
                    response, self.max_bytes, message="Connector source exceeds the configured byte limit"
                )
        except ValidationError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise DependencyUnavailableError("HTTP connector request failed") from exc
        media_type = response.headers.get("content-type", "application/octet-stream").split(";", 1)[0]
        return (
            bytes(content),
            media_type,
            {
                "source_url": sanitize_source_url(source, drop_query=True),
                "source_size": len(content),
                "etag": response.headers.get("etag"),
                "last_modified": response.headers.get("last-modified"),
            },
        )

    def close(self) -> None:
        if self._owns_client:
            self.client.close()


class ConnectorRegistry:
    def __init__(self, connectors: tuple[Connector, ...] = ()) -> None:
        self._connectors: dict[str, Connector] = {}
        for connector in connectors:
            self.register(connector)

    def register(self, connector: Connector) -> None:
        scheme = connector.scheme.casefold()
        if not scheme:
            raise ValueError("Connector scheme cannot be empty")
        if scheme in self._connectors:
            raise ValueError(f"Connector already registered for {scheme}")
        self._connectors[scheme] = connector

    def fetch(self, source: str) -> tuple[bytes, str, JsonObject]:
        scheme = urlparse(source).scheme.casefold()
        connector = self._connectors.get(scheme)
        if connector is None:
            raise ValidationError(f"No connector registered for scheme: {scheme or '<missing>'}")
        return connector.fetch(source)

    @property
    def schemes(self) -> frozenset[str]:
        return frozenset(self._connectors)

    def close(self, already_closed: set[int] | None = None) -> None:
        # ``already_closed`` is the runtime's shared close set: a connector that is
        # also owned elsewhere (registered as an event sink too) is closed exactly
        # once across all owners.
        for connector in self._connectors.values():
            identity = id(connector)
            if already_closed is not None:
                if identity in already_closed:
                    continue
                already_closed.add(identity)
            close = getattr(connector, "close", None)
            if callable(close):
                close()
