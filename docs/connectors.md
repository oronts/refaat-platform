# Connectors and source ingestion

A connector converts an external source reference into bytes, a media type, and source metadata. The ingestion service then uses the same extraction, chunking, embedding, graph, blob, versioning, and indexing transaction used by direct document ingestion.

## Built-in schemes

- `file://` under `REFAAT_CONNECTOR_FILE_ROOT`;
- `https://`;
- `http://` for explicitly permitted deployments;
- third-party schemes registered through `refaat.connectors` entry points.

## Security behavior

The file connector resolves every path against its configured root and rejects escape attempts. HTTP connectors:

- require an absolute HTTP(S) URL;
- reject URL credentials;
- block private, loopback, link-local, multicast, and reserved destinations by default;
- disable redirects;
- enforce declared and streamed byte limits;
- expose ETag and Last-Modified metadata when present.

Production network policy should still restrict egress and provide DNS controls. Application-level URL checks are one defense layer, not a replacement for network isolation.

## Plugin contract

```python
from refaat.domain.models import JsonObject


class CompanyWikiConnector:
    scheme = "company-wiki"

    def fetch(self, source: str) -> tuple[bytes, str, JsonObject]: ...
```

Register the implementation through the `refaat.connectors` entry-point group and allowlist its distribution with `REFAAT_PLUGIN_ALLOWLIST`.

Connector plugins are trusted code. They must implement their own authentication, pagination, retry, checkpoint, and rate-limit semantics without exposing credentials in returned metadata.
