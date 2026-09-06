"""One serializer for the answer cache, shared by every cache backend.

The relational store and any external backend (Redis) encode a :class:`CacheEntry`
through exactly these functions, so a value written by one backend and read by
another is byte-for-byte the same object. Keeping a single variant also keeps the
duplicate-body audit green: there is one encode body and one decode body.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, cast

from refaat.domain.models import CacheEntry, Citation, JsonObject, QueryResponse


def response_to_json(response: QueryResponse) -> JsonObject:
    return {
        "answer": response.answer,
        "citations": [
            {
                "chunk_id": citation.chunk_id,
                "document_id": citation.document_id,
                "label": citation.label,
                "quote": citation.quote,
            }
            for citation in response.citations
        ],
        "routes": list(response.routes),
        "cached": response.cached,
        "corpus_version": response.corpus_version,
        "trace_id": response.trace_id,
        "usage": response.usage,
        "abstained": response.abstained,
        "degraded": response.degraded,
    }


def response_from_json(data: Mapping[str, Any]) -> QueryResponse:
    return QueryResponse(
        answer=str(data["answer"]),
        citations=tuple(Citation(**item) for item in data.get("citations", [])),
        routes=tuple(str(route) for route in data.get("routes", [])),
        cached=bool(data.get("cached", False)),
        corpus_version=int(data["corpus_version"]),
        trace_id=str(data["trace_id"]),
        usage=cast(JsonObject, data.get("usage", {})),
        abstained=bool(data.get("abstained", False)),
        degraded=bool(data.get("degraded", False)),
    )


def cache_entry_to_json(entry: CacheEntry) -> JsonObject:
    return {
        "id": entry.id,
        "workspace_id": entry.workspace_id,
        "corpus_id": entry.corpus_id,
        "corpus_version": entry.corpus_version,
        "user_scope": entry.user_scope,
        "query": entry.query,
        "query_hash": entry.query_hash,
        "query_embedding": list(entry.query_embedding),
        "response": response_to_json(entry.response),
        "expires_at": entry.expires_at.isoformat(),
        "created_at": entry.created_at.isoformat(),
    }


def cache_entry_from_json(data: Mapping[str, Any]) -> CacheEntry:
    return CacheEntry(
        id=str(data["id"]),
        workspace_id=str(data["workspace_id"]),
        corpus_id=str(data["corpus_id"]),
        corpus_version=int(data["corpus_version"]),
        user_scope=str(data["user_scope"]),
        query=str(data["query"]),
        query_hash=str(data["query_hash"]),
        query_embedding=tuple(float(value) for value in data["query_embedding"]),
        response=response_from_json(cast(Mapping[str, Any], data["response"])),
        expires_at=datetime.fromisoformat(str(data["expires_at"])),
        created_at=datetime.fromisoformat(str(data["created_at"])),
    )
