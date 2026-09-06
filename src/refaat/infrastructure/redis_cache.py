"""Redis-backed answer cache and model-call cache.

Both adapters degrade to a miss on a transient Redis error so a query still serves, and both erase
fail-closed: a right-to-erasure request enumerates the real keys with ``SCAN`` and removes them with
``UNLINK``, raising rather than reporting a false erasure. Key segments are fixed-width digests so a
raw workspace or corpus id can never inject a glob metacharacter into a ``SCAN`` pattern, and every
key namespace carries a serde-version segment so a format change never reads a stale payload shape.
The ``redis`` package is imported lazily, never at module load, so it stays off the core import path.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from refaat.domain.models import CacheEntry, JsonObject
from refaat.infrastructure.cache_payload import cache_entry_from_json, cache_entry_to_json
from refaat.observability import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from refaat.clock import Clock

logger = get_logger(__name__)

_NS = "refaat"
_SCAN_COUNT = 500
_UNLINK_BATCH = 500
_MAX_DRAIN_PASSES = 8


# redis-py lets URL query params override client kwargs, so a URL could widen the bounded timeout we set.
_OVERRIDABLE_URL_KEYS = frozenset({"socket_timeout", "socket_connect_timeout", "retry_on_timeout", "retry"})


def _sanitize_redis_url(url: str) -> str:
    parts = urlsplit(url)
    kept = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k.lower() not in _OVERRIDABLE_URL_KEYS]
    return urlunsplit(parts._replace(query=urlencode(kept)))


def build_redis_client(url: str, timeout: float) -> Any:
    import redis
    from redis.backoff import NoBackoff
    from redis.retry import Retry

    return redis.Redis.from_url(
        _sanitize_redis_url(url),
        socket_connect_timeout=timeout,
        socket_timeout=timeout,
        retry_on_timeout=False,
        # Pin no retries so a black-holed server costs one bounded call, not a client-default backoff.
        retry=Retry(NoBackoff(), 0),
        decode_responses=True,
    )


def ping_or_warn(client: Any) -> None:
    """Log a warning if Redis is unreachable at startup, without failing: degrade-to-miss stays in force."""
    try:
        client.ping()
    except Exception as error:
        logger.warning("redis_cache_unreachable_at_startup", error=str(error))


def _transient_errors() -> tuple[type[BaseException], ...]:
    import redis

    return (redis.exceptions.RedisError, OSError)


def _seg(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


class _RedisBase:
    def __init__(self, client: Any, *, clock: Clock) -> None:
        self._client = client
        self._clock = clock
        self._transient = _transient_errors()
        # A read also decodes, so a non-UTF-8 value (a corrupt or foreign write) is a miss, not a crash.
        self._read_errors: tuple[type[BaseException], ...] = (*self._transient, UnicodeDecodeError)
        self._warned = False

    def _degraded(self, error: BaseException) -> None:
        # Log once per adapter so a persistent outage or misconfiguration is visible, without flooding.
        if not self._warned:
            self._warned = True
            logger.warning("redis_cache_degraded", error=str(error))

    def _get(self, key: str) -> str | None:
        try:
            return cast("str | None", self._client.get(key))
        except self._read_errors as error:
            self._degraded(error)
            return None

    def _mget(self, keys: list[str]) -> list[str | None]:
        if not keys:
            return []
        try:
            return cast("list[str | None]", self._client.mget(keys))
        except self._read_errors as error:
            self._degraded(error)
            return [None] * len(keys)

    def _scan_keys(self, pattern: str) -> list[str]:
        keys: list[str] = []
        cursor = 0
        while True:
            cursor, batch = self._client.scan(cursor=cursor, match=pattern, count=_SCAN_COUNT)
            keys.extend(batch)
            if cursor == 0:
                return keys

    def _unlink_all(self, keys: list[str]) -> int:
        deleted = 0
        for start in range(0, len(keys), _UNLINK_BATCH):
            chunk = keys[start : start + _UNLINK_BATCH]
            if chunk:
                deleted += int(self._client.unlink(*chunk))
        return deleted

    def _drain(self, pattern: str) -> int:
        # Re-scan until a pass finds nothing, so a key written during an earlier pass is still swept.
        # This narrows the erase window to a write landing strictly after the final empty scan, which
        # is the same window the single-statement SQL delete has. Fail-closed: a Redis error propagates.
        deleted = 0
        for _ in range(_MAX_DRAIN_PASSES):
            keys = self._scan_keys(pattern)
            if not keys:
                break
            deleted += self._unlink_all(keys)
        return deleted

    def _best_effort_unlink(self, pattern: str) -> None:
        try:
            self._unlink_all(self._scan_keys(pattern))
        except self._transient as error:
            self._degraded(error)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._client.close()


class RedisCacheStore(_RedisBase):
    def _entry_key(self, workspace_id: str, corpus_id: str, version: int, scope: str, query_hash: str) -> str:
        return f"{_NS}:qc1:{_seg(workspace_id)}:{_seg(corpus_id)}:{_seg(version)}:{_seg(scope)}:{_seg(query_hash)}"

    def _zset_key(self, workspace_id: str, corpus_id: str, version: int, scope: str) -> str:
        return f"{_NS}:qcz1:{_seg(workspace_id)}:{_seg(corpus_id)}:{_seg(version)}:{_seg(scope)}"

    def _live_entry(self, raw: str | None) -> CacheEntry | None:
        # One guard for decode, missing fields, AND the expiry comparison: a malformed or offset-naive
        # expires_at from a bad writer or an upgrade is a miss, never a crash on a served query.
        try:
            if raw is None:
                return None
            entry = cache_entry_from_json(json.loads(raw))
            if entry.expires_at <= self._clock.now():
                return None
            return entry
        except (ValueError, KeyError, TypeError):
            return None

    def get_exact_cache(
        self, workspace_id: str, corpus_id: str, version: int, user_scope: str, query_hash: str
    ) -> CacheEntry | None:
        return self._live_entry(self._get(self._entry_key(workspace_id, corpus_id, version, user_scope, query_hash)))

    def list_cache_candidates(
        self, workspace_id: str, corpus_id: str, version: int, user_scope: str, limit: int
    ) -> list[CacheEntry]:
        try:
            members = self._client.zrevrange(
                self._zset_key(workspace_id, corpus_id, version, user_scope), 0, max(0, limit - 1)
            )
        except self._transient as error:
            self._degraded(error)
            return []
        raws = self._mget(list(members))
        return [entry for raw in raws if (entry := self._live_entry(raw)) is not None]

    def put_cache(self, entry: CacheEntry) -> None:
        key = self._entry_key(
            entry.workspace_id, entry.corpus_id, entry.corpus_version, entry.user_scope, entry.query_hash
        )
        zkey = self._zset_key(entry.workspace_id, entry.corpus_id, entry.corpus_version, entry.user_scope)
        # ceil, not int: _persist reads the clock twice, so int() would truncate the TTL by a second.
        ex = max(1, math.ceil((entry.expires_at - self._clock.now()).total_seconds()))
        # Members are scored by created_at; drop any older than the TTL window so the per-scope index
        # cannot grow without bound as a hot scope accumulates distinct queries.
        stale_before = (self._clock.now() - timedelta(seconds=ex)).timestamp()
        try:
            self._client.set(key, json.dumps(cache_entry_to_json(entry)), ex=ex)
            self._client.zadd(zkey, {key: entry.created_at.timestamp()})
            # gt=True: only ever extend the index TTL, so a shorter-lived entry cannot expire the
            # shared scope index out from under a still-fresh longer-lived one.
            self._client.expire(zkey, ex, gt=True)
            self._client.zremrangebyscore(zkey, "-inf", stale_before)
        except self._transient as error:
            self._degraded(error)

    def delete_query_cache_for_corpus(self, workspace_id: str, corpus_id: str) -> int:
        deleted = self._drain(f"{_NS}:qc1:{_seg(workspace_id)}:{_seg(corpus_id)}:*")
        self._best_effort_unlink(f"{_NS}:qcz1:{_seg(workspace_id)}:{_seg(corpus_id)}:*")
        return deleted


class RedisModelCallCache(_RedisBase):
    def _gen_key(self, workspace_id: str, model_id: str, cache_key: str) -> str:
        return f"{_NS}:mc1:g:{_seg(workspace_id)}:{_seg(model_id)}:{cache_key}"

    def _emb_key(self, cache_key: str) -> str:
        return f"{_NS}:mc1:e:{cache_key}"

    def _put(self, key: str, payload: JsonObject, ttl_seconds: int) -> None:
        ex = max(1, int(ttl_seconds))
        wrapper = {"payload": payload, "expires_at": (self._clock.now() + timedelta(seconds=ex)).isoformat()}
        try:
            self._client.set(key, json.dumps(wrapper), ex=ex)
        except self._transient as error:
            self._degraded(error)

    def _unwrap(self, raw: str | None) -> JsonObject | None:
        if raw is None:
            return None
        try:
            wrapper = json.loads(raw)
            if datetime.fromisoformat(str(wrapper["expires_at"])) <= self._clock.now():
                return None
            return cast(JsonObject, wrapper["payload"])
        except (ValueError, KeyError, TypeError):
            return None

    def get_model_call(self, workspace_id: str, model_id: str, cache_key: str) -> JsonObject | None:
        return self._unwrap(self._get(self._gen_key(workspace_id, model_id, cache_key)))

    def put_model_call(
        self, workspace_id: str, model_id: str, cache_key: str, payload: JsonObject, ttl_seconds: int
    ) -> None:
        self._put(self._gen_key(workspace_id, model_id, cache_key), dict(payload), ttl_seconds)

    def get_embedding(self, model_id: str, cache_key: str) -> tuple[float, ...] | None:
        payload = self._unwrap(self._get(self._emb_key(cache_key)))
        if payload is None:
            return None
        try:
            return tuple(float(value) for value in payload["vector"])
        except (KeyError, TypeError, ValueError):
            return None

    def put_embedding(self, model_id: str, cache_key: str, vector: tuple[float, ...], ttl_seconds: int) -> None:
        self._put(self._emb_key(cache_key), {"vector": list(vector)}, ttl_seconds)

    def delete_generation_model_cache(self, workspace_id: str) -> int:
        return self._drain(f"{_NS}:mc1:g:{_seg(workspace_id)}:*")

    def delete_embedding_model_cache(self, key_digests: Sequence[str]) -> int:
        keys = list(dict.fromkeys(str(key) for key in key_digests))
        if not keys:
            return 0
        return self._unlink_all([self._emb_key(key) for key in keys])
