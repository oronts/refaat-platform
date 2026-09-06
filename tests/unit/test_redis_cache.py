"""The Redis cache adapters degrade to a miss on an outage and erase fail-closed.

Driven against an injected in-memory fake so the real RedisCacheStore and RedisModelCallCache code
runs offline; a live-Redis conformance run lives in tests/integration/test_redis_cache_conformance.py.
"""

from __future__ import annotations

import fnmatch
import json
from datetime import UTC, datetime, timedelta

import pytest
import redis

from refaat.domain.models import CacheEntry, Citation, QueryResponse
from refaat.infrastructure.cache_payload import cache_entry_from_json, cache_entry_to_json, response_from_json
from refaat.infrastructure.database.store import SQLStore
from refaat.infrastructure.redis_cache import RedisCacheStore, RedisModelCallCache


class FakeClock:
    def __init__(self, now: datetime) -> None:
        self._now = now

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.zsets: dict[str, dict[str, float]] = {}
        self.fail = False
        self.closed = False

    def _guard(self) -> None:
        if self.fail:
            raise redis.exceptions.ConnectionError("down")

    def get(self, key: str) -> str | None:
        self._guard()
        return self.values.get(key)

    def mget(self, keys: list[str]) -> list[str | None]:
        self._guard()
        return [self.values.get(key) for key in keys]

    def close(self) -> None:
        self.closed = True

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        self._guard()
        self.values[key] = value

    def unlink(self, *keys: str) -> int:
        self._guard()
        deleted = 0
        for key in keys:
            if self.values.pop(key, None) is not None:
                deleted += 1
            if self.zsets.pop(key, None) is not None:
                deleted += 1
        return deleted

    def scan(self, cursor: int = 0, match: str | None = None, count: int | None = None) -> tuple[int, list[str]]:
        self._guard()
        universe = list(self.values) + list(self.zsets)
        keys = [key for key in universe if match is None or fnmatch.fnmatch(key, match)]
        return (0, keys)

    def zadd(self, key: str, mapping: dict[str, float]) -> None:
        self._guard()
        self.zsets.setdefault(key, {}).update(mapping)

    def zrevrange(self, key: str, start: int, end: int) -> list[str]:
        self._guard()
        members = sorted(self.zsets.get(key, {}).items(), key=lambda item: item[1], reverse=True)
        window = members[start:] if end == -1 else members[start : end + 1]
        return [member for member, _ in window]

    def zremrangebyscore(self, key: str, low: str, high: float) -> int:
        self._guard()
        members = self.zsets.get(key, {})
        floor = float("-inf") if low == "-inf" else float(low)
        doomed = [member for member, score in members.items() if floor <= score <= float(high)]
        for member in doomed:
            del members[member]
        return len(doomed)

    def expire(self, key: str, seconds: int, **_kwargs: object) -> bool:
        self._guard()
        return True


def _entry(query_hash: str, *, ws: str = "acme", corpus: str = "c1", ttl: int = 3600, now: datetime) -> CacheEntry:
    return CacheEntry(
        id=f"cache_{query_hash}",
        workspace_id=ws,
        corpus_id=corpus,
        corpus_version=1,
        user_scope="scope",
        query="what is x",
        query_hash=query_hash,
        query_embedding=(0.1, 0.2, 0.3),
        response=QueryResponse(
            answer="x is grounded",
            citations=(Citation(chunk_id="chunk_1", document_id="doc_1", label="1", quote="q"),),
            routes=("rag",),
            cached=False,
            corpus_version=1,
            trace_id="trace_1",
            usage={"prompt_tokens": 3},
            abstained=False,
        ),
        expires_at=now + timedelta(seconds=ttl),
        created_at=now,
    )


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(datetime(2026, 1, 1, tzinfo=UTC))


def test_answer_cache_round_trip(clock: FakeClock) -> None:
    store = RedisCacheStore(FakeRedis(), clock=clock)
    entry = _entry("qh1", now=clock.now())
    store.put_cache(entry)
    got = store.get_exact_cache("acme", "c1", 1, "scope", "qh1")
    assert got is not None
    assert got.response.answer == "x is grounded"
    assert got.response.citations[0].chunk_id == "chunk_1"
    assert got.query_embedding == (0.1, 0.2, 0.3)


def test_answer_cache_miss_returns_none(clock: FakeClock) -> None:
    store = RedisCacheStore(FakeRedis(), clock=clock)
    assert store.get_exact_cache("acme", "c1", 1, "scope", "absent") is None


def test_list_candidates_returns_recent_first(clock: FakeClock) -> None:
    store = RedisCacheStore(FakeRedis(), clock=clock)
    store.put_cache(_entry("older", now=clock.now()))
    clock.advance(10)
    store.put_cache(_entry("newer", now=clock.now()))
    got = store.list_cache_candidates("acme", "c1", 1, "scope", limit=5)
    assert got[0].query_hash == "newer"


def test_clock_filter_expires_a_stale_entry(clock: FakeClock) -> None:
    store = RedisCacheStore(FakeRedis(), clock=clock)
    store.put_cache(_entry("qh", ttl=60, now=clock.now()))
    clock.advance(61)
    assert store.get_exact_cache("acme", "c1", 1, "scope", "qh") is None


def test_deserialization_failure_is_a_miss(clock: FakeClock) -> None:
    fake = FakeRedis()
    store = RedisCacheStore(fake, clock=clock)
    key = store._entry_key("acme", "c1", 1, "scope", "qh")
    fake.values[key] = "not json{{{"
    assert store.get_exact_cache("acme", "c1", 1, "scope", "qh") is None


def test_a_malformed_naive_expires_at_is_a_miss_not_a_crash(clock: FakeClock) -> None:
    fake = FakeRedis()
    store = RedisCacheStore(fake, clock=clock)
    key = store._entry_key("acme", "c1", 1, "scope", "qh")
    # An offset-naive expires_at (a bad writer or a pre-upgrade row) would raise on comparison with the
    # tz-aware clock; the read must treat it as a miss, never crash the query.
    fake.values[key] = json.dumps(
        {
            "id": "x",
            "workspace_id": "acme",
            "corpus_id": "c1",
            "corpus_version": 1,
            "user_scope": "scope",
            "query": "q",
            "query_hash": "qh",
            "query_embedding": [0.1],
            "response": {"answer": "a", "corpus_version": 1, "trace_id": "t"},
            "expires_at": "2099-01-01T00:00:00",
            "created_at": "2099-01-01T00:00:00",
        }
    )
    assert store.get_exact_cache("acme", "c1", 1, "scope", "qh") is None


def test_put_trims_the_stale_zset_index(clock: FakeClock) -> None:
    fake = FakeRedis()
    store = RedisCacheStore(fake, clock=clock)
    store.put_cache(_entry("old", ttl=60, now=clock.now()))
    clock.advance(61)
    store.put_cache(_entry("new", ttl=60, now=clock.now()))
    zkey = store._zset_key("acme", "c1", 1, "scope")
    members = {store._entry_key("acme", "c1", 1, "scope", qh) for qh in ("new",)}
    assert set(fake.zsets[zkey]) == members


def test_close_disconnects_the_client(clock: FakeClock) -> None:
    fake = FakeRedis()
    RedisCacheStore(fake, clock=clock).close()
    assert fake.closed is True


def test_url_sanitize_strips_timeout_and_retry_overrides() -> None:
    from refaat.infrastructure.redis_cache import _sanitize_redis_url

    dirty = "redis://host:6379/0?socket_timeout=3600&socket_connect_timeout=3600&retry_on_timeout=True&db=2"
    cleaned = _sanitize_redis_url(dirty)
    assert "socket_timeout" not in cleaned
    assert "socket_connect_timeout" not in cleaned
    assert "retry_on_timeout" not in cleaned
    assert "db=2" in cleaned


def test_degraded_flag_round_trips(clock: FakeClock) -> None:
    from dataclasses import replace

    store = RedisCacheStore(FakeRedis(), clock=clock)
    entry = _entry("qh", now=clock.now())
    store.put_cache(replace(entry, response=replace(entry.response, degraded=True)))
    got = store.get_exact_cache("acme", "c1", 1, "scope", "qh")
    assert got is not None
    assert got.response.degraded is True


def test_erasure_drains_a_key_written_during_the_scan(clock: FakeClock) -> None:
    class _RacyRedis(FakeRedis):
        def __init__(self) -> None:
            super().__init__()
            self._injected = False
            self._late_key = ""

        def unlink(self, *keys: str) -> int:
            deleted = super().unlink(*keys)
            if not self._injected and self._late_key:
                self._injected = True
                self.values[self._late_key] = "{}"
            return deleted

    fake = _RacyRedis()
    store = RedisCacheStore(fake, clock=clock)
    store.put_cache(_entry("early", now=clock.now()))
    fake._late_key = store._entry_key("acme", "c1", 1, "scope", "late")
    deleted = store.delete_query_cache_for_corpus("acme", "c1")
    assert deleted == 2
    assert store.get_exact_cache("acme", "c1", 1, "scope", "late") is None


def test_reads_and_writes_degrade_to_miss_on_outage(clock: FakeClock) -> None:
    fake = FakeRedis()
    store = RedisCacheStore(fake, clock=clock)
    fake.fail = True
    assert store.get_exact_cache("acme", "c1", 1, "scope", "qh") is None
    assert store.list_cache_candidates("acme", "c1", 1, "scope", limit=5) == []
    store.put_cache(_entry("qh", now=clock.now()))  # best-effort, must not raise


def test_answer_erasure_deletes_a_corpus_and_isolates_others(clock: FakeClock) -> None:
    store = RedisCacheStore(FakeRedis(), clock=clock)
    store.put_cache(_entry("a", corpus="c1", now=clock.now()))
    store.put_cache(_entry("b", corpus="c1", now=clock.now()))
    store.put_cache(_entry("keep", corpus="c2", now=clock.now()))
    deleted = store.delete_query_cache_for_corpus("acme", "c1")
    assert deleted == 2
    assert store.get_exact_cache("acme", "c1", 1, "scope", "a") is None
    assert store.get_exact_cache("acme", "c2", 1, "scope", "keep") is not None


def test_answer_erasure_is_fail_closed(clock: FakeClock) -> None:
    fake = FakeRedis()
    store = RedisCacheStore(fake, clock=clock)
    store.put_cache(_entry("a", now=clock.now()))
    fake.fail = True
    with pytest.raises(redis.exceptions.RedisError):
        store.delete_query_cache_for_corpus("acme", "c1")


def test_model_call_round_trip_and_isolation(clock: FakeClock) -> None:
    cache = RedisModelCallCache(FakeRedis(), clock=clock)
    cache.put_model_call("acme", "gpt", "k1", {"answer": "hi", "cited_chunk_ids": ["c"], "usage": {}}, 3600)
    assert cache.get_model_call("acme", "gpt", "k1") == {"answer": "hi", "cited_chunk_ids": ["c"], "usage": {}}
    assert cache.get_model_call("other", "gpt", "k1") is None


def test_embedding_round_trip_and_clock_filter(clock: FakeClock) -> None:
    cache = RedisModelCallCache(FakeRedis(), clock=clock)
    cache.put_embedding("emb", "ek", (1.0, 2.0), 60)
    assert cache.get_embedding("emb", "ek") == (1.0, 2.0)
    clock.advance(61)
    assert cache.get_embedding("emb", "ek") is None


def test_model_call_degrades_to_miss_on_outage(clock: FakeClock) -> None:
    fake = FakeRedis()
    cache = RedisModelCallCache(fake, clock=clock)
    fake.fail = True
    assert cache.get_model_call("acme", "gpt", "k") is None
    cache.put_model_call("acme", "gpt", "k", {"answer": "x"}, 3600)  # best-effort


def test_generation_erasure_purges_a_workspace_fail_closed(clock: FakeClock) -> None:
    fake = FakeRedis()
    cache = RedisModelCallCache(fake, clock=clock)
    cache.put_model_call("acme", "gpt", "k1", {"answer": "a"}, 3600)
    cache.put_model_call("acme", "gpt", "k2", {"answer": "b"}, 3600)
    cache.put_model_call("other", "gpt", "k1", {"answer": "c"}, 3600)
    assert cache.delete_generation_model_cache("acme") == 2
    assert cache.get_model_call("acme", "gpt", "k1") is None
    assert cache.get_model_call("other", "gpt", "k1") is not None
    fake.fail = True
    with pytest.raises(redis.exceptions.RedisError):
        cache.delete_generation_model_cache("acme")


def test_embedding_erasure_is_empty_safe_and_exact(clock: FakeClock) -> None:
    cache = RedisModelCallCache(FakeRedis(), clock=clock)
    cache.put_embedding("emb", "ek1", (1.0,), 3600)
    cache.put_embedding("emb", "ek2", (2.0,), 3600)
    assert cache.delete_embedding_model_cache([]) == 0
    assert cache.delete_embedding_model_cache(["ek1"]) == 1
    assert cache.get_embedding("emb", "ek1") is None
    assert cache.get_embedding("emb", "ek2") == (2.0,)


def test_generation_erasure_does_not_touch_embeddings(clock: FakeClock) -> None:
    cache = RedisModelCallCache(FakeRedis(), clock=clock)
    cache.put_model_call("acme", "gpt", "k", {"answer": "a"}, 3600)
    cache.put_embedding("emb", "ek", (1.0,), 3600)
    cache.delete_generation_model_cache("acme")
    assert cache.get_embedding("emb", "ek") == (1.0,)


def test_serde_parity_with_the_sql_store(clock: FakeClock) -> None:
    entry = _entry("qh", now=clock.now())
    assert cache_entry_from_json(cache_entry_to_json(entry)) == entry
    # The store and the shared serde encode a response identically, so a value written by one backend
    # and read by another is the same object.
    assert SQLStore._serialize_response(entry.response) == cache_entry_to_json(entry)["response"]


def test_response_from_json_decodes_a_legacy_row_shape() -> None:
    # A golden row in the exact shape the pre-extraction inline SQL serde wrote, so the byte-identity
    # of the extraction rests on decoding a real historical shape, not on comparing a function to itself.
    golden = {
        "answer": "grounded",
        "citations": [{"chunk_id": "c1", "document_id": "d1", "label": "1", "quote": "q"}],
        "routes": ["rag"],
        "cached": False,
        "corpus_version": 3,
        "trace_id": "t1",
        "usage": {"prompt_tokens": 5},
        "abstained": False,
    }
    response = response_from_json(golden)
    assert response.answer == "grounded"
    assert response.corpus_version == 3
    assert response.routes == ("rag",)
    assert response.citations[0].chunk_id == "c1"
    assert response.usage == {"prompt_tokens": 5}
    assert response.abstained is False
