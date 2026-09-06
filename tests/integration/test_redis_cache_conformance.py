"""Conformance run for the Redis cache adapters against a live server.

Skipped unless REFAAT_TEST_REDIS_URL points at a reachable Redis, so the offline gate never needs one;
the offline adapter logic is covered by tests/unit/test_redis_cache.py against an injected fake client.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest

from refaat.clock import SystemClock
from refaat.domain.models import CacheEntry, Citation, QueryResponse
from refaat.infrastructure.redis_cache import RedisCacheStore, RedisModelCallCache, build_redis_client

_URL = os.environ.get("REFAAT_TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(not _URL, reason="REFAAT_TEST_REDIS_URL is not set")


@pytest.fixture
def client() -> Iterator[object]:
    assert _URL is not None
    connection = build_redis_client(_URL, 1.0)
    try:
        connection.ping()
    except Exception:
        pytest.skip("Redis is configured but not reachable")
    try:
        yield connection
    finally:
        connection.close()


def _entry(query_hash: str, *, ws: str, corpus: str, now: datetime) -> CacheEntry:
    return CacheEntry(
        id=f"cache_{query_hash}",
        workspace_id=ws,
        corpus_id=corpus,
        corpus_version=1,
        user_scope="scope",
        query="q",
        query_hash=query_hash,
        query_embedding=(0.1, 0.2),
        response=QueryResponse(
            answer="grounded",
            citations=(Citation(chunk_id="c1", document_id="d1", label="1", quote="q"),),
            routes=("rag",),
            cached=False,
            corpus_version=1,
            trace_id="t1",
            usage={},
            abstained=False,
        ),
        expires_at=now + timedelta(hours=1),
        created_at=now,
    )


def test_answer_cache_round_trip_and_erasure_on_real_redis(client: object) -> None:
    store = RedisCacheStore(client, clock=SystemClock())
    ws, keep = f"ws-{uuid.uuid4().hex}", f"c-{uuid.uuid4().hex}"
    corpus = f"c-{uuid.uuid4().hex}"
    now = datetime.now(UTC)
    try:
        store.put_cache(_entry("a", ws=ws, corpus=corpus, now=now))
        store.put_cache(_entry("b", ws=ws, corpus=corpus, now=now))
        store.put_cache(_entry("keep", ws=ws, corpus=keep, now=now))
        assert store.get_exact_cache(ws, corpus, 1, "scope", "a") is not None
        assert {e.query_hash for e in store.list_cache_candidates(ws, corpus, 1, "scope", 10)} == {"a", "b"}
        assert store.delete_query_cache_for_corpus(ws, corpus) == 2
        assert store.get_exact_cache(ws, corpus, 1, "scope", "a") is None
        assert store.get_exact_cache(ws, keep, 1, "scope", "keep") is not None
    finally:
        store.delete_query_cache_for_corpus(ws, corpus)
        store.delete_query_cache_for_corpus(ws, keep)


def test_model_call_cache_round_trip_and_erasure_on_real_redis(client: object) -> None:
    cache = RedisModelCallCache(client, clock=SystemClock())
    ws = f"ws-{uuid.uuid4().hex}"
    ek1, ek2 = f"ek-{uuid.uuid4().hex}", f"ek-{uuid.uuid4().hex}"
    try:
        cache.put_model_call(ws, "gpt", "k1", {"answer": "x"}, 3600)
        cache.put_embedding("emb", ek1, (1.0, 2.0), 3600)
        cache.put_embedding("emb", ek2, (3.0,), 3600)
        assert cache.get_model_call(ws, "gpt", "k1") == {"answer": "x"}
        assert cache.get_embedding("emb", ek1) == (1.0, 2.0)
        assert cache.delete_generation_model_cache(ws) == 1
        assert cache.get_model_call(ws, "gpt", "k1") is None
        assert cache.delete_embedding_model_cache([ek1]) == 1
        assert cache.get_embedding("emb", ek1) is None
        assert cache.get_embedding("emb", ek2) == (3.0,)
    finally:
        cache.delete_generation_model_cache(ws)
        cache.delete_embedding_model_cache([ek1, ek2])
