# Cache backends

Refaat keeps two caches: the version-aware answer cache (CAG) and the content-addressed model-call
cache. Both default to the relational store, so a deployment that configures nothing keeps every cache
in SQLite or Postgres and behaves exactly as before. A deployment that wants a shared, in-memory cache
across many workers can point either cache at Redis, or bring its own backend through a typed port.

## Selecting a backend

Two settings select the backend for each cache independently. Both default to `sql`.

| Setting | Values | Selects |
| --- | --- | --- |
| `REFAAT_ANSWER_CACHE_BACKEND` | `sql`, `redis` | the CAG answer cache |
| `REFAAT_MODEL_CALL_CACHE_BACKEND` | `sql`, `redis` | the embedding and generation cache |

A Redis backend requires `REFAAT_REDIS_URL`. The model-call cache is a pure cost cache that only ever
holds a copy of a deterministic provider round-trip, so `REFAAT_MODEL_CALL_CACHE_BACKEND=redis`
requires `REFAAT_MODEL_CALL_CACHE_ENABLED=true`; selecting a Redis model cache that is never written
would add a failure surface with no benefit. `REFAAT_REDIS_TIMEOUT_SECONDS` bounds every Redis call so
an unreachable server fails fast rather than hanging the request.

The `redis` extra installs the client:

```bash
uv pip install --python .venv/bin/python -e ".[redis]"
```

## Behavior under Redis

The Redis adapters are built for a shared production cache and hold two guarantees.

- **Reads and writes degrade to a miss.** A transient Redis error on a read, a candidate list, or a
  write is swallowed: the query recomputes and serves rather than failing. The model-call cache is a
  cost optimization, so an outage costs money, never correctness.
- **Erasure is fail-closed.** A right-to-erasure request enumerates the real keys with `SCAN`, re-scans
  until a pass finds nothing so a key written during an earlier pass is still caught, and removes them
  with `UNLINK`, raising on a Redis error rather than reporting a false erasure. The answer cache for a
  corpus, a workspace's cached generations, and an erased text's embeddings are all swept. Erasure
  sweeps both the store and the currently configured backend, so a live deployment strands nothing. A
  backend migration must run erasure before the old backend is decommissioned: erasure can only reach a
  backend it is still configured for, so switching selectors back to `sql` and removing the Redis URL
  leaves any not-yet-erased Redis entries to expire on their own TTL. `SCAN` is not a snapshot: an answer
  generation already in flight when the erase runs can still write one stale entry after the drain, but
  it is keyed under the pre-erase corpus version, so it is never served and expires by its TTL. This is
  logical invalidation plus physical deletion of the keys present at erase time, not write-fencing of an
  in-flight generator; a deployment that requires physical fencing must quiesce writers or re-run the
  erase after draining in-flight requests.

Key segments are fixed-width digests, so a workspace or corpus id can never inject a glob metacharacter
into a `SCAN` pattern, and each key namespace carries a serde version so a format change never reads a
stale payload shape. Every entry carries its expiry, honored on read against the runtime clock in
addition to the native Redis TTL.

## Bringing your own backend

Each cache is a typed port, so a deployment can implement a backend for any store and inject it in
process:

```python
from refaat.application.runtime import RuntimeBuilder

runtime = RuntimeBuilder(settings).with_cache_store(MyCacheStore()).with_model_call_cache(MyModelCallCache()).build()
```

A `CacheStore` implements `get_exact_cache`, `list_cache_candidates`, `put_cache`, and
`delete_query_cache_for_corpus`. A `ModelCallCache` implements `get_model_call`, `put_model_call`,
`get_embedding`, `put_embedding`, `delete_generation_model_cache`, and `delete_embedding_model_cache`.
The erasure members are not optional: a backend that omits them would let a right-to-erasure request
silently leave cached answers or embeddings behind. See the Ports catalog for the full contract.

## Operational notes

- **Erasure enumerates with `SCAN`.** A right-to-erasure runs one or more `SCAN` passes over the
  keyspace while the corpus destroy lock is held. On a very large shared Redis this adds latency to an
  erase; it stays fail-closed, and the per-call timeout bounds each pass.
- **Erasure durability follows Redis durability.** The `ERASED` receipt is issued on the primary's
  `UNLINK` acknowledgement. A deployment with asynchronous replication or an on-disk `AOF` or `RDB`
  snapshot should treat the erasure as complete only once that state has propagated or been rewritten,
  the same consideration that applies to plaintext at rest anywhere else.
- **Degradation is logged.** A transient Redis error is logged once per adapter as
  `redis_cache_degraded`, and an unreachable server at startup logs `redis_cache_unreachable_at_startup`,
  so a persistent misconfiguration is visible rather than silently costing full model spend.
