# Custom search backend

Refaat ships two search backends: an in-process `LocalSearchIndex` (SQLite plus in-memory vectors) and
`PgVectorSearchIndex` (PostgreSQL with pgvector), selected by `database_url`. A third backend, such as
OpenSearch, Elasticsearch, Qdrant, or a managed vector service, plugs in through the `SearchIndex`
port and the `with_search_index` seam without touching the core.

## The port

A backend implements four methods: `replace_document`, `delete_document`, `semantic_search`, and
`lexical_search`. The two search methods return `SearchHit` values whose `chunk` is a stored chunk, so
the runtime resolves every citation against a real row. Run `refaat plugin describe SearchIndex` for
the exact members.

## Injection

A self-contained backend is injected directly. A backend that needs the runtime store, for chunk
hydration and the canonical ACL and metadata pushdown, is injected as a factory whose `build` receives
the store and embedder at the single composition point:

```python
from refaat.application.runtime import RuntimeBuilder
from refaat.infrastructure.opensearch_index import OpenSearchIndexFactory

runtime = (
    RuntimeBuilder(settings)
    .with_search_index(OpenSearchIndexFactory(hosts=["https://opensearch:9200"], index_name="refaat_chunks"))
    .build()
)
```

The built-in backends stay the default when nothing is injected, so the runtime is byte-identical
without a custom backend. An injected backend may expose a `cache_id`, and when it does, that id scopes
the answer cache so a cached answer is never reused across two backends that could select different
evidence.

## The three rules a backend must uphold

1. **Every hit resolves to a stored chunk.** Generate candidates in the engine, then hydrate them back
   through the store and return `SearchHit` values for real rows. The reference `OpenSearchSearchIndex`
   uses a knn query for semantic search and a match query for lexical search, then re-scores the
   hydrated chunks locally so scores share a basis with the built-in backends.
2. **Fail closed, never fall back to a full scan.** On a transport error, a timeout, or a partial-shard
   result the backend raises `DependencyUnavailableError` rather than quietly scanning the store or
   serving a truncated result, which would bypass the engine and its pushdown. The reference adapter
   sends `allow_partial_search_results=False` and rejects any response with a failed or missing shard.
3. **Negotiate capabilities honestly.** A backend that cannot serve a mode raises
   `CapabilityNotSupportedError` from that method and declares it with
   `SearchIndexCapabilities(supports_lexical=False)` (or `supports_semantic=False`) rather than
   returning an empty result. The retriever reads the declaration and skips that mode, so a
   semantic-only backend is servable. A backend that declares nothing is treated as a full in-process
   hybrid, so the built-in backends are unaffected.

## Durable healing

A backend that also exposes `reconcile_document(store, workspace_id, corpus_id, document_id)` is wired
to the reconcile sink, so the durable `document.indexed` outbox event rebuilds its index rows from the
committed chunks after a crash between the relational commit and the index write. Reading the current
committed chunks under the ingest overwrite lock makes healing version-safe: an out-of-order replay of
an older event still writes the document's latest state. Cross-process version-safety needs a shared
lock, so it holds on PostgreSQL (an advisory lock); on SQLite the lock is process-local, so a separate
worker process does not serialize against an in-process ingest.

## Conformance

The shipped behavioral contract asserts these invariants for any backend, and the Local, pgvector, and
OpenSearch backends all pass it. Seed a backend through the normal ingestion path, then run the
contract:

```python
from refaat.conformance import search_index_conformance, SearchIndexConformanceCase

case = SearchIndexConformanceCase(
    index=runtime.queries.retriever.index,
    workspace_id=ws,
    corpus_id=corpus,
    query_embedding=runtime.queries.embedder.embed(["a seeded query"])[0],
    all_chunk_ids=frozenset(chunk.id for chunk in runtime.store.list_chunks(ws, corpus)),
    document_id=document_id,
    document_chunk_ids=document_chunk_ids,
    lexical_term="a-seeded-term",
)
assert search_index_conformance(case) == []
```

The contract checks scope isolation, the citation guarantee, descending score order, and, on both the
semantic and the lexical path, an ACL allow-set that fails closed on the empty set, a document allow-set
that never leaks another document, and a metadata filter that never leaks. It also enforces the
declared-capability rule and reports a non-discriminating case rather than passing it silently. The case
must be seeded so the unfiltered result spans more than the single scoped document, or the scope checks
have nothing to catch. Conformance is evidence that a backend upholds the contract, not a substitute for
testing the backend against a real cluster: an approximate-nearest-neighbour engine's recall and its
in-cluster filtering must still be validated against a running instance.
