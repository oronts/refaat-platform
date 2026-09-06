# Building a production RAG system with Refaat

This guide maps a production retrieval system onto Refaat's existing runtime instead of proposing a second architecture around it.

## 1. Define the evidence boundary

A workspace is the top-level tenant boundary. A corpus is a versioned evidence collection within that workspace. Keep corpora narrow enough that access, retention, embedding, and evaluation policies are coherent.

```python
with Refaat() as runtime:
    project = runtime.project(
        workspace="acme",
        corpus="engineering-handbook",
        corpus_settings={"language": "en", "classification": "internal"},
    )
```

Do not use user profiles, session summaries, or harness state as substitutes for source documents. They can influence context selection but are not citation targets.

## 2. Retain originals and version every change

The ingestion service stores the original blob before replacing search state. It computes a checksum, extracts text, chunks it, embeds chunks, extracts graph state, updates the search index, and commits relational records. Re-ingesting the same source and checksum is idempotent. Replacing or deleting evidence advances the corpus version.

Applications should use stable `source_id` values derived from the upstream record, not temporary filenames.

```python
project.ingest_source(
    "https://knowledge.example.com/policies/release",
    source_id="policy:release",
    metadata={"classification": "internal", "owner": "platform"},
)
```

For private destinations, enforce network-level egress policy in addition to Refaat's URL checks.

## 3. Select production adapters

A common production composition is:

| Port | Recommended production choice |
|---|---|
| Relational/search store | PostgreSQL with pgvector |
| Original blobs | Versioned S3-compatible object storage |
| Distributed rate limit | Redis |
| Embeddings | OpenAI-compatible embedding endpoint or an organization adapter |
| Generation | OpenAI-compatible chat endpoint or an organization adapter |
| Graph extraction | Co-occurrence for deterministic baseline; typed model extraction when evaluated |
| Telemetry | OTLP collector |

Changing the embedding model or dimensions is a data migration. Rebuild into a new corpus/index version and evaluate it before switching traffic.

## 4. Tune through effective policy

A query resolves one effective `RetrievalPolicy` containing:

- `top_k`;
- candidate limit;
- token budget;
- semantic, lexical, and graph weights.

Explicit request values override defaults. An active harness can supply an allowlisted default `top_k` and weights. Weights are normalized before fusion.

```python
answer = project.ask(
    "What controls are required before production?",
    top_k=8,
    token_budget=3_000,
)
```

The effective policy is part of the cache-context fingerprint. A query with `top_k=1` cannot receive an answer cached under `top_k=12`.

## 5. Preserve grounding

The generator receives a canonical `ContextPacket`. It can cite only selected chunk IDs. Refaat validates the cited IDs and returns structured citations:

```python
for citation in answer.citations:
    print(citation.label, citation.document_id, citation.quote)
```

Citation validation proves that the answer is bound to retrieved evidence. It does not replace domain review or guarantee that the source itself is correct.

## 6. Evaluate the complete path

Evaluation cases contain a query and expected terms. A run records:

- case count;
- answer hit rate;
- citation recall;
- grounded rate.

Use representative domain queries, version the case set, and run it before model, chunking, retrieval-weight, prompt-supplement, or graph changes.

```python
suite_id = project.create_evaluation_suite("release-regression", cases)
run = project.run_evaluation(suite_id)
```

The built-in harness proposer requires at least ten cases before suggesting a bounded change.

## 7. Operate API and workers independently

Run migrations once, then deploy API and worker roles from the same image. Scale workers using unique `REFAAT_WORKER_ID` values. Monitor leases, retries, outbox backlog, connector failures, cache routes, selected context, and evaluation regressions.

## 8. Validate recovery and deletion

PostgreSQL and object storage must share one recovery policy. A restored database that points to missing original blobs is not a complete recovery. Test document deletion, corpus replay, vector rebuild, key rotation, and point-in-time restore before production launch.

See [Deployment](../deployment.md), [Backup and restore](../operations/backup-restore.md), and [Production checklist](../production-checklist.md).
