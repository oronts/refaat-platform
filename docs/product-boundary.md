# Product boundary

**REFAAT — Retrieval & Evidence Foundation for Adaptive Agentic Tooling** is a Python-native evidence, context, and
durability runtime. It is not an agent UI, model gateway, vector database, unrestricted code sandbox, or framework fork.

## Combined context model

**RAG** supplies lexical and semantic evidence. **GraphRAG** expands relationships extracted during ingestion.
**CAG** reuses grounded answers only when corpus version, identity/session scope, policy, and harness state match.
**KAG** adds graph facts, executable skills, durable session state, persistent goals, bounded user preferences, and
supplemental harness state.

The planner chooses useful routes, fuses scores, deduplicates evidence, and packs a canonical packet under a hard token
budget. Generation must return verifiable chunk citations. Bounded RLM can recursively process externalized corpus
context while preserving those same evidence identities.

## What Refaat owns

- canonical ingestion, evidence, context, citation, and evaluation contracts;
- versioned corpora, content checksums, and idempotent replacement;
- adaptive retrieval, graph expansion, diversity, and token budgeting;
- exact and semantic grounded-answer caches;
- durable sessions, goals, compaction, jobs, schedules, agent mailboxes, and heartbeats;
- bounded recursive context execution with hard limits and node traces;
- bounded feedback learning and reversible harness snapshots;
- authorization, tenancy, audit/outbox records, traces, and structured logs;
- Python API, SDKs, CLI, FastAPI, plugin entry points, and thin framework adapters.

## Replaceable adapters

Embeddings, generation, extraction, graph extraction, search, blob storage, connectors, event sinks, rate limiting, and
agent-framework integration are typed ports. Operators can use the deterministic offline stack, OpenAI-compatible
providers, PostgreSQL/pgvector, S3, Redis, local models, or reviewed plugins without changing application services.

## Deliberate non-goals

Refaat does not infer truth from model output, execute arbitrary generated host code, mutate its immutable base prompt,
promise exactly-once delivery, or represent SQLite scans as a production ANN index. Production uses explicit
migrations, PostgreSQL/pgvector, durable leases, idempotent handlers, and at-least-once outbox delivery.
