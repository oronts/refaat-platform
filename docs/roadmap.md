# Roadmap and non-goals

The current repository already includes adaptive RAG/CAG/KAG/GraphRAG, bounded RLM, durable jobs and agents, skills,
evaluations, bounded harness snapshots, typed programs with evaluation and optimization, the durable program registry
with promotion, approval, and rollback, governed ReAct with capability-based tool governance and human approvals,
per-principal authorization and multi-tenancy enforcement across every evidence path, provider-neutral model adapters,
streaming, reranking, the model-call cache and run budgets, the resource-gated multimodal, fine-tuning, and
sandboxed-code RLM features, FastAPI, SDKs, CLI, OpenTelemetry wiring, Compose, and Helm. The roadmap focuses
on hardening and ecosystem depth rather than renaming planned work as completed capability.

## Near-term hardening

- resumable reindex and embedding-model migration workflows;
- richer PostgreSQL lexical indexes, partitioning, and tenant-scale benchmarks;
- connector checkpoint, pagination, deletion, and incremental-sync contracts;
- source-specific connector packages with security and conformance suites;
- calibrated retrieval, graph, citation, latency, and answer evaluation packs;
- policy hooks for OGuardAI and customer DLP/PII systems;
- distributed worker notifications to reduce database polling;
- model-provider circuit breakers (per-run cost and token budgets already ship);
- isolated execution service for untrusted tools where required;
- Refaat Studio as a separate client of the public API.

## Non-goals

Refaat is not a model host, vector database, secret manager, general workflow UI, unrestricted browser/terminal agent, or
replacement for business-specific authorization. It ships the enforcement seams, the policy decision and enforcement
points, while identity, the policy engine, connectors, and models plug in, so it is not the source of truth for identity
or authorization. It does not promise that one retrieval strategy fits every corpus or
that grounded citations guarantee domain truth. The runtime makes strategies observable, configurable, evaluated, and
replaceable.
