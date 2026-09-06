# Bounded RLM and autonomy

## Recursive context, not unrestricted code execution

Refaat's recursive language-model service keeps the complete corpus outside the model context. It recursively partitions and ranks evidence, performs grounded subcalls, carries sub-results into synthesis, and validates final citations against canonical chunks.

The caller controls four hard limits:

- `max_depth` — maximum recursion depth;
- `max_nodes` — maximum generation calls;
- `branch_factor` — maximum child groups;
- `leaf_token_budget` — evidence budget for each generation node.

A deployment can additionally bound the corpus a run loads into memory with `rlm_max_input_chunks` (unset by default, so unbounded). When set, a run whose corpus exceeds the ceiling fails closed with an error before the full chunk set is materialized, so a very large corpus never forces the whole set into memory. The check reads at most one chunk beyond the ceiling in a single query, so there is no separate counting pass, and the error names only the configured limit, not the corpus size. The access-control filter that scopes evidence to the caller still runs before any recursion.

A corpus that is within the ceiling still loads its document list and its co-occurrence graph in full. The document rows are lightweight and bounded by the chunk count. The graph is bounded per document by `graph_max_entities` for the built-in extractor, and is not separately bounded for a custom or model graph extractor. The ordinary query path performs the same whole-corpus graph read, so tightly bounding it is a shared retrieval concern rather than an RLM-only setting.

Every run returns node traces with parent relationships, depth, route, input size, selected chunks, cited chunks, and a bounded answer excerpt. Usage records include total model calls, input chunks, input tokens, deepest level, grounding fallbacks, model IDs, and effective limits.

## Synchronous and durable execution

```python
result = project.rlm("Compare the complete policy set")
job_id = project.enqueue_rlm("Compare the complete policy set")
```

The FastAPI surface exposes `/v1/corpora/{corpus_id}/rlm` and `/v1/corpora/{corpus_id}/rlm/jobs`. The durable worker handles `rlm.run` jobs with the same runtime service.

## Trust boundary

Refaat does not execute arbitrary Python emitted by a model. This avoids making unrestricted host execution part of the grounding primitive. Reviewed skills and plugins are the executable extension boundary.

An opt-in governed code capability is available on the in-process RLM path and is off by default. It runs only when an operator installs an executing sandbox and passes an admitting policy, and it never fabricates the grounded answer from code output. See [Sandboxed code in the bounded RLM](sandboxed-rlm.md) for the safe-by-default posture and the isolation boundary.

## Long-running autonomy

Durable sessions, goals, agent records, mailboxes, schedules, jobs, leases, heartbeats, retries, retained subagents, and compaction preserve progress across API and terminal restarts. These primitives do not grant authority automatically. The embedding application must define:

- allowed tools and data;
- spending and model-call budgets;
- approval gates;
- maximum autonomous duration;
- cancellation and escalation rules;
- audit retention;
- sandbox and network boundaries for executable tools.

## Self-improvement boundary

RLM execution does not mutate the base policy. Evaluation evidence may create a new supplemental harness snapshot under the bounded schema described in [Harness learning and rollback](harness.md). Activation remains explicit and reversible.
