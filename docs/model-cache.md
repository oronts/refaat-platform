# Model-call cache and cost budgeting

Refaat can cache provider round-trips and enforce a per-run token or cost ceiling. Both are opt-in and transparent: with the defaults nothing is installed and the runtime is byte-identical to a build without them.

## What it does

Two decorators sit in front of the model ports:

- `CachingEmbedder` wraps the embedder. It serves identical embedding calls from a content-addressed store table without a provider round-trip and returns vectors in input order.
- `CachingGenerator` wraps the generator. It serves identical generation calls from the same table and, when a run budget is open, charges the meter with the real spend of every miss.

Both delegate `model_id` (and `dimensions`) to the inner adapter unchanged, so the answer-cache identity and the citation contract are untouched. A cache hit returns byte-identical output to a miss, because decode is deterministic at temperature zero.

### Cache key correctness

The generation key is a canonical digest of the complete generation input: the backend `cache_identity` (base URL plus model), the tenant `workspace_id`, the system prompt and query, every retrieved hit with its full chunk (including the document id the prompt renders) and scores, the graph facts, session summary, profile, supplemental harness, instructions, demonstrations, and response schema. Equal key therefore implies equal provider input, so no two distinct requests can ever share a cache entry, and the built-in generator renders the profile and harness in one sorted form so equal key also implies equal prompt bytes.

Scoping closes two leaks. Keying on `cache_identity` rather than a bare model name means two backends serving the same model name (a different `base_url`) never read each other's rows. Scoping the generation table by `workspace_id` means a tenant can only ever read rows it wrote. Embeddings are content-addressed and backend-scoped but not workspace-scoped on purpose: an embedding of identical text under the same backend is identical, so a cross-tenant hit leaks no vector value a tenant could not compute itself. The one residual signal is existence: a faster, zero-charge hit lets a tenant infer that some other tenant previously embedded that exact text. This same-text membership side channel is an accepted tradeoff for a shared embedding cache; scope embeddings by workspace, like generation rows, if membership inference is in the threat model.

Wrapping is a positive opt-in. Only an adapter that exposes a `cache_identity` property is ever cached, so the offline deterministic embedder and extractive generator, and any custom deterministic adapter, stay unwrapped and can never be served a stale value after a config change. A generator or embedder that is already wrapped is never double-wrapped, and a wrapper owns its inner adapter's close, so the underlying adapter is closed exactly once.

## Configuration

| Setting | Default | Effect |
| --- | --- | --- |
| `REFAAT_MODEL_CALL_CACHE_ENABLED` | `false` | Install the caching decorators around real provider adapters. |
| `REFAAT_RUN_TOKEN_BUDGET` | unset | Per-run token ceiling for the multi-attempt loops. |
| `REFAAT_RUN_COST_BUDGET_USD` | unset | Per-run USD ceiling, evaluated against the price map. |
| `REFAAT_MODEL_PRICES` | empty | Map of `model_id` to `{input_per_1k, output_per_1k}` USD. |

A token budget needs no price map. A cost budget does: a `REFAAT_RUN_COST_BUDGET_USD` set over a real generation model that carries no `REFAAT_MODEL_PRICES` entry would never trip, since an unpriced model accrues zero cost, so startup validation fails closed and requires the configured model to be priced.

## Budget semantics

A `CostMeter` is carried on a context variable for the duration of a run. The shared query path, the Best-of and Refine loop, the governed ReAct loop, and every native optimizer's candidate loop open a budget scope read from settings, so a plain `ask()` is bounded as well as a program run. Nested scopes share the one outer meter instead of resetting the budget, so Best-of wrapping ReAct spends against a single ceiling.

Token and cost are known only after a call, so the meter is charge only and never raises mid-call. Enforcement lives at the loop boundary: before starting the next attempt, candidate, or tool iteration, a loop that finds the meter already exhausted stops and selects the best result gathered so far. The first unit always runs, so a run never returns nothing, and the final grounded generation in a governed run always runs, so an exactly-exhausted run still answers rather than crashing.

There is no hard pre-flight ceiling and no dedicated budget-exceeded error. The contract is deliberately charge only plus stop at boundary: a request is never rejected before it runs and a run in progress is never aborted, so there is no 429 for an exceeded budget. A budget bounds how many further units a multi-attempt loop starts, not whether the current unit completes.

The meter charges only on a real miss, across every model path: generation, embeddings, streaming, and model tool planning. A cache hit charges nothing, because it was already paid. On the deterministic offline stack the adapters are unwrapped, so the meter never trips unless a ceiling is set for a real provider.

## Durability

Cached rows carry an expiry and are reaped by the `cache.purge` worker job, which now purges both the answer cache and the model-call cache. The table is declared in the schema and created by the standard migration.
