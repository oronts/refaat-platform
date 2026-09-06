# Quotas and cost governance

Refaat can cap how much a workspace, a principal, or a corpus consumes over a fixed window. The
quota governor is a persisted, opt-in, per-scope admission and cost control. It is distinct from the
per-run cost meter, which bounds a single run, and from the edge rate limiter, which throttles HTTP
requests per address. A quota accumulates across runs and workers, is enforced fail closed, and is
byte-identical when it is not configured.

## What it governs

Eight dimensions:

- `generation_tokens`: the provider tokens a run spends, summed across the run.
- `cost_usd`: the model cost of a run, metered as integer micro-USD so accounting is exact.
- `requests`: a count of admitted query requests (query, streaming query, and RLM).
- `ingest_requests`: a count of admitted ingests, so a tenant cannot exhaust embedding resources
  through unbounded ingestion. It bounds the ingest and indexing work, charged once per ingest before
  the embedding runs; like `requests` it counts each admitted attempt, so an ingest that is admitted
  and then fails validation still spends its unit. It is a separate dimension, so a `requests` rule
  keeps its meaning and never counts ingests. Ingestion carries no request principal, so a
  `principal`-scoped rule skips it; a `workspace` or `corpus` rule governs it. A connector fetch on a
  source ingest runs before the charge and is bounded instead by the writes kill switch and the
  connector safety limits. Internal shadow re-ingestion during optimization carries no quota.
- `program_requests`: a count of admitted served program requests. Each served HTTP request that
  executes a program pipeline is charged once (run, streaming run, deployment run, and evaluate), so an
  operator can cap the rate of program executions independently of the token and cost budgets its inner
  queries already meter. It is a separate dimension so it never double-counts against the `requests` an
  inner query charges. The charge is at the served surface, not inside the shared program-run primitive,
  so an internal evaluation, optimization, shadow or canary, or nested program-skill run never consumes
  the counter. A direct in-process or CLI program run is trusted local code, outside this served-surface
  control, like the edge rate limiter.
- `skill_requests`: a count of admitted served skill runs, charged once at the `SkillService` funnel.
  Because that funnel is the single chokepoint every served skill invocation passes through, it covers
  the HTTP skill-run route, the MCP bridge, and an in-process skill run, unlike `program_requests` which
  charges at the HTTP route because its run primitive is reused internally. A program exposed as a skill
  is capped here rather than under `program_requests`, so the two surfaces cap independently: a program
  run over the skill route charges `skill_requests`, never `program_requests`. Its inner grounded query
  still charges `requests`, exactly as a program run's inner query does, since that query is a real
  query. A skill invoked as a governed tool inside a program run reaches the skill's own run directly,
  not the funnel, so it does not consume the counter. The MCP bridge carries no resolved identity, so a
  `principal` or `workspace_principal` scoped `skill_requests` rule denies an MCP skill run fail closed;
  scope such a rule to the `workspace` or `corpus` to govern the MCP surface.
- `optimization_requests`: a count of admitted optimization enqueues, charged once at the served HTTP
  optimization route. An optimization run spawns many candidate evaluations, so capping the enqueue rate
  bounds that fan-out; the candidate runs themselves execute on the worker and are internal, so they do
  not consume it.
- `fine_tune_requests`: a count of admitted fine-tune requests, charged once at the fine-tune request
  route, the initiating admission of a fine-tune. Enqueuing an already requested and approved run is not
  re-charged, so a limit of N permits N fine-tunes rather than stranding approved runs. Fine-tune carries
  no corpus, so a `workspace` or `principal` rule governs it and a `corpus`-scoped rule denies it fail
  closed.

Four scopes, each keyed so that a workspace boundary is never crossed:

- `workspace`: one counter per workspace.
- `principal`: one counter per resolved principal, prefixed by the workspace.
- `workspace_principal`: the same as principal, kept as a separate name for clarity.
- `corpus`: one counter per validated corpus, prefixed by the workspace.

Every scope key embeds the workspace id, which is globally unique and server derived, so a principal
or corpus id that is only unique within a workspace can never share a counter across tenants. The
tenant travels only into the audit event, never into a row predicate.

## Enabling it

Off by default. Set the backend and the rules:

```
REFAAT_QUOTA_BACKEND=database
REFAAT_QUOTA_RULES='[{"scope":"workspace","dimension":"requests","limit":1000,"window_seconds":3600}]'
REFAAT_QUOTA_RESERVATION_LEASE_SECONDS=900
```

Each rule is `{scope, dimension, limit, window_seconds}` with positive integer `limit` and
`window_seconds`. A malformed rule fails at startup rather than governing nothing. The rules are an
immutable snapshot loaded once at build time, so there is no rule read that can fail open.

A custom backend can be installed with `RuntimeBuilder.with_quota_policy`, for example a Redis backed
one. The default database backend uses two tables, `quota_counters` and `quota_reservations`, over
the same store the rest of the runtime uses.

## Where it is enforced

The quota consults these seams, after the incident kill switch and before any model or embedding
spend: `QueryService.query`, `QueryService.stream_query`, and `RecursiveContextService.run` for the
query dimensions, `IngestionService.ingest` and `ingest_image` for `ingest_requests`, and the served
HTTP program routes (run, run stream, deployment run, and evaluate) for `program_requests`, and
`SkillService.run` for `skill_requests` (covering the HTTP route, the MCP bridge, and in-process). A
program run is charged at the served route rather than inside the shared `BoundProgram.run` primitive
(reused by evaluation, optimization, and the shadow), while a skill run is charged at the `SkillService`
funnel because that funnel is not reused by an internal governed-tool call. The served HTTP optimization
route charges `optimization_requests` and the served HTTP fine-tune request route charges `fine_tune_requests`.
A request counts against a `requests` quota before the answer cache is read, so a cache hit still
consumes its request budget. Token and cost reservations wrap only the generation path, so a cache hit
reserves nothing.

Every served, user-reachable expensive path is now metered: query, ingest, program, skill, optimization,
and fine-tune. Only the internal worker executions those enqueues schedule stay uncharged, since their
model spend is already governed by the token and cost dimensions.

## Reserve then reconcile

Token and cost cannot be known before the model call, so the governor reserves the run ceiling up
front and reconciles down to the measured spend afterward. The ceiling is the configured per-run
budget: a `generation_tokens` quota requires `REFAAT_RUN_TOKEN_BUDGET`, and a `cost_usd` quota
requires `REFAAT_RUN_COST_BUDGET_USD`. You cannot reserve a ceiling you do not have, so this is
enforced at startup.

A reservation is a durable, leased, idempotent row, not an in-memory handle. The reserve is a single
atomic accumulate and check: an `UPDATE ... WHERE spent + reserved + estimate <= limit` decided by
the affected row count, inside one immediate transaction. Two near-exhausted concurrent reservations,
even across two worker processes, can never both pass. On commit the reservation moves the measured
actual into `spent` and returns the reserved ceiling to baseline. On failure it releases the
reservation with no spend recorded.

## Fail closed

An active quota denies on every failure path:

- An exhausted counter denies with HTTP 429 (`QuotaExceededError`).
- Any store read or write error denies rather than serving.
- A principal scoped rule that meets a principal bearing seam with no principal denies.
- A corpus rule over a corpus that has not been validated denies, so an arbitrary client corpus id
  can never mint a fresh counter row.
- An unpriced model under a `cost_usd` quota fails at startup, and as defense in depth charges the
  full ceiling and denies at runtime, so an unpriced run consumes its worst case rather than nothing.

## Crash safety

A crash between reserve and reconcile leaves an open, leased reservation. The `quota.reap` worker job
reclaims a reservation whose lease has expired by charging its ceiling to `spent` and returning the
reserved amount to baseline. Charging the worst case is fail closed, and returning the reservation to
baseline means a crash never permanently blocks the tenant: capacity returns in the next window. Set
`REFAAT_QUOTA_RESERVATION_LEASE_SECONDS` above the longest run wall clock. Too short charges a live
long run its full ceiling, too long delays capacity reclaim after a crash.

## Owner decisions

These defaults were chosen conservatively and can be overridden.

- Backend: `database`, portable and with one authoritative counter. Redis is a documented future
  backend, not built.
- Window: a fixed, aligned tumbling window with a deterministic reset, matching the existing rate
  limiter precedent. A sliding window is tighter but breaks the single-row atomic guard.
- Default limits: none. An operator configures every rule. With no rules the runtime is
  byte-identical.
- Unpriced model under a cost quota: fail closed at build and at runtime, rather than exempting
  unpriced models from cost quotas.
- Expired lease: charge the ceiling on expiry, a fail-closed choice, rather than releasing to zero,
  which would leak served spend.
- Wired seams: query, stream, RLM run, ingest, the served program routes (run, stream, deployment run,
  evaluate), the skill funnel, the optimization enqueue route, and the fine-tune request route. Every
  served, user-reachable expensive path is metered; only the internal worker executions the enqueues
  schedule stay uncharged.

## The cache is not partitioned by quota

A quota is an admission gate, not an input that shapes an answer. An admitted request produces the
same answer whether or not a quota governs it, and a denied request never reaches the cache. The
quota therefore does not enter the answer-cache identity, which keeps the cache from fragmenting and
keeps the runtime byte-identical when no quota is configured.
