# Execution resilience

Refaat can bound how long a single governed tool call is allowed to run. The bound is opt-in and off
by default, so a runtime that configures nothing runs a tool inline exactly as before.

## Per-tool execution timeout

Set `REFAAT_TOOL_TIMEOUT_SECONDS` to a positive number to give every governed tool call a caller
deadline. With it unset (the default) the governed loop runs a tool inline and the runtime is
byte-identical to before the feature existed.

A plain skill's `run` is arbitrary blocking code. Refaat can stop waiting for it at the deadline, but
it cannot terminate it: Python has no safe way to kill a running thread, and the skill's I/O may
already have reached an external system. So the deadline reports one of three outcomes and never
conflates them.

| Outcome | When | What Refaat does |
| --- | --- | --- |
| Confirmed no effect | The bounded worker pool was full, so the skill provably never started | Records a denied step with reason `tool_timeout:confirmed_never_started`; the tool never ran, so a caller is free to retry |
| Unknown, read tool | The skill started and was abandoned at the deadline, and its declared side effect is read-only | Records a denied step with reason `tool_timeout:unknown_abandoned` and continues to the grounded answer from the evidence already gathered |
| Unknown, write-capable tool | The skill started and was abandoned at the deadline, and its capability declares a write, sensitive-write, irreversible, or external side effect | Raises `ToolTimeoutError` (HTTP 409) rather than continue, so no caller reads the timeout as a safe nothing-happened retry |

The distinction that matters for data integrity is confirmed versus unknown. A confirmed no-effect is
safe to retry. An unknown outcome on a write-capable tool means the remote action may have been
accepted, so the typed error is a 409 (a conflict that needs reconciliation), never a 504 (which a
generic mesh or SDK would auto-retry). The HTTP body carries first-class fields so a client cannot
mistake it for a gateway timeout: `outcome`, `side_effect`, `execution_terminated` (always false, since
the work was not terminated), and `retryable` (false).

### Honest boundaries

- The abandoned worker keeps running until its own `run` returns. A permanently wedged worker holds its
  slot until the process ends; the bound on concurrent abandoned workers is
  `REFAAT_TOOL_TIMEOUT_MAX_WORKERS` (default 8). Once the pool is full, further tool calls report the
  confirmed no-effect outcome rather than starting.
- A secret-bearing tool's secret bundle is scrubbed exactly once, and only after its `run` returns, so
  an abandoned worker's still-live bundle is never scrubbed out from under it. A wedged worker retains
  its bundle until it self-terminates.
- The worker pool is an in-process bound, the same in-process boundary the edge rate limiter documents.
  A cross-process bound is a follow-up.
- An abandoned worker keeps charging the run meter after the deadline, so its spend after abandonment is
  not reconciled into the per-run quota. The quota is a boundary check, and an abandoned worker is by
  definition outside the served run's accounting; the charge itself stays atomic under the meter lock.
- The streaming endpoint fails a write timeout closed (it raises, and the SSE error event is emitted with
  no answer served) but the SSE error body does not yet carry the structured `outcome`, `side_effect`,
  `execution_terminated`, and `retryable` fields the non-streaming 409 does. Surfacing them on the SSE
  error is a follow-up; the non-streaming path carries the full contract today.

## Per-provider circuit breaker

Set `REFAAT_CIRCUIT_BREAKER_BACKEND=in_memory` to fast-fail a wedged provider endpoint instead of
letting every request wait out its own timeout. With the default `none` no breaker is installed, no
adapter is wrapped, and the runtime is byte-identical. The deterministic offline stack is never wrapped
even when a breaker is configured, so the offline behavior is unchanged.

The breaker keys on a provider endpoint, the normalized base URL authority plus its path prefix, so an
embedder and a generator on the same endpoint share one breaker while a path-multiplexed gateway keeps
independent breakers per path. It sits innermost, before any caching wrapper, and is transparent to the
cache identity, so turning it on never changes a cached answer or a model-call cache key.

The state machine has three states:

- **Closed.** Calls flow. `REFAAT_CIRCUIT_BREAKER_FAILURE_THRESHOLD` consecutive contributing failures
  open the circuit; any success resets the streak.
- **Open.** Every call to that endpoint fast-fails with a 503 `provider_circuit_open` (a
  `DependencyUnavailableError`, so it flows through the same degradation path a provider outage does)
  without touching the provider, until `REFAAT_CIRCUIT_BREAKER_RESET_TIMEOUT_SECONDS` elapses.
- **Half-open.** After the cooldown, exactly one probe is admitted; concurrent callers still fast-fail.
  A clean probe closes the circuit, and a failed or inconclusive probe re-opens it with a fresh cooldown.
  A probe that never reports (a lost worker) self-heals after
  `REFAAT_CIRCUIT_BREAKER_HALF_OPEN_DEADLINE_SECONDS`, so a lost record can never wedge the endpoint.

Only a genuine unreachable-or-failing endpoint contributes to opening: a transport error (connect,
read, write, timeout, or a remote protocol error) or a 5xx status. A client-side fault (a 4xx status, a
local protocol error, an unsupported scheme, or a response that did not parse) is inconclusive and never
opens a closed circuit. A 429 does not contribute unless `REFAAT_CIRCUIT_BREAKER_COUNT_429=true`.

### Honest boundaries

- The breaker covers the generate and embed calls, which transitively cover HyPE, RAPTOR, the judge, and
  the streamed answer. The graph extractor, the fine-tuning provider, the vision captioner, and the
  tool-call planning stream are not wrapped in this slice; each still bounds on the per-call request
  timeout, so an unwrapped path fails closed on a wedged provider, just without the fast-fail.
- Breaker state is per process, the same in-process boundary the edge rate limiter documents. A shadow
  or canary build never draws down or trips the production breaker (an injected breaker is refused in the
  reproduced composition). A cross-process breaker is a follow-up.
