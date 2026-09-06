# Durable agents and long-running work

Refaat keeps long work in durable records rather than an in-memory agent loop.

## Jobs

A job records its workspace, kind, payload, state, run time, attempt count, maximum attempts, lease owner, lease expiry,
heartbeat, result, and last error. A worker atomically claims one eligible job. A heartbeat extends the lease while a
synchronous handler runs. A retryable failure releases the lease and applies bounded exponential backoff. Exhausted
or unsupported jobs enter a terminal failed state instead of starving valid work.

Built-in handlers are:

- `evaluation.run`;
- `program.evaluate`;
- `session.compact`;
- `outbox.publish`;
- `audit.publish`;
- `cache.purge`;
- `deferred_erasure.sweep`;
- `shadow_runs.purge`;
- `quota.reap`;
- `blob.gc`;
- `rlm.run`;
- `optimization.run`;
- `program.shadow_compare`;
- `fine_tune.run`.

Applications can register additional typed handlers programmatically. Job handlers must be idempotent because lease
recovery and at-least-once delivery can repeat work after a process failure.

## Schedules

Schedules persist a job kind, payload, interval, next run, and enabled flag. Workers dispatch due schedules before
claiming work. This release intentionally implements interval schedules only; cron and calendar semantics belong in a
plugin or external scheduler with explicit timezone behavior.

## Agents and direct communication

Retained agents have durable goal and state records, optional parent IDs, and heartbeat timestamps. An agent can send
a typed message directly to another agent mailbox without routing content through a user. Delivery is persistent and
an explicit acknowledgement changes message state.

Refaat supplies coordination primitives, not unrestricted authority. The embedding application defines wake-up policy,
models, tools, spending limits, approval gates, cancellation, and escalation.

## Sessions and compaction

Goals and turns survive API, worker, and terminal restarts. Once the configured threshold is reached, older turns are
compacted into a bounded summary while recent turns remain verbatim. The built-in compactor is deterministic and
conservative; a model-backed implementation can replace the same port when covered by representative evaluations.

## Worker lifecycle

The worker handles `SIGINT` and `SIGTERM`, stops claiming work, and closes owned model, HTTP, store, and telemetry
resources. Each production worker must use a unique `REFAAT_WORKER_ID` and share PostgreSQL with the API processes.
