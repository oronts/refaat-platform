# Incident kill switches

A kill switch lets an operator immediately deny a class of operations during an incident without a
redeploy or restart. Switches are stored in the database and read at request time behind a short cache,
so engaging one reaches every worker within one cache interval. The feature is off by default and the
runtime is byte-identical until an operator enables it and engages a switch.

## Enabling it

Set `REFAAT_KILL_SWITCH_BACKEND=database` to install the store-backed registry, or inject a custom
registry with `RuntimeBuilder.with_kill_switch_registry`. With the default `none` nothing is installed,
no table is read, and every enforcement point short-circuits.

Two settings tune the read:

- `REFAAT_KILL_SWITCH_CACHE_TTL_SECONDS` (default `2.0`): how long a worker serves a cached snapshot
  before it re-reads. Engaging or clearing a switch takes effect within one interval. Lower it to trade
  more reads for tighter propagation.
- `REFAAT_KILL_SWITCH_STALE_DENY_SECONDS` (default `0.0`): how long to serve the last known good
  snapshot after a read fails. The default `0.0` is strict fail-closed: any read failure past the TTL
  denies every guarded operation. A positive value is an explicit decision to relax that default and
  ride out a brief database outage before denying.

## Scopes

A switch names a scope and a workspace. An empty workspace is a global switch that applies to every
workspace. The scopes are the workspace freeze, external egress, tool execution, writes, and generation.

This build enforces four scopes:

- **workspace freeze**: every serving and mutation path for the workspace is denied, across the HTTP
  surface, the in-process library, and the durable worker.
- **generation**: the whole generated-answer path is denied, the query, the streaming query, and the
  RLM, a cached answer included, because an operator engaging a generation switch during an incident
  means to stop serving generated answers. Retrieval alone, which returns raw evidence and generates
  nothing, still serves, so an operator can keep inspecting evidence while generation is dark.
- **writes**: every corpus and learned-state mutation is denied, ingest (a connector-sourced ingest and
  an image ingest included), delete, erase, feedback, and writing or activating a harness snapshot (the
  active harness is generation-config that feeds every answer). The query and retrieval read paths are
  untouched, so a workspace under a writes switch is read-only rather than fully frozen.
- **tool execution**: governed tool execution is denied. Inside a governed ReAct run each proposed tool
  is recorded as a denied step and the agent still answers from evidence without it, since generation is
  a separate scope; on the direct skill-run path, which has nothing to continue to, the call is refused.
  Retrieval and generation are untouched. The opt-in RLM sandboxed-code phase is part of the RLM
  generation operation, so it is stopped by the generation or workspace-freeze scope rather than this one.

External egress ships in a later slice; the management surface refuses to engage a scope whose
enforcement is not yet wired, so a switch is never recorded without a seam that honors it.

## Enforcement and fail-closed behavior

A workspace freeze denies every operation in the workspace's AI and data plane: serving query evidence,
running model inference or training, or mutating the evidence corpus or its derived and learned state.
That covers query, retrieval, the RLM, skill runs, evaluation runs, fine-tune runs, ingest, delete,
erase, feedback, and harness configuration. It is enforced at three chokepoints, so it is not an
HTTP-only control:

- the HTTP edge, where every `/v1` route resolves an api key through one dependency;
- the in-process runtime, at the entry of each evidence-serving and mutation service (query, retrieval,
  RLM, skill run, ingest, delete, erase), so a program run, the SDK, or a direct service call is frozen
  too, not only a call routed through the project wrapper;
- the durable worker, before it dispatches any claimed job. A frozen job is returned to the queue
  without counting the attempt, so a freeze does not exhaust a job's retries during the incident.

Queuing a durable job is not blocked, since a queued job cannot execute while the freeze holds: the
worker refuses it at dispatch. So a freeze stops all execution for the workspace, and any work queued
during the incident runs only after the freeze lifts.

Read-only in-process diagnostics, listing documents and corpus statistics, are intentionally allowed so
an operator can inspect a frozen workspace during an incident. They mutate nothing and serve no query
evidence, and their HTTP forms are frozen at the edge, so a remote client cannot reach them.

The governance and administration control plane stays available during a freeze so an operator keeps the
tools to respond to the incident: recording or revoking tool approvals, placing or lifting legal holds,
promoting or rolling back a program deployment, and managing evaluation suites, fine-tune metadata, api
keys, and agents. None of these serves query evidence or runs the model, and handcuffing them would
handcuff incident response. The one governance-adjacent operation that does run external compute, a
fine-tune run, is in the data plane and is frozen; recording and approving a fine-tune stays available.

The freeze is enforced at the public API surfaces: the HTTP edge, the durable worker, the runtime and
project methods, the data-plane service entries, and the program run surface (running a program through
`project.programs.get(name).run()`). Reaching past those into a raw low-level in-process object, calling
a store method directly on `runtime.store`, or constructing a program object and invoking its
`execute()` directly instead of through the program run surface, is trusted in-process code outside the
freeze boundary, the same category as a malicious installed plugin in the threat model. It is not
reachable through the public API or over HTTP.

Fail-closed holds throughout. The registry primes once at startup, so a runtime never serves without a
known baseline and a failed prime fails startup. A read failure past the TTL denies every scope and
workspace unless a stale-serve window is configured. An unreadable snapshot is treated as frozen.

## Managing switches

Engaging and clearing a switch are admin-only and audited. A workspace admin engages only for its own
workspace over HTTP. A global switch and lifting a workspace freeze are operator only, through the CLI
or the database, because a workspace freeze darkens its own HTTP surface (the freeze is checked before
the request is authorized), so a workspace credential cannot lift the freeze that locked it out.

HTTP:

- `POST /v1/kill-switches` with `{ "scope": "workspace_freeze" }` engages a switch for the caller's
  workspace and returns its id.
- `GET /v1/kill-switches` lists the workspace's active switches.
- `POST /v1/kill-switches/{scope}/disengage` clears one (unusable for a workspace freeze, which darkens
  this route; use the CLI).

CLI (the operator surface):

```
refaat kill-switch engage workspace_freeze <actor> [--workspace <ws>] [--reason <text>]
refaat kill-switch disengage workspace_freeze <actor> [--workspace <ws>]
refaat kill-switch list [--workspace <ws>]
```

An omitted workspace is the global sentinel. The reason is stored on the row for the incident record
and never enters an audit event.

## Audit

Engaging emits a redacted `killswitch.engaged` event and clearing emits `killswitch.disengaged`. Each
event carries ids, the scope as a fixed value, the actor, and timestamps, never the human reason.
Engaging is protection-increasing, so its audit is best-effort and the switch lands even under a
degraded sink; clearing is protection-decreasing, so under `audit_fail_closed` a sink outage keeps the
switch on rather than lifting it unaudited.
