# Admin status plane

The admin status plane is a read-only, redacted view of the current posture and health of every
governance and operational subsystem, so an operator can see what is engaged, active, or healthy
right now without querying each subsystem by hand. It is purely additive: it opens no write, mutates
no state, and folds into no artifact id or cache identity, so it is byte-identical to the rest of
the runtime.

This surface is experimental until it has been through an adversarial review.

## What it aggregates

One entry per subsystem, each a `SubsystemStatus` carrying a fixed name, a health `state`, a
`configured` boolean, and a small allowlisted `detail`:

- Kill switches: whether the workspace is frozen, and per-scope deny booleans.
- Quotas: the rule count and the workspace-scope usage counters.
- Audit: durable and fail-closed flags, and the outbox backlog depth, oldest undelivered age, and
  stuck count under durable audit.
- Legal holds: the active hold count and whether management is enabled.
- Approvals: the active grant count and open request count.
- Jobs: pending, running, stuck-lease, and failed counts.
- Connectors, degradation, model routing, output governance, egress transform, source trust,
  identity and authorization, plugin pins, and secrets: presence plus a coarse posture.

Alongside the subsystems it reports build info (version, environment, and a fixed installed-feature
map) and schema info (the applied migration revision, the code head, and whether the database is at
head or was auto-created). A database behind the code head reports schema `state` as `active` with
`at_head` false, so schema health is read from the `at_head` boolean, not from `state` alone.

Retention policy and key-rotation state are named in the enterprise intent but have no first-class
governed object to read today, so they are out of scope here. Adding a reader for either later is
additive and does not disturb this surface.

## States

- `not_configured`: the subsystem seam is absent or the feature is off. This is the byte-identical
  default, never an error.
- `active`: configured and healthy, or present with no liveness check needed.
- `engaged`: a live kill-switch freeze applies.
- `degraded`: configured but a bounded, caught fault (the database was unreachable, or a backlog is
  stale). A present-but-unreachable subsystem reports `degraded`, never a false `active`.
- `unknown`: the accessor itself raised. The entry carries an empty detail and the rest of the
  plane still reports.
- `unmanaged`: the schema was auto-created and carries no migration revision row.

## Guarantees

Degrade-to-unknown, never raise. Every accessor is wrapped, so one unreadable subsystem reports
`unknown` while the rest report normally and the plane still returns. Build and schema readers catch
internally and report `unknown` on failure. The plane stays available during an incident.

Redaction by projection. Each accessor emits only counts, booleans, enum values, cache identifiers,
timestamps, and schema revisions, sourced from count and boolean readers, never a secret, evidence
text, a raw principal, a hold or switch reason, an actor id, or a raw quota scope key. The
scalar-only check on `SubsystemStatus` is defense in depth, not the guarantee.

The cache identifiers a status entry may carry are public, content-addressed identifiers already
folded into the answer cache key, never a secret. A third-party model router, output governor, or
context transform must keep its `cache_id` a non-secret public identifier per that contract. Those
identifiers appear only in the operator view, never on the HTTP surface.

## Configured versus healthy

`configured` reports presence, derived from the in-memory seam object, which never raises. `state`
reports the health verdict. A subsystem can be `configured` and `degraded` at the same time when its
seam is present but a live count or liveness probe fails. Presence is never rendered as a health
verdict.

## Access model

The HTTP route `GET /v1/status` requires an admin-scoped key. The workspace is derived server-side
from the key, never from a body, query, or header field, and the HTTP view is workspace-scoped: a
workspace admin sees only its own state and a set of coarse governance booleans, never the global
kill-switch state or the process-wide governance inventory.

`GET /v1/status` funnels through the same principal dependency as every other `/v1` route, so a
frozen workspace's status darkens at the edge (503) exactly like the rest of its surface. The
during-freeze surface is the operator CLI, which reads the runtime in-process and is freeze-exempt.

## CLI

```
refaat status                       # operator view: process-wide inventory and global kill-switch state
refaat status --workspace-id <id>   # a single workspace's posture
```

The CLI is the operator surface. An empty workspace or `--operator` reads the full governance
inventory and the global kill-switch state. Output is JSON.

## Relationship to health checks

`GET /v1/status` is admin posture. `GET /health/ready` is an unauthenticated liveness probe. The
plane infers reachability per subsystem from its own read: a count reader that raises degrades that
subsystem rather than the whole plane. It is not a substitute for the health endpoint.
