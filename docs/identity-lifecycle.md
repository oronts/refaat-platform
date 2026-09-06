# Identity lifecycle

Refaat does not authenticate. It consumes an already-authenticated identity and turns it into a canonical `Principal` through a pluggable resolver. This page covers where identity comes from, how the HTTP boundary stays trustworthy, and how a principal reaches every enforcement point. For the decision rules, see [Authorization](authorization.md).

## The resolver port

`PrincipalResolver` has one method:

```python
class PrincipalResolver(Protocol):
    def resolve(self, context: Mapping[str, Any]) -> Principal | None: ...
```

A deployment adapts its identity provider to this port. The context is trusted, server-side data assembled by Refaat, never a client request body. The resolver returns `None` when no identity is present; the fail-closed decision under an active policy is made at the HTTP boundary.

Install it on the builder:

```python
runtime = (
    RuntimeBuilder(settings)
    .with_access_policy(CanonicalAccessPolicy())
    .with_principal_resolver(HeaderPrincipalResolver())
    .build()
)
```

## The HTTP identity boundary

Over HTTP, identity comes only from the resolver, driven by the authenticated api-key context plus request headers. It is never read from the request body, and the OpenAPI document declares no principal field, so a client cannot widen its own access by sending groups, roles, a tenant, or a kind.

The reference `HeaderPrincipalResolver` reads a JSON claims object from a configured trusted header, `x-refaat-principal` by default:

```
x-refaat-principal: {"id":"alice","tenant":"acme","groups":["research"]}
```

A hardened deployment sets this header at a trusted upstream auth proxy after it authenticates the caller, and strips any client-supplied copy at the edge. The header is server-side by the time Refaat reads it.

Fail-closed at the boundary: when an access policy is installed and no identity resolves, the request is denied rather than served unfiltered. A resolver is how identity is supplied, so a deployment with a policy but no resolver denies every request at the boundary. This closes the case where a policy is installed but a request arrives with no usable identity.

## Building a principal from claims

`principal_from_claims(claims)` builds a `Principal` from a trusted claims mapping and is fail-safe: it returns `None` when the claims name no identity, defaults an unknown `kind` to `HUMAN`, and reads only `id`, `kind`, `tenant`, `groups`, and `roles`. Use it inside a custom resolver that already holds verified claims from your identity provider.

## Service and system identities

A principal is not always a person. `PrincipalKind` is `HUMAN`, `SERVICE`, or `SYSTEM`, so a background job or a machine client is a first-class principal with its own entitlements. The audit trail records the principal kind, so a service action is distinguishable from a human one.

## Threading to enforcement

Once resolved, the principal flows into every evidence path: `project.ask`, `project.retrieve`, and `project.rlm` accept it directly, and the HTTP evidence routes pass the resolved identity into the same request. From there it reaches the retrieval filter, the graph and RLM scoping, citation resolution, the cache scope, session memory, learning, and every program and deployment run. A run with no principal under an active policy is denied by default under strict authorization; with no policy installed it is byte-identical to an unauthorized run.

## Durable jobs and execution-time re-resolution

A durable `rlm.run` job runs later, on a worker, with no live request to resolve identity from. Under an active policy the worker fails every evidence-serving durable job closed, because it carries no principal. The one exception is `rlm.run`, and only when a `PrincipalDirectory` is installed.

`PrincipalDirectory` re-resolves a principal by stable id, the execution-time counterpart to the request-time `PrincipalResolver`:

```python
class PrincipalDirectory(Protocol):
    def refresh(self, principal_id: str) -> Principal | None: ...
```

The durable path is opt-in and needs both ports. The resolver captures the identity when the job is enqueued: `enqueue_rlm` records a redacted snapshot of the resolved principal on the job, the principal id, tenant, deciding policy, a timestamp, and an injective entitlement digest, never a full principal or a secret. The directory re-resolves that principal at execution. Both are required. With a policy but no directory the enqueue endpoint stays fail-closed with a 403, and a directory with no resolver also yields a 403 at enqueue, because no identity resolves to capture. The worker gate keys on the directory.

The snapshot is written server-side only, by `enqueue_rlm`. The generic `POST /v1/jobs` and `POST /v1/schedules` surfaces stay denied under a policy, so a client cannot enqueue an `rlm.run` job carrying a forged snapshot.

At execution the worker re-resolves the snapshot principal and admits the run only when the identity comes back fresh: the same id, the same tenant, an unchanged entitlement digest, and a snapshot no older than the risk class allows. Only then does the job run, under the freshly re-resolved principal, which scopes its evidence exactly as a live request would. Every staleness or mismatch is a terminal denial: a revoked identity, a changed tenant, a changed scope, a missing snapshot, or a snapshot past its freshness window all fail the job with no retry. A transient directory outage is the one retryable case, so a recovered directory drains the job on a later attempt. The gate audits both outcomes, `identity.reresolved` on the allow path and `identity.refused` on a denial, each redacted to ids, the policy, and a timestamp. The allow-path audit is fail-closed under `audit_fail_closed`, so an authorized durable run never serves unaudited.

Operational note: an `rlm.run` job re-resolves under the READ risk class, whose default freshness window is 86400 seconds. A durable job left undrained past that window re-resolves as stale and terminally fails, so operators must drain the durable RLM queue within the window. A configurable freshness policy is a later refinement.

## Related pages

- [Authorization](authorization.md)
- [Access and multi-tenancy](access-and-multitenancy.md)
- [Audit and provenance](audit-and-provenance.md)
