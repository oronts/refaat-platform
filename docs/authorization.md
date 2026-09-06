# Authorization

Refaat enforces per-principal access control over evidence: retrieval, graph facts, RLM recursion, citations, the answer cache, session memory, learned preferences, and every program and HTTP surface. It is the enforcement layer, not the identity provider. A deployment brings its own identity and its own policy; Refaat owns the seams that apply the decision consistently and fail closed.

Two principles govern the whole design:

- **Byte-identical when unconfigured.** With no access policy installed, the runtime behaves exactly as it did before authorization existed. Artifact ids, the answer-cache identity, and every pre-existing result are unchanged.
- **Fail closed when active.** Once a policy is installed, any ambiguity denies. An untagged or malformed evidence unit is not served, a model never influences a decision, a missing identity at the HTTP boundary is refused, and under the default strict authorization an in-process call that omits the principal is denied rather than served unfiltered.

## Activation

Authorization is a first-class production feature, not an experimental flag. It activates when an `AccessPolicy` is installed through `RuntimeBuilder.with_access_policy(...)`. A request that carries a `Principal` is scoped to that principal. A request that omits the principal is denied by default under strict authorization; set `strict_authorization` to `false` to serve it unfiltered as trusted in-process code (see below).

The in-process Python API passes a principal directly. The HTTP surface resolves the principal server-side through a `PrincipalResolver` and never reads identity from the request body; with a policy installed, a request that resolves to no identity is refused at the boundary. See [Identity lifecycle](identity-lifecycle.md).

```python
from refaat.application.runtime import RuntimeBuilder
from refaat.config import Settings
from refaat.domain.authz import CanonicalAccessPolicy, Principal

runtime = RuntimeBuilder(Settings()).with_access_policy(CanonicalAccessPolicy()).build()
project = runtime.project(workspace="acme", corpus="handbook")

alice = Principal(id="alice", tenant="acme", groups=frozenset({"research"}))
answer = project.ask("What is the rollback policy?", principal=alice)
```

Without `with_access_policy` the same call is byte-identical to an unauthorized runtime. With a policy installed and `principal=None`, strict authorization (on by default) denies the call; see below.

### Strict authorization

By default (`strict_authorization` is `true`) an in-process call that omits the principal under an active policy is denied with an `AuthorizationError` rather than served unfiltered, across every in-process surface that serves evidence or its metadata or reaches a model: `ask`, `stream`, `retrieve`, `rlm`, a program run, the `list_documents` and `stats` inventory reads, and `submit_feedback` (which reads the selected chunks). The denial runs first on each path, before the answer cache is read, before the embedder or a model is called, and before any quota is charged, so a strict-denied call neither returns a cached unfiltered answer nor reaches an external model. The HTTP boundary already fails closed through its identity dependency and resolves the principal server-side, so this setting only tightens the in-process SDK and CLI path. Set `strict_authorization` to `false` to restore the trusted-in-process behavior, where such a call runs unfiltered because the caller is trusted application code; that path is byte-identical to before, and the no-policy path is unaffected either way.

## Vocabulary

The canonical value objects live in `refaat.domain.authz`. They are pure, frozen, and framework-neutral.

### Principal

An already-authenticated identity, resolved from trusted context. Refaat never authenticates it and never constructs it from a client payload.

| Field | Meaning |
|---|---|
| `id` | Stable identity id |
| `kind` | `HUMAN`, `SERVICE`, or `SYSTEM` |
| `tenant` | The tenant boundary, or `None` |
| `groups` | Policy-relevant group entitlements |
| `roles` | Policy-relevant role entitlements |
| `attributes` | Extra typed attributes for a custom policy |

Only `groups`, `roles`, and `tenant` feed the cache scope and the audit trail. Arbitrary identity claims such as an email never enter a cache key, so the key stays stable and minimal.

### AccessTag and the `acl` block

Every evidence unit carries one canonical access tag, derived at ingestion from an `acl` block in the document metadata. The permission then travels source to document to chunk to graph fact to retrieval to citation.

```python
project.runtime.ingest(
    workspace_id,
    corpus_id,
    DocumentInput(
        source_id="q3-forecast.md",
        title="Q3 forecast",
        media_type="text/markdown",
        content=content_bytes,
        metadata={
            "acl": {
                "classification": "confidential",
                "tenant": "acme",
                "allow_groups": ["finance"],
                "deny_groups": ["contractors"],
            }
        },
    ),
)
```

Fields on the tag: `classification`, `tenant`, `allow_groups`, `allow_roles`, `deny_groups`, `deny_users`.

### Classification

Ordered sensitivity, higher is more sensitive: `PUBLIC` (0), `INTERNAL` (1), `CONFIDENTIAL` (2), `RESTRICTED` (3). Classification is distinct from source trust: a document can be `PUBLIC` and `AUTHORITATIVE`, or `CONFIDENTIAL` and `UNVERIFIED`, so the two are never overloaded. Classification also drives deterministic model routing (see [Data-classification model routing](model-routing.md)).

## The Policy Decision Point

The `AccessPolicy` protocol has one method, `decide(principal, tag) -> AccessDecision`. The shipped `CanonicalAccessPolicy` applies these rules in order, and any ambiguity denies:

1. **Tenant mismatch denies.** A tag with a tenant different from the principal's tenant is refused. This is also enforced as a hard pre-policy invariant (below), so it holds even under a custom policy.
2. **Explicit deny beats any allow.** A match in `deny_users` or `deny_groups` denies.
3. **Allow list, when present, is required.** If the tag has an `allow_groups` or `allow_roles` list, the principal is allowed only when it satisfies one of them.
4. **No allow list:** `PUBLIC` is visible within the workspace; `INTERNAL` fails closed because its organizational meaning belongs to the customer policy; `CONFIDENTIAL` and `RESTRICTED` fail closed as sensitive without a grant.

The decision is pure and stateless, so it is reproducible and auditable. The `AccessDecision` records a machine-readable `reason`, `policy_id`, and `policy_version` internally; sensitive ACL internals are never surfaced to the caller.

### Bring your own policy

`CanonicalAccessPolicy` is the default, not a lock-in. A deployment can install a stricter policy, or route the decision to an external engine such as OPA or Cedar, behind the same `decide` contract, with no engine entering Refaat core:

```python
class OpaBridgePolicy:
    name = "opa"
    version = "2025-08"

    def decide(self, principal, tag):
        allowed = opa_client.check(principal, tag)  # your call
        return AccessDecision(allowed=allowed, reason="opa", policy_id=self.name, policy_version=self.version)


runtime = RuntimeBuilder(settings).with_access_policy(OpaBridgePolicy()).build()
```

The tenant and workspace boundary is enforced by the framework before `decide` runs, so even a tenant-blind custom policy cannot leak across tenants.

## The Policy Enforcement Points

The decision is one function; the enforcement is many seams that all call it. Every path that touches evidence or an identity-scoped store is covered:

| Surface | Enforcement |
|---|---|
| Retrieval | A central post-fusion filter drops denied chunks over the whole fused pool, so semantic, lexical, HyPE, and RAPTOR routes are all covered, plus a pre-fusion filter. Denied evidence never influences fusion, ranking, generation, or a citation. |
| Graph | `graph_context` removes entities from unauthorized documents and any relation touching a dropped entity before boosts and facts are computed, so the graph is never a side channel. |
| RLM | The corpus is filtered to authorized chunks before any recursion, so an unauthorized chunk never enters a sub-call, a partition, or synthesis. |
| Citations | A defense-in-depth filter re-authorizes every cited chunk, so a citation can only resolve to authorized evidence. |
| Answer cache | The cache scope folds an `auth_scope_hash` over the principal's normalized entitlements plus the policy identity, and a cache hit re-authorizes every cited chunk against the current ACL. |
| Session memory | A compacted summary and each turn are bound to the writer's auth scope; a summary is surfaced only to a reader with the same scope, at every read site. |
| Learning | The learned-preference load and write are principal-bound, and a write is refused for a chunk the principal cannot see. See [Learning and preferences](learning-and-preferences.md). |
| Programs | The principal threads end to end through typed programs, governed ReAct, program-as-skill, and the deployment router, so a program run as one principal never returns another principal's evidence. |
| HTTP | Identity is resolved server-side; read and management endpoints filter by the principal or fail closed. |
| Async jobs and worker | An evidence-serving durable job (evaluation, program-evaluate, RLM, optimization, shadow-compare) fails closed at worker dispatch under an active policy rather than running unfiltered. |

## Fail-closed details

- **Untagged evidence.** An evidence unit with no `acl` block is treated as `INTERNAL`, which the canonical policy denies. Enabling authorization on an untagged corpus is fail closed until the evidence is tagged. A deployment may override the default tag.
- **Malformed ACL.** A present but malformed or unknown-classification `acl` block parses to no tag and is denied, so a broken ACL never silently becomes public.
- **Tenant and workspace as a hard invariant.** Cross-workspace queries fail through workspace-scoped storage. Cross-tenant access is refused by a framework pre-policy guard that runs before `decide` on every evidence path, so a custom tenant-blind policy still cannot leak across tenants.
- **Over-classified evidence.** Deterministic model routing refuses to send evidence above a configured classification ceiling to a model, without revealing the classification level to the caller.

## Cache correctness under changing permissions

The answer cache is version-safe against permission change, independent of content:

- **`auth_scope_hash`** over the principal's entitlements plus the policy id and version folds into the cache identity, so two principals never share a cached answer and losing an entitlement misses old entries.
- **`permission_fingerprint`** makes an ACL-only re-ingest advance the corpus version even when the content bytes are identical, so a changed ACL re-tags chunks and invalidates stale answers.
- **Serve-time re-authorization** re-checks every cited chunk on a cache hit, so a hit whose evidence is now forbidden is treated as a miss and regenerated against the current ACLs.

## What Refaat does not do

- It does not authenticate. A deployment authenticates upstream and adapts its identity provider to the `PrincipalResolver` port.
- It does not own an identity store, a connector permission model, or a governance engine. These plug in.
- A model never decides a permission. Every value that gates access is deterministic and set outside the model.

## How the guarantee is enforced

The authorization invariant is simple and general: every path that runs retrieval or reads an identity-scoped store, and every fetch-by-id that substitutes evidence text, binds to the principal or fails closed under an active policy, and a signal that only reveals the existence of forbidden evidence is itself a leak. Enforcement funnels through a single `chunk_allowed` chokepoint in `domain/authz.py`, so every seam in the table above applies the same decision rather than reimplementing it.

This is an adversarially verified property backed by a regression suite, not a proof of impossibility: the review is thorough but finite, and the suites lock every closed vector so it cannot regress. They live in `tests/integration/test_authz_enforcement_e2e.py`, `tests/integration/test_authz_bypass_matrix_e2e.py`, `tests/integration/test_authz_program_run.py`, `tests/integration/test_audit_events_e2e.py`, `tests/integration/test_model_routing_e2e.py`, and `tests/e2e/test_authz_http_identity.py`.

## Related pages

- [Access and multi-tenancy](access-and-multitenancy.md)
- [Tool governance](tool-governance.md)
- [Approvals](approvals.md)
- [Identity lifecycle](identity-lifecycle.md)
- [Audit and provenance](audit-and-provenance.md)
- [Data-classification model routing](model-routing.md)
- [Learning and preferences](learning-and-preferences.md)
- [Enterprise foundation](enterprise-foundation.md)
