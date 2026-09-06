# Access and multi-tenancy

Refaat isolates evidence along three boundaries that stack: workspace, tenant, and the per-evidence ACL. The first two are hard invariants enforced before any pluggable policy runs; the third is the policy decision. This page covers how to model tenants and how the boundaries compose. For the decision rules and enforcement seams, see [Authorization](authorization.md).

## The three boundaries

1. **Workspace.** Every stored row is workspace-scoped at the storage layer. A query for workspace A can never read workspace B, regardless of policy or principal. A not-found is uniform, so a missing resource in another workspace is indistinguishable from one that does not exist.
2. **Tenant.** A principal carries an optional `tenant`, and an evidence unit's `acl` can carry a `tenant`. A tag whose tenant differs from the principal's tenant is refused by a framework pre-policy guard that runs before the policy's `decide`. This holds on every evidence path (retrieval including HyPE and RAPTOR, graph, RLM, citations, and the cache) and even under a custom tenant-blind policy, so cross-tenant isolation is a property of the framework, not of any one policy.
3. **ACL.** Within a tenant and workspace, the `AccessPolicy` decides visibility per principal from the evidence's classification, allow lists, and deny lists.

## Modeling tenants

There are two common shapes. Pick per deployment.

### One workspace per tenant

The strongest isolation. Each tenant gets its own workspace, so tenant data never shares storage scope. Use this when tenants are separate customers and you want no shared rows.

```python
acme = runtime.project(workspace="tenant-acme", corpus="handbook")
globex = runtime.project(workspace="tenant-globex", corpus="handbook")
```

### Shared workspace, tenant on the principal and the ACL

A single workspace holds many tenants, and the `tenant` field on the principal and on each evidence `acl` enforces the boundary. Use this for a shared knowledge base with per-tenant slices.

```python
project.runtime.ingest(
    workspace_id,
    corpus_id,
    DocumentInput(
        source_id="acme-contract.md",
        title="Acme contract",
        media_type="text/markdown",
        content=content_bytes,
        metadata={"acl": {"classification": "confidential", "tenant": "acme", "allow_groups": ["legal"]}},
    ),
)

alice = Principal(id="alice", tenant="acme", groups=frozenset({"legal"}))
bob = Principal(id="bob", tenant="globex", groups=frozenset({"legal"}))
# alice sees the Acme contract; bob (tenant globex) is refused before the policy even runs.
```

## Roles and groups

`allow_groups` and `allow_roles` grant, `deny_groups` and `deny_users` deny, and a deny always beats an allow. Groups and roles are two independent axes so you can model both team membership and job function. An unknown group or role never broadens access: with an allow list present, the principal must satisfy it, and with no allow list, anything above `PUBLIC` fails closed.

## Classification levels

`PUBLIC`, `INTERNAL`, `CONFIDENTIAL`, `RESTRICTED`, ordered by sensitivity. `PUBLIC` is visible within the workspace with no allow list. `INTERNAL` with no allow list fails closed under the canonical policy, because its organizational meaning belongs to your policy; grant it explicitly with an allow list or a custom policy. `CONFIDENTIAL` and `RESTRICTED` always require a grant.

## Composition

The boundaries are checked in a fixed order, so the result is predictable: workspace scope at the query, then the tenant pre-policy guard, then the deny lists, then the allow lists, then the classification default. A request that fails any earlier check never reaches a later one, and every failure denies.

## Related pages

- [Authorization](authorization.md)
- [Identity lifecycle](identity-lifecycle.md)
- [Audit and provenance](audit-and-provenance.md)
