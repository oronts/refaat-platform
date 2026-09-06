# Program registry, promotion, and rollback

The registry is a durable, workspace-scoped store of immutable program artifacts and environment aliases. Production activation is an explicit promotion: it points an environment alias at a saved artifact and never mutates the artifact. Rollback is atomic and moves the alias back to the previously deployed artifact. Every promotion and rollback is recorded in an append-only deployment history, so lineage is auditable.

## Saving artifacts

Program artifacts are content-addressed. Saving is idempotent: the same definition always resolves to the same id, and saving it again is a no-op.

```python
from refaat.programs import GroundedRagProgram, Tunables

registry = project.registry
base_id = registry.save(GroundedRagProgram().artifact())
tuned_id = registry.save(GroundedRagProgram(tunables=Tunables(top_k=4)).artifact())
```

An artifact chosen by the optimizer can be rebuilt from its tunables and saved:

```python
best = result.best  # from project.programs.get(...).optimize(dataset)
best_id = registry.save(program.with_tunables(Tunables(**best.tunables)).artifact())
```

## Promotion and rollback

Environments are `development`, `staging`, and `production`. Promotion requires an artifact that is already registered and that belongs to the named program. Promotion to `production` additionally requires an explicit approval of that artifact; development and staging do not.

```python
registry.approve("grounded-rag", "production", tuned_id, approver="reviewer")
registry.promote("grounded-rag", "production", tuned_id)
current = registry.resolve("grounded-rag", "production")  # the deployed artifact document

deployment = registry.rollback("grounded-rag", "production")  # back to the prior artifact
history = registry.history("grounded-rag", "production")  # newest first
```

## Guarantees

- Artifacts are immutable and content-addressed. Promotion and rollback change only the alias.
- Rollback is atomic: it appends a deployment to the most recent previously deployed artifact, and fails cleanly if there is no prior artifact.
- The registry is workspace-scoped. A workspace can only resolve or promote artifacts it has saved, which is covered by an isolation end-to-end test.
- Production promotion is gated by an explicit, recorded approval. Approvals and deployments are append-only, so who approved or promoted what, and when, is auditable.
- Nothing is pickled. Artifacts are canonical JSON only.

## Command line

```bash
refaat program approve <workspace_id> <corpus_id> grounded-rag production
refaat program promote <workspace_id> <corpus_id> grounded-rag production
refaat program resolve <workspace_id> grounded-rag production
refaat program rollback <workspace_id> grounded-rag production
```

`promote` saves the current definition of the named program and promotes it. `resolve` and `rollback` operate on the workspace alias directly.

## Shadow and canary routing

A canary attaches a candidate artifact to an environment that already has an active deployment. Each request is bucketed deterministically (a sha256 digest of workspace, program, environment, and a sticky identity), and the candidate is served when the bucket falls under the configured percent. A percent of 0 is a pure shadow: the candidate still runs and is compared, but it is never served. Both the active and the candidate run through the same query pipeline, so both cite real stored chunks; the unserved result is used only to compute a deterministic comparison (answer equality, citation overlap, grounding, route equality, and a context-token delta). The comparison and the routing decision are appended to append-only telemetry, and a `program.shadow_compared` outbox event is emitted for external observers.

The layer never touches the answer-cache identity. With no canary attached, the router collapses to resolve-active-and-run through the unchanged pipeline, so every artifact id, cache key, and route is byte-identical to running the resolved active artifact directly. Attaching a canary to production reuses the same approval gate as promotion: the candidate must be approved first. A definition-hash identity guard refuses to serve any artifact whose stored definition does not match the rehydrated program, so only the registered, approved artifact is ever served. Promotion, approval, and rollback are unchanged; graduation is an explicit approve, promote, then `clear_canary`.

By default the comparison runs inline on the request path. When `defer_shadow` is enabled and no access policy is active, the router serves the request first and defers the comparison to a durable `program.shadow_compare` job, so the shadow re-derivation never delays the served response. The deferred job re-runs both artifacts under the same suppression and isolated-budget scopes as the inline shadow, records only the hash-only telemetry row and never any answer text, and is bounded by `shadow_defer_timeout_seconds` so a comparison the worker cannot reach in time is skipped rather than computed stale. Deferral is off by default and never fires under an active access policy, so the inline path is unchanged for every existing deployment.

```python
project.deployment.set_canary("grounded-rag", "staging", candidate_id, percent=0)  # pure shadow
served = project.deployment.run("grounded-rag", "staging", inputs={"question": "how does rollback work"})
telemetry = project.deployment.shadow_runs("grounded-rag", "staging")
project.deployment.clear_canary("grounded-rag", "staging")
```

```bash
refaat program canary set <workspace_id> <corpus_id> grounded-rag staging <candidate_artifact_id> --percent 0
refaat program canary status <workspace_id> <corpus_id> grounded-rag staging
refaat program deploy-run <workspace_id> <corpus_id> grounded-rag staging "how does rollback work"
refaat program canary clear <workspace_id> <corpus_id> grounded-rag staging
```

## Where this is going

The approval gate, shadow and canary routing, and the offline telemetry comparison documented above build on this registry and ship today. Further deployment policy and richer production telemetry are described in the repository review and in [Roadmap and non-goals](roadmap.md).
