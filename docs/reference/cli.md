# CLI reference

The `refaat` command uses the same composition root, persistence, ingestion, query, evaluation, RLM, skill, and worker services as the Python and HTTP interfaces. Every data command prints JSON.

## Service and diagnostics

```bash
refaat doctor
refaat serve --host 127.0.0.1 --port 8000
refaat worker
refaat worker --once
```

`doctor` validates startup configuration, opens the store, and reports selected storage/model providers, registered media types, and connector schemes.

## Workspace bootstrap

```bash
refaat init --slug acme --name "Acme"
```

The output contains `workspace_id`, `api_key_id`, and the one-time raw administrator API key.

## Corpora

```bash
refaat corpus create <workspace-id> handbook
refaat corpus list <workspace-id>
refaat corpus derivations list <workspace-id> <corpus-id>
refaat corpus derivations get <workspace-id> <corpus-id> <derivation-key>
refaat corpus derivations verify <workspace-id> <corpus-id>
refaat corpus derivations compare <workspace-id> <corpus-id> <left-key> <right-key>
refaat corpus derivations verify-document <workspace-id> <corpus-id> <document-id>
```

The `derivations` commands are read-only and require `derivation_manifest_enabled`; `verify` reports `no_manifest`, `match`, or `drift`, and `verify-document` localizes which document's chunk set drifted, reporting `no_document`, `no_digest`, `match`, or `drift`.

## Ingestion

```bash
refaat ingest \
  <workspace-id> \
  <corpus-id> \
  ./policy.md \
  --source-id policy:release \
  --title "Release policy"
```

```bash
refaat ingest-source \
  <workspace-id> \
  <corpus-id> \
  https://knowledge.example.com/release \
  --source-id policy:release
```

```bash
refaat ingest-image \
  <workspace-id> \
  <corpus-id> \
  ./diagram.png \
  --source-id diagram:architecture \
  --caption "System architecture" \
  --alt-text "The system architecture diagram"
```

`ingest` requires one existing file. Directory ingestion is available through `RefaatProject.ingest(...)`. `ingest-image` stores a local image as a citable document backed by one descriptor chunk.

## Query and RLM

```bash
refaat query \
  <workspace-id> \
  <corpus-id> \
  "What evidence is required?" \
  --user-id alice \
  --session-id <session-id>
```

```bash
refaat rlm \
  <workspace-id> \
  <corpus-id> \
  "Compare release and incident controls" \
  --max-depth 2 \
  --max-nodes 16 \
  --branch-factor 4 \
  --leaf-token-budget 1500
```

Add `--background` to enqueue the RLM run for a worker.

## Evaluations

```bash
refaat evaluation create \
  <workspace-id> \
  release-regression \
  ./cases.json

refaat evaluation run \
  <workspace-id> \
  <suite-id> \
  <corpus-id>
```

The cases file must contain a JSON array.

## Skills

```bash
refaat skill list
refaat skill create release_guard "Checks release evidence"
```

Generated skill projects are written under `REFAAT_SKILL_ROOT` and include a package, entry point, test, README, and Refaat dependency declaration. Review and install generated code explicitly; creation does not dynamically install or execute it.

## Programs, optimization, and deployment

```bash
refaat program list <workspace-id> <corpus-id>
refaat program run <workspace-id> <corpus-id> grounded-rag "What evidence is required?"
refaat program eval <workspace-id> <corpus-id> grounded-rag ./cases.json
refaat program optimize <workspace-id> <corpus-id> grounded-rag ./cases.json
```

Durable and resumable optimization enqueues a worker run and is polled by run id:

```bash
refaat program optimize-job <workspace-id> <corpus-id> grounded-rag ./cases.json
refaat program optimization <workspace-id> <run-id>
```

Registry promotion, approval, resolution, and atomic rollback:

```bash
refaat program promote <workspace-id> <corpus-id> grounded-rag staging
refaat program approve <workspace-id> <corpus-id> grounded-rag production
refaat program resolve <workspace-id> grounded-rag production
refaat program rollback <workspace-id> grounded-rag production
```

Shadow and canary routing (percent 0 is pure shadow: the candidate is computed and compared but never served):

```bash
refaat program canary set <workspace-id> <corpus-id> grounded-rag staging <candidate-artifact-id> --percent 0
refaat program canary status <workspace-id> <corpus-id> grounded-rag staging
refaat program canary clear <workspace-id> <corpus-id> grounded-rag staging
refaat program deploy-run <workspace-id> <corpus-id> grounded-rag staging "how does rollback work"
```

Promotion, approval, and rollback are the same operations exposed over the HTTP and Python surfaces. See [Program optimization](../program-optimization.md) and [Program registry, promotion, and rollback](../program-registry.md) for the semantics.

## Fine-tuning

```bash
refaat fine-tune request <workspace-id> <base-model> ./training.json
refaat fine-tune approve <workspace-id> <run-id>
refaat fine-tune enqueue <workspace-id> <run-id>
refaat fine-tune run <workspace-id> <run-id>
refaat fine-tune status <workspace-id> <run-id>
```

Governed fine-tuning is approval-gated: `request` records a `PENDING` run over a JSON training dataset, `approve` authorizes it, `enqueue` runs it as a durable worker job or `run` executes it inline (both refuse an unapproved run), and `status` reports its approval, model reference, and provenance. A completed run registers an immutable, un-promoted model artifact. See [Governed fine-tuning](../fine-tuning.md).

## Data lifecycle and operations

```bash
refaat document erase <workspace-id> <corpus-id> <document-id>
refaat legal-hold place <workspace-id> <corpus-id> <placed-by> --document-id <document-id> --reason "matter 14"
refaat legal-hold lift <workspace-id> <hold-id> <lifted-by>
refaat legal-hold list <workspace-id>
refaat kill-switch engage workspace_freeze <engaged-by> --workspace-id <workspace-id> --reason "incident 42"
refaat kill-switch disengage workspace_freeze <disengaged-by> --workspace-id <workspace-id>
refaat kill-switch list --workspace-id <workspace-id>
refaat status <workspace-id>
```

Erasure removes every auto-derived surface a document produced. A legal hold blocks erasure, deletion, and overwrite; placing one without a document id holds the whole corpus, and lifting the last covering hold completes a deferred erasure. A kill switch freezes a workspace's AI and data plane; omit the workspace id to read or set the global state. `refaat status` prints the read-only status plane and stays available during a freeze. These are the admin-scoped compliance and incident commands. See [Right to erasure](../erasure.md), [Legal holds](../legal-holds.md), [Kill switches](../kill-switches.md), and [Admin status plane](../admin-status.md).

## Plugin authoring

```bash
refaat plugin ports
refaat plugin describe ChunkingStrategy
refaat plugin new ChunkingStrategy acme_chunker --out ./plugins
refaat plugin verify acme_chunker:AcmeChunker --port ChunkingStrategy
```

`plugin ports` lists every typed port a plugin can implement, with its injection tier and seam. `plugin describe <Port>` shows one port's required members, its default, and the invariant an adapter must uphold. `plugin new <Port> <name>` scaffolds an installable adapter project from the catalog. `plugin verify <module:Class> --port <Port>` imports the named adapter (which runs the module's top-level code) and checks it implements every member the port declares, exiting non-zero if not. The discovery commands need no database. The full catalog is the [Ports catalog](ports-catalog.md) reference, kept in lockstep with the code by the `ports_catalog` gate.

## Full executable flow

```bash
uv run python examples/07_cli_workflow.py
```

That example persists state across independent CLI processes and covers diagnostics, bootstrap, corpus creation, ingestion, query, RLM, evaluations, and skills.
