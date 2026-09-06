# Runnable product examples

The `examples/` directory is an executable product tour, not pseudocode. Every required script creates isolated temporary state, uses real Refaat services, asserts its own outcomes, and emits one JSON object suitable for automation.

Run every example:

```bash
PYTHONPATH=src python scripts/examples_check.py
```

The release gate executes every script in the catalog in isolated subprocesses. The authoritative catalog with per-script assertions is `examples/manifest.json` in the repository. The E2E suite validates the shipped catalog and ensures this executable gate remains mandatory, while API, SDK, CLI, and worker journeys have dedicated E2E tests.

## 01: Local RAG, GraphRAG, CAG, and version invalidation

```bash
uv run python examples/01_local_rag_graph_cache.py
```

Demonstrates:

- directory ingestion of six connected operational documents;
- deterministic embeddings and lexical retrieval;
- graph extraction, relations, graph facts, and graph-aware ranking;
- grounded generation and structured citations;
- exact answer-cache reuse;
- document replacement and corpus-version advancement;
- automatic rejection of the stale cache entry after evidence changes.

## 02: Sessions, compaction, feedback, and personalization

```bash
uv run python examples/02_sessions_feedback.py
```

Demonstrates persistent goals, user/session turns, automatic compaction, retained recent turns, explicit positive feedback, bounded user preferences, and profile-aware retrieval. Personal preferences affect ranking but are not converted into factual evidence.

## 03: Bounded RLM, agents, mailboxes, and jobs

```bash
uv run python examples/03_rlm_agents_jobs.py
```

Demonstrates recursive context decomposition with limits, preserved evidence IDs, parent and retained subagents, direct agent messages, acknowledgement, a durable background RLM job, worker execution, and persistence after the runtime is reopened.

## 04: Evaluations and reversible harness adaptation

```bash
uv run python examples/04_evaluations_harness.py
```

Demonstrates a ten-case regression suite, deterministic metrics, an evidence-backed proposal, a new immutable harness snapshot, activation, and rollback to the prior snapshot. The base system policy is not rewritten.

## 05: Programmatic extensions

```bash
uv run python examples/05_programmatic_extensions.py
```

Demonstrates `RuntimeBuilder` registration of a custom `memory://` connector, Markdown front-matter extractor, executable skill, and event sink. The durable worker publishes the real `document.indexed` outbox event to the custom sink.

## 06: Real FastAPI server and SDKs

```bash
uv run python examples/06_live_api_sdk.py
```

Starts a real Uvicorn subprocess on a temporary TCP port, waits for readiness, bootstraps a workspace, ingests and queries through `RefaatClient`, runs RLM and a session, then retrieves and queries through `AsyncRefaatClient`.

## 07: Persistent CLI workflow

```bash
uv run python examples/07_cli_workflow.py
```

Runs `doctor`, `init`, corpus creation, ingestion, query, RLM, evaluation creation/run, and skill listing as independent CLI processes against the same durable database.

## 08: Framework integrations

```bash
uv run python examples/08_framework_integrations.py
```

Exercises the framework-neutral retriever and CrewAI-compatible callable against a real API server. LangChain and LlamaIndex adapters execute when `refaat[integrations]` is installed; otherwise the result clearly reports the missing optional dependency.

## 09: Typed programs

```bash
uv run python examples/09_typed_program.py
```

Demonstrates typed signatures, content-addressed program artifacts, and grounded execution of the built-in `grounded-rag` program.

## 10: Program evaluation

```bash
uv run python examples/10_program_evaluation.py
```

Demonstrates content-addressed datasets, the deterministic metric bundle with hard constraints and weighted objectives, and a scored evaluation report.

## 11: Program optimization

```bash
uv run python examples/11_program_optimization.py
```

Demonstrates deterministic, model-free retrieval-policy search that produces immutable candidates, ranks them by hard constraints then score then token cost, and never mutates the active program.

## 12: Program registry, promotion, and rollback

```bash
uv run python examples/12_program_registry.py
```

Demonstrates the durable, workspace-scoped registry: content-addressed artifacts, environment promotion with an approval gate, and atomic rollback with append-only lineage.

## 13: Durable program evaluation

```bash
uv run python examples/13_durable_evaluation.py
```

Demonstrates enqueuing a program evaluation as a durable worker job whose report is persisted with the job and survives a restart.

## 14: Full-library RAG gauntlet

```bash
uv run python examples/14_full_rag_gauntlet.py
```

Exercises the whole library end to end over one fabricated knowledge base: ingestion, retrieval, programs, evaluation, optimization, and the registry, including a measurable contextual-retrieval ranking difference.

## 15: Program as skill and trajectory evaluation

```bash
uv run python examples/15_program_as_skill_trajectory.py
```

Demonstrates a promoted program wrapped as a governed tool that a ReAct program calls, scored with a trajectory bundle over a full multi-step run.

## 16: Reference application

```bash
uv run python examples/16_reference_app.py
```

Demonstrates one enterprise FastAPI service driven over the SDK together with a real LangChain agent and a real LangGraph state machine, over one corpus.

## 17: Durable and resumable optimization

```bash
uv run python examples/17_durable_optimization.py
```

Demonstrates a checkpointed optimization run on the worker that resumes after a restart and produces a result byte-identical to the synchronous `optimize()`.

## 18: Oronts showcase

```bash
uv run python examples/18_showcase_rag.py
```

Drives the full retrieval and program feature set over the Oronts knowledge base, offline and deterministically through an in-process client, and asserts grounded citations plus a measurable contextual-retrieval effect.

## 19: Governed fine-tuning

```bash
uv run python examples/19_governed_fine_tuning.py
```

Demonstrates an approval-gated, provenance-tracked fine-tune run: it records its lineage, refuses to execute before an explicit approval, and registers an immutable model artifact without auto-promoting it.

## 20: Multimodal generation

```bash
uv run python examples/20_multimodal_generation.py
```

Demonstrates an image ingested as one citable descriptor chunk, answered over deterministically and offline, so a multimodal corpus keeps the same grounding and citation contract as text.

## 21: Sandboxed code in the bounded RLM

```bash
uv run python examples/21_sandboxed_code_rlm.py
```

Demonstrates the deny-by-default code sandbox in the bounded RLM: nothing executes unless an executing sandbox is explicitly configured, and a governed computation stays audit-only and never becomes a citation.

## 22: Authorization-aware evidence

```bash
uv run python examples/22_authorization_aware_evidence.py
```

Demonstrates per-principal ACL isolation over one corpus: two confidential documents grant different groups, so Alice retrieves and cites only her document and Bob only his, their cited evidence is disjoint. With no policy the read is unfiltered, and this example opts out of strict authorization so a principal-less read under the policy stays unfiltered for the demonstration (the default denies it).

## 23: Governed tool with a human approval

```bash
uv run python examples/23_governed_tool_approval.py
```

Demonstrates a durable single-use human approval on a governed tool: the tool is denied until a separate approver records an approval for the exact action fingerprint, then it runs exactly once and the consumed approval leaves the next run denied.

## 24: Governed model routing

```bash
uv run python examples/24_governed_model_routing.py
```

Demonstrates a classification-ceiling model router: evidence at or below the ceiling is answered and cited, while evidence above the ceiling is refused before it reaches the model, with a generic refusal message. An unrouted control over the same evidence generates and surfaces the marker, proving the refusal is router-caused.

## 26: Audit and provenance

```bash
uv run python examples/26_audit_and_provenance.py
```

Demonstrates a redacted-by-construction audit event per served query, carrying only ids, counts, policy, and timestamps and never evidence text. The event is delivered inline, and with the durable outbox enabled it is persisted and published exactly once by the worker.

## 27: Durable job identity freshness

```bash
uv run python examples/27_durable_job_freshness.py
```

Demonstrates a durable RLM job whose principal identity snapshot is captured server-side at enqueue and re-resolved at execution. A fresh job runs and cites only authorized evidence; a job whose scope changed after enqueue is refused rather than served under stale permissions.

## 28: Enterprise governance end to end

```bash
uv run python examples/28_enterprise_governance.py
```

Demonstrates authorization isolation, a classification-ceiling model router, and redacted audit composed on one governed runtime. Authorization isolates per-group evidence, an authorized principal is still refused generation over evidence above the routing ceiling, and every served decision emits a redacted audit event, with controls proving each guard is causal.

## Example data

`examples/data/company/` contains a fictional but internally connected company knowledge set covering architecture, release policy, incident response, security/privacy, customer onboarding, and service ownership. `examples/data/evaluation_cases.json` contains ten regression cases used by the evaluation and CLI examples.

The data is intentionally deterministic so CI can verify retrieval, graph, cache, evaluation, and persistence behavior without external network calls or paid models.
