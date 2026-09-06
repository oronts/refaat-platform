# Refaat documentation

**REFAAT — Retrieval & Evidence Foundation for Adaptive Agentic Tooling** is the Python-native evidence, context, and durable-agent foundation from Oronts.

Refaat provides one programmable runtime for RAG, GraphRAG, CAG, KAG, bounded recursive language-model execution, typed programs with evaluation, optimization, and a durable registry, governed agent tools with capability-based approvals, per-principal authorization and multi-tenancy, evaluations, executable skills, sessions, jobs, schedules, agent mailboxes, and OpenTelemetry-compatible operations. The local path works without a paid model; production adapters replace storage and model components without changing the application contract. Every replaceable part is a typed Protocol port in `refaat.ports`: model vendors and agent frameworks are adapters, and custom connectors, extractors, chunking strategies, skills, and event sinks plug in through a `RuntimeBuilder` injection point or an allowlisted entry point. See [Authoring a plugin](guides/authoring-a-plugin.md).

## Choose a path

| Goal | Start here |
|---|---|
| Run the complete local workflow | [Getting started](getting-started.md) |
| Execute the shipped product journeys | [Runnable examples](examples.md) |
| Understand the system boundary | [Product boundary](product-boundary.md) and [architecture](architecture.md) |
| Build retrieval and GraphRAG | [Adaptive context](adaptive-context.md), [ingestion](ingestion.md), and [production RAG guide](guides/production-rag.md) |
| Add memory or long-running agents | [Sessions and memory](sessions-and-memory.md), [agents and jobs](agents-and-jobs.md), and [RLM/autonomy](rlm-and-autonomy.md) |
| Build and optimize typed programs | [Programs](programs.md), [program evaluation](program-evaluation.md), [optimization](program-optimization.md), and [registry](program-registry.md) |
| Run governed agent tools | [Agentic tools and function calling](agentic-tools.md) |
| Enforce access and multi-tenancy | [Authorization](authorization.md), [access and multi-tenancy](access-and-multitenancy.md), and [enterprise foundation](enterprise-foundation.md) |
| Integrate an application | [Python API](reference/python-api.md), [HTTP API](reference/http-api.md), [CLI](reference/cli.md), and [framework integrations](framework-integrations.md) |
| Add custom behavior | [Authoring a plugin](guides/authoring-a-plugin.md), [custom extensions](guides/custom-extensions.md), [custom chunking](guides/custom-chunking.md), and [plugins and skills](plugins-and-skills.md) |
| Operate in production | [Deployment](deployment.md), [security](security.md), [backup and restore](operations/backup-restore.md), and [runbooks](operations/runbooks.md) |
| Contribute or release | [Testing](testing.md), [contributing](development/contributing.md), and [release process](development/release-process.md) |

## Core guarantees

Refaat is designed around a small set of explicit guarantees:

1. Evidence has stable workspace, corpus, document, and chunk identities.
2. Document replacement and deletion advance the corpus version.
3. Cached answers are isolated by corpus version and effective context state.
4. Generators can cite only chunks that were present in the selected context.
5. The immutable base grounding policy cannot be rewritten by feedback or an agent.
6. Long-running work is durable, leased, retryable, and visible through persisted state.
7. Connectors, extractors, models, event sinks, skills, chunking strategies, and framework adapters remain replaceable.
8. Missing quality tools are recorded as unavailable, never silently counted as passes.

The project remains a production-oriented foundation rather than a claim of production certification for every organization. Capacity, recovery, data classification, provider, threat-model, and compliance validation remain deployment responsibilities.
