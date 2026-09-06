# The enterprise foundation

Refaat is a framework-neutral runtime you build on top of, not a framework you build inside. It owns the parts of a production AI system that every team otherwise rebuilds: evidence, citations, caching, evaluation, learning, durability, and governance. Your agent framework and your model provider stay replaceable adapters. This page ties the pieces together and links each claim to the page that documents it in depth. Everything stated here is backed by shipped code and exercised by the offline test and example suites.

## One runtime under every framework

The framework adapters translate the outer retrieval and answering interfaces, while the core stays the authority for identity, cache scope, fusion, graph traversal, preferences, budgeting, and citations. The same grounded backend drives every adapter, so an answer produced through LangChain, LangGraph, CrewAI, or LlamaIndex carries the same real citations that resolve to real stored chunks.

- LangChain: `as_langchain_retriever` returns a `BaseRetriever`, and `as_langchain_tool` returns a `StructuredTool` that answers with canonical citation ids.
- LangGraph: dependency-free node and router primitives plus two lazy graph assemblers (`build_grounded_graph`, `build_react_subgraph`) run a real branching state machine over the same runtime.
- LlamaIndex: `as_llamaindex_retriever` returns a `BaseRetriever` whose node ids are canonical Refaat chunk ids.
- CrewAI and custom agents: `as_crewai_answer_tool` and `as_crewai_tool` are plain callables that answer or retrieve through the grounded pipeline, so they compose in any framework.

Each optional framework package is imported lazily, so the base package does not depend on any agent ecosystem, and constructing an adapter without its dependency raises a clear installation error. See [Framework integrations](framework-integrations.md).

## Any OpenAI-compatible model provider

Refaat is not tied to one vendor. The embedding, generation, and graph-extraction adapters speak the OpenAI-compatible HTTP protocol, so OpenAI, Mistral, Together, Groq, Ollama, and vLLM all work by configuration alone. There is no per-provider adapter class: only the base URL, key, and model change. Each role can point at a different provider, so you can run embeddings on one vendor and generation on another. For a provider that is not OpenAI-compatible, inject a custom adapter through the builder. With no configuration, every role runs offline and deterministically. See [Model providers](providers.md).

## Every input format

Ingestion accepts plain text, Markdown, JSON, CSV, XML, YAML, TOML, source code, HTML, PDF, and DOCX, plus images and audio and video through optional local media adapters. Each format flows through the same pipeline: extract canonical text and safe metadata, chunk, embed, extract a graph, and commit transactionally with a version bump. An image can be ingested as one citable descriptor chunk. See [Ingestion and extraction](ingestion.md) for the per-format how-to and [Multimodal ingestion](guides/multimodal.md) for images.

## What Refaat owns

- Evidence and citations: hybrid retrieval fuses semantic, lexical, and graph signals, and every answer cites real stored chunks by canonical id. A hallucinated id is dropped, never returned. See [Adaptive RAG, CAG, KAG, and GraphRAG](adaptive-context.md).
- Caching and token savings: the version-safe answer cache serves an exact or semantic hit without re-generating, an optional model-call cache serves identical embedding and generation calls from a table, and per-run token and cost budgets stop a multi-attempt loop before it overspends. See [Model-call cache and budgeting](model-cache.md).
- Learning: harness snapshots capture evidence-backed retrieval and prompt adjustments with activation and rollback, feedback records per-user signal, and native optimization searches immutable candidates without ever mutating the active program. See [Harness learning and rollback](harness.md), [Evaluations](evaluations.md), and [Program optimization](program-optimization.md).
- Governance and durability: typed programs and a content-addressed artifact registry with explicit promotion, approval, and atomic rollback, plus governed ReAct with a side-effect ceiling, capability-based tool governance, and durable single-use human approval with separation of duties and four-eyes. See [Program registry and promotion](program-registry.md), [Tool governance](tool-governance.md), and [Approvals](approvals.md).
- Authorization: per-principal access control over every evidence path, with a pluggable policy and a pluggable identity resolver, byte-identical when unconfigured and fail closed when active. See [Authorization](authorization.md).

## Horizontal scale

Refaat scales out through durable, leased work rather than in-memory state.

- Durable jobs run on a leased worker with heartbeats, bounded retries, and a per-process-unique lease owner, so two replicas never double-run one job. See [Durable agents and jobs](agents-and-jobs.md).
- PostgreSQL with pgvector is the multi-process backend, fronted by a bounded connection pool, and the vector table self-heals from committed chunk rows through a reconcile sink after a crash. See [Ingestion and extraction](ingestion.md) and [Deployment](deployment.md).
- The API rate limiter keys on the trusted client IP. A multi-worker deployment requires the shared Redis backend, enforced at startup, so the limit holds across workers.

## Clean and maintainable

There is one composition root: `RuntimeBuilder.build()` returns a `Refaat`, and the API, worker, CLI, and embedded Python application all use it, so there is exactly one implementation of retrieval, generation, cache, and citations. Adapters are typed ports, artifacts are canonical JSON with no pickle, and optimizers produce immutable candidates. This is what keeps the runtime free of vendor lock-in. See [Architecture](architecture.md) and [Public surfaces overview](api-and-sdk.md).

## Multi-tenant

Multi-tenancy is a first-class shipped property, not a bolt-on. Every corpus, document, chunk, artifact, deployment, API key, scope, and answer-cache identity is workspace-scoped. A query in one workspace never returns another workspace's chunks, the answer caches do not cross, and registry resolution is per workspace. The tenant-per-workspace pattern is the recommended isolation model: create one workspace per tenant and issue that tenant a scoped API key.

How to onboard a new tenant:

1. Create the tenant workspace: `workspace_id = runtime.create_workspace("acme", "Acme")`.
2. Issue a scoped API key for the tenant: `key_id, raw = runtime.issue_api_key(workspace_id, "acme-admin", scopes=("read", "write", "admin"))`. Return the raw key to the tenant once; only its hash is stored.
3. Create the tenant's corpus and ingest: `project = runtime.project(workspace="acme", corpus="handbook")` then `project.ingest(path)`.
4. All of the tenant's reads and writes carry its workspace id, so its evidence, caches, and artifacts are isolated from every other tenant.

Use case: a SaaS product gives each customer its own workspace. Customer A's documents, retrieval, cached answers, and promoted programs are invisible to customer B, and a leaked or revoked key affects only its own workspace. See [Security model](security.md).

## Authorization and access control

Below the workspace boundary, Refaat enforces per-principal access control over every evidence path. Refaat is the enforcement layer, not the identity provider: a deployment brings its identity through a `PrincipalResolver` and its policy through an `AccessPolicy`, and can route the decision to an external engine such as OPA or Cedar behind the same contract. Evidence is tagged at ingestion with an `acl` block (classification plus allow and deny lists and tenant), and the permission travels source to document to chunk to graph fact to retrieval to citation.

The decision is one pluggable function; the enforcement is many seams that all call it: retrieval including HyPE and RAPTOR, graph facts, RLM recursion, citations, the answer cache, session memory, learned preferences, and every program and HTTP surface. It is byte-identical when unconfigured and fail closed when active: an untagged or malformed unit is denied, a cross-tenant request is refused by a framework pre-policy guard even under a custom policy, the HTTP surface never trusts a client-supplied identity, and a model never decides a permission.

- [Authorization](authorization.md): the decision rules, the enforcement points, and the fail-closed semantics.
- [Access and multi-tenancy](access-and-multitenancy.md): the workspace, tenant, and ACL boundaries and how to model tenants.
- [Identity lifecycle](identity-lifecycle.md): the resolver port and the trusted HTTP identity boundary.
- [Audit and provenance](audit-and-provenance.md): the redacted audit event contract and its pluggable sink.
- [Data-classification model routing](model-routing.md): deterministic routing that refuses over-classified evidence to a model.
- [Learning and preferences](learning-and-preferences.md): principal-bound learned ranking.

## Multi-language

The vector retrieval path is language-neutral when paired with a multilingual embedding and generation model, so a query in one language can retrieve evidence written in another. To scope retrieval to a specific language, tag each document with a `language` metadata field and apply the shipped metadata hard filter, which restricts candidates before scoring and still cites real stored chunks.

How to build a multilingual RAG:

1. Ingest documents with a language tag: pass `metadata={"language": "de"}` on each `DocumentInput`.
2. Configure a multilingual embedding and generation model through the OpenAI-compatible provider layer (see [Model providers](providers.md)).
3. Scope a query to one language by activating a harness filter: `project.create_harness_snapshot(supplemental={"retrieval": {"filters": {"language": "de"}}}, evidence={"reason": "language scope"}, activate=True)`.
4. Retrieval returns only documents in that language, and citations resolve to those stored chunks.

The co-occurrence graph extractor is language-aware. `core/text.py` ships a per-language stopword registry as sensible defaults: English is the default, with curated German, French, Spanish, and Arabic sets. The graph extractor selects its stopword set from a document's `language` metadata tag, falling back to the `default_language` setting (`en` by default), so a document tagged `language="de"` builds its graph from German content words rather than German function words. The tokenizer is Unicode-aware for space-delimited scripts (Latin, Cyrillic, Arabic, and Latin diacritics), so those languages tokenize without extra configuration. An English or untagged corpus resolves to the exact English set and is unchanged. A parallel per-language query-stopword registry ships alongside it as scaffolding; the lexical and query-transform path does not yet consume it, so query-term filtering uses the English set regardless of language.

Stopwords are configurable per project. The shipped registry is the default, and each project can override a language or add a new one without editing the library, through the `stopwords_by_language` setting (`REFAAT_STOPWORDS_BY_LANGUAGE`) or `RuntimeBuilder.with_stopwords({...})`. Each entry maps a short language code to a list of stopwords; codes normalize to their primary subtag, words casefold to match the tokenizer, and the override merges onto the shipped defaults so untouched languages keep their curated sets. The override affects the graph co-occurrence stopwords and is additive: with nothing configured the runtime passes the module default to the extractor and every unconfigured corpus is byte-identical. Use case: a project ingesting Italian content supplies its own Italian set with `with_stopwords({"it": ["della", "sono", "come", "questo"]})`.

Honest limitation: a script without spaces (CJK) still tokenizes a run as a single token, so word segmentation for those scripts is a future enhancement, not a current claim. Non-Latin sentence splitting is likewise limited: `SENTENCE_PATTERN` keys on a Latin capital lookahead, so sentence boundaries in non-Latin scripts are approximate. Use case: a support corpus holds the same policy in several languages, and a filter serves each user answers grounded only in their own language's documents.

## Multi-channel

Refaat is reachable from many channels, all built on the same grounded backend with the same evidence and citation contract:

- the HTTP API for language and process isolation;
- the synchronous and asynchronous SDK clients;
- the CLI, with full parity to the HTTP surface;
- the LangChain, LangGraph, CrewAI, and LlamaIndex adapters embedded in an agent.

Sessions carry per-user and per-session conversational context, so a conversation stays coherent across turns on any channel. Any delivery surface you build (a web app, a chat widget, a Slack bot, a voice front end) sits on top of the same runtime, so the grounding, caching, learning, and governance are identical no matter how the user reaches it. See [Public surfaces overview](api-and-sdk.md) and [Sessions, memory, and preferences](sessions-and-memory.md).
