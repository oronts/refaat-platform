# Extending Refaat

Refaat owns evidence, context, cache, citations, learning, durability, and evaluation. Everything outside that core is a typed seam. A model vendor, a document format, a search backend, a governance decision, and a chunking rule are all adapters that plug into the runtime, and the dependency flows inward: the core calls a Protocol, never a specific vendor. This page is the map of every seam, how each one plugs in, and the two guarantees that make the extension surface safe to use in production.

## The port model

Every external concern is a typed `Protocol` in `src/refaat/ports.py`. There are twenty-seven of them (plus one opt-in `CacheIdentified` marker that an adapter exposes only when it is genuinely cacheable). A port names the exact methods the core will call and nothing else, so an adapter is judged by structure, not by inheritance. Two properties hold across the whole set:

- Every port has a byte-identical default. The offline stack (deterministic embeddings, extractive generation, a co-occurrence graph, local vector search, a local blob store) satisfies the same Protocols a production vendor does. Building a runtime with no injection produces the reference behavior the gate certifies.
- A port receives only what it needs. A `ChunkingStrategy` proposes text, ordinal, and structure but never mints a chunk id; the pipeline derives every id and checksum from `(document_id, checksum, ordinal)`, so a strategy cannot forge an id or collide with a reserved namespace. A `Summarizer` and an `ImageCaptioner` see passage text or image bytes but never a stored chunk id, so a summary or a caption can never carry or fabricate a citation.

The result is that adding a capability is an adapter, not a fork. You implement one Protocol and hand the object to the runtime through one of three tiers.

## The three injection tiers

### Tier 1: entry-point plugin groups

A separately packaged plugin publishes a Python entry point, and `RuntimeBuilder.build` discovers it at startup through `PluginRegistry`. There are five groups:

```toml
[project.entry-points."refaat.skills"]
release_guard = "acme_refaat.skills:ReleaseGuardSkill"

[project.entry-points."refaat.extractors"]
runbook = "acme_refaat.extractors:RunbookExtractor"

[project.entry-points."refaat.connectors"]
knowledge = "acme_refaat.connectors:KnowledgeConnector"

[project.entry-points."refaat.event_sinks"]
audit = "acme_refaat.events:AuditSink"

[project.entry-points."refaat.chunkers"]
legal_clause = "acme_refaat.chunking:LegalClauseStrategy"
```

Discovery is gated by the allowlist described below, so an installed package does not load until an operator names it. This tier is for reusable, independently versioned components.

### Tier 2: builder `with_*` injection

An application that owns a component in its own process injects it directly on the builder. There are thirty-five `with_*` methods. They cover the model seams and the governance seams, and they also accept the same in-process kinds that Tier 1 discovers as packages.

```python
from refaat.application.runtime import RuntimeBuilder

runtime = (
    RuntimeBuilder(settings)
    .with_embedder(MyEmbedder())
    .with_generator(MyGenerator())
    .with_graph_extractor(MyGraphExtractor())
    .with_reranker(MyReranker())
    .with_chunking_strategy(MyChunkingStrategy())
    .with_access_policy(MyAccessPolicy())
    .with_principal_resolver(MyResolver())
    .with_audit_sink(MyAuditSink())
    .build()
)
```

Model seams (`with_embedder`, `with_generator`, `with_graph_extractor`, `with_reranker`, `with_cross_encoder_scorer`, `with_query_transform`, `with_hype_generator`, `with_raptor_builder`, `with_judge_model`, `with_image_captioner`, `with_multimodal_generator`, `with_fine_tuning_provider`, `with_code_sandbox`) and governance seams (`with_access_policy`, `with_principal_resolver`, `with_secret_provider`, `with_audit_sink`, `with_model_router`, `with_output_governor`, `with_context_transform`, `with_degradation_policy`, `with_kill_switch_registry`, `with_quota_policy`, `with_egress_policy`, and more) are builder-only: there is no entry-point discovery for them, because installing a model provider or an authorization decision from an unreviewed package is not a posture Refaat offers. The four in-process registration seams that overlap Tier 1 (`with_connector`, `with_extractor`, `with_skill`, `with_event_sink`) plus `with_chunking_strategy` let you register an application-owned object without packaging it as a plugin. The registries reject a duplicate connector scheme, media type, or skill name, and a composition failure closes every resource already built.

### Tier 3: configuration selection

Two storage backends are chosen by configuration, not by a builder call:

- The search index is selected from `REFAAT_DATABASE_URL`: a `postgresql` URL builds the pgvector index, anything else builds the local vector index. There is no `with_search_index` today.
- The blob store is selected from `REFAAT_BLOB_BACKEND` (`local` or `s3`). There is no `with_blob_store` today.

The built-in chunking strategies are also config-selected: `REFAAT_CHUNKING_STRATEGY` picks `flat`, `parent_child`, or `sentence_window`, all byte-identical defaults. When the value names neither a built-in nor a discovered `refaat.chunkers` plugin, startup refuses rather than silently falling back.

## What plugs in, and how

| Concern | Port | How it plugs in | Default |
|---|---|---|---|
| Source connector | `Connector` | `refaat.connectors` entry point or `with_connector` | file and http/https connectors |
| Document extractor | `DocumentExtractor` | `refaat.extractors` entry point or `with_extractor` | built-in text and structured extractors |
| Skill (governed tool) | `Skill` | `refaat.skills` entry point or `with_skill` | deny-all registry |
| Event sink | `EventSink` | `refaat.event_sinks` entry point or `with_event_sink` | logging sink |
| Chunking strategy | `ChunkingStrategy` | `refaat.chunkers` entry point, `with_chunking_strategy`, or `REFAAT_CHUNKING_STRATEGY` | `flat` (adaptive) |
| Embedder | `Embedder` | `with_embedder` or `REFAAT_EMBEDDING_PROVIDER` | deterministic hash embeddings |
| Generator | `Generator` | `with_generator` or `REFAAT_GENERATION_PROVIDER` | extractive generator |
| Graph extractor | `GraphExtractor` | `with_graph_extractor` | co-occurrence extractor |
| Reranker | `Reranker` | `with_reranker` or `REFAAT_RERANKER_PROVIDER` | none |
| Cross-encoder scorer | `CrossEncoderScorer` | `with_cross_encoder_scorer` | none |
| Query transform | `QueryTransform` | `with_query_transform` | none |
| HyPE question generator | `HypotheticalQuestionGenerator` | `with_hype_generator` | deterministic templated generator |
| RAPTOR tree summarizer | `Summarizer` (via `with_raptor_builder`) | `with_raptor_builder` | deterministic extractive summarizer |
| Judge model | `JudgeModel` | `with_judge_model` | deterministic Faithfulness proxy |
| Image captioner | `ImageCaptioner` | `with_image_captioner` or `REFAAT_MULTIMODAL_*` | deterministic descriptor |
| Multimodal generator | `Generator` | `with_multimodal_generator` or `REFAAT_MULTIMODAL_GENERATION` | none |
| Fine-tuning provider | `FineTuningProvider` | `with_fine_tuning_provider` or `REFAAT_FINE_TUNING_PROVIDER` | deterministic recorder (no training) |
| Code sandbox | `CodeSandbox` | `with_code_sandbox` or `REFAAT_CODE_SANDBOX` | denies every call, executes nothing |
| Secret provider | `SecretProvider` | `with_secret_provider` | none installed, no secret resolved |
| Access policy | (authz policy) | `with_access_policy` | none, enforcement off |
| Principal resolver | `PrincipalResolver` | `with_principal_resolver` | none, no server-side identity |
| Audit sink | (audit sink) | `with_audit_sink` | none, no events emitted |
| Model router | (routing policy) | `with_model_router` | none, no classification ceiling |
| Output governor | (output policy) | `with_output_governor` | none |
| Search index | `SearchIndex` | config-selected by `REFAAT_DATABASE_URL` (no `with_*` today) | local vector index |
| Blob store | `BlobStore` | config-selected by `REFAAT_BLOB_BACKEND` (no `with_*` today) | local blob store |
| Answer cache store | `CacheStore` | Protocol defined, not yet wired to a setter | relational store |
| Model-call cache | `ModelCallCache` | Protocol defined, not yet wired to a setter | relational store when enabled |

The last two rows are honest about the current surface: `CacheStore` and `ModelCallCache` are typed as ports so a future backend can slot in, but there is no injection point for them yet. The runtime uses the relational implementation.

Example 29, `examples/29_custom_chunking_strategy.py`, runs a custom `ChunkingStrategy` end to end, and the guide at [guides/custom-chunking.md](guides/custom-chunking.md) walks the same path in prose. The broader connector, extractor, skill, and event-sink walkthrough lives in [guides/custom-extensions.md](guides/custom-extensions.md).

## The allowlist and code-hash pin security model

Entry-point discovery is not automatic. `REFAAT_PLUGIN_ALLOWLIST` decides which entry-point names may load, and an empty allowlist (the default) disables third-party discovery entirely. An operator lists each approved name explicitly, so installing a package is not the same as trusting it.

A name is not an identity. Any installed distribution can publish an entry point under an allowlisted name, and name-only discovery would import whatever module that name points at. Plugin pinning closes that gap. A pin binds each allowlisted entry point to an exact target module, the distribution that must provide it, its version, and a sha256 over the distribution's shipped source, so a discovered entry point loads only when it matches its pin:

```python
runtime = (
    RuntimeBuilder(settings)
    .with_plugin_pins(
        {
            "refaat.skills:release_guard": {
                "value": "acme_refaat.skills:ReleaseGuardSkill",
                "dist_name": "acme-refaat",
                "dist_version": "2.3.1",
                "code_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            }
        }
    )
    .build()
)
```

Pinning is opt-in and total. With no pins configured, discovery is the name-only allowlist and is byte-identical to before: no distribution metadata is read and no file is hashed. When any pin is set, every allowlisted entry point must be pinned, and the registry refuses two distributions that publish the same allowlisted name so a squatter cannot shadow a legitimate plugin. When pinning is active, each load or rejection can emit a redacted `plugin.loaded` or `plugin.rejected` audit event. The full field-by-field model is documented in [plugin-pinning.md](plugin-pinning.md).

## The governance guarantee

Governance seams follow one rule: byte-identical when absent, fail-closed when active.

Byte-identical when absent means a runtime with no access policy, no resolver, no audit sink, no router, and no sandbox behaves exactly as the certified offline stack. The policy-enforcement point in retrieval does not run, no per-request audit event is emitted, no classification ceiling is applied, and the code sandbox refuses every call. Nothing about the extension surface changes the answer, the citations, or the cache identity of an unconfigured runtime.

Fail-closed when active means a configured seam denies rather than serves under ambiguity. A principal resolver installed with no access policy is refused at build time, because it would resolve identity while enforcing nothing and serve evidence unfiltered. Fail-closed audit with no sink is refused at build time, because it would let every governed operation proceed unaudited. When an access policy is active, an untagged or malformed ACL is treated as its most restrictive class, not as public. The model never decides a permission: classification and routing ceilings are derived deterministically from evidence tags, and a tool capability's required secrets come from the registered capability, never from model output or a tool argument.

These two properties are why the extension surface is safe to grow. An unconfigured seam cannot change certified behavior, and a configured seam cannot fail open.
