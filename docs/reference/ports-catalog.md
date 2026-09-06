# Ports catalog

Every replaceable concern in Refaat is a typed Protocol port. This page is the reference for each one: what a plugin implements, how it is injected, and the invariant its adapter must uphold. The ports_catalog gate keeps this catalog in lockstep with the live ports, so a new port cannot ship without an entry here.

## Entry-point plugins

Discovered through the named `refaat.*` entry-point group, gated by `REFAAT_PLUGIN_ALLOWLIST` and optional code-hash pins. Also injectable in-process on the builder.

| Port | Members | Seam | Default | Invariant |
|---|---|---|---|---|
| `ChunkingStrategy` | `chunk`, `strategy_id` | `refaat.chunkers` | refaat.core.chunking.AdaptiveChunkingStrategy | propose only text, ordinal, and structure; the pipeline mints every id and checksum, so a strategy cannot forge a chunk id. Also injectable via with_chunking_strategy. |
| `Connector` | `fetch`, `scheme` | `refaat.connectors` | refaat.infrastructure.connectors.FileConnector | outbound fetches must route through the egress policy; a hand-rolled network call bypasses governance. |
| `DocumentExtractor` | `extract`, `media_types` | `refaat.extractors` | built-in text and PDF extractors | convert bytes to text and metadata for the declared media types. |
| `EventSink` | `emit` | `refaat.event_sinks` | refaat.infrastructure.events.LoggingEventSink | delivery is at least once, so a sink must be idempotent on the event id. |
| `Skill` | `description`, `name`, `run` | `refaat.skills` | the built-in corpus_health skill | a skill receives the canonical context packet and explicit JSON arguments; an external skill routes egress through the policy. |

## Builder injection

Injected in-process through the named `RuntimeBuilder` method.

| Port | Members | Seam | Default | Invariant |
|---|---|---|---|---|
| `CacheStore` | `delete_query_cache_for_corpus`, `get_exact_cache`, `list_cache_candidates`, `put_cache` | `with_cache_store` | the relational store | the version-aware answer cache; the erasure member must delete a corpus's cached answers or right-to-erasure silently leaves them behind. Byte-identical when unset (the store is the default). |
| `CodeSandbox` | `execute`, `executes`, `name` | `with_code_sandbox` | refaat.infrastructure.sandbox.DenyingSandbox | deny by default; an executing sandbox must bound resources and never let a result become a citation. |
| `CrossEncoderScorer` | `model_id`, `score` | `with_cross_encoder_scorer` | injected scorer, no offline default | score is a pure relevance signal over (query, passage); model_id folds into the cache id. |
| `Embedder` | `dimensions`, `embed`, `model_id` | `with_embedder` | refaat.infrastructure.embeddings.DeterministicEmbedder | dimensions must be constant and equal every vector length; embed must be deterministic for identical input, and a real backend should also implement CacheIdentified. |
| `FineTuningProvider` | `provider_id`, `submit` | `with_fine_tuning_provider` | refaat.infrastructure.fine_tuning.DeterministicFineTuner | a run is approval-gated and its model is an immutable un-promoted artifact. |
| `Generator` | `generate`, `model_id`, `stream` | `with_generator` | refaat.infrastructure.generation.ExtractiveGenerator | citation contract: an answer may cite only chunk ids present in the request evidence; an invented id must not survive. |
| `GraphExtractor` | `extract`, `model_id` | `with_graph_extractor` | refaat.infrastructure.graph.CooccurrenceGraphExtractor | entities and relations must reference only the supplied document text. |
| `GraphReadRepository` | `graph_index_present`, `graph_neighborhood`, `list_graph`, `max_entity_name_length` | `with_graph_read_repository` | refaat.infrastructure.database.store.SQLStore | read-only graph seam behind the query and RLM hot paths. graph_neighborhood must never under-return a name graph_context would match (over-return is harmless, graph_context re-filters). An adapter may expose an optional cache_id that folds into the answer-cache identity since graph reads change results; the default composite store exposes none, so the path is byte-identical. |
| `HypotheticalQuestionGenerator` | `provider_id`, `questions` | `with_hype_generator` | deterministic templated generator | a hypothetical question is embedded and matched but is never a chunk and never cited; provider_id folds into the answer-cache identity. |
| `ImageCaptioner` | `caption`, `provider_id` | `with_image_captioner` | honest deterministic descriptor when unconfigured | a caption becomes a stored descriptor chunk; the raw image bytes are never cited. |
| `JudgeModel` | `assess`, `model_id` | `with_judge_model` | deterministic faithfulness proxy | a score is advisory and never rewrites a grounded answer or a citation. |
| `ModelCallCache` | `delete_embedding_model_cache`, `delete_generation_model_cache`, `get_embedding`, `get_model_call`, `put_embedding`, `put_model_call` | `with_model_call_cache` | the relational store | a content-addressed cache for provider round-trips; a hit must be byte-identical to a miss and the erasure members must purge a workspace's generations and an erased text's embeddings. |
| `QueryTransform` | `cache_id`, `expand` | `with_query_transform` | deterministic transform catalog | a variant query is never retrieved or cited; cache_id folds into the answer-cache identity. |
| `Reranker` | `rerank` | `with_reranker` | refaat.infrastructure.reranking.LexicalReranker | reranking reorders candidates only; it must not introduce a chunk that was not retrieved. |
| `SearchIndex` | `delete_document`, `lexical_search`, `replace_document`, `semantic_search` | `with_search_index` | refaat.infrastructure.search.LocalSearchIndex | every returned hit must resolve to a stored chunk in scope, the ACL and metadata pushdown must fail closed on both the semantic and the lexical path, a transport or partial-shard error must raise rather than fall back to a scan, and a deleted document must never resurface. The built-in Local and pgvector backends are selected by database_url; a third backend is injected via with_search_index, as an index or a factory whose build(store, embedder) receives the runtime store for id-to-chunk hydration. |
| `SecretProvider` | `available`, `provider_id`, `resolve` | `with_secret_provider` | no provider unless configured | a secret is supplied at execution and must never enter an artifact, a cache key, or an audit event. |

## Config selection

Selected by the named settings value inside the builder. No `with_*` seam today.

| Port | Members | Seam | Default | Invariant |
|---|---|---|---|---|
| `BlobStore` | `delete`, `list_blobs`, `modified_at`, `open`, `put` | `blob_backend` | refaat.infrastructure.blob.LocalBlobStore | put then open must round-trip the exact bytes; selected by blob_backend, no with_* seam yet. |

## No injection seam yet

Constructed internally or a marker. No external injection seam today.

| Port | Members | Seam | Default | Invariant |
|---|---|---|---|---|
| `CacheIdentified` | `cache_identity` | n/a | marker on OpenAI-compatible embedder and generator | cache_identity must fingerprint the backend and model so two backends sharing a bare model name never collide in the answer cache; opt-in marker, no injection seam. |
| `FileGuard` | `resolve` | n/a | the built-in path guard | resolve must reject a path that escapes the configured root. |
| `Resolver` | `resolve` | n/a | system DNS resolver | used only to pin an outbound host to a validated address for SSRF defense. |
| `StreamingToolCallingGenerator` | `model_id`, `stream_tool_call` | n/a | a real streaming function-calling provider only | a partial argument fragment is opaque until the terminating delta; only a real provider implements it. |
| `Summarizer` | `provider_id`, `summarize` | n/a | refaat.infrastructure.summarization.DeterministicExtractiveSummarizer | a summary node is embedded and matched but resolves to real leaf chunks and is never cited; injected through the raptor builder, no direct with_* seam. |
| `ToolCallingGenerator` | `model_id`, `propose_tool_call` | n/a | a real function-calling provider only | propose only from the provided tool specs; kept separate from Generator so the offline generator never fakes a tool choice. |
| `ToolPlanner` | `plan`, `tools` | n/a | refaat.programs.react.KeywordToolPlanner | plan must return only a tool named in tools() and None to stop; constructed inside the programs layer, no with_* seam. |

## Authoring an adapter

Implement the port's members, register it (an entry-point group for a packaged plugin, or a `with_*` method for an in-process object), add its name to the allowlist, and pin it. See [Authoring a plugin](../guides/authoring-a-plugin.md) and [Extending Refaat](../extending.md).
