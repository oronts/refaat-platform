# Configuration reference

Refaat uses `pydantic-settings`, the `REFAAT_` environment prefix, `.env` files, and `__` as the nested-value delimiter. Lists are normally supplied as JSON arrays.

## Runtime and persistence

| Variable | Default | Purpose |
|---|---|---|
| `REFAAT_ENVIRONMENT` | `development` | `development`, `test`, or `production` |
| `REFAAT_DATABASE_URL` | `sqlite:///./data/refaat.db` | SQLAlchemy database URL |
| `REFAAT_AUTO_CREATE_SCHEMA` | `true` | Local schema convenience; rejected in production |
| `REFAAT_BLOB_BACKEND` | `local` | `local` or `s3` |
| `REFAAT_BLOB_ROOT` | `./data/blobs` | Local original-document root |
| `REFAAT_BLOB_GC_GRACE_SECONDS` | `86400` | Minimum age before the `blob.gc` job reclaims an unreferenced blob |
| `REFAAT_S3_BUCKET` | unset | Required for S3 mode |
| `REFAAT_S3_ENDPOINT_URL` | unset | S3-compatible endpoint |
| `REFAAT_S3_REGION` | `eu-central-1` | Object-store region |
| `REFAAT_S3_ACCESS_KEY_ID` | unset | Optional static access key |
| `REFAAT_S3_SECRET_ACCESS_KEY` | unset | Optional static secret key |

Use workload identity instead of static S3 credentials where the adapter/platform supports it.

## Connectors and HTTP

| Variable | Default | Purpose |
|---|---:|---|
| `REFAAT_CONNECTOR_FILE_ROOT` | `./data/connectors` | Root for `file://` ingestion |
| `REFAAT_CONNECTOR_MAX_BYTES` | `20971520` | Maximum connector response/file size |
| `REFAAT_ALLOW_PRIVATE_CONNECTOR_HOSTS` | `false` | Permit private destinations; use only with network controls |
| `REFAAT_REQUEST_TIMEOUT_SECONDS` | `60` | Model and connector request timeout |

HTTP/HTTPS connectors reject URL credentials and redirects and stream under the configured byte limit. Network egress policy remains required in production.

## Egress governance

| Variable | Default | Purpose |
|---|---:|---|
| `REFAAT_EGRESS_POLICY` | `none` | `none` installs nothing (byte-identical); `allowlist` installs the default-deny allowlist policy over every outbound httpx seam |
| `REFAAT_EGRESS_ALLOWLIST` | `[]` | JSON list of `{host, port?, schemes?, purposes?}` rules the runtime is permitted to reach |

With the policy off the runtime is byte-identical. With `allowlist` set, an un-allowlisted destination is denied before the socket opens at every model, embedding, graph, multimodal, fine-tuning, connector, webhook, and S3-endpoint seam, a policy-read failure denies, and a configured model, embedding, or fine-tuning `base_url` host that is not on the allowlist fails startup. Rule hosts are casefolded; a URL with no explicit port normalizes to the scheme default before the port comparison; a leading-dot suffix such as `.example.com` matches the apex and any subdomain and is boundary-safe. See [Deployment hardening](deployment-hardening.md).

## Models and retrieval

| Variable | Default | Purpose |
|---|---:|---|
| `REFAAT_EMBEDDING_PROVIDER` | `deterministic` | `deterministic` or `openai_compatible` |
| `REFAAT_EMBEDDING_MODEL` | `refaat-hash-v1` | Embedding model identifier |
| `REFAAT_EMBEDDING_DIMENSIONS` | `384` | Vector dimension |
| `REFAAT_GENERATION_PROVIDER` | `extractive` | `extractive` or `openai_compatible` |
| `REFAAT_GENERATION_MODEL` | `extractive-v1` | Generation model identifier |
| `REFAAT_GRAPH_PROVIDER` | `cooccurrence` | `cooccurrence` or `openai_compatible` |
| `REFAAT_GRAPH_MAX_ENTITIES` | `80` | Maximum entities the co-occurrence graph extractor keeps per document |
| `REFAAT_GRAPH_MIN_TOKEN_LENGTH` | `5` | Minimum token length for a co-occurrence graph entity |
| `REFAAT_DEFAULT_LANGUAGE` | `en` | Fallback language for lexical stopwords and the co-occurrence graph when a document has no `language` metadata tag |
| `REFAAT_STOPWORDS_BY_LANGUAGE` | unset | Per-project override or extension of the built-in graph-extractor stopword registry, a mapping of short language code to a list of stopwords; merges onto the shipped defaults, unset keeps them byte-identical. Also settable in code via `RuntimeBuilder.with_stopwords({...})` |
| `REFAAT_OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint root |
| `REFAAT_OPENAI_API_KEY` | unset | Provider credential |
| `REFAAT_EMBEDDING_BASE_URL` | unset | Per-role embedding endpoint; falls back to `REFAAT_OPENAI_BASE_URL` |
| `REFAAT_EMBEDDING_API_KEY` | unset | Per-role embedding credential; falls back to `REFAAT_OPENAI_API_KEY` |
| `REFAAT_EMBEDDING_SEND_DIMENSIONS` | `true` | Send the `dimensions` field; disable for providers that reject it |
| `REFAAT_GENERATION_BASE_URL` | unset | Per-role generation endpoint; falls back to `REFAAT_OPENAI_BASE_URL` |
| `REFAAT_GENERATION_API_KEY` | unset | Per-role generation credential; falls back to `REFAAT_OPENAI_API_KEY` |
| `REFAAT_MODEL_MAX_RETRIES` | `2` | Bounded retries on rate-limit and transient model errors |
| `REFAAT_MODEL_RETRY_BASE_DELAY_SECONDS` | `0.25` | Base delay for model retry backoff |
| `REFAAT_JUDGE_PROVIDER` | `none` | `ModelJudge` backend: `none` (deterministic `Faithfulness` proxy) or `generator` |
| `REFAAT_DEFAULT_CHUNK_TOKENS` | `420` | Target chunk size |
| `REFAAT_CHUNK_OVERLAP_TOKENS` | `64` | Adjacent overlap; must be smaller than chunk size |
| `REFAAT_CHUNKING_STRATEGY` | `flat` | `flat`, `parent_child`, `sentence_window`, or a registered `refaat.chunkers` plugin name |
| `REFAAT_SENTENCE_WINDOW_RADIUS` | `1` | Sentence-window neighbor radius, range 0 to 8 |
| `REFAAT_QUERY_TOKEN_BUDGET` | `4000` | Default packed-context budget |
| `REFAAT_QUERY_TOP_K` | `12` | Default selected chunk count |
| `REFAAT_SEMANTIC_WEIGHT` | `0.55` | Semantic fusion input |
| `REFAAT_LEXICAL_WEIGHT` | `0.30` | Lexical fusion input |
| `REFAAT_GRAPH_WEIGHT` | `0.15` | Graph fusion input |
| `REFAAT_FUSION_STRATEGY` | `rrf` | Candidate fusion: `rrf` (Reciprocal Rank Fusion) or `linear` (weighted sum) |
| `REFAAT_RRF_K` | `60` | RRF rank constant; a larger value flattens rank influence |
| `REFAAT_AUTOCUT_ENABLED` | `false` | Trim candidates at the largest relative score drop before packing |
| `REFAAT_AUTOCUT_THRESHOLD` | `0.5` | Minimum relative gap for autocut to fire |
| `REFAAT_BM25_K1` | `1.2` | BM25 term-frequency saturation, range 0 to 5 |
| `REFAAT_BM25_B` | `0.75` | BM25 length-normalization, range 0 to 1 |
| `REFAAT_QUERY_EXPANSION` | `true` | Retrieve each query variant and RRF-merge the rankings |
| `REFAAT_MAX_QUERY_VARIANTS` | `4` | Cap on retrieval variants including the original, range 1 to 8 |
| `REFAAT_QUERY_TRANSFORM_PROVIDER` | `sub_query` | Pre-retrieval transform: `sub_query` (clause decomposition), `hyde` (Hypothetical Document Embeddings), `query_rewrite` (single restated query), `multi_query` (RRF-merged phrasings), or `step_back` (broader concept query) |
| `REFAAT_RERANKER_PROVIDER` | `none` | Post-fusion reranker: `none`, `lexical` (offline query-term coverage), or `cross_encoder` (model-backed, requires an injected scorer) |
| `REFAAT_COMPRESSION_ENABLED` | `false` | Context compression before packing |
| `REFAAT_COMPRESSION_MIN_SALIENCE` | `0.0` | Minimum salience a sentence must reach to survive compression, range 0 to 1 |
| `REFAAT_HYPE_ENABLED` | `false` | Ingestion-time hypothetical prompt embeddings |
| `REFAAT_RAPTOR_ENABLED` | `false` | Ingestion-time recursive summary tree |
| `REFAAT_GRAPH_TERM_INDEX_ENABLED` | `false` | Opt-in term-indexed graph read: fetch only the query-relevant subgraph instead of the full-corpus scan. Byte-identical when off. See [Adaptive context](adaptive-context.md) |
| `REFAAT_GRAPH_TERM_INDEX_MAX_QUERY_CHARS` | `256` | A query longer than this falls back to the full graph scan, so the substring candidate set stays bounded |
| `REFAAT_DERIVATION_MANIFEST_ENABLED` | `false` | Record a per-corpus derivation manifest (the materialization recipe, row counts, and a content checksum) refreshed on every ingest, delete, and erase |
| `REFAAT_CONTEXTUAL_RETRIEVAL` | `false` | Prefix the title before embedding each chunk at ingestion |
| `REFAAT_ABSTAIN_BELOW_SCORE` | `0.0` | Abstain when the top hit is below this score; `0` disables, range 0 to 1 |
| `REFAAT_CITATION_POLICY` | `optional` | Refuse a zero-citation answer: `optional` serves it, `required_when_evidence` refuses when evidence was retrieved, `required` refuses always |
| `REFAAT_ACL_OVERFETCH_FACTOR` | `4` | Under an active policy, over-fetch `limit * factor` candidates so authorized results are not starved by top_k truncation in an ACL-dense corpus |
| `REFAAT_ACL_OVERFETCH_HARD_CAP` | `2000` | Hard ceiling on the ACL over-fetch escalation |
| `REFAAT_ROUTER_LONG_QUERY_TOKENS` | `24` | Router: token count above which a query is treated as long |
| `REFAAT_ROUTER_MULTI_CLAUSE_THRESHOLD` | `2` | Router: clause count that marks a multi-clause query |
| `REFAAT_ROUTER_ENTITY_THRESHOLD` | `2` | Router: entity count that marks an entity-heavy query |
| `REFAAT_CACHE_TTL_SECONDS` | `3600` | Answer-cache expiry |
| `REFAAT_SEMANTIC_CACHE_THRESHOLD` | `0.96` | Minimum cosine similarity for semantic CAG |

Retrieval weights must be non-negative and have a positive total. The runtime normalizes them before fusion. The default fusion strategy is Reciprocal Rank Fusion, which combines the lexical, semantic, and graph rankings by position; `linear` selects the weighted-sum strategy instead. Autocut is opt-in and never empties a non-empty candidate list. The offline lexical ranker is BM25; `REFAAT_BM25_K1` and `REFAAT_BM25_B` tune its term-frequency saturation and length normalization. The defaults reproduce the previous fixed scorer exactly. Query expansion is a deterministic, model-free pre-retrieval stage: a compound query is decomposed into one sub-query per content-bearing clause, each variant is retrieved, and the per-variant rankings are merged with Reciprocal Rank Fusion and normalized by divide-by-max into the [0, 1] candidate scores the reranker and budgeter consume. An ordinary single-clause question yields only the original variant, so its retrieval path is byte-identical to the single-query behavior; disabling expansion or capping `REFAAT_MAX_QUERY_VARIANTS` at 1 has the same effect. `REFAAT_QUERY_TRANSFORM_PROVIDER` chooses which transform runs at that pre-retrieval seam: the default `sub_query` decomposition, or `hyde`, which pairs the query with a hypothetical answer whose embedding replaces the semantic search vector while the original query still drives lexical search. Under `hyde` the offline default is deterministic and model-free; a configured OpenAI-compatible generator writes the hypothetical instead. The hypothetical is embed-only: it is never lexically searched, retrieved, or cited, and the transform identity folds into the answer-cache scope. Three further providers sit behind the same seam: `query_rewrite` restates the query into a single search query, `multi_query` fans out to several phrasings merged by Reciprocal Rank Fusion, and `step_back` pairs the query with a broader concept query. Each keeps a deterministic offline default (the `query_rewrite` and `multi_query` defaults keep the single-query path byte-identical) and routes to a model-backed variant only when an OpenAI-compatible generator is configured. In every case the variant text feeds only semantic and lexical search over stored chunks, never becomes a chunk, and is never cited, and the transform identity folds into the answer-cache scope.

## Model-call cache and run budget

| Variable | Default | Purpose |
|---|---:|---|
| `REFAAT_MODEL_CALL_CACHE_ENABLED` | `false` | Serve identical embedding and generation calls from the `model_call_cache` table |
| `REFAAT_ANSWER_CACHE_BACKEND` | `sql` | `sql` or `redis`; selects the CAG answer-cache backend. See [Cache backends](cache-backends.md) |
| `REFAAT_MODEL_CALL_CACHE_BACKEND` | `sql` | `sql` or `redis`; selects the model-call cache backend. `redis` requires `REFAAT_MODEL_CALL_CACHE_ENABLED=true` |
| `REFAAT_REDIS_TIMEOUT_SECONDS` | `1.0` | Connect and socket timeout for a Redis cache backend, so an unreachable server fails fast |
| `REFAAT_RUN_TOKEN_BUDGET` | unset | Hard per-run token ceiling; a multi-attempt loop stops and returns the best result so far |
| `REFAAT_RLM_MAX_INPUT_CHUNKS` | unset | Fail an RLM run closed before it loads the corpus when the corpus exceeds this many chunks; unset is unbounded |
| `REFAAT_RUN_COST_BUDGET_USD` | unset | Hard per-run cost ceiling; requires priced models via `REFAAT_MODEL_PRICES` |
| `REFAAT_TOOL_TIMEOUT_SECONDS` | unset | Per-tool execution deadline; unset runs a tool inline (byte-identical). See [Execution resilience](resilience.md) |
| `REFAAT_TOOL_TIMEOUT_MAX_WORKERS` | `8` | Concurrent bound on abandoned tool workers; a full pool reports a confirmed no-effect rather than starting |
| `REFAAT_CIRCUIT_BREAKER_BACKEND` | `none` | `none` or `in_memory`; `in_memory` fast-fails a wedged provider endpoint. Byte-identical when `none`. See [Execution resilience](resilience.md) |
| `REFAAT_CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `5` | Consecutive contributing failures that open a provider endpoint circuit |
| `REFAAT_CIRCUIT_BREAKER_RESET_TIMEOUT_SECONDS` | `30.0` | How long an open circuit fast-fails before admitting one half-open probe |
| `REFAAT_CIRCUIT_BREAKER_HALF_OPEN_DEADLINE_SECONDS` | `60.0` | A half-open probe that never reports self-heals after this, so a lost probe cannot wedge the endpoint |
| `REFAAT_CIRCUIT_BREAKER_COUNT_429` | `false` | Whether a 429 contributes to opening the circuit |
| `REFAAT_MODEL_PRICES` | `{}` | Per-model `{input_per_1k, output_per_1k}` rates used to price a cost budget |

Both the model-call cache and the run budget are opt-in and byte-identical to the previous behavior when unset. The cache wraps only real provider adapters and is keyed so a hit can never be served across a different backend, model, or workspace. A cost budget over a real model fails startup unless that model has a positive priced entry, so an unpriced budget cannot silently pass. See [Model-call cache and run budget](model-cache.md) for the full contract.

## Resource-gated features

These features are off by default and activate only when their real resource is configured. The offline default path is byte-identical.

| Variable | Default | Purpose |
|---|---:|---|
| `REFAAT_FINE_TUNING_PROVIDER` | `deterministic` | `deterministic` provenance stub, or `openai` |
| `REFAAT_FINE_TUNING_BASE_URL` | unset | Fine-tuning endpoint root |
| `REFAAT_FINE_TUNING_API_KEY` | unset | Fine-tuning credential; falls back to `REFAAT_OPENAI_API_KEY` |
| `REFAAT_FINE_TUNE_POLL_SECONDS` | `60` | Delay before the worker re-polls a still-running provider fine-tune job |
| `REFAAT_IMAGE_CAPTIONER` | `none` | `none` or `openai_vision`; captions an ingested image |
| `REFAAT_MULTIMODAL_GENERATION` | `none` | `none` or `openai_vision`; answers over image pixels |
| `REFAAT_MULTIMODAL_MODEL` | `gpt-4o-mini` | Vision model identifier |
| `REFAAT_MULTIMODAL_BASE_URL` | unset | Vision endpoint root; falls back to `REFAAT_OPENAI_BASE_URL` |
| `REFAAT_MULTIMODAL_API_KEY` | unset | Vision credential; falls back to `REFAAT_OPENAI_API_KEY` |
| `REFAAT_IMAGE_MAX_BYTES` | `8388608` | Maximum ingested image size |
| `REFAAT_CODE_SANDBOX` | `deny` | `deny`, `safe_expression`, or `subprocess` for the RLM code phase |
| `REFAAT_CODE_SANDBOX_MAX_NODES` | `256` | AST node ceiling for the `safe_expression` sandbox |
| `REFAAT_CODE_SANDBOX_MAX_OUTPUT_BYTES` | `65536` | Output byte ceiling for a sandboxed run |
| `REFAAT_CODE_SANDBOX_MAX_WALL_SECONDS` | `2.0` | Wall-clock ceiling for a sandboxed run |

The `openai_vision` captioner and generation providers, and the `openai` fine-tuning provider, require their credential in production, validated at startup. The code sandbox defaults to `deny`, which executes nothing. See [Multimodal generation](multimodal.md), [Governed fine-tuning](fine-tuning.md), and [Sandboxed code in the bounded RLM](sandboxed-rlm.md).

## API and access control

| Variable | Default | Purpose |
|---|---:|---|
| `REFAAT_BOOTSTRAP_TOKEN` | unset | Enables initial workspace provisioning |
| `REFAAT_API_KEY_PEPPER` | development placeholder | Server-side API-key hashing pepper |
| `REFAAT_CORS_ORIGINS` | `[]` | Allowed origins |
| `REFAAT_TRUSTED_HOSTS` | local/test hosts | Trusted host middleware values |
| `REFAAT_MAX_BODY_BYTES` | `20971520` | Maximum request body |
| `REFAAT_RATE_LIMIT_REQUESTS` | `120` | Requests per rate-limit window |
| `REFAAT_RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate-limit window |
| `REFAAT_RATE_LIMIT_TRUSTED_PROXIES` | `[]` | Proxy peers whose `X-Forwarded-For` is trusted |
| `REFAAT_RATE_LIMIT_MAX_BUCKETS` | `10000` | In-memory limiter active-bucket ceiling |
| `REFAAT_API_WORKERS` | `1` | Declared API worker or replica count; more than one requires Redis in production |
| `REFAAT_REDIS_URL` | unset | Distributed rate-limit backend, and the server for a Redis cache backend |

The edge rate limiter keys on the trusted client IP, never on the API key, so a caller cannot bypass the limit by rotating forged keys. The socket peer is authoritative; `X-Forwarded-For` is consulted only when that peer is in `REFAAT_RATE_LIMIT_TRUSTED_PROXIES`, every `X-Forwarded-For` header is read so a spoofed second header cannot hide, each entry is parsed and canonicalized as an IP (a malformed entry is dropped and an IPv6 address collapses to one spelling), and then the right-most entry that is not itself a trusted proxy is used. Callers behind one shared NAT share a bucket, so per-tenant and true multi-process limiting belong on the Redis backend. The in-memory limiter sweeps only empty and stale buckets; it never evicts a bucket whose window is still active, so cycling addresses cannot reset an active count, and when it is at capacity with every bucket active it rejects a new address rather than forgetting an existing one.

Because that limiter is per process, `REFAAT_API_WORKERS` greater than one selects the shared Redis limiter regardless of environment: the app refuses to start with more than one worker unless `REFAAT_REDIS_URL` is set, so a multi-process deployment enforces one shared limit. The `serve` command launches uvicorn with the configured `REFAAT_API_WORKERS` (except under reload, which forces a single process), so the setting is honored rather than only validated. An externally spawned multi-process launch (gunicorn, or an external `uvicorn --workers`) starts each worker as its own process with `REFAAT_API_WORKERS` at its default of one, so the app cannot detect the peers: such a deployment must set `REFAAT_REDIS_URL` itself, otherwise each worker runs an independent in-memory limiter and the shared limit is not enforced.

Production rejects SQLite, automatic schema creation, wildcard CORS, the default/short API-key pepper, and a configured bootstrap token shorter than 24 characters.

## Sessions, jobs, and workers

| Variable | Default | Purpose |
|---|---:|---|
| `REFAAT_SESSION_COMPACTION_TURNS` | `20` | Turn count that triggers compaction |
| `REFAAT_RETAINED_SESSION_TURNS` | `8` | Recent turns kept after compaction |
| `REFAAT_WORKER_ID` | `hostname-pid-random` | Unique worker lease identity |
| `REFAAT_WORKER_POLL_SECONDS` | `1.0` | Idle poll interval |
| `REFAAT_JOB_LEASE_SECONDS` | `60` | Work lease duration |
| `REFAAT_JOB_MAX_ATTEMPTS` | `5` | Default retry limit |
| `REFAAT_SHADOW_RUN_RETENTION_DAYS` | `30` | Age after which the `shadow_runs.purge` job trims shadow telemetry |
| `REFAAT_OUTBOX_RETENTION_DAYS` | `7` | Age after which delivered outbox rows are purged |

The retained-turn count must be smaller than the compaction threshold. Every production worker needs a unique lease identity. The default is generated per process from the hostname, the PID, and a random suffix, so two replicas never share a lease owner even inside one host. Setting `REFAAT_WORKER_ID` overrides it: the Helm chart injects a per-pod value from the downward API, and a scaled Compose deployment relies on the generated default rather than a shared constant.

## Plugins, skills, logs, and telemetry

| Variable | Default | Purpose |
|---|---|---|
| `REFAAT_PLUGIN_ALLOWLIST` | `[]` | Approved entry-point names; empty disables discovery |
| `REFAAT_SKILL_ROOT` | `./data/skills` | Generated skill-project root |
| `REFAAT_LOG_LEVEL` | `INFO` | Application log level |
| `REFAAT_LOG_JSON` | `true` | Structured JSON logs |
| `REFAAT_OTEL_ENABLED` | `false` | Enable OpenTelemetry instrumentation |
| `REFAAT_OTEL_SERVICE_NAME` | `refaat` | Service name |
| `REFAAT_OTEL_EXPORTER_ENDPOINT` | unset | OTLP gRPC endpoint |

Enabling telemetry without the `otel` extra raises an actionable dependency error.

## Validation

`RuntimeBuilder` calls `Settings.validate_for_startup()` automatically. Custom composition code should do the same before creating external resources.

```python
from refaat.config import Settings

settings = Settings()
settings.validate_for_startup()
```
