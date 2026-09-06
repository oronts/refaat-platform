# Model providers

Refaat is not tied to one model vendor. The embedding, generation, and graph-extraction adapters speak the OpenAI-compatible HTTP protocol, which is the de-facto standard exposed by OpenAI, Mistral, Together, Groq, Fireworks, and local servers such as Ollama, vLLM, and LM Studio. Point the base URL at any of them. For a provider that is not OpenAI-compatible, inject a custom adapter through the builder (see the last section).

Two independent decisions control the model layer:

1. The provider kind per role. Embeddings are `deterministic` or `openai_compatible`, generation is `extractive` or `openai_compatible`, and graph extraction is `cooccurrence` or `openai_compatible`. The offline kinds need no network and run in CI.
2. The endpoint per role. A base URL and API key. Embeddings and generation can point at different providers.

## Shared endpoint

The simplest configuration uses one endpoint for every role.

```bash
REFAAT_EMBEDDING_PROVIDER=openai_compatible
REFAAT_GENERATION_PROVIDER=openai_compatible
REFAAT_OPENAI_BASE_URL=https://api.openai.com/v1
REFAAT_OPENAI_API_KEY=sk-...
REFAAT_EMBEDDING_MODEL=text-embedding-3-small
REFAAT_GENERATION_MODEL=gpt-4o-mini
```

## Per-role endpoints

Embeddings and generation are configured independently. Each per-role base URL and key falls back to the shared `REFAAT_OPENAI_BASE_URL` and `REFAAT_OPENAI_API_KEY` when unset, so you can keep embeddings on one provider and run generation on another.

```bash
# Embeddings on OpenAI
REFAAT_EMBEDDING_PROVIDER=openai_compatible
REFAAT_EMBEDDING_BASE_URL=https://api.openai.com/v1
REFAAT_EMBEDDING_API_KEY=sk-openai-...
REFAAT_EMBEDDING_MODEL=text-embedding-3-small

# Generation on Mistral
REFAAT_GENERATION_PROVIDER=openai_compatible
REFAAT_GENERATION_BASE_URL=https://api.mistral.ai/v1
REFAAT_GENERATION_API_KEY=...
REFAAT_GENERATION_MODEL=mistral-large-latest
```

The graph extractor uses the generation endpoint, so a per-role generation base URL and key also apply to model-assisted entity and relation extraction.

## The embedding dimensions field

OpenAI embedding endpoints accept a `dimensions` field that truncates the output vector. Many other providers, including Mistral, reject an unknown `dimensions` field. Refaat sends it only when `REFAAT_EMBEDDING_SEND_DIMENSIONS` is true, which is the default. Set it to false for providers that reject it, and set `REFAAT_EMBEDDING_DIMENSIONS` to the native output size of the model you select, because that value defines the vector length the search index expects.

```bash
REFAAT_EMBEDDING_PROVIDER=openai_compatible
REFAAT_EMBEDDING_BASE_URL=https://api.mistral.ai/v1
REFAAT_EMBEDDING_API_KEY=...
REFAAT_EMBEDDING_MODEL=mistral-embed
REFAAT_EMBEDDING_SEND_DIMENSIONS=false
REFAAT_EMBEDDING_DIMENSIONS=1024
```

## Provider examples

Every example below is an OpenAI-compatible endpoint. Only the base URL, key, and model change. Refaat speaks the OpenAI-compatible protocol, so any endpoint that implements it works by configuration alone, with no code change and no per-provider adapter class. Each configuration is verified offline by composing the adapter from settings and asserting the base URL and model. That is a claim about the configuration path, not a claim that Refaat has been tested against each provider's live API.

### OpenAI

```bash
REFAAT_GENERATION_PROVIDER=openai_compatible
REFAAT_GENERATION_BASE_URL=https://api.openai.com/v1
REFAAT_GENERATION_API_KEY=sk-...
REFAAT_GENERATION_MODEL=gpt-4o-mini
REFAAT_EMBEDDING_PROVIDER=openai_compatible
REFAAT_EMBEDDING_BASE_URL=https://api.openai.com/v1
REFAAT_EMBEDDING_API_KEY=sk-...
REFAAT_EMBEDDING_MODEL=text-embedding-3-small
```

### Mistral

```bash
REFAAT_GENERATION_PROVIDER=openai_compatible
REFAAT_GENERATION_BASE_URL=https://api.mistral.ai/v1
REFAAT_GENERATION_API_KEY=...
REFAAT_GENERATION_MODEL=mistral-large-latest
REFAAT_EMBEDDING_PROVIDER=openai_compatible
REFAAT_EMBEDDING_BASE_URL=https://api.mistral.ai/v1
REFAAT_EMBEDDING_API_KEY=...
REFAAT_EMBEDDING_MODEL=mistral-embed
REFAAT_EMBEDDING_SEND_DIMENSIONS=false
REFAAT_EMBEDDING_DIMENSIONS=1024
```

### Together

```bash
REFAAT_GENERATION_PROVIDER=openai_compatible
REFAAT_GENERATION_BASE_URL=https://api.together.xyz/v1
REFAAT_GENERATION_API_KEY=...
REFAAT_GENERATION_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo
```

### Groq

```bash
REFAAT_GENERATION_PROVIDER=openai_compatible
REFAAT_GENERATION_BASE_URL=https://api.groq.com/openai/v1
REFAAT_GENERATION_API_KEY=...
REFAAT_GENERATION_MODEL=llama-3.3-70b-versatile
```

### Ollama (local)

Ollama serves an OpenAI-compatible endpoint on port 11434 and ignores the API key.

```bash
REFAAT_GENERATION_PROVIDER=openai_compatible
REFAAT_GENERATION_BASE_URL=http://127.0.0.1:11434/v1
REFAAT_GENERATION_API_KEY=ollama
REFAAT_GENERATION_MODEL=llama3.1
REFAAT_EMBEDDING_PROVIDER=openai_compatible
REFAAT_EMBEDDING_BASE_URL=http://127.0.0.1:11434/v1
REFAAT_EMBEDDING_API_KEY=ollama
REFAAT_EMBEDDING_MODEL=nomic-embed-text
REFAAT_EMBEDDING_SEND_DIMENSIONS=false
REFAAT_EMBEDDING_DIMENSIONS=768
```

### vLLM (self-hosted)

```bash
REFAAT_GENERATION_PROVIDER=openai_compatible
REFAAT_GENERATION_BASE_URL=http://vllm.internal:8000/v1
REFAAT_GENERATION_API_KEY=local
REFAAT_GENERATION_MODEL=your-served-model
```

### Anthropic-compatible endpoints

Refaat's model adapters call the OpenAI chat-completions and embeddings shapes. The direct Anthropic Messages API uses a different request shape, so point Refaat at an OpenAI-compatible gateway in front of the model rather than at the native Anthropic endpoint. A gateway that exposes the OpenAI chat-completions shape (for example a self-hosted proxy or an aggregator that fronts several vendors) works like any other row above: set the base URL to the gateway and the model to the served model name.

```bash
REFAAT_GENERATION_PROVIDER=openai_compatible
REFAAT_GENERATION_BASE_URL=https://your-openai-compatible-gateway/v1
REFAAT_GENERATION_API_KEY=...
REFAAT_GENERATION_MODEL=your-served-model
```

For a provider that is not OpenAI-compatible and has no such gateway, inject a custom adapter through the builder as shown in the last section.

## Retries and timeouts

Transient provider failures are retried automatically. A rate-limit or transient server response (429, 500, 502, 503, 504) and a connection error are retried with exponential backoff, up to `REFAAT_MODEL_MAX_RETRIES` times (default 2) starting from `REFAAT_MODEL_RETRY_BASE_DELAY_SECONDS` seconds (default 0.25). A non-retryable response such as 400 or 401 fails immediately, and `REFAAT_REQUEST_TIMEOUT_SECONDS` bounds each attempt.

## Offline default

With no configuration, every role runs offline and deterministically: hash embeddings, extractive generation, and a co-occurrence graph, backed by SQLite. This is the default used by tests, CI, and local development. It ingests, embeds, builds a graph, retrieves, generates a grounded answer with citations, and serves a version-safe cache hit with no paid provider. Switch any single role to `openai_compatible` when you want a real model for that role, and leave the others deterministic.

## Language awareness

The vector path is language-neutral when paired with a multilingual embedding and generation model, so a query in one language can retrieve evidence written in another. The co-occurrence graph extractor is language-aware through a per-language stopword registry in `core/text.py`: English is the default, with curated German, French, Spanish, and Arabic sets. Tag a document with a `language` metadata field (`metadata={"language": "de"}`) and the graph extractor selects that language's stopword set, otherwise it falls back to `REFAAT_DEFAULT_LANGUAGE` (`en` by default). The tokenizer is Unicode-aware for space-delimited scripts. An English or untagged corpus resolves to the exact English set and is unchanged. A parallel per-language query-stopword registry ships alongside it as scaffolding; the lexical and query-transform path does not yet consume it, so query-term filtering uses the English set regardless of language. Known limitations: language-aware lexical filtering, word segmentation for space-free scripts such as CJK (a run tokenizes as one token), and non-Latin sentence splitting are future enhancements, not current claims.

The registry ships as defaults, not a fixed list. Each project can override a shipped language or add a new one without editing the library, through the `stopwords_by_language` setting (`REFAAT_STOPWORDS_BY_LANGUAGE`, a mapping of short language code to a list of stopwords) or `RuntimeBuilder.with_stopwords({...})`. An override merges onto the shipped defaults, codes normalize to their primary subtag, and words casefold to match the tokenizer. It is additive: with nothing configured the runtime passes the module default to the extractor and every unconfigured corpus is byte-identical. Use case: a project ingesting Italian content supplies its own Italian stopwords, for example `with_stopwords({"it": ["della", "sono", "come", "questo"]})`.

## Model-lift query transform (HyDE)

`REFAAT_QUERY_TRANSFORM_PROVIDER=hyde` turns on Hypothetical Document Embeddings at the pre-retrieval seam. It is
model-lift: with the offline extractive stack it runs a deterministic, model-free transform that builds the hypothetical
from the query's content terms, so it works in CI with no provider. When `REFAAT_GENERATION_PROVIDER=openai_compatible`,
the same knob routes to a model-backed variant that writes the hypothetical passage with the configured generator through
a context-free generation call, and blank or failed generation degrades to the original-only path. The hypothetical only
moves the semantic embedding target: it is never lexically searched, retrieved, or cited, so citations stay grounded in
real indexed chunks. The extractive generator is never routed into the model HyDE path. See
[Adaptive context](adaptive-context.md) for the retrieval-side details.

## More model-lift query transforms

Three more transforms sit behind the same seam and follow the same rule: with the offline extractive stack each runs a deterministic, model-free default that works in CI with no provider, and only `REFAAT_GENERATION_PROVIDER=openai_compatible` routes to the model-backed variant. The extractive generator is never routed into a model path.

- `REFAAT_QUERY_TRANSFORM_PROVIDER=query_rewrite` restates the query for retrieval. Offline it returns the normalized query as a single variant, so the single-query path stays byte-identical; with a real generator it returns the model rewrite.
- `REFAAT_QUERY_TRANSFORM_PROVIDER=multi_query` generates several phrasings and merges the per-variant rankings with Reciprocal Rank Fusion. Offline it returns the single original variant; with a real generator it fans out, capped by `REFAAT_MAX_QUERY_VARIANTS`.
- `REFAAT_QUERY_TRANSFORM_PROVIDER=step_back` pairs the query with a broader concept query and merges both. Offline the broadening is deterministic (content terms only); with a real generator the model writes the broader question.

In all three the variant text feeds only semantic and lexical search over stored chunks. A rewritten, expanded, or broader query never becomes a chunk and is never cited, so citations stay grounded in real indexed chunks, and each transform folds its identity into the answer-cache scope. See [Adaptive context](adaptive-context.md) for the retrieval-side details.

## Custom adapters for any provider or framework

When a provider is not OpenAI-compatible, or when you want to reuse an embedder or a language model from another framework, inject the adapter directly. Any object implementing the typed `Embedder`, `Generator`, or `GraphExtractor` port is accepted, and it overrides the configured provider for that role.

```python
from refaat.application.runtime import RuntimeBuilder
from refaat.config import Settings


class MyProviderEmbedder:
    @property
    def model_id(self) -> str:
        return "my-provider-embed"

    @property
    def dimensions(self) -> int:
        return 1024

    def embed(self, texts):
        # call your provider or framework here, return one vector per text
        ...


runtime = RuntimeBuilder(Settings()).with_embedder(MyProviderEmbedder()).build()
```

`with_generator` and `with_graph_extractor` work the same way. This is how Refaat stays framework-neutral: the OpenAI-compatible path covers most providers by configuration alone, and the port injection covers everything else, including a wrapper around a LangChain, LlamaIndex, or vendor SDK object. See [Custom extensions](guides/custom-extensions.md) for the full port contracts.
