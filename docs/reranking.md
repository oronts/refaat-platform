# Reranking

Retrieval fuses lexical, vector, and graph scores into a ranked candidate set. A reranker is an optional stage that rescreens those candidates before the context is packed under the token budget, so the most relevant evidence survives when the budget is tight. It is off by default and adds no cost when absent.

## Enabling the built-in reranker

The built-in `lexical` reranker is deterministic and offline: it rescreens the fused candidates by query-term coverage and reorders them, with no model call, so it runs in CI and local development.

```bash
REFAAT_RERANKER_PROVIDER=lexical
```

When a reranker is active, the query response `routes` include `rerank`.

## The built-in cross-encoder reranker

The `cross_encoder` reranker is model-backed but dependency-light: it rescreens the fused candidates with a cross-encoder relevance score, yet no cross-encoder model library is bundled into the core. The model is reached through an injected `CrossEncoderScorer` seam, a callable that scores `(query, documents)` pairs and returns one relevance score per document. This keeps the offline gate deterministic while letting you plug in a real sentence-transformers cross-encoder or a hosted rerank endpoint.

```python
from collections.abc import Sequence

from refaat.application.runtime import RuntimeBuilder
from refaat.config import Settings


class SentenceTransformersScorer:
    @property
    def model_id(self) -> str:
        return "bge-reranker-large"

    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        # call your cross-encoder here and return one score per document, aligned with the input order
        ...


runtime = (
    RuntimeBuilder(Settings(reranker_provider="cross_encoder"))
    .with_cross_encoder_scorer(SentenceTransformersScorer())
    .build()
)
```

Selecting `reranker_provider="cross_encoder"` without an injected scorer fails loudly at build time rather than falling back silently or constructing a network client. You can also inject a fully built `CrossEncoderReranker` through `with_reranker`, which takes precedence over the config value. The reranker degrades to the input order on an empty candidate set or when the scorer raises `DependencyUnavailableError`, so a scorer outage never crashes a query.

The cross-encoder `cache_id` is `cross_encoder:<model_id>`, so two cross-encoders with different models produce distinct answer-cache scopes.

## A custom or model-based reranker

Any object implementing the `Reranker` port is accepted, so a cross-encoder or a hosted rerank endpoint, for example Cohere Rerank or Jina, can replace the built-in one. Inject it through the builder.

```python
from refaat.application.runtime import RuntimeBuilder
from refaat.config import Settings


class MyCrossEncoderReranker:
    def rerank(self, query, hits):
        # score each hit against the query with your model, return them reordered
        ...


runtime = RuntimeBuilder(Settings()).with_reranker(MyCrossEncoderReranker()).build()
```

The reranker receives the query and the fused `SearchHit` candidates and returns them reordered. It runs after fusion and before token-budget packing, so it changes which evidence survives the budget without changing retrieval or the citation contract.

## Notes

- The active reranker's identity is part of the answer-cache identity, so a reranked answer and a non-reranked answer are cached separately, and changing the reranker does not serve a stale cached answer. The built-in reranker encodes its weight in a `cache_id`; a custom reranker can expose a `cache_id` property to encode its own tunables, otherwise its class name is used.
- A reranker never mutates the corpus or the index; it only reorders in-memory candidates for a single query.
