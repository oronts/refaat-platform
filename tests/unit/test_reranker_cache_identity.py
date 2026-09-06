from __future__ import annotations

import json
from collections.abc import Sequence

from refaat.application.services import QueryService
from refaat.core.harness import BASE_SYSTEM_PROMPT_HASH
from refaat.core.retrieval import RetrievalPolicy
from refaat.core.text import sha256_text
from refaat.infrastructure.reranking import CrossEncoderReranker, LexicalReranker


class _Scorer:
    def __init__(self, model_id: str) -> None:
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        return [0.0 for _ in documents]


_POLICY = RetrievalPolicy(
    top_k=12,
    candidate_limit=48,
    token_budget=4_000,
    semantic_weight=0.55,
    lexical_weight=0.30,
    graph_weight=0.15,
    fusion="rrf",
    rrf_k=60,
    autocut_enabled=False,
    autocut_threshold=0.5,
    metadata_filter=None,
    bm25_k1=1.2,
    bm25_b=0.75,
    query_expansion=True,
    max_query_variants=4,
    compression_enabled=False,
    compression_min_salience=0.0,
    expansion_mode="flat",
    window_radius=1,
)


def _hash(reranker: str) -> str:
    return QueryService._context_hash(
        policy=_POLICY,
        profile={},
        supplemental={},
        session_summary=None,
        embedding_model="emb-v1",
        generation_model="gen-v1",
        reranker=reranker,
        query_transform="none",
        guidance=None,
        response_schema=None,
    )


def test_none_reranker_context_hash_is_byte_identical() -> None:
    payload = {
        "base_prompt_hash": BASE_SYSTEM_PROMPT_HASH,
        "embedding_model": "emb-v1",
        "generation_model": "gen-v1",
        "policy": _POLICY.as_json(),
        "profile": {},
        "reranker": "none",
        "query_transform": "none",
        "session_summary": None,
        "supplemental_harness": {},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert _hash("none") == sha256_text(canonical)


def test_cross_encoder_cache_id_folds_and_differs_from_none() -> None:
    cross = CrossEncoderReranker(_Scorer("bge-reranker-large")).cache_id
    assert cross == "cross_encoder:bge-reranker-large"
    assert _hash(cross) != _hash("none")


def test_cross_encoder_model_id_changes_cache_scope() -> None:
    a = CrossEncoderReranker(_Scorer("model-a")).cache_id
    b = CrossEncoderReranker(_Scorer("model-b")).cache_id
    assert _hash(a) != _hash(b)


def test_cross_encoder_distinct_from_lexical() -> None:
    cross = CrossEncoderReranker(_Scorer("model-a")).cache_id
    lexical = LexicalReranker().cache_id
    assert _hash(cross) != _hash(lexical)
