from __future__ import annotations

import json
from typing import Any, cast

from refaat.application.services import QueryService
from refaat.config import Settings
from refaat.core.harness import BASE_SYSTEM_PROMPT_HASH
from refaat.core.retrieval import AdaptiveRetriever
from refaat.core.text import sha256_text
from refaat.domain.models import QueryRequest
from refaat.infrastructure.query_transform import DeterministicHydeQueryTransform, DeterministicQueryTransform


def _policy() -> Any:
    settings = Settings(api_key_pepper="test-pepper-with-enough-entropy")
    retriever = AdaptiveRetriever(cast(Any, None), cast(Any, None), settings)
    return retriever.resolve_policy(QueryRequest(workspace_id="w", corpus_id="c", query="q"), {})


def _hash(policy: Any, query_transform: str) -> str:
    return QueryService._context_hash(
        policy=policy,
        profile={},
        supplemental={},
        session_summary=None,
        embedding_model="e",
        generation_model="g",
        reranker="none",
        query_transform=query_transform,
        guidance=None,
        response_schema=None,
    )


def test_folded_token_equals_the_active_transform_cache_id() -> None:
    assert DeterministicQueryTransform().cache_id == "deterministic-v1"
    assert DeterministicHydeQueryTransform().cache_id == "hyde-deterministic-v1"


def test_swapping_to_hyde_changes_the_context_hash() -> None:
    policy = _policy()
    sub_query = _hash(policy, DeterministicQueryTransform().cache_id)
    hyde = _hash(policy, DeterministicHydeQueryTransform().cache_id)
    assert sub_query != hyde


def test_none_transform_folds_the_exact_canonical_payload() -> None:
    policy = _policy()
    expected_payload = {
        "base_prompt_hash": BASE_SYSTEM_PROMPT_HASH,
        "embedding_model": "e",
        "generation_model": "g",
        "policy": policy.as_json(),
        "profile": {},
        "query_transform": "none",
        "reranker": "none",
        "session_summary": None,
        "supplemental_harness": {},
    }
    expected = sha256_text(json.dumps(expected_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    assert _hash(policy, "none") == expected
    assert _hash(policy, "none") != _hash(policy, DeterministicQueryTransform().cache_id)
    assert _hash(policy, "none") != _hash(policy, DeterministicHydeQueryTransform().cache_id)


def test_same_config_yields_identical_hash() -> None:
    policy = _policy()
    cache_id = DeterministicHydeQueryTransform().cache_id
    assert _hash(policy, cache_id) == _hash(policy, cache_id)
