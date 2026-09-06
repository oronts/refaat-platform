from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, cast

from refaat.application.runtime import RuntimeBuilder
from refaat.application.services import QueryService
from refaat.config import Settings
from refaat.core.harness import BASE_SYSTEM_PROMPT_HASH
from refaat.core.retrieval import AdaptiveRetriever
from refaat.core.text import content_terms, normalize_text, sha256_text, tokens
from refaat.domain.models import GenerationRequest, GenerationResult, QueryRequest
from refaat.infrastructure.generation import ExtractiveGenerator
from refaat.infrastructure.query_transform import (
    DeterministicMultiQueryTransform,
    DeterministicQueryTransform,
    DeterministicRewriteQueryTransform,
    DeterministicStepBackQueryTransform,
    ModelMultiQueryTransform,
    ModelRewriteQueryTransform,
    ModelStepBackQueryTransform,
)

QUERY = "How does rollback restore the prior artifact in production?"


class _FakeGenerator:
    def __init__(self, answer: str) -> None:
        self._answer = answer
        self.calls: list[GenerationRequest] = []

    @property
    def model_id(self) -> str:
        return "fake-tier2-model"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request)
        return GenerationResult(answer=self._answer, cited_chunk_ids=())

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        yield self._answer


# --- Technique 1: query rewriting ---------------------------------------------


def test_deterministic_rewrite_returns_normalized_single_original() -> None:
    variants = DeterministicRewriteQueryTransform().expand(QUERY, 4)
    assert len(variants) == 1
    variant = variants[0]
    assert variant.kind == "original"
    assert variant.text == normalize_text(QUERY)
    assert variant.embed_text is None


def test_deterministic_rewrite_cache_id_is_stable() -> None:
    assert DeterministicRewriteQueryTransform().cache_id == "rewrite-deterministic-v1"


def test_model_rewrite_returns_the_original_plus_the_rewrite() -> None:
    generator = _FakeGenerator("rollback production artifact restore procedure")
    variants = ModelRewriteQueryTransform(generator).expand(QUERY, 4)
    # The original plus the rewrite, so the rewrite actually reaches retrieval: a lone
    # variant is short-circuited to the original by the pre-retrieval seam.
    assert len(variants) == 2
    assert variants[0].kind == "original"
    assert variants[0].text == QUERY
    rewrite = variants[1]
    assert rewrite.kind == "rewrite"
    assert rewrite.text == "rollback production artifact restore procedure"
    assert rewrite.embed_text is None
    assert len(generator.calls) == 1
    assert generator.calls[0].context.hits == ()


def test_model_rewrite_blank_output_degrades_to_normalized_original() -> None:
    variants = ModelRewriteQueryTransform(_FakeGenerator("   ")).expand(QUERY, 4)
    assert len(variants) == 1
    assert variants[0].kind == "original"
    assert variants[0].text == normalize_text(QUERY)


def test_model_rewrite_cache_id_encodes_model_id() -> None:
    assert ModelRewriteQueryTransform(_FakeGenerator("x")).cache_id == "rewrite-model:fake-tier2-model"


# --- Technique 2: multi-query expansion ---------------------------------------


def test_deterministic_multi_query_returns_single_original_identity() -> None:
    variants = DeterministicMultiQueryTransform().expand(QUERY, 4)
    assert len(variants) == 1
    assert variants[0].kind == "original"
    assert variants[0].text == QUERY
    assert variants[0].embed_text is None


def test_deterministic_multi_query_cache_id_is_stable() -> None:
    assert DeterministicMultiQueryTransform().cache_id == "multiquery-deterministic-v1"


def test_model_multi_query_fans_out_original_plus_generated() -> None:
    generator = _FakeGenerator("what triggers a rollback\nhow is the prior artifact stored\nrollback recovery steps")
    variants = ModelMultiQueryTransform(generator).expand(QUERY, 8)
    assert len(variants) == 4
    assert variants[0].text == QUERY
    assert [variant.kind for variant in variants] == ["original", "original", "original", "original"]
    assert all(variant.embed_text is None for variant in variants)
    assert variants[1].text == "what triggers a rollback"
    assert len(generator.calls) == 1


def test_model_multi_query_total_capped_at_max_variants() -> None:
    generator = _FakeGenerator("a one\nb two\nc three\nd four\ne five")
    variants = ModelMultiQueryTransform(generator).expand(QUERY, 3)
    assert len(variants) == 3
    assert variants[0].text == QUERY


def test_model_multi_query_dedupes_and_drops_the_original() -> None:
    generator = _FakeGenerator(f"{QUERY}\nrollback recovery steps\nrollback recovery steps")
    variants = ModelMultiQueryTransform(generator).expand(QUERY, 8)
    texts = [variant.text for variant in variants]
    assert texts == [QUERY, "rollback recovery steps"]


def test_model_multi_query_blank_output_degrades_to_original_only() -> None:
    variants = ModelMultiQueryTransform(_FakeGenerator("   ")).expand(QUERY, 8)
    assert [variant.kind for variant in variants] == ["original"]


def test_model_multi_query_single_variant_budget_does_not_call_generator() -> None:
    generator = _FakeGenerator("unused\nvariants")
    variants = ModelMultiQueryTransform(generator).expand(QUERY, 1)
    assert [variant.kind for variant in variants] == ["original"]
    assert generator.calls == []


def test_model_multi_query_cache_id_encodes_model_id() -> None:
    assert ModelMultiQueryTransform(_FakeGenerator("x")).cache_id == "multiquery-model:fake-tier2-model"


# --- Technique 3: step-back prompting -----------------------------------------


def test_deterministic_stepback_yields_broader_second_variant() -> None:
    variants = DeterministicStepBackQueryTransform().expand(QUERY, 4)
    assert len(variants) == 2
    original, broader = variants
    assert original.kind == "original"
    assert original.text == QUERY
    assert original.embed_text is None
    assert broader.kind == "stepback"
    assert broader.embed_text is None
    assert broader.text != QUERY
    assert broader.text == " ".join(content_terms(QUERY))
    assert set(tokens(broader.text)) < set(tokens(QUERY))


def test_deterministic_stepback_cache_id_is_stable() -> None:
    assert DeterministicStepBackQueryTransform().cache_id == "stepback-deterministic-v1"


def test_deterministic_stepback_single_budget_short_circuits() -> None:
    variants = DeterministicStepBackQueryTransform().expand(QUERY, 1)
    assert [variant.kind for variant in variants] == ["original"]


def test_deterministic_stepback_no_content_terms_degrades_to_original_only() -> None:
    variants = DeterministicStepBackQueryTransform().expand("the of and to", 4)
    assert [variant.kind for variant in variants] == ["original"]


def test_model_stepback_returns_original_plus_generated_broader() -> None:
    generator = _FakeGenerator("What is artifact versioning?")
    variants = ModelStepBackQueryTransform(generator).expand(QUERY, 4)
    assert len(variants) == 2
    original, broader = variants
    assert original.kind == "original"
    assert broader.kind == "stepback"
    assert broader.text == "What is artifact versioning?"
    assert broader.embed_text is None
    assert len(generator.calls) == 1
    assert generator.calls[0].context.hits == ()


def test_model_stepback_blank_output_degrades_to_original_only() -> None:
    variants = ModelStepBackQueryTransform(_FakeGenerator("   ")).expand(QUERY, 4)
    assert [variant.kind for variant in variants] == ["original"]


def test_model_stepback_single_variant_budget_does_not_call_generator() -> None:
    generator = _FakeGenerator("broader question")
    variants = ModelStepBackQueryTransform(generator).expand(QUERY, 1)
    assert [variant.kind for variant in variants] == ["original"]
    assert generator.calls == []


def test_model_stepback_cache_id_encodes_model_id() -> None:
    assert ModelStepBackQueryTransform(_FakeGenerator("x")).cache_id == "stepback-model:fake-tier2-model"


# --- selector wiring ----------------------------------------------------------


def _select(settings: Settings) -> object:
    return RuntimeBuilder(settings)._query_transform(ExtractiveGenerator())


def _settings(**overrides: Any) -> Settings:
    return Settings(api_key_pepper="test-pepper-with-enough-entropy", **overrides)


def test_query_rewrite_offline_selects_deterministic_rewrite() -> None:
    assert isinstance(_select(_settings(query_transform_provider="query_rewrite")), DeterministicRewriteQueryTransform)


def test_multi_query_offline_selects_deterministic_multi_query() -> None:
    assert isinstance(_select(_settings(query_transform_provider="multi_query")), DeterministicMultiQueryTransform)


def test_step_back_offline_selects_deterministic_stepback() -> None:
    assert isinstance(_select(_settings(query_transform_provider="step_back")), DeterministicStepBackQueryTransform)


def test_query_rewrite_with_real_generation_selects_model_rewrite() -> None:
    settings = _settings(query_transform_provider="query_rewrite", generation_provider="openai_compatible")
    assert isinstance(RuntimeBuilder(settings)._query_transform(_FakeGenerator("x")), ModelRewriteQueryTransform)


def test_multi_query_with_real_generation_selects_model_multi_query() -> None:
    settings = _settings(query_transform_provider="multi_query", generation_provider="openai_compatible")
    assert isinstance(RuntimeBuilder(settings)._query_transform(_FakeGenerator("x")), ModelMultiQueryTransform)


def test_step_back_with_real_generation_selects_model_stepback() -> None:
    settings = _settings(query_transform_provider="step_back", generation_provider="openai_compatible")
    assert isinstance(RuntimeBuilder(settings)._query_transform(_FakeGenerator("x")), ModelStepBackQueryTransform)


def test_extractive_generator_never_routed_into_a_model_transform() -> None:
    for provider in ("query_rewrite", "multi_query", "step_back"):
        transform = _select(_settings(query_transform_provider=provider))
        assert "model" not in transform.cache_id  # type: ignore[attr-defined]


# --- cache identity (canonical-payload pins, not tautologies) -----------------


def _policy() -> Any:
    settings = _settings()
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


def test_each_tier2_cache_id_changes_the_context_hash() -> None:
    policy = _policy()
    baseline = _hash(policy, "none")
    sub_query = _hash(policy, DeterministicQueryTransform().cache_id)
    cache_ids = [
        DeterministicRewriteQueryTransform().cache_id,
        ModelRewriteQueryTransform(_FakeGenerator("x")).cache_id,
        DeterministicMultiQueryTransform().cache_id,
        ModelMultiQueryTransform(_FakeGenerator("x")).cache_id,
        DeterministicStepBackQueryTransform().cache_id,
        ModelStepBackQueryTransform(_FakeGenerator("x")).cache_id,
    ]
    hashes = [_hash(policy, cache_id) for cache_id in cache_ids]
    assert baseline not in hashes
    assert sub_query not in hashes
    assert len(set(hashes)) == len(hashes)


def test_swapping_providers_changes_the_hash() -> None:
    policy = _policy()
    rewrite = _hash(policy, DeterministicRewriteQueryTransform().cache_id)
    multi = _hash(policy, DeterministicMultiQueryTransform().cache_id)
    stepback = _hash(policy, DeterministicStepBackQueryTransform().cache_id)
    assert rewrite != multi
    assert multi != stepback
    assert rewrite != stepback
