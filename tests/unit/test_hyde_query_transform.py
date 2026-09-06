from __future__ import annotations

from collections.abc import Iterator

from refaat.application.runtime import RuntimeBuilder
from refaat.config import Settings
from refaat.domain.models import GenerationRequest, GenerationResult
from refaat.infrastructure.embeddings import DeterministicEmbedder
from refaat.infrastructure.generation import ExtractiveGenerator
from refaat.infrastructure.query_transform import (
    DeterministicHydeQueryTransform,
    DeterministicQueryTransform,
    ModelHydeQueryTransform,
    _templated_hypothesis,
)

QUERY = "How does rollback restore the prior artifact?"


class _FakeGenerator:
    def __init__(self, answer: str) -> None:
        self._answer = answer
        self.calls: list[GenerationRequest] = []

    @property
    def model_id(self) -> str:
        return "fake-hyde-model"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request)
        return GenerationResult(answer=self._answer, cited_chunk_ids=())

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        yield self._answer


def _embed(text: str) -> tuple[float, ...]:
    return DeterministicEmbedder().embed([text])[0]


def test_deterministic_hyde_yields_original_and_hypothetical() -> None:
    variants = DeterministicHydeQueryTransform().expand(QUERY, 4)
    assert len(variants) == 2

    original, hypothetical = variants
    assert original.kind == "original"
    assert original.text == QUERY
    assert original.embed_text is None

    assert hypothetical.kind == "hypothetical"
    assert hypothetical.text == QUERY
    assert hypothetical.embed_text is not None
    assert hypothetical.embed_text != QUERY


def test_deterministic_hyde_cache_id_is_stable() -> None:
    assert DeterministicHydeQueryTransform().cache_id == "hyde-deterministic-v1"


def test_max_variants_one_short_circuits_to_original_only() -> None:
    variants = DeterministicHydeQueryTransform().expand(QUERY, 1)
    assert len(variants) == 1
    assert variants[0].kind == "original"
    assert variants[0].embed_text is None


def test_blank_query_degrades_to_original_only() -> None:
    variants = DeterministicHydeQueryTransform().expand("the of and to", 4)
    assert [variant.kind for variant in variants] == ["original"]


def test_embedding_target_genuinely_moves_under_deterministic_embedder() -> None:
    hypothesis = _templated_hypothesis(QUERY)
    assert hypothesis
    assert _embed(hypothesis) != _embed(QUERY)


def test_model_hyde_places_generator_output_only_in_embed_text() -> None:
    generator = _FakeGenerator("A rollback swaps the active alias back to the prior artifact digest.")
    variants = ModelHydeQueryTransform(generator).expand(QUERY, 4)
    assert len(variants) == 2

    original, hypothetical = variants
    assert original.embed_text is None
    assert hypothetical.text == QUERY
    assert hypothetical.embed_text == "A rollback swaps the active alias back to the prior artifact digest."
    assert len(generator.calls) == 1
    assert generator.calls[0].context.hits == ()


def test_model_hyde_cache_id_encodes_model_id() -> None:
    transform = ModelHydeQueryTransform(_FakeGenerator("anything"))
    assert transform.cache_id == "hyde-model:fake-hyde-model"


def test_model_hyde_blank_output_degrades_to_original_only() -> None:
    variants = ModelHydeQueryTransform(_FakeGenerator("   ")).expand(QUERY, 4)
    assert [variant.kind for variant in variants] == ["original"]


def test_model_hyde_does_not_call_generator_on_single_variant_budget() -> None:
    generator = _FakeGenerator("unused hypothesis")
    variants = ModelHydeQueryTransform(generator).expand(QUERY, 1)
    assert [variant.kind for variant in variants] == ["original"]
    assert generator.calls == []


def _select(settings: Settings) -> object:
    return RuntimeBuilder(settings)._query_transform(ExtractiveGenerator())


def test_default_provider_selects_sub_query_transform() -> None:
    assert isinstance(_select(Settings(api_key_pepper="test-pepper-with-enough-entropy")), DeterministicQueryTransform)


def test_hyde_offline_selects_deterministic_hyde() -> None:
    settings = Settings(api_key_pepper="test-pepper-with-enough-entropy", query_transform_provider="hyde")
    assert isinstance(_select(settings), DeterministicHydeQueryTransform)


def test_hyde_with_real_generation_selects_model_hyde() -> None:
    settings = Settings(
        api_key_pepper="test-pepper-with-enough-entropy",
        query_transform_provider="hyde",
        generation_provider="openai_compatible",
    )
    transform = RuntimeBuilder(settings)._query_transform(_FakeGenerator("passage"))
    assert isinstance(transform, ModelHydeQueryTransform)


def test_custom_query_transform_injection_wins() -> None:
    settings = Settings(api_key_pepper="test-pepper-with-enough-entropy", query_transform_provider="hyde")
    custom = DeterministicQueryTransform()
    builder = RuntimeBuilder(settings).with_query_transform(custom)
    assert builder._query_transform(ExtractiveGenerator()) is custom
