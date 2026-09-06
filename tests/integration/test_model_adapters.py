from __future__ import annotations

import json

import httpx
import pytest

from refaat.application.runtime import RuntimeBuilder
from refaat.config import Settings
from refaat.domain.models import Chunk, ContextPacket, GenerationRequest, SearchHit
from refaat.errors import DependencyUnavailableError
from refaat.infrastructure.embeddings import DeterministicEmbedder, OpenAICompatibleEmbedder
from refaat.infrastructure.generation import OpenAICompatibleGenerator
from refaat.infrastructure.graph import OpenAICompatibleGraphExtractor


def test_openai_compatible_embedding_contract() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/embeddings"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            },
        )

    embedder = OpenAICompatibleEmbedder(
        base_url="https://models.test/v1",
        api_key="secret",
        model="embed",
        dimensions=2,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert embedder.embed(["a", "b"]) == [(1.0, 0.0), (0.0, 1.0)]


def test_embedding_dimensions_field_is_omitted_for_non_openai_endpoints() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.clear()
        captured.update(json.loads(request.read().decode()))
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    def embedder(send_dimensions: bool) -> OpenAICompatibleEmbedder:
        return OpenAICompatibleEmbedder(
            base_url="https://api.mistral.ai/v1",
            api_key="secret",
            model="mistral-embed",
            dimensions=2,
            send_dimensions=send_dimensions,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

    embedder(True).embed(["a"])
    assert captured["dimensions"] == 2

    embedder(False).embed(["a"])
    assert "dimensions" not in captured


def test_openai_embedder_retries_transient_failures() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, json={"error": "unavailable"})
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    embedder = OpenAICompatibleEmbedder(
        base_url="https://api.mistral.ai/v1",
        api_key="secret",
        model="mistral-embed",
        dimensions=2,
        retries=2,
        retry_base_delay=0.0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert embedder.embed(["a"]) == [(1.0, 0.0)]
    assert calls["n"] == 3


def test_openai_embedder_raises_after_exhausting_retries() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"})

    embedder = OpenAICompatibleEmbedder(
        base_url="https://api.mistral.ai/v1",
        api_key="secret",
        model="mistral-embed",
        dimensions=2,
        retries=1,
        retry_base_delay=0.0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(DependencyUnavailableError):
        embedder.embed(["a"])


def test_builder_wires_per_role_provider_endpoints() -> None:
    settings = Settings(
        embedding_provider="openai_compatible",
        embedding_base_url="https://embeddings.test/v1",
        embedding_api_key="embed-key",
        generation_provider="openai_compatible",
        generation_base_url="https://generation.test/v1",
        generation_api_key="gen-key",
    )
    builder = RuntimeBuilder(settings)
    embedder = builder._embedder()
    generator = builder._generator()
    assert isinstance(embedder, OpenAICompatibleEmbedder)
    assert isinstance(generator, OpenAICompatibleGenerator)
    try:
        assert embedder.base_url == "https://embeddings.test/v1"
        assert generator.base_url == "https://generation.test/v1"
    finally:
        embedder.close()
        generator.close()


def test_builder_accepts_a_custom_model_adapter() -> None:
    custom = DeterministicEmbedder(dimensions=64, model_id="custom-provider")
    builder = RuntimeBuilder(Settings(embedding_provider="openai_compatible")).with_embedder(custom)
    assert builder._embedder() is custom


def test_openai_compatible_generation_and_graph_contracts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read().decode()
        if "knowledge_graph" in body:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"entities":[{"name":"Refaat","kind":"product"},'
                                    '{"name":"Oronts","kind":"company"}],'
                                    '"relations":[{"source":"Oronts","target":"Refaat",'
                                    '"kind":"builds","weight":1.0}]}'
                                )
                            }
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Grounded answer [chunk:chunk_1]"}}],
                "usage": {"total_tokens": 20},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    graph = OpenAICompatibleGraphExtractor(
        base_url="https://models.test/v1",
        api_key="secret",
        model="model",
        client=client,
    ).extract(corpus_id="corpus", document_id="doc", text="Oronts builds Refaat")
    assert len(graph.entities) == 2
    assert graph.relations[0].kind == "builds"

    chunk = Chunk(
        id="chunk_1",
        corpus_id="corpus",
        document_id="doc",
        ordinal=0,
        text="Evidence",
        token_count=1,
        checksum="x",
        metadata={},
        embedding=(1.0, 0.0),
    )
    context = ContextPacket(
        query="question",
        hits=(SearchHit(chunk=chunk, score=1.0),),
        graph_facts=(),
        session_summary=None,
        profile={},
        supplemental_harness={},
        token_count=1,
        routes=("semantic",),
    )
    result = OpenAICompatibleGenerator(
        base_url="https://models.test/v1",
        api_key="secret",
        model="model",
        client=client,
    ).generate(GenerationRequest(query="question", context=context, system_prompt="ground"))
    assert result.cited_chunk_ids == ("chunk_1",)


def test_embedder_wraps_malformed_success_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"embedding": [1.0]}]})

    embedder = OpenAICompatibleEmbedder(
        base_url="https://models.test/v1",
        api_key="secret",
        model="model",
        dimensions=1,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(DependencyUnavailableError):
        embedder.embed(["a"])


def test_generator_wraps_non_json_success_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="upstream gateway error")

    chunk = Chunk(
        id="chunk_1",
        corpus_id="corpus",
        document_id="doc",
        ordinal=0,
        text="Evidence",
        token_count=1,
        checksum="x",
        metadata={},
        embedding=(1.0, 0.0),
    )
    context = ContextPacket(
        query="question",
        hits=(SearchHit(chunk=chunk, score=1.0),),
        graph_facts=(),
        session_summary=None,
        profile={},
        supplemental_harness={},
        token_count=1,
        routes=("semantic",),
    )
    generator = OpenAICompatibleGenerator(
        base_url="https://models.test/v1",
        api_key="secret",
        model="model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(DependencyUnavailableError):
        generator.generate(GenerationRequest(query="question", context=context, system_prompt="ground"))
