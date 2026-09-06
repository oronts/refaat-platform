from __future__ import annotations

import pytest

from refaat.application.runtime import RuntimeBuilder
from refaat.config import Settings
from refaat.infrastructure.embeddings import OpenAICompatibleEmbedder
from refaat.infrastructure.generation import OpenAICompatibleGenerator
from refaat.infrastructure.graph import OpenAICompatibleGraphExtractor

# Each row is a provider reachable purely by configuration through the OpenAI-compatible
# protocol. The assertion is that the builder composes the OpenAI-compatible adapter with
# the configured base URL and model. No provider is contacted: adapters make a network
# call only on use, and these tests only construct them.
GENERATION_PROVIDERS = [
    ("openai", "https://api.openai.com/v1", "gpt-4o-mini"),
    ("mistral", "https://api.mistral.ai/v1", "mistral-large-latest"),
    ("together", "https://api.together.xyz/v1", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    ("groq", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
    ("ollama", "http://127.0.0.1:11434/v1", "llama3.1"),
    ("vllm", "http://vllm.internal:8000/v1", "your-served-model"),
]

EMBEDDING_PROVIDERS = [
    ("openai", "https://api.openai.com/v1", "text-embedding-3-small", 1536, True),
    ("mistral", "https://api.mistral.ai/v1", "mistral-embed", 1024, False),
    ("ollama", "http://127.0.0.1:11434/v1", "nomic-embed-text", 768, False),
]


@pytest.mark.parametrize(("name", "base_url", "model"), GENERATION_PROVIDERS)
def test_generation_provider_composes_by_config(name: str, base_url: str, model: str) -> None:
    settings = Settings(
        generation_provider="openai_compatible",
        generation_base_url=base_url,
        generation_model=model,
        openai_api_key="configuration-only-key",
    )
    builder = RuntimeBuilder(settings)

    generator = builder._generator()

    assert isinstance(generator, OpenAICompatibleGenerator), name
    assert generator.base_url == base_url
    assert generator.model_id == model
    generator.close()


@pytest.mark.parametrize(("name", "base_url", "model", "dimensions", "send_dimensions"), EMBEDDING_PROVIDERS)
def test_embedding_provider_composes_by_config(
    name: str, base_url: str, model: str, dimensions: int, send_dimensions: bool
) -> None:
    settings = Settings(
        embedding_provider="openai_compatible",
        embedding_base_url=base_url,
        embedding_model=model,
        embedding_dimensions=dimensions,
        embedding_send_dimensions=send_dimensions,
        openai_api_key="configuration-only-key",
    )
    builder = RuntimeBuilder(settings)

    embedder = builder._embedder()

    assert isinstance(embedder, OpenAICompatibleEmbedder), name
    assert embedder.base_url == base_url
    assert embedder.model_id == model
    assert embedder.dimensions == dimensions
    embedder.close()


def test_per_role_endpoints_point_at_different_providers() -> None:
    # Embeddings on OpenAI, generation on Mistral, graph extraction on the generation
    # endpoint. Each role composes independently against its own base URL and model.
    settings = Settings(
        embedding_provider="openai_compatible",
        embedding_base_url="https://api.openai.com/v1",
        embedding_model="text-embedding-3-small",
        embedding_api_key="embed-key",
        generation_provider="openai_compatible",
        generation_base_url="https://api.mistral.ai/v1",
        generation_model="mistral-large-latest",
        generation_api_key="generation-key",
        graph_provider="openai_compatible",
    )
    builder = RuntimeBuilder(settings)

    embedder = builder._embedder()
    generator = builder._generator()
    graph = builder._graph_extractor()

    assert isinstance(embedder, OpenAICompatibleEmbedder)
    assert embedder.base_url == "https://api.openai.com/v1"
    assert isinstance(generator, OpenAICompatibleGenerator)
    assert generator.base_url == "https://api.mistral.ai/v1"
    assert isinstance(graph, OpenAICompatibleGraphExtractor)
    assert graph.base_url == "https://api.mistral.ai/v1"
    assert graph.model_id == "mistral-large-latest"
    embedder.close()
    generator.close()
    graph.close()


def test_shared_endpoint_falls_back_for_both_roles() -> None:
    settings = Settings(
        embedding_provider="openai_compatible",
        generation_provider="openai_compatible",
        openai_base_url="https://api.groq.com/openai/v1",
        openai_api_key="shared-key",
        embedding_model="nomic-embed-text",
        generation_model="llama-3.3-70b-versatile",
    )
    builder = RuntimeBuilder(settings)

    embedder = builder._embedder()
    generator = builder._generator()

    assert embedder.base_url == "https://api.groq.com/openai/v1"
    assert generator.base_url == "https://api.groq.com/openai/v1"
    embedder.close()
    generator.close()
