"""The shadow optimizer reproduces the safe production composition or refuses, never writing to production.

Two write-to-production invariants are proven here: a store-bound RaptorTreeBuilder is refused (it would
write RAPTOR nodes into the production database), and a reproduced production embedder or generator is
borrowed (the shadow never closes its shared client). The default no-injection path stays byte-identical.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from refaat.application.raptor import RaptorTreeBuilder
from refaat.application.runtime import Refaat, RefaatProject, RuntimeBuilder
from refaat.clock import SystemClock
from refaat.config import Settings
from refaat.core.chunking import BUILTIN_CHUNKING_STRATEGIES
from refaat.domain.authz import CanonicalAccessPolicy
from refaat.domain.models import DocumentInput, GenerationRequest, GenerationResult
from refaat.errors import AuthorizationError
from refaat.evaluation import Dataset
from refaat.infrastructure.database.store import SQLStore
from refaat.infrastructure.embeddings import DeterministicEmbedder
from refaat.infrastructure.generation import ExtractiveGenerator
from refaat.infrastructure.graph import CooccurrenceGraphExtractor
from refaat.infrastructure.summarization import DeterministicExtractiveSummarizer
from refaat.optimization import ChunkingOptimizer, GraphOptimizer

pytestmark = pytest.mark.integration

_DOCS = [
    "Runbook: escalate a Sev1 incident to the on-call engineer within fifteen minutes and open a bridge.",
    "The rollback plan requires an approved change ticket before a production release is reverted.",
    "On-call response time targets acknowledge within five minutes and mitigate within thirty minutes.",
    "Incident severity levels range from Sev1 for a full outage to Sev4 for a minor degradation.",
]

_DATASET = Dataset.from_records(
    "chunking",
    [
        {"query": "incident escalation", "expected_terms": ["incident"]},
        {"query": "on-call response time", "expected_terms": ["minutes"]},
    ],
)


def _ingest(project: RefaatProject) -> None:
    for index, body in enumerate(_DOCS):
        project.runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id=f"doc{index}.txt",
                title=f"Doc {index}",
                media_type="text/plain",
                content=body.encode(),
            ),
        )


class _RecordingRaptorBuilder(RaptorTreeBuilder):
    """A store-bound builder that records every rebuild instead of writing, to catch a production write."""

    def __init__(self, store: SQLStore) -> None:
        super().__init__(store=store, embedder=DeterministicEmbedder(), summarizer=DeterministicExtractiveSummarizer())
        self.rebuild_calls = 0

    def rebuild(self, workspace_id: str, corpus_id: str) -> int:
        self.rebuild_calls += 1
        return 0


class _FlagEmbedder:
    """Delegates to a real deterministic embedder and flips a flag when closed."""

    def __init__(self) -> None:
        self._inner = DeterministicEmbedder()
        self.closed = False

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return self._inner.embed(texts)

    def close(self) -> None:
        self.closed = True


class _FlagGenerator:
    """Delegates to the real extractive generator and flips a flag when closed."""

    def __init__(self) -> None:
        self._inner = ExtractiveGenerator()
        self.closed = False

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return self._inner.generate(request)

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        return self._inner.stream(request)

    def close(self) -> None:
        self.closed = True


class _RecordingEmbedder:
    """Delegates to a real deterministic embedder and counts embed calls, to prove reproduction is real."""

    def __init__(self) -> None:
        self._inner = DeterministicEmbedder()
        self.embed_calls = 0

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        self.embed_calls += 1
        return self._inner.embed(texts)


def test_default_runtime_omits_shadow_composition_key_and_still_optimizes(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    _ingest(project)

    result = ChunkingOptimizer(project).optimize(_DATASET, chunk_sizes=(64, 96), overlaps=(12,))

    payload = result.to_json()
    assert "shadow_composition" not in payload
    assert result.candidate_count >= 1
    assert result.shadow_composition is not None
    assert result.shadow_composition.is_empty() is True


def test_blocker1_store_bound_raptor_builder_is_refused(settings: Settings, tmp_path: Path) -> None:
    marker_store = SQLStore(f"sqlite:///{tmp_path / 'prod_marker.db'}", SystemClock())
    raptor = _RecordingRaptorBuilder(marker_store)
    runtime = RuntimeBuilder(settings).with_raptor_builder(raptor).build()
    try:
        project = runtime.project(workspace="acme", corpus="handbook")
        _ingest(project)
        assert raptor.rebuild_calls > 0  # production ingestion drives the injected builder
        raptor.rebuild_calls = 0  # baseline for the optimization window

        result = ChunkingOptimizer(project).optimize(_DATASET, chunk_sizes=(64, 96), overlaps=(12,))

        # BLOCKER-1: the shadow never reproduced the store-bound builder, so no production RAPTOR write ran.
        assert raptor.rebuild_calls == 0
        assert result.shadow_composition is not None
        assert "raptor_builder" in {component.name for component in result.shadow_composition.omitted}
    finally:
        runtime.close()
        marker_store.close()


def test_blocker2_reproduced_adapters_are_borrowed_not_closed_by_shadow(settings: Settings) -> None:
    embedder = _FlagEmbedder()
    generator = _FlagGenerator()
    runtime = RuntimeBuilder(settings).with_embedder(embedder).with_generator(generator).build()
    try:
        project = runtime.project(workspace="acme", corpus="handbook")
        _ingest(project)

        result = ChunkingOptimizer(project).optimize(_DATASET, chunk_sizes=(64, 96, 128), overlaps=(12,))

        assert result.candidate_count == 3  # several shadow builds and closes happened
        # BLOCKER-2: no shadow.close() closed the shared production adapters.
        assert embedder.closed is False
        assert generator.closed is False
        # The production runtime still serves after the optimization.
        response = project.ask("incident escalation")
        assert isinstance(response.answer, str)
        assert {"embedder", "generator"} <= {c.name for c in result.shadow_composition.reproduced}
    finally:
        runtime.close()

    # runtime.close() closes each borrowed-into-shadow adapter exactly once, here.
    assert embedder.closed is True
    assert generator.closed is True


def test_reproduced_embedder_actually_reaches_the_shadow(settings: Settings) -> None:
    embedder = _RecordingEmbedder()
    runtime = RuntimeBuilder(settings).with_embedder(embedder).build()
    try:
        project = runtime.project(workspace="acme", corpus="handbook")
        _ingest(project)
        embedder.embed_calls = 0  # ignore production-side embedding

        result = ChunkingOptimizer(project).optimize(_DATASET, chunk_sizes=(64, 96), overlaps=(12,))

        # The shadow re-ingested and evaluated over the SAME injected embedder, so reproduction is real.
        assert embedder.embed_calls > 0
        assert "embedder" in {c.name for c in result.shadow_composition.reproduced}
    finally:
        runtime.close()


def test_chunking_optimizer_refuses_principalless_run_under_strict_policy(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings).with_access_policy(CanonicalAccessPolicy()).build()
    try:
        project = runtime.project(workspace="acme", corpus="handbook")
        with pytest.raises(AuthorizationError):
            ChunkingOptimizer(project).optimize(_DATASET, chunk_sizes=(64, 96), overlaps=(12,))
    finally:
        runtime.close()


def test_chunking_optimizer_runs_under_a_non_strict_policy_and_refuses_it_in_the_shadow(settings: Settings) -> None:
    lenient = settings.model_copy(update={"strict_authorization": False})
    runtime = RuntimeBuilder(lenient).with_access_policy(CanonicalAccessPolicy()).build()
    try:
        project = runtime.project(workspace="acme", corpus="handbook")
        _ingest(project)

        result = ChunkingOptimizer(project).optimize(_DATASET, chunk_sizes=(64, 96), overlaps=(12,))

        assert result.candidate_count >= 1
        assert "access_policy" in {c.name for c in result.shadow_composition.omitted}
    finally:
        runtime.close()


def test_graph_optimizer_refuses_a_custom_graph_extractor(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings).with_graph_extractor(CooccurrenceGraphExtractor()).build()
    try:
        project = runtime.project(workspace="acme", corpus="handbook")
        _ingest(project)
        with pytest.raises(ValueError, match="custom graph extractor"):
            GraphOptimizer(project).optimize(_DATASET, max_entities=(40, 80), min_token_lengths=(4,))
    finally:
        runtime.close()


def test_chunking_optimizer_refuses_a_custom_chunking_strategy(settings: Settings) -> None:
    strategy = BUILTIN_CHUNKING_STRATEGIES["flat"](settings)
    runtime = RuntimeBuilder(settings).with_chunking_strategy(strategy).build()
    try:
        project = runtime.project(workspace="acme", corpus="handbook")
        _ingest(project)
        with pytest.raises(ValueError, match="custom chunking strategy"):
            ChunkingOptimizer(project).optimize(_DATASET, chunk_sizes=(64, 96), overlaps=(12,))
    finally:
        runtime.close()
