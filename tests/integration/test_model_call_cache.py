from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select

from refaat.application.budget import budget_scope
from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.config import Settings
from refaat.domain.erasure import ErasureStatus
from refaat.domain.models import (
    Chunk,
    ContextPacket,
    DocumentInput,
    GenerationRequest,
    GenerationResult,
    SearchHit,
    ToolProposal,
    ToolSpec,
)
from refaat.infrastructure.database.schema import model_call_cache
from refaat.infrastructure.database.store import SQLStore
from refaat.infrastructure.hype import DeterministicHypotheticalGenerator, ModelHypotheticalGenerator
from refaat.infrastructure.judging import GeneratorJudge
from refaat.infrastructure.model_cache import (
    CachingEmbedder,
    CachingGenerator,
    _estimate_tokens,
    _request_prompt_tokens,
)
from refaat.infrastructure.query_transform import ModelRewriteQueryTransform
from refaat.infrastructure.summarization import DeterministicExtractiveSummarizer, ModelSummarizer

pytestmark = pytest.mark.integration


class _FakeClock:
    def __init__(self, start: datetime) -> None:
        self._now = start

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)


class _CountingEmbedder:
    def __init__(self, dimensions: int = 4, model_id: str = "fake-embed-v1", identity: str | None = None) -> None:
        self._dimensions = dimensions
        self._model_id = model_id
        self._identity = identity or f"fake-embed:{model_id}:{dimensions}"
        self.sent: list[str] = []
        self.closed = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def cache_identity(self) -> str:
        return self._identity

    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        self.sent.extend(texts)
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> tuple[float, ...]:
        seed = sum(ord(char) for char in text)
        return tuple(float((seed + index) % 7) for index in range(self._dimensions))

    def close(self) -> None:
        self.closed += 1


class _CountingGenerator:
    def __init__(
        self,
        *,
        model_id: str = "fake-gen-v1",
        identity: str | None = None,
        usage: dict[str, int] | None = None,
    ) -> None:
        self._model_id = model_id
        self._identity = identity or f"fake-gen:{model_id}"
        self._usage = usage or {"prompt_tokens": 5, "completion_tokens": 3}
        self.calls = 0
        self.closed = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def cache_identity(self) -> str:
        return self._identity

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls += 1
        cited = (request.context.hits[0].chunk.id,) if request.context.hits else ()
        return GenerationResult(
            answer=f"answer to {request.query} #{self.calls}",
            cited_chunk_ids=cited,
            usage=dict(self._usage),
        )

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        yield f"answer to {request.query}"

    def close(self) -> None:
        self.closed += 1


class _ToolGenerator(_CountingGenerator):
    def propose_tool_call(self, query: str, tools: Sequence[ToolSpec], used: frozenset[str]) -> ToolProposal | None:
        del tools, used
        self.calls += 1
        return ToolProposal(tool="lookup", arguments={"q": query})


class _MultiPieceGenerator(_CountingGenerator):
    def stream(self, request: GenerationRequest) -> Iterator[str]:
        del request
        yield from ("alpha", "beta", "gamma", "delta")


class _WorkspaceRecordingGenerator:
    """A cacheable generator that records the workspace on every request it sees."""

    def __init__(self, *, answer: str = "recorded answer", model_id: str = "rec-gen-v1") -> None:
        self._answer = answer
        self._model_id = model_id
        self.workspaces: list[str] = []
        self.calls = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def cache_identity(self) -> str:
        return f"rec:https://x/v1:{self._model_id}"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls += 1
        self.workspaces.append(request.workspace_id)
        cited = (request.context.hits[0].chunk.id,) if request.context.hits else ()
        return GenerationResult(
            answer=self._answer, cited_chunk_ids=cited, usage={"prompt_tokens": 1, "completion_tokens": 1}
        )

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        del request
        yield self._answer


def _store(tmp_path: Path, clock: _FakeClock | None = None) -> SQLStore:
    store = SQLStore(f"sqlite:///{tmp_path / 'cache.db'}", clock or _FakeClock(datetime(2026, 1, 1, tzinfo=UTC)))
    store.initialize()
    return store


def _request(
    query: str,
    chunk_id: str = "chunk_1",
    text: str = "Evidence text",
    *,
    document_id: str = "doc",
    workspace_id: str = "ws",
    embedding: tuple[float, ...] = (1.0, 0.0),
) -> GenerationRequest:
    chunk = Chunk(
        id=chunk_id,
        corpus_id="corpus",
        document_id=document_id,
        ordinal=0,
        text=text,
        token_count=2,
        checksum="x",
        metadata={},
        embedding=embedding,
    )
    context = ContextPacket(
        query=query,
        hits=(SearchHit(chunk=chunk, score=1.0),),
        graph_facts=(),
        session_summary=None,
        profile={},
        supplemental_harness={},
        token_count=2,
        routes=("semantic",),
    )
    return GenerationRequest(query=query, context=context, system_prompt="ground", workspace_id=workspace_id)


def _row_count(store: SQLStore) -> int:
    with store.engine.connect() as conn:
        return int(conn.execute(select(func.count()).select_from(model_call_cache)).scalar_one())


def test_generation_hit_is_byte_identical_to_a_miss_and_calls_inner_once(tmp_path: Path) -> None:
    store = _store(tmp_path)
    inner = _CountingGenerator()
    wrapped = CachingGenerator(inner, store, ttl_seconds=3600)
    request = _request("what is the policy")

    miss = wrapped.generate(request)
    hit = wrapped.generate(request)

    assert inner.calls == 1
    assert hit == miss
    assert hit.answer == "answer to what is the policy #1"
    assert wrapped.model_id == inner.model_id


def test_cross_backend_same_model_never_shares_a_cache_entry(tmp_path: Path) -> None:
    store = _store(tmp_path)
    # Same bare model name, different backend endpoint => distinct cache_identity.
    inner_a = _CountingGenerator(model_id="gpt", identity="openai-compat-gen:https://a.example/v1:gpt")
    inner_b = _CountingGenerator(model_id="gpt", identity="openai-compat-gen:https://b.example/v1:gpt")
    request = _request("shared question")

    first = CachingGenerator(inner_a, store, ttl_seconds=3600).generate(request)
    second = CachingGenerator(inner_b, store, ttl_seconds=3600).generate(request)

    assert inner_a.calls == 1
    assert inner_b.calls == 1
    assert first.answer.endswith("#1")
    assert second.answer.endswith("#1")
    assert _row_count(store) == 2


def test_resolved_cache_identity_sees_through_the_caching_wrappers(tmp_path: Path) -> None:
    from refaat.infrastructure.model_cache import cache_identity_of, resolved_cache_identity

    store = _store(tmp_path)
    embedder = _CountingEmbedder(identity="openai-compat-embed:https://a.example/v1:e")
    generator = _CountingGenerator(identity="openai-compat-gen:https://a.example/v1:g")
    wrapped_embedder = CachingEmbedder(embedder, store, ttl_seconds=3600)
    wrapped_generator = CachingGenerator(generator, store, ttl_seconds=3600)

    # By design cache_identity_of does not see a wrapper's identity (wrapping is never a re-cache opt-in).
    assert cache_identity_of(wrapped_embedder) is None
    assert cache_identity_of(wrapped_generator) is None
    # resolved_cache_identity sees through to the inner backend identity, so the durable plan digest keeps
    # two same-model different-backend runs apart even when the adapters are wrapped for model-call caching.
    assert resolved_cache_identity(wrapped_embedder) == embedder.cache_identity
    assert resolved_cache_identity(wrapped_generator) == generator.cache_identity
    # A raw adapter is unchanged: resolved_cache_identity matches cache_identity_of.
    assert resolved_cache_identity(generator) == cache_identity_of(generator)


def test_answer_cache_is_scoped_by_backend_identity_through_the_wrapper(tmp_path: Path) -> None:
    from refaat.domain.models import DocumentInput

    shared_db = f"sqlite:///{tmp_path / 'shared.db'}"

    def _runtime(tag: str) -> Refaat:
        settings = Settings(database_url=shared_db, model_call_cache_enabled=True)
        return (
            RuntimeBuilder(settings)
            .with_generator(_CountingGenerator(identity=f"openai-compat-gen:https://{tag}.example/v1:g"))
            .with_embedder(_CountingEmbedder(identity=f"openai-compat-embed:https://{tag}.example/v1:e"))
            .build()
        )

    runtime_a = _runtime("a")
    try:
        project_a = runtime_a.project(workspace="acme", corpus="kb")
        runtime_a.ingest(
            project_a.workspace_id,
            project_a.corpus_id,
            DocumentInput(
                source_id="d.txt",
                title="D",
                media_type="text/plain",
                content=b"Rollback restores the previously approved harness snapshot before deployment.",
            ),
        )
        assert project_a.ask("how does rollback work").cached is False
        # Same runtime and backend identity: a repeat is a cache hit.
        assert project_a.ask("how does rollback work").cached is True
    finally:
        runtime_a.close()

    runtime_b = _runtime("b")
    try:
        # Same workspace, corpus, and query over the shared store, but a DIFFERENT backend identity resolved
        # through the caching wrapper: the answer cache no longer cross-serves runtime A's cached answer.
        project_b = runtime_b.project(workspace="acme", corpus="kb")
        assert project_b.ask("how does rollback work").cached is False
    finally:
        runtime_b.close()


def test_generation_cache_is_scoped_by_workspace_no_cross_tenant_serve(tmp_path: Path) -> None:
    store = _store(tmp_path)
    inner = _CountingGenerator()
    wrapped = CachingGenerator(inner, store, ttl_seconds=3600)

    wrapped.generate(_request("secret question", workspace_id="tenant-a"))
    # Same everything except the tenant: must be a miss, never served tenant-a's row.
    wrapped.generate(_request("secret question", workspace_id="tenant-b"))

    assert inner.calls == 2
    assert _row_count(store) == 2


def test_generation_key_separates_a_previously_omitted_document_id(tmp_path: Path) -> None:
    store = _store(tmp_path)
    inner = _CountingGenerator()
    wrapped = CachingGenerator(inner, store, ttl_seconds=3600)

    wrapped.generate(_request("q", document_id="doc-1"))
    # document_id is rendered into the prompt but was previously omitted from the key.
    wrapped.generate(_request("q", document_id="doc-2"))

    assert inner.calls == 2
    assert _row_count(store) == 2


def test_generation_key_separates_a_different_hit_text(tmp_path: Path) -> None:
    store = _store(tmp_path)
    inner = _CountingGenerator()
    wrapped = CachingGenerator(inner, store, ttl_seconds=3600)

    wrapped.generate(_request("q", text="Evidence alpha"))
    wrapped.generate(_request("q", text="Evidence beta"))

    assert inner.calls == 2


def test_generation_key_separates_a_different_hit_embedding(tmp_path: Path) -> None:
    store = _store(tmp_path)
    inner = _CountingGenerator()
    wrapped = CachingGenerator(inner, store, ttl_seconds=3600)

    # Same id/text/checksum, different embedding vector (post embedder swap): a custom
    # generator that reads the raw vector must not be served the stale first entry.
    wrapped.generate(_request("q", embedding=(1.0, 0.0)))
    wrapped.generate(_request("q", embedding=(0.0, 1.0)))

    assert inner.calls == 2
    assert _row_count(store) == 2


def test_embedding_hit_equals_miss_and_dimensions_delegate(tmp_path: Path) -> None:
    store = _store(tmp_path)
    inner = _CountingEmbedder(dimensions=6)
    wrapped = CachingEmbedder(inner, store, ttl_seconds=3600)

    miss = wrapped.embed(["alpha"])
    hit = wrapped.embed(["alpha"])

    assert inner.sent == ["alpha"]
    assert hit == miss
    assert wrapped.dimensions == inner.dimensions == 6


def test_cross_backend_embeddings_never_share_a_row(tmp_path: Path) -> None:
    store = _store(tmp_path)
    inner_a = _CountingEmbedder(identity="openai-compat-embed:https://a/v1:m:4:1")
    inner_b = _CountingEmbedder(identity="openai-compat-embed:https://b/v1:m:4:1")

    CachingEmbedder(inner_a, store, ttl_seconds=3600).embed(["shared"])
    CachingEmbedder(inner_b, store, ttl_seconds=3600).embed(["shared"])

    assert inner_a.sent == ["shared"]
    assert inner_b.sent == ["shared"]
    assert _row_count(store) == 2


def test_batched_embed_sends_only_misses_and_preserves_order(tmp_path: Path) -> None:
    store = _store(tmp_path)
    inner = _CountingEmbedder()
    wrapped = CachingEmbedder(inner, store, ttl_seconds=3600)

    result = wrapped.embed(["a", "b", "a", "c"])

    assert inner.sent == ["a", "b", "c"]
    assert result[0] == result[2]
    assert len(result) == 4

    again = wrapped.embed(["a"])
    assert inner.sent == ["a", "b", "c"]
    assert again[0] == result[0]


def test_store_round_trip_ttl_and_purge(tmp_path: Path) -> None:
    clock = _FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
    store = _store(tmp_path, clock)
    store.put_model_call("ws", "model-a", "key1", {"answer": "hi", "cited_chunk_ids": [], "usage": {}}, ttl_seconds=10)
    store.put_embedding("model-a", "key2", (1.0, 2.0), ttl_seconds=10)

    assert store.get_model_call("ws", "model-a", "key1") == {"answer": "hi", "cited_chunk_ids": [], "usage": {}}
    assert store.get_embedding("model-a", "key2") == (1.0, 2.0)
    assert store.get_model_call("ws", "model-b", "key1") is None
    # A different tenant cannot read the row.
    assert store.get_model_call("other", "model-a", "key1") is None

    clock.advance(20)
    assert store.get_model_call("ws", "model-a", "key1") is None
    assert store.get_embedding("model-a", "key2") is None
    assert store.purge_expired_model_call_cache() == 2
    assert _row_count(store) == 0


def test_cache_hit_charges_zero_against_an_active_meter(tmp_path: Path) -> None:
    store = _store(tmp_path)
    inner = _CountingGenerator(usage={"prompt_tokens": 4, "completion_tokens": 2})
    wrapped = CachingGenerator(inner, store, ttl_seconds=3600)
    request = _request("meter question")

    with budget_scope(1000) as meter:
        wrapped.generate(request)
        wrapped.generate(request)
        assert meter is not None
        assert meter.tokens_used == 6


def test_charge_only_generate_never_raises_on_overrun_and_calls_inner(tmp_path: Path) -> None:
    store = _store(tmp_path)
    inner = _CountingGenerator()
    wrapped = CachingGenerator(inner, store, ttl_seconds=3600)

    with budget_scope(10) as meter:
        assert meter is not None
        meter.charge({"prompt_tokens": 10, "completion_tokens": 0})
        # Already exhausted, yet a mid-call charge-only wrapper must not raise and
        # must still serve the caller (enforcement is a loop-boundary concern).
        result = wrapped.generate(_request("still answered"))
    assert result.answer.endswith("#1")
    assert inner.calls == 1


def test_streaming_charges_the_active_meter(tmp_path: Path) -> None:
    store = _store(tmp_path)
    inner = _CountingGenerator()
    wrapped = CachingGenerator(inner, store, ttl_seconds=3600)

    with budget_scope(1000) as meter:
        list(wrapped.stream(_request("stream this out loud")))
        assert meter is not None
        assert meter.tokens_used > 0


def test_tool_planning_charges_the_active_meter(tmp_path: Path) -> None:
    inner = _ToolGenerator()
    wrapped = CachingGenerator(inner, None, ttl_seconds=3600)

    with budget_scope(1000) as meter:
        proposal = wrapped.propose_tool_call("look up the rollback policy", (), frozenset())
        assert proposal is not None
        assert meter is not None
        assert meter.tokens_used > 0


def test_embedding_miss_charges_the_active_meter(tmp_path: Path) -> None:
    store = _store(tmp_path)
    inner = _CountingEmbedder()
    wrapped = CachingEmbedder(inner, store, ttl_seconds=3600)

    with budget_scope(1000) as meter:
        wrapped.embed(["alpha beta gamma"])
        assert meter is not None
        assert meter.tokens_used == 3
        # A hit charges nothing more.
        wrapped.embed(["alpha beta gamma"])
        assert meter.tokens_used == 3


def _ingest(runtime, project) -> None:  # type: ignore[no-untyped-def]
    runtime.ingest(
        project.workspace_id,
        project.corpus_id,
        DocumentInput(
            source_id="ops.txt",
            title="Ops",
            media_type="text/plain",
            content=b"Rollback restores the previously approved snapshot before deployment.",
        ),
    )


def test_disabled_path_writes_no_model_call_rows(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings).build()
    try:
        project = runtime.project(workspace="acme", corpus="handbook")
        _ingest(runtime, project)
        answer = project.ask("how does rollback work")
        assert answer.citations
        assert _row_count(runtime.store) == 0
    finally:
        runtime.close()


def test_caching_enabled_does_not_wrap_the_deterministic_stack(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings.model_copy(update={"model_call_cache_enabled": True})).build()
    try:
        project = runtime.project(workspace="acme", corpus="handbook")
        _ingest(runtime, project)
        project.ask("how does rollback work")
        assert _row_count(runtime.store) == 0
    finally:
        runtime.close()


def test_caching_enabled_wraps_an_injected_generator_that_opts_in(settings: Settings) -> None:
    generator = _CountingGenerator()
    runtime = (
        RuntimeBuilder(settings.model_copy(update={"model_call_cache_enabled": True})).with_generator(generator).build()
    )
    try:
        project = runtime.project(workspace="acme", corpus="handbook")
        _ingest(runtime, project)
        project.ask("how does rollback work")
        assert _row_count(runtime.store) >= 1
    finally:
        runtime.close()


class _OpaqueGenerator:
    """A custom deterministic generator that does NOT expose cache_identity."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "opaque-v1"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls += 1
        cited = (request.context.hits[0].chunk.id,) if request.context.hits else ()
        return GenerationResult(answer="fixed answer", cited_chunk_ids=cited, usage={})

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        yield "fixed answer"


def test_caching_does_not_wrap_a_custom_adapter_without_cache_identity(settings: Settings) -> None:
    generator = _OpaqueGenerator()
    runtime = (
        RuntimeBuilder(settings.model_copy(update={"model_call_cache_enabled": True})).with_generator(generator).build()
    )
    try:
        project = runtime.project(workspace="acme", corpus="handbook")
        _ingest(runtime, project)
        project.ask("how does rollback work")
        # No cache_identity => never cached, so no stale-serve risk after a config change.
        assert _row_count(runtime.store) == 0
    finally:
        runtime.close()


def test_double_wrap_is_guarded(tmp_path: Path) -> None:
    store = _store(tmp_path)
    inner = _CountingGenerator()
    from refaat.application.runtime import RuntimeBuilder as _Builder

    builder = _Builder(Settings(model_call_cache_enabled=True))
    once = builder._wrap_generator(inner, store)
    twice = builder._wrap_generator(once, store)
    assert isinstance(once, CachingGenerator)
    assert twice is once


def test_wrapped_adapter_is_closed_exactly_once(settings: Settings) -> None:
    generator = _CountingGenerator()
    embedder = _CountingEmbedder()
    runtime = (
        RuntimeBuilder(settings.model_copy(update={"model_call_cache_enabled": True}))
        .with_generator(generator)
        .with_embedder(embedder)
        .build()
    )
    runtime.project(workspace="acme", corpus="handbook")
    runtime.close()
    assert generator.closed == 1
    assert embedder.closed == 1


def test_generation_key_folds_the_full_request_via_a_probe(tmp_path: Path) -> None:
    """A generator that reads a previously-unkeyed field is not served a stale hit."""
    store = _store(tmp_path)

    class _SessionSensitiveGenerator(_CountingGenerator):
        def generate(self, request: GenerationRequest) -> GenerationResult:
            summary = request.context.session_summary or "none"
            self.calls += 1
            cited = (request.context.hits[0].chunk.id,) if request.context.hits else ()
            return GenerationResult(answer=f"summary={summary}", cited_chunk_ids=cited, usage={})

    inner = _SessionSensitiveGenerator()
    wrapped = CachingGenerator(inner, store, ttl_seconds=3600)
    base = _request("q")
    a = wrapped.generate(replace(base, context=replace(base.context, session_summary="alpha")))
    b = wrapped.generate(replace(base, context=replace(base.context, session_summary="beta")))
    assert a.answer == "summary=alpha"
    assert b.answer == "summary=beta"
    assert inner.calls == 2


# R1 fail-safe: an unthreaded (empty-workspace) generation is never cacheable, so a
# forgotten tenant thread can never cross-serve.
def test_empty_workspace_generation_is_uncacheable_fail_safe(tmp_path: Path) -> None:
    store = _store(tmp_path)
    inner = _CountingGenerator()
    wrapped = CachingGenerator(inner, store, ttl_seconds=3600)

    wrapped.generate(_request("shared secret", workspace_id=""))
    wrapped.generate(_request("shared secret", workspace_id=""))

    # Two callers that both forgot to thread the tenant share scope "" and must not
    # cross-serve: the wrapper never reads or writes a row for an empty workspace.
    assert inner.calls == 2
    assert _row_count(store) == 0


# R1 threading: HyPE ingestion generations carry the tenant, so tenant A is scoped
# away from tenant B and never cross-served.
def test_hype_generation_threads_workspace_no_cross_tenant_serve(tmp_path: Path) -> None:
    store = _store(tmp_path)
    recorder = _WorkspaceRecordingGenerator(answer="What is x?\nHow does x work?")
    wrapped = CachingGenerator(recorder, store, ttl_seconds=3600)
    hype = ModelHypotheticalGenerator(wrapped, DeterministicHypotheticalGenerator())

    hype.questions("a passage about the rollback subject", workspace_id="tenant-a")
    hype.questions("a passage about the rollback subject", workspace_id="tenant-b")

    assert recorder.workspaces == ["tenant-a", "tenant-b"]
    assert recorder.calls == 2
    assert _row_count(store) == 2


# R1 threading: RAPTOR summary generations carry the tenant.
def test_raptor_summary_generation_threads_workspace_no_cross_tenant_serve(tmp_path: Path) -> None:
    store = _store(tmp_path)
    recorder = _WorkspaceRecordingGenerator(answer="a concise summary of the cluster")
    wrapped = CachingGenerator(recorder, store, ttl_seconds=3600)
    summarizer = ModelSummarizer(wrapped, DeterministicExtractiveSummarizer())

    summarizer.summarize(["a passage about rollback"], workspace_id="tenant-a")
    summarizer.summarize(["a passage about rollback"], workspace_id="tenant-b")

    assert recorder.workspaces == ["tenant-a", "tenant-b"]
    assert _row_count(store) == 2


# R1 threading: model query-transform generations carry the tenant.
def test_query_transform_generation_threads_workspace_no_cross_tenant_serve(tmp_path: Path) -> None:
    store = _store(tmp_path)
    recorder = _WorkspaceRecordingGenerator(answer="restated query")
    wrapped = CachingGenerator(recorder, store, ttl_seconds=3600)
    transform = ModelRewriteQueryTransform(wrapped)

    transform.expand("original question", 4, workspace_id="tenant-a")
    transform.expand("original question", 4, workspace_id="tenant-b")

    assert recorder.workspaces == ["tenant-a", "tenant-b"]
    assert _row_count(store) == 2


# R1 fail-safe: the judge builds an unthreaded request, so the fail-safe makes it
# uncacheable and tenant A's judge verdict can never be served to tenant B.
def test_judge_generation_is_uncacheable_when_unthreaded(tmp_path: Path) -> None:
    store = _store(tmp_path)
    recorder = _WorkspaceRecordingGenerator(answer="score: 0.9\nfully supported")
    wrapped = CachingGenerator(recorder, store, ttl_seconds=3600)
    judge = GeneratorJudge(wrapped)

    first = judge.assess("q", "an answer", ["evidence"])
    second = judge.assess("q", "an answer", ["evidence"])

    assert first.score == second.score == 0.9
    assert recorder.workspaces == ["", ""]
    assert recorder.calls == 2
    assert _row_count(store) == 0


# R1 threading, end to end: RLM generations carry the tenant across two workspaces.
def test_rlm_generation_threads_workspace_no_cross_tenant_serve(settings: Settings) -> None:
    generator = _WorkspaceRecordingGenerator(answer="Rollback restores the previously approved snapshot.")
    runtime = (
        RuntimeBuilder(settings.model_copy(update={"model_call_cache_enabled": True})).with_generator(generator).build()
    )
    try:
        for slug in ("tenant-a", "tenant-b"):
            project = runtime.project(workspace=slug, corpus="ops")
            _ingest(runtime, project)
            project.rlm("how does rollback work")
        assert generator.workspaces
        assert "" not in generator.workspaces
        assert len(set(generator.workspaces)) == 2
    finally:
        runtime.close()


# R4: a partially-consumed stream still charges the deterministic prompt estimate.
def test_streaming_charges_prompt_up_front_on_partial_consume(tmp_path: Path) -> None:
    del tmp_path
    inner = _MultiPieceGenerator()
    wrapped = CachingGenerator(inner, None, ttl_seconds=3600)
    request = _request("a somewhat longer question about the rollback policy")

    with budget_scope(10_000) as meter:
        stream = wrapped.stream(request)
        assert next(stream) == "alpha"
        stream.close()
        assert meter is not None
        # Prompt is charged before the first yield; only the one consumed completion
        # piece is charged, never the unconsumed beta/gamma/delta.
        assert meter.tokens_used == _request_prompt_tokens(request) + _estimate_tokens("alpha")
        assert meter.tokens_used >= _request_prompt_tokens(request)


# R5: the tool-planning charge folds in the serialized manifest, so a bigger manifest
# costs more.
def test_tool_planning_charge_grows_with_the_manifest(tmp_path: Path) -> None:
    del tmp_path
    inner = _ToolGenerator()
    wrapped = CachingGenerator(inner, None, ttl_seconds=3600)
    big_manifest = tuple(
        ToolSpec(
            name=f"tool_{index}",
            description="a detailed description of a tool " * 4,
            parameters={"type": "object", "properties": {"q": {"type": "string"}, "n": {"type": "integer"}}},
        )
        for index in range(12)
    )

    with budget_scope(1_000_000) as small_meter:
        wrapped.propose_tool_call("look up the rollback policy", (), frozenset())
        assert small_meter is not None
        small = small_meter.tokens_used
    with budget_scope(1_000_000) as big_meter:
        wrapped.propose_tool_call("look up the rollback policy", big_manifest, frozenset())
        assert big_meter is not None
        big = big_meter.tokens_used

    assert big > small


# R7: the context query is folded into the generation key.
def test_generation_key_separates_a_different_context_query(tmp_path: Path) -> None:
    store = _store(tmp_path)
    inner = _CountingGenerator()
    wrapped = CachingGenerator(inner, store, ttl_seconds=3600)
    base = _request("q")

    wrapped.generate(replace(base, context=replace(base.context, query="alpha")))
    wrapped.generate(replace(base, context=replace(base.context, query="beta")))

    assert inner.calls == 2
    assert _row_count(store) == 2


# R8: scores are keyed exactly, so a difference past the 6th decimal separates entries.
def test_generation_key_separates_scores_past_the_sixth_decimal(tmp_path: Path) -> None:
    store = _store(tmp_path)
    inner = _CountingGenerator()
    wrapped = CachingGenerator(inner, store, ttl_seconds=3600)
    first = _request("q")
    hit = first.context.hits[0]
    second = replace(first, context=replace(first.context, hits=(replace(hit, score=hit.score + 1e-9),)))

    wrapped.generate(first)
    wrapped.generate(second)

    assert inner.calls == 2
    assert _row_count(store) == 2


# R9: one adapter filling both roles closes its shared inner exactly once.
class _DualAdapter:
    """A single object that is both a cacheable Embedder and a cacheable Generator."""

    def __init__(self) -> None:
        self.closed = 0

    @property
    def model_id(self) -> str:
        return "dual-v1"

    @property
    def dimensions(self) -> int:
        return 4

    @property
    def cache_identity(self) -> str:
        return "dual:https://x/v1:dual-v1:4"

    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [(1.0, 0.0, 0.0, 0.0) for _ in texts]

    def generate(self, request: GenerationRequest) -> GenerationResult:
        cited = (request.context.hits[0].chunk.id,) if request.context.hits else ()
        return GenerationResult(answer="dual answer", cited_chunk_ids=cited, usage={})

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        del request
        yield "dual answer"

    def close(self) -> None:
        self.closed += 1


def test_same_adapter_in_both_roles_closes_inner_once(settings: Settings) -> None:
    dual = _DualAdapter()
    runtime = (
        RuntimeBuilder(settings.model_copy(update={"model_call_cache_enabled": True}))
        .with_generator(dual)
        .with_embedder(dual)
        .build()
    )
    runtime.project(workspace="acme", corpus="handbook")
    runtime.close()
    assert dual.closed == 1


class _InMemoryCacheStore:
    def __init__(self) -> None:
        self.entries: dict[tuple[str, str, int, str, str], object] = {}

    def get_exact_cache(self, workspace_id, corpus_id, version, user_scope, query_hash):  # type: ignore[no-untyped-def]
        return self.entries.get((workspace_id, corpus_id, version, user_scope, query_hash))

    def list_cache_candidates(  # type: ignore[no-untyped-def]
        self, workspace_id, corpus_id, version, user_scope, limit
    ):
        target = (workspace_id, corpus_id, version, user_scope)
        scoped = [entry for (w, c, v, s, _), entry in self.entries.items() if (w, c, v, s) == target]
        return scoped[:limit]

    def put_cache(self, entry) -> None:  # type: ignore[no-untyped-def]
        key = (entry.workspace_id, entry.corpus_id, entry.corpus_version, entry.user_scope, entry.query_hash)
        self.entries[key] = entry

    def delete_query_cache_for_corpus(self, workspace_id, corpus_id) -> int:  # type: ignore[no-untyped-def]
        doomed = [k for k in self.entries if k[0] == workspace_id and k[1] == corpus_id]
        for key in doomed:
            del self.entries[key]
        return len(doomed)


class _InMemoryModelCallCache:
    def __init__(self) -> None:
        self.generations: dict[tuple[str, str, str], object] = {}
        self.embeddings: dict[str, tuple[float, ...]] = {}

    def get_model_call(self, workspace_id, model_id, cache_key):  # type: ignore[no-untyped-def]
        return self.generations.get((workspace_id, model_id, cache_key))

    def put_model_call(  # type: ignore[no-untyped-def]
        self, workspace_id, model_id, cache_key, payload, ttl_seconds
    ) -> None:
        self.generations[(workspace_id, model_id, cache_key)] = dict(payload)

    def get_embedding(self, model_id, cache_key):  # type: ignore[no-untyped-def]
        return self.embeddings.get(cache_key)

    def put_embedding(self, model_id, cache_key, vector, ttl_seconds) -> None:  # type: ignore[no-untyped-def]
        self.embeddings[cache_key] = tuple(vector)

    def delete_generation_model_cache(self, workspace_id) -> int:  # type: ignore[no-untyped-def]
        doomed = [k for k in self.generations if k[0] == workspace_id]
        for key in doomed:
            del self.generations[key]
        return len(doomed)

    def delete_embedding_model_cache(self, key_digests) -> int:  # type: ignore[no-untyped-def]
        deleted = 0
        for key in key_digests:
            if self.embeddings.pop(key, None) is not None:
                deleted += 1
        return deleted


def test_injected_cache_backends_route_reads_writes_and_erasure(settings: Settings) -> None:
    cache_store = _InMemoryCacheStore()
    model_cache = _InMemoryModelCallCache()
    runtime = (
        RuntimeBuilder(settings.model_copy(update={"model_call_cache_enabled": True}))
        .with_embedder(_CountingEmbedder())
        .with_generator(_CountingGenerator())
        .with_cache_store(cache_store)
        .with_model_call_cache(model_cache)
        .build()
    )
    try:
        project = runtime.project(workspace="acme", corpus="handbook")
        result = runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id="ops.txt",
                title="Ops",
                media_type="text/plain",
                content=b"Rollback restores the previously approved snapshot before deployment.",
            ),
        )
        first = project.ask("how does rollback work")
        assert first.cached is False
        assert first.citations
        assert cache_store.entries, "the answer cache write did not route to the injected backend"
        assert model_cache.embeddings, "the embedding cache write did not route to the injected backend"
        assert model_cache.generations, "the generation cache write did not route to the injected backend"
        assert project.ask("how does rollback work").cached is True

        runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id="ops.txt",
                title="Ops",
                media_type="text/plain",
                content=b"Rollback now restores a different, newer approved snapshot before deployment.",
            ),
        )
        assert project.ask("how does rollback work").cached is False

        warm_embeddings = len(model_cache.embeddings)
        receipt = project.erase(result.document_id)
        assert receipt.status is ErasureStatus.ERASED
        assert not [k for k in cache_store.entries if k[1] == project.corpus_id]
        assert not model_cache.generations
        assert len(model_cache.embeddings) < warm_embeddings
        assert project.ask("how does rollback work").cached is False
        assert project.erase(result.document_id).status is ErasureStatus.ERASED
    finally:
        runtime.close()


def test_reproduce_unwraps_an_injected_caching_wrapper_to_its_inner(settings: Settings) -> None:
    from refaat.application.composition import RuntimeComposition
    from refaat.infrastructure.model_cache import CachingEmbedder, CachingGenerator

    inner_embed = _CountingEmbedder()
    inner_gen = _CountingGenerator()
    composition = RuntimeComposition(
        embedder=CachingEmbedder(inner_embed, _InMemoryModelCallCache(), ttl_seconds=3600),  # type: ignore[arg-type]
        generator=CachingGenerator(inner_gen, _InMemoryModelCallCache(), ttl_seconds=3600),  # type: ignore[arg-type]
    )
    builder = RuntimeBuilder(settings)
    report = builder.reproduce_from(composition)
    # The shadow borrows the raw inner, never the wrapper, so a production cache client cannot be written.
    assert builder._custom_embedder is inner_embed
    assert builder._custom_generator is inner_gen
    assert id(inner_embed) in builder._borrowed_component_ids
    assert id(inner_gen) in builder._borrowed_component_ids
    assert {"embedder", "generator"} <= {component.name for component in report.reproduced}
