from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextvars import copy_context
from dataclasses import replace
from typing import Any

import pytest

from refaat.application.budget import budget_scope
from refaat.application.runtime import Refaat, RefaatProject, RuntimeBuilder
from refaat.config import Settings
from refaat.domain.models import (
    Demonstration,
    DocumentInput,
    GenerationRequest,
    GenerationResult,
    QueryRequest,
    ReActPolicy,
    ToolCallDelta,
    ToolSpec,
)
from refaat.errors import ValidationError
from refaat.infrastructure.model_cache import CachingGenerator, _estimate_tokens, _request_prompt_tokens
from refaat.programs.corrective import CorrectiveRagProgram
from refaat.programs.ensemble import EnsembleProgram
from refaat.programs.knn import KNNFewShotProgram
from refaat.programs.orchestrator import MultiAgentOrchestrator, TaskRouter
from refaat.programs.policy import BoundedInferencePolicy
from refaat.programs.program import GroundedRagProgram, Program, ProgramResult, Tunables
from refaat.programs.react import GovernedReActProgram, KeywordToolPlanner
from refaat.programs.router import RouterProgram
from refaat.programs.self_rag import SelfRagProgram, _SelfRagAnswer
from refaat.programs.structured import StructuredOutputProgram

pytestmark = pytest.mark.integration

_SIMPLE = "what is rollback"
_COMPLEX = "How does Kafka connect to Postgres, and how does Redis cache the results"


class _StreamingMeteredGenerator:
    """A cacheable generator that streams several pieces and reports usage."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "stream-metered-v1"

    @property
    def cache_identity(self) -> str:
        return "streammeter:https://x/v1:stream-metered-v1"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls += 1
        cited = (request.context.hits[0].chunk.id,) if request.context.hits else ()
        return GenerationResult(
            answer="Rollback restores the snapshot.",
            cited_chunk_ids=cited,
            usage={"prompt_tokens": 4, "completion_tokens": 2},
        )

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        del request
        yield from ("Rollback ", "restores ", "the snapshot.")


def _seed(runtime: Refaat, corpus: str = "handbook") -> RefaatProject:
    project = runtime.project(workspace="acme", corpus=corpus)
    documents = {
        "rollback.txt": "Rollback restores the previously approved harness snapshot before deployment.",
        "kafka.txt": "Kafka connects to Postgres through a sink connector for durable storage.",
        "redis.txt": "Redis caches query results so repeated reads avoid recomputation.",
        "postgres.txt": "Postgres stores the canonical rows that Kafka streams from upstream topics.",
    }
    for source_id, text in documents.items():
        runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(source_id=source_id, title=source_id, media_type="text/plain", content=text.encode()),
        )
    return project


# A streamed run under a run budget survives being driven across per-next context copies
# (the ASGI threadpool boundary), raising no ValueError, and charges the streamed
# completion tokens on the one run meter.
def test_streamed_budget_survives_per_next_context_copy_and_charges(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    from refaat.application import services as services_mod

    captured: dict[str, Any] = {}
    real = services_mod.resolve_stream_meter

    def capturing(*args: Any, **kwargs: Any) -> Any:
        meter = real(*args, **kwargs)
        captured["meter"] = meter
        return meter

    monkeypatch.setattr(services_mod, "resolve_stream_meter", capturing)

    generator = _StreamingMeteredGenerator()
    runtime = (
        RuntimeBuilder(settings.model_copy(update={"run_token_budget": 100_000})).with_generator(generator).build()
    )
    try:
        project = _seed(runtime, corpus="streambudget")
        request = QueryRequest(
            workspace_id=project.workspace_id, corpus_id=project.corpus_id, query="how does rollback work"
        )
        stream = runtime.stream_query(request)
        events: list[dict[str, Any]] = []
        # Drive each next() inside a FRESH context copy, exactly as Starlette's
        # iterate_in_threadpool does. A budget_scope whose ContextVar reset straddled
        # the yields would raise a ValueError here on teardown.
        while True:
            ctx = copy_context()
            try:
                events.append(ctx.run(next, stream))
            except StopIteration:
                break
        meter = captured["meter"]
        assert meter is not None
        assert meter.tokens_used > 0
        assert any(event["type"] == "delta" for event in events)
        assert events[-1]["type"] == "final"
    finally:
        runtime.close()


def _drain(events: Iterator[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(events)


def _final(events: list[dict[str, Any]]) -> dict[str, Any]:
    return next(event for event in events if event["type"] == "final")


def _spy_stream(runtime: Refaat) -> list[QueryRequest]:
    captured: list[QueryRequest] = []
    original = runtime.stream_query

    def spy(request: QueryRequest) -> Iterator[dict[str, Any]]:
        captured.append(request)
        return original(request)

    runtime.stream_query = spy  # type: ignore[method-assign]
    return captured


# A streamed RouterProgram applies routing and folds router_id into the answer cache
# scope, so two distinct router configs get distinct, isolated cache scopes.
def test_streamed_router_routes_and_isolates_cache_scope_by_config(runtime: Refaat) -> None:
    project = _seed(runtime)
    captured = _spy_stream(runtime)

    r1 = RouterProgram(tunables=Tunables(router={"long_query_tokens": 24}))
    r2 = RouterProgram(tunables=Tunables(router={"long_query_tokens": 3}))

    final_1 = _final(_drain(r1.stream(project, inputs={"question": _SIMPLE}, user_id=None, session_id=None)))
    final_1_repeat = _final(_drain(r1.stream(project, inputs={"question": _SIMPLE}, user_id=None, session_id=None)))
    final_2 = _final(_drain(r2.stream(project, inputs={"question": _SIMPLE}, user_id=None, session_id=None)))

    # Routing is applied on the stream path: R1 sees the short query as direct
    # (linear, no expansion); R2's low threshold makes the same query deep.
    req_r1, _, req_r2 = captured[0], captured[1], captured[2]
    assert req_r1.fusion == "linear"
    assert req_r1.query_expansion is False
    assert req_r2.fusion == "rrf"
    assert req_r2.query_expansion is True

    # router_id is folded and distinct per config.
    assert req_r1.router_id is not None
    assert req_r1.router_id.startswith("router:")
    assert req_r2.router_id is not None
    assert req_r2.router_id.startswith("router:")
    assert req_r1.router_id != req_r2.router_id

    # A repeat under R1 is an exact hit in the router's own scope; R2's different
    # router_id isolates its scope, so it does not read R1's entry.
    assert "cag:exact" not in final_1["routes"]
    assert "cag:exact" in final_1_repeat["routes"]
    assert "cag:exact" not in final_2["routes"]


def _execute_request_and_result(
    runtime: Refaat, project: RefaatProject, program: Program, question: str
) -> tuple[QueryRequest, ProgramResult]:
    captured: list[QueryRequest] = []
    original = runtime.query

    def spy(request: QueryRequest) -> Any:
        captured.append(request)
        return original(request)

    runtime.query = spy  # type: ignore[method-assign]
    try:
        result = program.execute(project, inputs={"question": question}, user_id=None, session_id=None)
    finally:
        runtime.query = original  # type: ignore[method-assign]
    return captured[-1], result


def _stream_request_and_final(
    runtime: Refaat, project: RefaatProject, program: Program, question: str
) -> tuple[QueryRequest, dict[str, Any]]:
    captured: list[QueryRequest] = []
    original = runtime.stream_query

    def spy(request: QueryRequest) -> Iterator[dict[str, Any]]:
        captured.append(request)
        return original(request)

    runtime.stream_query = spy  # type: ignore[method-assign]
    try:
        events = _drain(program.stream(project, inputs={"question": question}, user_id=None, session_id=None))
    finally:
        runtime.stream_query = original  # type: ignore[method-assign]
    return captured[-1], _final(events)


def _parity_programs() -> list[Program]:
    pool = (
        Demonstration(question="how does rollback restore the harness snapshot", answer="It restores the snapshot."),
        Demonstration(question="deployment approval gate for production", answer="Promotion needs an approval."),
    )
    return [
        GroundedRagProgram(),
        RouterProgram(),
        CorrectiveRagProgram(),
        KNNFewShotProgram(GroundedRagProgram(), pool, k=2),
        _SelfRagAnswer(tunables=Tunables(), query_expansion=True, max_query_variants=4),
    ]


# For a parity program the streamed answer request must equal the executed one AND, on
# the deterministic offline stack, the streamed final answer must equal the executed
# answer and cite the same chunks. A parity program whose _query_request stops matching
# its execute() (a removed gate, a dropped field) diverges here and fails.
def test_streamed_answer_matches_execute_for_parity_programs(runtime: Refaat) -> None:
    project = _seed(runtime)
    distinguishing = 0
    for program in _parity_programs():
        exec_request, exec_result = _execute_request_and_result(runtime, project, program, _COMPLEX)
        stream_request, final = _stream_request_and_final(runtime, project, program, _COMPLEX)
        assert exec_request == stream_request, f"stream/execute request diverged for {program.name}"
        assert final["answer"] == exec_result.answer, f"streamed answer diverged for {program.name}"
        streamed_citations = [citation["chunk_id"] for citation in final["citations"]]
        executed_citations = [citation.chunk_id for citation in exec_result.citations]
        assert streamed_citations == executed_citations, f"streamed citations diverged for {program.name}"
        if (
            exec_request.query_expansion is not None
            or exec_request.demonstrations
            or exec_request.response_schema is not None
            or exec_request.router_id is not None
        ):
            distinguishing += 1

    # Non-vacuous: several parity programs inject an answer-affecting field beyond the
    # base request, so a broken _query_request returning base defaults would diverge here.
    assert distinguishing >= 4


def _all_program_subclasses() -> set[type[Program]]:
    seen: set[type[Program]] = set()
    stack = list(Program.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
    # Test modules define their own Program doubles; the shipped contract only covers
    # the library's own programs.
    return {cls for cls in seen if cls.__module__.startswith("refaat.")}


# A program that selects among several results (a composition or a multi-attempt policy)
# has no single request that reproduces execute(), so it must refuse streaming. A KNN
# wrapping such an inner must reach the inner's refusal, not bypass it.
def _refusing_programs() -> list[Program]:
    pool = (Demonstration(question="how does rollback restore the snapshot", answer="It restores the snapshot."),)
    return [
        EnsembleProgram([GroundedRagProgram()]),
        MultiAgentOrchestrator({"generalist": GroundedRagProgram()}, TaskRouter({"generalist": ["rollback"]})),
        BoundedInferencePolicy(GroundedRagProgram(), mode="best_of", attempts=3),
        BoundedInferencePolicy(RouterProgram(), mode="refine", attempts=2),
        StructuredOutputProgram(GroundedRagProgram()),
        SelfRagProgram(),
        KNNFewShotProgram(EnsembleProgram([GroundedRagProgram()]), pool, k=1),
    ]


def test_multi_result_programs_refuse_streaming(runtime: Refaat) -> None:
    project = _seed(runtime)
    for program in _refusing_programs():
        with pytest.raises(ValidationError):
            list(program.stream(project, inputs={"question": _SIMPLE}, user_id=None, session_id=None))


# Two StructuredOutput programs with different top_k execute different requests (matching their
# distinct artifact ids), and a default StructuredOutput executes exactly the plain inner request.
def test_structured_output_executes_inner_with_outer_tunables(runtime: Refaat) -> None:
    project = _seed(runtime)
    inner = GroundedRagProgram()
    default = StructuredOutputProgram(inner)
    tuned = StructuredOutputProgram(inner, tunables=Tunables(top_k=1, token_budget=120))

    plain_request, _ = _execute_request_and_result(runtime, project, inner, _COMPLEX)
    default_request, _ = _execute_request_and_result(runtime, project, default, _COMPLEX)
    tuned_request, _ = _execute_request_and_result(runtime, project, tuned, _COMPLEX)

    # A default structured program adds only the enforced output schema: strip it and the
    # executed request equals the plain inner request, so the default path is unchanged.
    assert default_request.response_schema is not None
    assert replace(default_request, response_schema=None) == plain_request
    # A tuned structured program executes its inner with the outer tunables, a distinct request.
    assert tuned_request.top_k == 1
    assert tuned_request.token_budget == 120
    assert tuned_request != default_request
    assert tuned.artifact().artifact_id != default.artifact().artifact_id
    assert default.artifact().artifact_id == StructuredOutputProgram(GroundedRagProgram()).artifact().artifact_id


# The governed trajectory answers on top_k and token_budget only. Its streamed final-answer
# request must equal its executed final-answer request AND equal that narrowed request, so
# removing its _query_request override (which would fall back to the base request that also
# carries instructions and demonstrations) fails here.
def test_governed_react_final_answer_request_is_narrowed_on_both_paths(runtime: Refaat) -> None:
    project = _seed(runtime)
    program = GovernedReActProgram(
        policy=ReActPolicy(max_iterations=2),
        planner=KeywordToolPlanner({}),
        skills={},
        tunables=Tunables(
            top_k=2,
            token_budget=256,
            instructions="answer tersely",
            demonstrations=(Demonstration(question="what is rollback", answer="It restores the snapshot."),),
        ),
    )
    expected = QueryRequest(
        workspace_id=project.workspace_id,
        corpus_id=project.corpus_id,
        query=_COMPLEX,
        user_id=None,
        session_id=None,
        top_k=2,
        token_budget=256,
    )
    exec_request, _ = _execute_request_and_result(runtime, project, program, _COMPLEX)
    stream_request, _ = _stream_request_and_final(runtime, project, program, _COMPLEX)
    assert exec_request == stream_request
    assert exec_request == expected
    assert exec_request.instructions is None
    assert not exec_request.demonstrations


# Completeness: every Program subclass is accounted for by the stream contract, so a new
# program cannot silently inherit a diverging base stream. A subclass is classified as
# PARITY (execute routes through self._query_request, reused by the base stream), REFUSE
# (overrides stream to raise on a multi-result selection), or GOVERNED_TRAJECTORY (its own
# stream emits the trajectory then delegates the final answer to self._query_request). A
# new subclass forces an author to classify it, and the behavioral parity and refuse tests
# above make removing a guard from a whitelisted class fail.
_PARITY_CLASSES = {
    GroundedRagProgram,
    RouterProgram,
    CorrectiveRagProgram,
    KNNFewShotProgram,
    _SelfRagAnswer,
}
_REFUSE_CLASSES = {
    BoundedInferencePolicy,
    EnsembleProgram,
    MultiAgentOrchestrator,
    StructuredOutputProgram,
    SelfRagProgram,
}
_GOVERNED_TRAJECTORY_CLASSES = {GovernedReActProgram}


def test_every_program_subclass_is_covered_by_the_stream_contract() -> None:
    covered = _PARITY_CLASSES | _REFUSE_CLASSES | _GOVERNED_TRAJECTORY_CLASSES
    actual = _all_program_subclasses()
    unclassified = {cls.__name__ for cls in actual ^ covered}
    assert actual == covered, (
        "an unclassified Program subclass: classify it as parity (execute routes through "
        "self._query_request), refuse (override stream to raise on a multi-result selection), "
        f"or governed trajectory (own stream, final answer via self._query_request): {unclassified}"
    )


class _MeteredGenerator:
    """A cacheable generator that reports fixed usage per generate."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "canary-metered-v1"

    @property
    def cache_identity(self) -> str:
        return "canarymeter:https://x/v1:canary-metered-v1"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls += 1
        cited = (request.context.hits[0].chunk.id,) if request.context.hits else ()
        return GenerationResult(
            answer="Rollback restores the snapshot.",
            cited_chunk_ids=cited,
            usage={"prompt_tokens": 4, "completion_tokens": 2},
        )

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        del request
        yield "Rollback restores the snapshot."


def _promote(project: RefaatProject, environment: str) -> tuple[str, str]:
    base = GroundedRagProgram().artifact()
    candidate = GroundedRagProgram(tunables=Tunables(top_k=2)).artifact()
    registry = project.registry
    registry.save(base)
    registry.save(candidate)
    registry.promote("grounded-rag", environment, base.artifact_id)
    return base.artifact_id, candidate.artifact_id


# An unserved canary shadow run does not charge the run budget; only the served
# run does, so a shared outer meter is charged exactly once per canary request.
def test_canary_shadow_does_not_charge_the_budget(settings: Settings) -> None:
    generator = _MeteredGenerator()
    runtime = RuntimeBuilder(settings.model_copy(update={"run_token_budget": 1000})).with_generator(generator).build()
    try:
        project = _seed(runtime, corpus="canary")
        _, candidate_id = _promote(project, "staging")
        project.deployment.set_canary("grounded-rag", "staging", candidate_id, percent=0)

        with budget_scope(1_000_000) as meter:
            result = project.deployment.run("grounded-rag", "staging", inputs={"question": "how does rollback work"})
            assert meter is not None
            served_charge = meter.tokens_used

        # Both the unserved shadow and the served run call the real generator.
        assert generator.calls == 2
        assert result.served == "active"
        # Yet only the served run charges: one generate worth of usage (4 + 2).
        assert served_charge == 6
    finally:
        runtime.close()


class _StreamingToolGenerator:
    """A cacheable generator that streams tool-call deltas."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "stream-tool-v1"

    @property
    def cache_identity(self) -> str:
        return "streamtool:https://x/v1:stream-tool-v1"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        cited = (request.context.hits[0].chunk.id,) if request.context.hits else ()
        return GenerationResult(answer="answer", cited_chunk_ids=cited, usage={"prompt_tokens": 1})

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        del request
        yield "answer"

    def stream_tool_call(self, query: str, tools: Sequence[ToolSpec], used: frozenset[str]) -> Iterator[ToolCallDelta]:
        del query, tools, used
        self.calls += 1
        yield ToolCallDelta(tool="lookup")
        yield ToolCallDelta(arguments_fragment='{"topic": "rollback"}')
        yield ToolCallDelta(done=True)


# A streamed tool call routes through the single cache/meter composition root, so
# the wrapped generator's stream_tool_call charges the active meter.
def test_streamed_tool_call_charges_the_active_meter() -> None:
    inner = _StreamingToolGenerator()
    wrapped = CachingGenerator(inner, None, ttl_seconds=3600)
    tools = (ToolSpec(name="lookup", description="lookup a fact", parameters={}),)

    with budget_scope(1000) as meter:
        deltas = list(wrapped.stream_tool_call("look up the rollback policy", tools, frozenset()))
        assert meter is not None
        assert meter.tokens_used > 0

    assert inner.calls == 1
    assert [delta.tool for delta in deltas if delta.tool] == ["lookup"]


# RuntimeBuilder's generator wrap gives the injected dual adapter a metered
# stream_tool_call, and the double-wrap guard leaves an already-wrapped one untouched.
def test_builder_wrap_meters_a_streamed_tool_call_and_guards_double_wrap(settings: Settings) -> None:
    from refaat.clock import SystemClock
    from refaat.infrastructure.database.store import SQLStore

    builder = RuntimeBuilder(settings.model_copy(update={"run_token_budget": 1000}))
    store = SQLStore(settings.database_url, SystemClock())
    store.initialize()
    try:
        inner = _StreamingToolGenerator()
        wrapped = builder._wrap_generator(inner, store)
        assert isinstance(wrapped, CachingGenerator)
        again = builder._wrap_generator(wrapped, store)
        assert again is wrapped

        tools = (ToolSpec(name="lookup", description="lookup a fact", parameters={}),)
        with budget_scope(1000) as meter:
            list(wrapped.stream_tool_call("look up rollback", tools, frozenset()))  # type: ignore[attr-defined]
            assert meter is not None
            assert meter.tokens_used > 0
    finally:
        store.close()


class _MultiPieceStreamGenerator:
    """Streams fixed pieces; generate reports a fixed real usage."""

    @property
    def model_id(self) -> str:
        return "multi-piece-v1"

    @property
    def cache_identity(self) -> str:
        return "multi:https://x/v1:multi-piece-v1"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        cited = (request.context.hits[0].chunk.id,) if request.context.hits else ()
        return GenerationResult(
            answer="alphabetagammadelta", cited_chunk_ids=cited, usage={"prompt_tokens": 5, "completion_tokens": 7}
        )

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        del request
        yield from ("alpha", "beta", "gamma", "delta")


def _tool_request() -> GenerationRequest:
    from refaat.domain.models import Chunk, ContextPacket, SearchHit

    chunk = Chunk(
        id="chunk_1",
        corpus_id="corpus",
        document_id="doc",
        ordinal=0,
        text="Evidence text about rollback policy",
        token_count=5,
        checksum="x",
        metadata={},
        embedding=(1.0, 0.0),
    )
    context = ContextPacket(
        query="q",
        hits=(SearchHit(chunk=chunk, score=1.0),),
        graph_facts=(),
        session_summary=None,
        profile={},
        supplemental_harness={},
        token_count=5,
        routes=("semantic",),
    )
    return GenerationRequest(
        query="a question about rollback", context=context, system_prompt="ground", workspace_id="ws"
    )


# Streamed metering is a documented deterministic estimate (prompt plus each
# completion piece), which is distinct from generate charging the provider's real
# usage. Both are prompt plus completion; only the streamed side is an estimate.
def test_streamed_metering_is_a_prompt_plus_completion_estimate_distinct_from_generate() -> None:
    inner = _MultiPieceStreamGenerator()
    wrapped = CachingGenerator(inner, None, ttl_seconds=3600)
    request = _tool_request()

    with budget_scope(1_000_000) as stream_meter:
        list(wrapped.stream(request))
        assert stream_meter is not None
        expected = _request_prompt_tokens(request) + sum(
            _estimate_tokens(piece) for piece in ("alpha", "beta", "gamma", "delta")
        )
        assert stream_meter.tokens_used == expected

    with budget_scope(1_000_000) as generate_meter:
        wrapped.generate(request)
        assert generate_meter is not None
        # generate charges the provider's real reported usage, not the estimate.
        assert generate_meter.tokens_used == 12
