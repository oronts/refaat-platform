"""Degradation on a generation provider outage across the query and RLM answer seams.

With no policy installed a provider outage propagates uncaught (the legacy refuse), on both the
query and RLM paths. RefuseAll refuses the same way explicitly. SafeDefaults serves a grounded,
cited degraded answer when retrieved evidence survives, abstains when it does not, emits one
redacted degradation audit, and never writes the degraded answer to the answer cache. The pinned
grounded-rag program artifact id is unchanged, which the wider suite proves for byte-identity when
unconfigured.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from refaat.application.rlm import RLM_DEGRADED_ABSTAIN_MESSAGE
from refaat.application.runtime import RefaatProject, RuntimeBuilder
from refaat.application.services import ABSTAIN_MESSAGE
from refaat.config import Settings
from refaat.domain.audit import AuditEvent
from refaat.domain.authz import CanonicalAccessPolicy, Principal
from refaat.domain.degradation import (
    DegradationDecision,
    DependencyFailure,
    RefuseAllDegradationPolicy,
    SafeDefaultsDegradationPolicy,
)
from refaat.domain.models import DocumentInput, GenerationRequest, GenerationResult, QueryRequest, RlmLimits
from refaat.domain.output_governance import OutputContext, OutputDecision
from refaat.errors import DependencyUnavailableError
from refaat.programs.program import GroundedRagProgram

pytestmark = pytest.mark.integration

PINNED_GROUNDED_RAG_ID = "sha256:b76d73f8d956bcd38ccc9d4c24372b07b387647a189f3a9e300cded186febbb4"
QUERY = "internal rollout plan timeline"
MARKER = "SECRET-DELTA-42"
CONTENT = (
    b"The internal rollout plan timeline covers the production release milestone and the "
    b"SECRET-DELTA-42 rollout details for the launch."
).replace(b"\n", b" ")
_RLM_LIMITS = RlmLimits(max_depth=1, max_nodes=4, branch_factor=2, leaf_token_budget=300)


class _OutageGenerator:
    """A generator that fails with a provider outage until it is marked recovered."""

    def __init__(self, *, recovered: bool = False) -> None:
        self.recovered = recovered
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "outage-v1"

    def _answer(self, request: GenerationRequest) -> GenerationResult:
        hits = request.context.hits
        if not hits:
            return GenerationResult(answer="ok", cited_chunk_ids=(), usage={"provider": "outage"})
        hit = hits[0]
        return GenerationResult(
            answer=f"{hit.chunk.text} [chunk:{hit.chunk.id}]",
            cited_chunk_ids=(hit.chunk.id,),
            usage={"provider": "outage"},
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls += 1
        if not self.recovered:
            raise DependencyUnavailableError("generation provider down")
        return self._answer(request)

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        self.calls += 1
        if not self.recovered:
            raise DependencyUnavailableError("generation provider down")
        yield self._answer(request).answer
        return {"provider": "outage"}


class _RecordingSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        self.events.append(event)


def _ingest(project: RefaatProject, *, content: bytes = CONTENT) -> None:
    project.runtime.ingest(
        project.workspace_id,
        project.corpus_id,
        DocumentInput(source_id="release.md", title="release.md", media_type="text/markdown", content=content),
    )


def _request(project: RefaatProject) -> QueryRequest:
    return QueryRequest(workspace_id=project.workspace_id, corpus_id=project.corpus_id, query=QUERY)


def test_no_policy_propagates_the_outage_on_the_query_path(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings).with_generator(_OutageGenerator()).build()
    try:
        project = runtime.project(workspace="acme", corpus="q")
        _ingest(project)
        with pytest.raises(DependencyUnavailableError):
            project.ask(QUERY)
    finally:
        runtime.close()


def test_no_policy_propagates_the_outage_on_the_rlm_path(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings).with_generator(_OutageGenerator()).build()
    try:
        project = runtime.project(workspace="acme", corpus="rlm")
        _ingest(project)
        with pytest.raises(DependencyUnavailableError):
            project.rlm(QUERY, limits=_RLM_LIMITS)
    finally:
        runtime.close()


def test_refuse_all_refuses_both_paths(settings: Settings) -> None:
    runtime = (
        RuntimeBuilder(settings)
        .with_generator(_OutageGenerator())
        .with_degradation_policy(RefuseAllDegradationPolicy())
        .build()
    )
    try:
        project = runtime.project(workspace="acme", corpus="refuse")
        _ingest(project)
        with pytest.raises(DependencyUnavailableError):
            project.ask(QUERY)
        with pytest.raises(DependencyUnavailableError):
            project.rlm(QUERY, limits=_RLM_LIMITS)
    finally:
        runtime.close()


def test_safe_defaults_serves_a_grounded_degraded_answer_and_audits_without_caching(settings: Settings) -> None:
    sink = _RecordingSink()
    generator = _OutageGenerator()
    runtime = (
        RuntimeBuilder(settings)
        .with_generator(generator)
        .with_degradation_policy(SafeDefaultsDegradationPolicy())
        .with_audit_sink(sink)
        .build()
    )
    try:
        project = runtime.project(workspace="acme", corpus="degrade")
        _ingest(project)
        response = project.ask(QUERY)
        assert response.degraded is True
        assert response.cached is False
        assert response.citations, "the degraded answer must carry grounded citations"
        assert all(citation.chunk_id for citation in response.citations)

        degradations = [event for event in sink.events if event.event_type == "degradation.decided"]
        assert len(degradations) == 1
        event = degradations[0]
        serialized = str(event.to_json())
        assert QUERY not in serialized
        assert MARKER not in serialized

        # The transient degraded answer was not cached: once the provider recovers the next
        # identical query regenerates a fresh, non-degraded, grounded answer.
        generator.recovered = True
        again = project.ask(QUERY)
        assert again.cached is False
        assert again.degraded is False
        assert again.citations
        # And the recovered answer is cached normally.
        assert project.ask(QUERY).cached is True
    finally:
        runtime.close()


def test_safe_defaults_abstains_when_grounding_does_not_survive(settings: Settings) -> None:
    runtime = (
        RuntimeBuilder(settings)
        .with_generator(_OutageGenerator())
        .with_degradation_policy(SafeDefaultsDegradationPolicy())
        .build()
    )
    try:
        # No document is ingested, so retrieval returns no evidence and grounding cannot survive.
        project = runtime.project(workspace="acme", corpus="empty")
        response = project.ask(QUERY)
        assert response.abstained is True
        assert response.citations == ()
        assert response.answer == ABSTAIN_MESSAGE
    finally:
        runtime.close()


def test_safe_defaults_serves_a_degraded_rlm_run(settings: Settings) -> None:
    sink = _RecordingSink()
    runtime = (
        RuntimeBuilder(settings)
        .with_generator(_OutageGenerator())
        .with_degradation_policy(SafeDefaultsDegradationPolicy())
        .with_audit_sink(sink)
        .build()
    )
    try:
        project = runtime.project(workspace="acme", corpus="rlmdeg")
        _ingest(project)
        result = project.rlm(QUERY, limits=_RLM_LIMITS)
        assert result.degraded is True
        assert result.citations, "the degraded RLM answer must cite real evidence"
        degradations = [event for event in sink.events if event.event_type == "degradation.decided"]
        assert len(degradations) == 1
    finally:
        runtime.close()


def test_degraded_rlm_serve_does_not_record_a_session_turn(settings: Settings) -> None:
    runtime = (
        RuntimeBuilder(settings)
        .with_generator(_OutageGenerator())
        .with_degradation_policy(SafeDefaultsDegradationPolicy())
        .build()
    )
    try:
        project = runtime.project(workspace="acme", corpus="rlmsess")
        _ingest(project)
        session_id = project.create_session()
        result = project.rlm(QUERY, session_id=session_id, limits=_RLM_LIMITS)
        assert result.degraded is True
        assert result.citations
        # Mirrors the query path: a transient-outage degraded answer is never written to session
        # memory, so a later same-principal query cannot surface it as context.
        assert runtime.store.list_session_turns(project.workspace_id, session_id) == []
    finally:
        runtime.close()


def test_recovered_rlm_run_still_records_session_turns(settings: Settings) -> None:
    runtime = (
        RuntimeBuilder(settings)
        .with_generator(_OutageGenerator(recovered=True))
        .with_degradation_policy(SafeDefaultsDegradationPolicy())
        .build()
    )
    try:
        project = runtime.project(workspace="acme", corpus="rlmsessok")
        _ingest(project)
        session_id = project.create_session()
        result = project.rlm(QUERY, session_id=session_id, limits=_RLM_LIMITS)
        assert result.degraded is False
        assert len(runtime.store.list_session_turns(project.workspace_id, session_id)) == 2
    finally:
        runtime.close()


class _AlwaysAbstainPolicy:
    cache_id = "always-abstain-test"

    def decide(self, failure: DependencyFailure) -> DegradationDecision:
        return DegradationDecision.abstain(failure.dependency_class, "test_abstain")


def test_custom_policy_abstains_on_an_rlm_outage(settings: Settings) -> None:
    runtime = (
        RuntimeBuilder(settings)
        .with_generator(_OutageGenerator())
        .with_degradation_policy(_AlwaysAbstainPolicy())
        .build()
    )
    try:
        project = runtime.project(workspace="acme", corpus="rlmabstain")
        _ingest(project)
        session_id = project.create_session()
        result = project.rlm(QUERY, session_id=session_id, limits=_RLM_LIMITS)
        assert result.degraded is True
        assert result.citations == ()
        assert result.answer == RLM_DEGRADED_ABSTAIN_MESSAGE
        assert runtime.store.list_session_turns(project.workspace_id, session_id) == []
    finally:
        runtime.close()


def test_streaming_serves_a_degraded_final_event(settings: Settings) -> None:
    runtime = (
        RuntimeBuilder(settings)
        .with_generator(_OutageGenerator())
        .with_degradation_policy(SafeDefaultsDegradationPolicy())
        .build()
    )
    try:
        project = runtime.project(workspace="acme", corpus="stream")
        _ingest(project)
        events = list(runtime.stream_query(_request(project)))
        final = next(event for event in events if event["type"] == "final")
        assert final["citations"], "the streamed degraded answer must carry citations"
        assert final["replaced"] is True
    finally:
        runtime.close()


def test_no_policy_streaming_propagates_the_outage(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings).with_generator(_OutageGenerator()).build()
    try:
        project = runtime.project(workspace="acme", corpus="streamoff")
        _ingest(project)
        with pytest.raises(DependencyUnavailableError):
            list(runtime.stream_query(_request(project)))
    finally:
        runtime.close()


def test_program_artifact_id_is_unchanged(settings: Settings) -> None:
    assert GroundedRagProgram().artifact().artifact_id == PINNED_GROUNDED_RAG_ID


class _DenyingGovernor:
    @property
    def cache_id(self) -> str:
        return "deny:v1"

    def review(self, output: OutputContext) -> OutputDecision:
        return OutputDecision(allowed=False, reason="output_denied")


def test_served_degraded_output_denied_is_still_audited(settings: Settings) -> None:
    sink = _RecordingSink()
    runtime = (
        RuntimeBuilder(settings)
        .with_generator(_OutageGenerator())
        .with_degradation_policy(SafeDefaultsDegradationPolicy())
        .with_access_policy(CanonicalAccessPolicy())
        .with_audit_sink(sink)
        .with_output_governor(_DenyingGovernor())
        .build()
    )
    try:
        alice = Principal(id="alice", tenant="merck", groups=frozenset({"research-a"}))
        project = runtime.project(workspace="merck", corpus="research")
        project.runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id="release.md",
                title="release.md",
                media_type="text/markdown",
                content=CONTENT,
                metadata={"acl": {"classification": "confidential", "allow_groups": ["research-a"]}},
            ),
        )
        project.ask(QUERY, principal=alice)
        # The served run degrades under SERVE_DEGRADED and the governor denies the fallback answer.
        # _finalize runs under shadow_execution() for write suppression, but the served denial must
        # still be audited: only an unserved deployment run suppresses audit.
        denied = [event for event in sink.events if event.event_type == "output.denied"]
        assert len(denied) == 1
    finally:
        runtime.close()
