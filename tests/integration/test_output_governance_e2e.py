"""Output governance reviews the final answer after generation, before return or cache.

An installed governor refuses or redacts a produced answer plus its resolved citations on
the query, streaming, program, and RLM paths, fail-closed when it raises, never caching a
denial. With no governor installed every path is byte-identical to the default, which the
wider suite proves. A refusal emits a redacted output.denied audit event that never carries
the answer text.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import fields

import pytest

from refaat.application.runtime import RefaatProject, RuntimeBuilder
from refaat.application.services import OUTPUT_REFUSAL_MESSAGE, OUTPUT_REFUSAL_REASON, OUTPUT_ROUTE
from refaat.config import Settings
from refaat.domain.audit import AuditEvent
from refaat.domain.authz import CanonicalAccessPolicy, Principal
from refaat.domain.models import DocumentInput, QueryRequest, RlmLimits
from refaat.domain.output_governance import OutputContext, OutputDecision
from refaat.infrastructure.output_governance import PatternDenylistGovernor

pytestmark = pytest.mark.integration

MARKER = "Bluejay"
QUERY = "internal project codename production rollout"
CONTENT = b"The internal project codename is Bluejay and Bluejay governs the production rollout plan for the release."


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        self.events.append(event)


class RaisingGovernor:
    @property
    def cache_id(self) -> str:
        return "raising:1"

    def review(self, output: OutputContext) -> OutputDecision:
        raise RuntimeError("governor is broken")


def _ingest(project: RefaatProject, *, acl: dict[str, object] | None = None) -> None:
    metadata = {"acl": acl} if acl is not None else {}
    project.runtime.ingest(
        project.workspace_id,
        project.corpus_id,
        DocumentInput(
            source_id="release.md",
            title="release.md",
            media_type="text/markdown",
            content=CONTENT,
            metadata=metadata,
        ),
    )


@pytest.fixture
def guarded(settings: Settings) -> Iterator[RefaatProject]:
    runtime = RuntimeBuilder(settings).with_output_governor(PatternDenylistGovernor([MARKER])).build()
    project = runtime.project(workspace="acme", corpus="guarded")
    _ingest(project)
    try:
        yield project
    finally:
        runtime.close()


def test_a_tripped_answer_is_refused_without_citations(guarded: RefaatProject) -> None:
    response = guarded.ask(QUERY)
    assert response.answer == OUTPUT_REFUSAL_MESSAGE
    assert response.citations == ()
    assert OUTPUT_ROUTE in response.routes
    assert MARKER.casefold() not in response.answer.casefold()
    assert response.abstained is False


def test_a_denied_output_is_not_cached(guarded: RefaatProject) -> None:
    first = guarded.ask(QUERY)
    assert first.answer == OUTPUT_REFUSAL_MESSAGE
    assert first.cached is False
    # A denial is never cached, so an identical repeat regenerates and refuses again rather
    # than being served from a cached denial row.
    second = guarded.ask(QUERY)
    assert second.answer == OUTPUT_REFUSAL_MESSAGE
    assert second.cached is False


def test_no_governor_answers_normally_and_a_governor_scopes_the_cache(settings: Settings) -> None:
    plain = RuntimeBuilder(settings).build()
    try:
        project = plain.project(workspace="acme", corpus="shared")
        _ingest(project)
        answer = project.ask(QUERY)
        assert MARKER.casefold() in answer.answer.casefold()
        assert answer.citations
        assert OUTPUT_ROUTE not in answer.routes
        # A no-governor runtime caches the answer, so an identical repeat is served from the
        # cache: the answer-cache key carries no output_governor term.
        again = project.ask(QUERY)
        assert again.cached is True
    finally:
        plain.close()

    guarded = RuntimeBuilder(settings).with_output_governor(PatternDenylistGovernor([MARKER])).build()
    try:
        project = guarded.project(workspace="acme", corpus="shared")
        response = project.ask(QUERY)
        # The governor identity scopes the cache, so the permissive cached answer is not
        # served; the query re-routes through the governor and refuses.
        assert response.answer == OUTPUT_REFUSAL_MESSAGE
        assert OUTPUT_ROUTE in response.routes
        assert response.cached is False
    finally:
        guarded.close()


def test_redact_mode_serves_a_redacted_grounded_answer(settings: Settings) -> None:
    runtime = (
        RuntimeBuilder(settings)
        .with_output_governor(PatternDenylistGovernor([MARKER], mode="redact", placeholder="[redacted]"))
        .build()
    )
    try:
        project = runtime.project(workspace="acme", corpus="redact")
        _ingest(project)
        response = project.ask(QUERY)
        assert MARKER.casefold() not in response.answer.casefold()
        assert "[redacted]" in response.answer
        assert OUTPUT_ROUTE in response.routes
        assert response.citations
    finally:
        runtime.close()


def test_a_governor_that_raises_is_treated_as_a_denial(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings).with_output_governor(RaisingGovernor()).build()
    try:
        project = runtime.project(workspace="acme", corpus="raising")
        _ingest(project)
        response = project.ask(QUERY)
        assert response.answer == OUTPUT_REFUSAL_MESSAGE
        assert OUTPUT_ROUTE in response.routes
        assert response.citations == ()
    finally:
        runtime.close()


def test_streaming_with_a_governor_buffers_and_never_yields_a_raw_token(guarded: RefaatProject) -> None:
    request = QueryRequest(workspace_id=guarded.workspace_id, corpus_id=guarded.corpus_id, query=QUERY)
    events = list(guarded.runtime.stream_query(request))
    deltas = [event for event in events if event["type"] == "delta"]
    # Exactly one governed delta: the buffered path emits the refusal as a single delta.
    assert len(deltas) == 1
    assert deltas[0]["text"] == OUTPUT_REFUSAL_MESSAGE
    # Adversarial: no raw pre-governed token (the marker) was ever streamed.
    streamed = "".join(event["text"] for event in deltas)
    assert MARKER.casefold() not in streamed.casefold()
    final = next(event for event in events if event["type"] == "final")
    assert final["answer"] == OUTPUT_REFUSAL_MESSAGE
    assert final["citations"] == []
    assert OUTPUT_ROUTE in final["routes"]


def test_streaming_without_a_governor_is_incremental_and_unredacted(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings).build()
    try:
        project = runtime.project(workspace="acme", corpus="stream-plain")
        _ingest(project)
        request = QueryRequest(workspace_id=project.workspace_id, corpus_id=project.corpus_id, query=QUERY)
        events = list(runtime.stream_query(request))
        streamed = "".join(event["text"] for event in events if event["type"] == "delta")
        # The marker reaches the caller incrementally: the no-governor path is not buffered.
        assert MARKER.casefold() in streamed.casefold()
        final = next(event for event in events if event["type"] == "final")
        assert OUTPUT_ROUTE not in final["routes"]
    finally:
        runtime.close()


def test_output_denied_audit_is_emitted_and_redacted(settings: Settings) -> None:
    sink = RecordingSink()
    alice = Principal(id="alice", tenant="acme", groups=frozenset({"eng"}))
    runtime = (
        RuntimeBuilder(settings)
        .with_access_policy(CanonicalAccessPolicy())
        .with_audit_sink(sink)
        .with_output_governor(PatternDenylistGovernor([MARKER]))
        .build()
    )
    try:
        project = runtime.project(workspace="acme", corpus="audited")
        _ingest(project, acl={"classification": "confidential", "allow_groups": ["eng"]})
        response = project.ask(QUERY, principal=alice)
        assert response.answer == OUTPUT_REFUSAL_MESSAGE
        denied = [event for event in sink.events if event.event_type == "output.denied"]
        assert len(denied) == 1
        event = denied[0]
        assert event.capability_name == "output_matched_denied_pattern"
        assert event.principal_id == "alice"
        assert event.citation_count == 0
        assert event.cited_document_ids == ()
        # Redacted by construction: no answer text or marker in any field.
        values = " ".join(str(getattr(event, field.name)) for field in fields(event))
        assert MARKER.casefold() not in values.casefold()
        assert "codename" not in values.casefold()
    finally:
        runtime.close()


def test_a_program_run_answer_that_trips_the_pattern_is_refused(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings).with_output_governor(PatternDenylistGovernor([MARKER])).build()
    try:
        project = runtime.project(workspace="acme", corpus="program")
        _ingest(project)
        result = project.programs.get("grounded-rag").run(question=QUERY)
        # A program's final answer routes through the query pipeline, so the QueryService
        # seam covers it: the refusal is returned and it is not grounded.
        assert result.answer == OUTPUT_REFUSAL_MESSAGE
        assert result.citations == ()
        assert result.grounded is False
    finally:
        runtime.close()


def test_an_rlm_answer_that_trips_the_pattern_is_refused_and_records_no_session_turn(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings).with_output_governor(PatternDenylistGovernor([MARKER])).build()
    try:
        project = runtime.project(workspace="acme", corpus="rlm")
        _ingest(project)
        session_id = project.create_session()
        result = project.rlm(
            QUERY,
            session_id=session_id,
            limits=RlmLimits(max_depth=1, max_nodes=4, branch_factor=2, leaf_token_budget=300),
        )
        assert result.answer == OUTPUT_REFUSAL_MESSAGE
        assert result.citations == ()
        assert result.usage.get("output_denied") == OUTPUT_REFUSAL_REASON
        # A denial returns before _record_session, so the session carries no turn.
        assert runtime.store.list_session_turns(project.workspace_id, session_id) == []
    finally:
        runtime.close()


class RedactingGovernor:
    @property
    def cache_id(self) -> str:
        return "redacting:1"

    def review(self, output: OutputContext) -> OutputDecision:
        return OutputDecision(allowed=True, reason="output_denied", redacted_answer="[redacted]")


def test_a_redacted_answer_blanks_its_citation_quotes(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings).with_output_governor(RedactingGovernor()).build()
    try:
        project = runtime.project(workspace="acme", corpus="redact")
        _ingest(project)
        response = project.ask(QUERY)
        assert response.answer == "[redacted]"
        assert response.citations, "provenance ids survive a redaction"
        assert all(citation.quote == "" for citation in response.citations)
        assert MARKER.casefold() not in str([c.quote for c in response.citations]).casefold()
    finally:
        runtime.close()
