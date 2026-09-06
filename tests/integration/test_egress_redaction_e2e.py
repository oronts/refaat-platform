"""Generation-evidence egress redaction across the query, stream, RLM, and CRAG-judge paths.

An installed context transform redacts configured spans from the evidence a generation model sees,
after the routing gate and before the generate or stream call, on every corpus-evidence egress: the
query path, both streaming branches, the RLM node egress, and the corrective-RAG judge. A citation
still resolves to the original stored chunk (redaction is an egress control against the model
provider, not a user-hiding control), and the local extractive fallback keeps the original evidence
so it grounds and cites real chunks. Under an active transform, image egress is refused. Redaction
provenance in usage carries only pattern-id digests and counts. With no transform installed every
path is byte-identical, which the wider suite proves; an installed transform scopes the answer cache.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

import pytest

from refaat.application.runtime import RefaatProject, RuntimeBuilder
from refaat.config import Settings
from refaat.domain.models import ContextPacket, DocumentInput, GenerationRequest, GenerationResult, RlmLimits
from refaat.infrastructure.context_transform import PatternRedactor
from refaat.infrastructure.output_governance import PatternDenylistGovernor
from refaat.programs import CorrectiveRagProgram, ModelConfidenceEvaluator
from refaat.programs.corrective import ConfidenceAssessment, DeterministicConfidenceEvaluator

pytestmark = pytest.mark.integration

MARKER = "BRAVO-SECRET-9"
QUERY = "internal launch code rollout"
CONTENT = (
    b"The internal launch code is BRAVO-SECRET-9 and BRAVO-SECRET-9 governs the production "
    b"rollout plan for the release milestone."
)

_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc``\x00\x00\x00"
    b"\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
)


class RecordingGenerator:
    """A generator that records the request it receives and cites the top retrieved chunk."""

    def __init__(self) -> None:
        self.seen: list[GenerationRequest] = []

    @property
    def model_id(self) -> str:
        return "recording-v1"

    def _result(self, request: GenerationRequest) -> GenerationResult:
        hits = request.context.hits
        if not hits:
            return GenerationResult(answer="no evidence", cited_chunk_ids=(), usage={"provider": "rec"})
        hit = hits[0]
        return GenerationResult(
            answer=f"{hit.chunk.text} [chunk:{hit.chunk.id}]",
            cited_chunk_ids=(hit.chunk.id,),
            usage={"provider": "rec", "total_tokens": 9},
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.seen.append(request)
        return self._result(request)

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        self.seen.append(request)
        yield self._result(request).answer
        return {"provider": "rec"}


class HallucinatingGenerator:
    """A generator that cites a chunk that does not exist, forcing the extractive fallback."""

    def __init__(self) -> None:
        self.seen: list[GenerationRequest] = []

    @property
    def model_id(self) -> str:
        return "hallucinating-v1"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.seen.append(request)
        return GenerationResult(answer="see [chunk:does-not-exist]", cited_chunk_ids=("does-not-exist",), usage={})

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        self.seen.append(request)
        yield "see [chunk:does-not-exist]"
        return {}


def _ingest(project: RefaatProject, *, content: bytes = CONTENT, source_id: str = "release.md") -> None:
    project.runtime.ingest(
        project.workspace_id,
        project.corpus_id,
        DocumentInput(source_id=source_id, title=source_id, media_type="text/markdown", content=content),
    )


def _no_marker_in_evidence(request: GenerationRequest) -> bool:
    fields = [hit.chunk.text for hit in request.context.hits]
    fields.extend(request.context.graph_facts)
    if request.context.session_summary:
        fields.append(request.context.session_summary)
    return all(MARKER not in field for field in fields)


def test_the_generation_model_sees_redacted_evidence_on_the_query_path(settings: Settings) -> None:
    rec = RecordingGenerator()
    runtime = RuntimeBuilder(settings).with_generator(rec).with_context_transform(PatternRedactor([MARKER])).build()
    try:
        project = runtime.project(workspace="acme", corpus="q")
        _ingest(project)
        response = project.ask(QUERY)
        assert rec.seen, "the generator was never called"
        assert _no_marker_in_evidence(rec.seen[-1])
        assert "[REDACTED]" in rec.seen[-1].context.hits[0].chunk.text
        assert MARKER not in response.answer
    finally:
        runtime.close()


def test_a_citation_still_resolves_to_the_original_stored_chunk(settings: Settings) -> None:
    rec = RecordingGenerator()
    runtime = RuntimeBuilder(settings).with_generator(rec).with_context_transform(PatternRedactor([MARKER])).build()
    try:
        project = runtime.project(workspace="acme", corpus="cite")
        _ingest(project)
        response = project.ask(QUERY)
        assert response.citations
        # The model saw redacted evidence, but the citation quote is the ORIGINAL stored chunk text,
        # so an authorized local reader still sees the real evidence.
        assert any(MARKER in citation.quote for citation in response.citations)
        assert MARKER not in response.answer
    finally:
        runtime.close()


def test_the_extractive_fallback_keeps_the_original_evidence(settings: Settings) -> None:
    # The primary generator cites a nonexistent chunk, so _finalize falls back to the in-process
    # ExtractiveGenerator. The fallback runs against the ORIGINAL request (never redacted), so its
    # grounded answer is built from the real evidence and cites the real chunk.
    generator = HallucinatingGenerator()
    runtime = (
        RuntimeBuilder(settings).with_generator(generator).with_context_transform(PatternRedactor([MARKER])).build()
    )
    try:
        project = runtime.project(workspace="acme", corpus="fallback")
        _ingest(project)
        response = project.ask(QUERY)
        # The model still saw redacted evidence.
        assert _no_marker_in_evidence(generator.seen[-1])
        # The local fallback answer is grounded in the original evidence and cites a real chunk.
        assert MARKER in response.answer
        assert response.citations
    finally:
        runtime.close()


def test_the_model_sees_redacted_evidence_on_the_plain_streaming_branch(settings: Settings) -> None:
    rec = RecordingGenerator()
    runtime = RuntimeBuilder(settings).with_generator(rec).with_context_transform(PatternRedactor([MARKER])).build()
    try:
        project = runtime.project(workspace="acme", corpus="stream")
        _ingest(project)
        events = list(runtime.stream_query(_request(project)))
        assert rec.seen
        assert _no_marker_in_evidence(rec.seen[-1])
        streamed = "".join(event["text"] for event in events if event["type"] == "delta")
        assert MARKER not in streamed
    finally:
        runtime.close()


def test_the_model_sees_redacted_evidence_on_the_buffered_governed_stream(settings: Settings) -> None:
    rec = RecordingGenerator()
    runtime = (
        RuntimeBuilder(settings)
        .with_generator(rec)
        .with_context_transform(PatternRedactor([MARKER]))
        # An empty-denylist governor always allows, so the buffered-governed branch runs (which
        # drains the stream) and returns the grounded answer.
        .with_output_governor(PatternDenylistGovernor([]))
        .build()
    )
    try:
        project = runtime.project(workspace="acme", corpus="buffered")
        _ingest(project)
        events = list(runtime.stream_query(_request(project)))
        assert rec.seen
        assert _no_marker_in_evidence(rec.seen[-1])
        assert "[REDACTED]" in rec.seen[-1].context.hits[0].chunk.text
        streamed = "".join(event["text"] for event in events if event["type"] == "delta")
        assert MARKER not in streamed
    finally:
        runtime.close()


def test_the_model_sees_redacted_evidence_on_the_rlm_path(settings: Settings) -> None:
    rec = RecordingGenerator()
    runtime = RuntimeBuilder(settings).with_generator(rec).with_context_transform(PatternRedactor([MARKER])).build()
    try:
        project = runtime.project(workspace="acme", corpus="rlm")
        _ingest(project)
        result = project.rlm(QUERY, limits=RlmLimits(max_depth=1, max_nodes=4, branch_factor=2, leaf_token_budget=300))
        assert rec.seen
        assert all(_no_marker_in_evidence(request) for request in rec.seen)
        assert MARKER not in result.answer
    finally:
        runtime.close()


def test_the_corrective_rag_judge_sees_redacted_evidence(settings: Settings) -> None:
    judge = RecordingGenerator()
    # A model-backed confidence evaluator whose judge model is the recording generator. The CRAG
    # program threads project.context_transform into it, so the judge sees redacted evidence.
    runtime = RuntimeBuilder(settings).with_context_transform(PatternRedactor([MARKER])).build()
    try:
        project = runtime.project(workspace="acme", corpus="crag")
        _ingest(project)
        program = CorrectiveRagProgram(evaluator=ModelConfidenceEvaluator(judge))
        program.execute(project, inputs={"question": QUERY}, user_id=None, session_id=None)
        assert judge.seen, "the judge model was never called"
        assert all(_no_marker_in_evidence(request) for request in judge.seen)
    finally:
        runtime.close()


def test_provenance_in_usage_carries_only_pattern_ids_and_counts(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings).with_context_transform(PatternRedactor([MARKER])).build()
    try:
        project = runtime.project(workspace="acme", corpus="prov")
        _ingest(project)
        response = project.ask(QUERY)
        provenance = response.usage.get("egress_redaction")
        assert provenance is not None
        redactions = provenance["redactions"]
        assert redactions
        assert all(set(item) == {"pattern_id", "count"} for item in redactions)
        assert all(item["count"] >= 1 for item in redactions)
        # No raw pattern or matched text anywhere in the usage provenance.
        assert MARKER not in str(provenance)
    finally:
        runtime.close()


def test_images_are_dropped_under_an_active_transform(settings: Settings) -> None:
    rec = RecordingGenerator()
    runtime = (
        RuntimeBuilder(settings)
        .with_multimodal_generator(rec)
        .with_context_transform(PatternRedactor([MARKER]))
        .build()
    )
    try:
        project = runtime.project(workspace="acme", corpus="img")
        project.ingest_image(
            content=_PNG,
            media_type="image/png",
            source_id="chart.png",
            caption="quarterly revenue chart for BRAVO-SECRET-9",
        )
        response = project.ask("quarterly revenue chart")
        assert rec.seen
        # A vision model would have received pixels, but the seam refuses image egress under an
        # active transform because a text redactor cannot vet pixels.
        assert rec.seen[-1].images == ()
        provenance = response.usage.get("egress_redaction")
        assert provenance is not None
        assert provenance["images_dropped"] >= 1
    finally:
        runtime.close()


def test_byte_identical_when_no_transform_is_installed(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings).build()
    try:
        project = runtime.project(workspace="acme", corpus="off")
        _ingest(project)
        response = project.ask(QUERY)
        # The default extractive generator sees the original evidence, so the marker reaches the
        # answer and no redaction provenance is recorded.
        assert MARKER in response.answer
        assert "egress_redaction" not in response.usage
    finally:
        runtime.close()


def test_a_redactor_scopes_the_answer_cache(settings: Settings) -> None:
    plain = RuntimeBuilder(settings).build()
    try:
        project = plain.project(workspace="acme", corpus="shared")
        _ingest(project)
        first = project.ask(QUERY)
        assert MARKER in first.answer
        again = project.ask(QUERY)
        assert again.cached is True
    finally:
        plain.close()

    guarded = RuntimeBuilder(settings).with_context_transform(PatternRedactor([MARKER])).build()
    try:
        project = guarded.project(workspace="acme", corpus="shared")
        response = project.ask(QUERY)
        # The redactor identity scopes the cache, so the permissive cached answer is not served; the
        # query regenerates against the redacted evidence.
        assert response.cached is False
        assert MARKER not in response.answer
        assert "[REDACTED]" in response.answer
    finally:
        guarded.close()


def test_changing_or_removing_the_redactor_rescopes_the_cache(settings: Settings) -> None:
    first = RuntimeBuilder(settings).with_context_transform(PatternRedactor([MARKER])).build()
    try:
        project = first.project(workspace="acme", corpus="rescope")
        _ingest(project)
        assert project.ask(QUERY).cached is False
        assert project.ask(QUERY).cached is True
    finally:
        first.close()

    other = RuntimeBuilder(settings).with_context_transform(PatternRedactor([MARKER], placeholder="[HIDDEN]")).build()
    try:
        project = other.project(workspace="acme", corpus="rescope")
        assert project.ask(QUERY).cached is False
    finally:
        other.close()

    plain = RuntimeBuilder(settings).build()
    try:
        project = plain.project(workspace="acme", corpus="rescope")
        assert project.ask(QUERY).cached is False
    finally:
        plain.close()


class _LegacyConfidenceEvaluator:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def cache_id(self) -> str:
        return "legacy-2arg"

    def assess(self, query: str, context: ContextPacket) -> ConfidenceAssessment:
        self.calls += 1
        return DeterministicConfidenceEvaluator().assess(query, context)


def test_a_legacy_two_argument_confidence_evaluator_still_runs(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings).build()
    try:
        project = runtime.project(workspace="acme", corpus="legacy")
        _ingest(project)
        evaluator = _LegacyConfidenceEvaluator()
        program = CorrectiveRagProgram(evaluator=evaluator)
        program.execute(project, inputs={"question": QUERY}, user_id=None, session_id=None)
        assert evaluator.calls == 1
    finally:
        runtime.close()


def test_config_auto_installs_the_reference_redactor(settings: Settings) -> None:
    configured = settings.model_copy(update={"egress_redaction_patterns": [MARKER]})
    runtime = RuntimeBuilder(configured).build()
    try:
        assert isinstance(runtime.queries.context_transform, PatternRedactor)
        assert runtime.rlm.context_transform is runtime.queries.context_transform
        project = runtime.project(workspace="acme", corpus="auto")
        _ingest(project)
        response = project.ask(QUERY)
        assert MARKER not in response.answer
        assert "[REDACTED]" in response.answer
    finally:
        runtime.close()


def test_a_compiled_regex_from_the_builder_redacts_evidence(settings: Settings) -> None:
    rec = RecordingGenerator()
    runtime = (
        RuntimeBuilder(settings)
        .with_generator(rec)
        .with_context_transform(PatternRedactor([re.compile(r"BRAVO-SECRET-\d+")]))
        .build()
    )
    try:
        project = runtime.project(workspace="acme", corpus="regex")
        _ingest(project)
        project.ask(QUERY)
        assert _no_marker_in_evidence(rec.seen[-1])
    finally:
        runtime.close()


def _request(project: RefaatProject) -> object:
    from refaat.domain.models import QueryRequest

    return QueryRequest(workspace_id=project.workspace_id, corpus_id=project.corpus_id, query=QUERY)
