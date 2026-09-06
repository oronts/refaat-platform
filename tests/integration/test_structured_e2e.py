from __future__ import annotations

from dataclasses import replace

import pytest

from refaat.application.runtime import Refaat
from refaat.domain.models import Citation, DocumentInput, GenerationRequest, QueryRequest, QueryResponse
from refaat.evaluation import Dataset, default_bundle
from refaat.infrastructure.generation import ExtractiveGenerator
from refaat.programs import (
    GroundedRagProgram,
    InputField,
    OutputField,
    Signature,
    StructuredOutputProgram,
)
from refaat.programs.program import GroundedRagOutput, Program, ProgramResult

pytestmark = pytest.mark.integration

_DOC = DocumentInput(
    source_id="ops.txt",
    title="Ops",
    media_type="text/plain",
    content=b"Rollback restores the previously approved harness snapshot before deployment.",
)


def _ingest(runtime: Refaat, corpus: str):
    project = runtime.project(workspace="acme", corpus=corpus)
    runtime.ingest(project.workspace_id, project.corpus_id, _DOC)
    return project


class _UnprojectableOutput(Signature):
    question: str = InputField(description="q")
    answer: str = OutputField(description="a", grounded=True)
    score: int = OutputField(description="required but never projectable", required=True)


class _AlwaysViolatesProgram(Program):
    name = "always-violates"
    input_signature = _UnprojectableOutput
    output_signature = _UnprojectableOutput

    def __init__(self, counter: list[int]) -> None:
        super().__init__()
        self._counter = counter

    def execute(self, project, *, inputs, user_id, session_id, principal=None) -> ProgramResult:
        self._counter.append(1)
        response = QueryResponse(
            answer="grounded text [chunk:c1]",
            citations=(Citation(chunk_id="c1", document_id="d", label="[0]", quote="q"),),
            routes=(),
            cached=False,
            corpus_version=1,
            trace_id="t",
        )
        return self._result(response)


def test_structured_program_validates_deterministically_over_the_offline_stack(runtime: Refaat) -> None:
    project = _ingest(runtime, "structured")
    program = StructuredOutputProgram(GroundedRagProgram())

    result = program.execute(project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None)

    assert result.grounded
    validator_names = {outcome.validator for outcome in result.validations}
    assert {"grounding", "structured-output"} <= validator_names
    assert all(outcome.passed for outcome in result.validations)
    assert result.citations
    retrieved = project.retrieve("how does rollback work")
    retrieved_ids = {hit.chunk.id for hit in retrieved.hits}
    assert all(citation.chunk_id in retrieved_ids for citation in result.citations)


def test_schema_violation_drives_the_bounded_refine_loop_to_the_cap(runtime: Refaat) -> None:
    project = _ingest(runtime, "violates")
    counter: list[int] = []
    program = StructuredOutputProgram(_AlwaysViolatesProgram(counter), max_attempts=3)

    result = program.execute(project, inputs={"question": "anything"}, user_id=None, session_id=None)

    assert len(counter) == 3
    assert not result.grounded


def test_built_in_program_populates_the_query_request_response_schema(runtime: Refaat) -> None:
    project = _ingest(runtime, "reachable")
    captured: dict[str, object] = {}
    original = runtime.query

    def spy(request: QueryRequest) -> QueryResponse:
        captured["response_schema"] = request.response_schema
        return original(request)

    runtime.query = spy  # type: ignore[method-assign]
    try:
        StructuredOutputProgram(GroundedRagProgram()).execute(
            project, inputs={"question": "rollback"}, user_id=None, session_id=None
        )
    finally:
        runtime.query = original  # type: ignore[method-assign]

    assert captured["response_schema"] == GroundedRagOutput.json_schema()


def test_extractive_generator_ignores_the_schema_honesty_boundary(runtime: Refaat) -> None:
    project = _ingest(runtime, "honesty")
    context = project.retrieve("rollback")
    generator = ExtractiveGenerator()
    plain = generator.generate(GenerationRequest(query="rollback", context=context, system_prompt="s"))
    with_schema = generator.generate(
        GenerationRequest(
            query="rollback",
            context=context,
            system_prompt="s",
            response_schema=GroundedRagOutput.json_schema(),
        )
    )

    assert plain.answer == with_schema.answer
    assert plain.cited_chunk_ids == with_schema.cited_chunk_ids


def test_schema_enforced_and_plain_asks_use_distinct_cache_scopes(runtime: Refaat) -> None:
    project = _ingest(runtime, "cache")
    plain_request = QueryRequest(workspace_id=project.workspace_id, corpus_id=project.corpus_id, query="rollback")
    schema_request = replace(plain_request, response_schema=GroundedRagOutput.json_schema())

    plain_prep = runtime.queries._prepare(plain_request)
    schema_prep = runtime.queries._prepare(schema_request)

    assert plain_prep.cache_scope != schema_prep.cache_scope


def test_structured_program_runs_through_the_bound_program_and_evaluator(runtime: Refaat) -> None:
    project = _ingest(runtime, "bound")
    bound = project.programs.register(StructuredOutputProgram(GroundedRagProgram()))

    run = bound.run(question="how does rollback work")
    assert run.grounded

    dataset = Dataset.from_records(
        "structured-regression",
        [{"query": "how does rollback work", "expected_terms": ["rollback"]}],
    )
    report = bound.evaluate(dataset, default_bundle())
    assert report.case_count == 1
    assert report.hard_constraints_passed is True
