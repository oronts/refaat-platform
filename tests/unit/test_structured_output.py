from __future__ import annotations

import pytest

from refaat.domain.models import Citation, QueryResponse
from refaat.errors import ValidationError
from refaat.programs import (
    BEST_OF,
    BoundedInferencePolicy,
    Demonstration,
    GroundedRagProgram,
    InputField,
    KNNFewShotProgram,
    OutputField,
    Signature,
    StructuredOutputProgram,
    StructuredValidator,
    validate_structured,
)
from refaat.programs.program import GroundedRagOutput
from refaat.programs.schema import compile_output_schema, validate_object
from refaat.programs.toolcall import validate_tool_arguments


class _TypedOutput(Signature):
    question: str = InputField(description="q")
    answer: str = OutputField(description="a", grounded=True)
    count: int = OutputField(description="c", required=True)


class _OptionalOutput(Signature):
    question: str = InputField(description="q")
    answer: str = OutputField(description="a", grounded=True)
    note: str = OutputField(description="n", required=False)


class _UnprojectableOutput(Signature):
    question: str = InputField(description="q")
    answer: str = OutputField(description="a", grounded=True)
    score: int = OutputField(description="not in the projection", required=True)


class _BadAnnotation(Signature):
    question: str = InputField(description="q")
    payload: dict = OutputField(description="unsupported")


def _response(answer: str, chunk_ids: tuple[str, ...]) -> QueryResponse:
    citations = tuple(
        Citation(chunk_id=cid, document_id="doc", label=f"[{i}]", quote="q") for i, cid in enumerate(chunk_ids)
    )
    return QueryResponse(answer=answer, citations=citations, routes=(), cached=False, corpus_version=1, trace_id="t")


def test_grounded_rag_output_compiles_to_the_expected_object_schema() -> None:
    assert GroundedRagOutput.json_schema() == {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "citations": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["answer", "citations"],
        "additionalProperties": False,
    }


def test_output_schema_digest_is_stable_and_content_addressed() -> None:
    assert GroundedRagOutput.output_schema_digest() == GroundedRagOutput.output_schema_digest()
    assert _TypedOutput.output_schema_digest() != GroundedRagOutput.output_schema_digest()


def test_optional_field_is_dropped_from_required() -> None:
    schema = _OptionalOutput.json_schema()
    assert schema["required"] == ["answer"]
    assert "note" in schema["properties"]


def test_typed_annotations_map_through_the_shared_table() -> None:
    assert _TypedOutput.json_schema()["properties"]["count"] == {"type": "integer"}


def test_unknown_annotation_fails_loud() -> None:
    with pytest.raises(ValidationError):
        _BadAnnotation.json_schema()


def test_validate_structured_reuses_the_single_shared_body() -> None:
    schema = GroundedRagOutput.json_schema()
    obj = {"answer": "a", "citations": ["c1"]}
    # One validation body: structured output, tool arguments, and the raw
    # validator agree byte-for-byte on the same input.
    assert validate_structured(obj, schema) == validate_object(obj, schema)
    assert validate_tool_arguments(obj, schema) == validate_object(obj, schema)


def test_validate_structured_accepts_conforming_and_rejects_violations() -> None:
    schema = GroundedRagOutput.json_schema()
    assert validate_structured({"answer": "a", "citations": ["c1"]}, schema) == (True, "valid")
    missing_ok, missing_reason = validate_structured({"citations": ["c1"]}, schema)
    assert not missing_ok
    assert "answer" in missing_reason
    type_ok, type_reason = validate_structured({"answer": 1, "citations": ["c1"]}, schema)
    assert not type_ok
    assert "answer" in type_reason


def test_compile_output_schema_preserves_declaration_order() -> None:
    schema = compile_output_schema([("answer", "str", True), ("count", "int", True), ("note", "str", False)])
    assert list(schema["properties"]) == ["answer", "count", "note"]
    assert schema["required"] == ["answer", "count"]


def test_structured_validator_passes_on_a_projectable_response() -> None:
    validator = StructuredValidator(GroundedRagOutput)
    outcome = validator.check(_response("grounded answer", ("chunk_a",)))
    assert outcome.passed
    assert outcome.validator == "structured-output"


def test_structured_validator_fails_when_a_required_field_is_unprojectable() -> None:
    validator = StructuredValidator(_UnprojectableOutput)
    outcome = validator.check(_response("answer", ("chunk_a",)))
    assert not outcome.passed
    assert "score" in outcome.detail


def test_structured_program_artifact_differs_from_its_inner() -> None:
    inner = GroundedRagProgram()
    assert StructuredOutputProgram(inner).artifact().artifact_id != inner.artifact().artifact_id


def test_two_output_schemas_produce_distinct_structured_artifacts() -> None:
    class _Other(GroundedRagProgram):
        name = "grounded-rag-other"
        output_signature = _TypedOutput

    one = StructuredOutputProgram(GroundedRagProgram()).artifact().artifact_id
    two = StructuredOutputProgram(_Other()).artifact().artifact_id
    assert one != two


def test_identical_structured_config_is_a_reproducible_artifact() -> None:
    one = StructuredOutputProgram(GroundedRagProgram(), max_attempts=2).artifact().artifact_id
    two = StructuredOutputProgram(GroundedRagProgram(), max_attempts=2).artifact().artifact_id
    three = StructuredOutputProgram(GroundedRagProgram(), max_attempts=3).artifact().artifact_id
    assert one == two
    assert one != three


def test_default_extra_validator_seam_is_a_true_no_op() -> None:
    inner = GroundedRagProgram()
    baseline = inner.artifact().artifact_id
    # An unused seam leaves validators and the artifact id byte-identical.
    assert [validator.name for validator in inner.validators()] == ["grounding"]
    assert inner.with_extra_validators(()).artifact().artifact_id == baseline
    assert inner.response_schema is None
    assert inner._definition_extra() == {}


def test_structured_validator_is_not_dropped_when_wrapping_a_bounded_policy() -> None:
    inner = BoundedInferencePolicy(GroundedRagProgram(), mode=BEST_OF, attempts=2)
    names = [validator.name for validator in StructuredOutputProgram(inner).validators()]
    assert "structured-output" in names


def test_structured_validator_is_not_dropped_when_wrapping_a_knn_program() -> None:
    inner = KNNFewShotProgram(GroundedRagProgram(), [Demonstration("q", "a")], k=1)
    names = [validator.name for validator in StructuredOutputProgram(inner).validators()]
    assert "structured-output" in names


def test_pure_validator_matches_jsonschema_cross_check() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = GroundedRagOutput.json_schema()
    good = {"answer": "a", "citations": ["c1"]}
    jsonschema.validate(good, schema)  # does not raise
    assert validate_structured(good, schema)[0] is True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"citations": ["c1"]}, schema)
    assert validate_structured({"citations": ["c1"]}, schema)[0] is False


def test_validate_object_rejects_unknown_declared_type() -> None:
    schema = {"type": "object", "properties": {"n": {"type": "integre"}}, "required": ["n"]}
    ok, reason = validate_object({"n": 5}, schema)
    assert ok is False
    assert "unsupported type" in reason


def test_validate_object_rejects_non_string_type_declaration() -> None:
    schema = {"type": "object", "properties": {"n": {"type": ["integer", "string"]}}, "required": ["n"]}
    ok, reason = validate_object({"n": {}}, schema)
    assert ok is False
    assert "unsupported type" in reason
