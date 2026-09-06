from __future__ import annotations

from refaat.domain.models import Citation, JsonObject, QueryResponse
from refaat.programs import (
    GroundedRagProgram,
    GroundingValidator,
    InputField,
    OutputField,
    Signature,
    Tunables,
)


class _Sig(Signature):
    question: str = InputField(description="q", tunable=True)
    answer: str = OutputField(description="a", grounded=True)


def test_signature_classifies_fields_and_hash_is_stable() -> None:
    assert [field.name for field in _Sig.input_fields()] == ["question"]
    assert [field.name for field in _Sig.output_fields()] == ["answer"]
    assert _Sig.input_fields()[0].metadata["tunable"] is True
    assert _Sig.output_fields()[0].metadata["grounded"] is True
    assert _Sig.signature_hash() == _Sig.signature_hash()


def test_program_artifact_is_content_addressed_and_tunable_sensitive() -> None:
    base = GroundedRagProgram().artifact()
    tuned = GroundedRagProgram(tunables=Tunables(top_k=3)).artifact()
    assert base.artifact_id.startswith("sha256:")
    assert base.artifact_id == GroundedRagProgram().artifact().artifact_id
    assert base.artifact_id != tuned.artifact_id
    assert base.immutable_policy_hash
    assert base.to_json()["validators"] == ["grounding"]


def test_tunables_round_trip_is_faithful_including_empty_instructions() -> None:
    # An explicit empty-string instructions normalizes to None so it round-trips
    # faithfully instead of surfacing as a "" != None inequality after from_json.
    empty = Tunables(instructions="")
    assert empty.instructions is None
    assert empty == Tunables.from_json(empty.as_json())
    assert empty.as_json() == Tunables().as_json()
    assert GroundedRagProgram(tunables=empty).artifact().artifact_id == GroundedRagProgram().artifact().artifact_id

    rich = Tunables(top_k=5, instructions="cite precisely", fusion="rrf", rrf_k=60)
    assert rich == Tunables.from_json(rich.as_json())


def _response(citation_count: int, answer: str = "grounded answer") -> QueryResponse:
    citations = tuple(
        Citation(chunk_id=f"chunk_{index}", document_id="doc", label=f"[{index}]", quote="quote")
        for index in range(citation_count)
    )
    return QueryResponse(
        answer=answer,
        citations=citations,
        routes=("semantic",),
        cached=False,
        corpus_version=1,
        trace_id="trace",
    )


def test_grounding_validator_enforces_citations_by_threshold() -> None:
    # A positive floor is a real fraction: the share of answer sentences carrying a resolved citation
    # marker must meet it, so 0.5 and 0.9 are no longer the same toggle.
    fully_cited = _response(2, answer="First point [chunk:chunk_0]. Second point [chunk:chunk_1].")
    half_cited = _response(1, answer="First point [chunk:chunk_0]. Second point is uncited here.")

    strict = GroundingValidator(0.9)
    assert strict.check(fully_cited).passed is True  # 2/2 = 1.0 >= 0.9
    assert strict.check(half_cited).passed is False  # 1/2 = 0.5 < 0.9
    assert strict.check(_response(0, answer="Ungrounded text.")).passed is False

    assert GroundingValidator(0.5).check(half_cited).passed is True  # 0.5 >= 0.5

    # A marker at a sentence boundary must not collapse two sentences into one and inflate coverage.
    boundary = _response(1, answer="Grounded fact. [chunk:chunk_0] Ungrounded fact.")
    assert strict.check(boundary).passed is False  # 1/2, not 1/1
    assert GroundingValidator(0.5).check(boundary).passed is True

    # A dangling marker whose id is not a resolved citation must not count toward coverage.
    dangling = _response(1, answer="Claim [chunk:ghost].")
    assert GroundingValidator(1.0).check(dangling).passed is False  # 0/1

    # A marker sitting in a whitespace/punctuation-only gap is not a covered sentence.
    pseudo = _response(1, answer=" [chunk:chunk_0] . Ungrounded fact.")
    assert GroundingValidator(0.5).check(pseudo).passed is False  # 0/1, the only real sentence is uncited

    # A marker-only answer with no real sentence has no coverage to satisfy a positive floor.
    marker_only = _response(1, answer=" [chunk:chunk_0] ")
    assert GroundingValidator(1.0).check(marker_only).passed is False

    # A leading marker before a sentence's first word does not cite it.
    prefix = _response(1, answer=" [chunk:chunk_0] Ungrounded fact.")
    assert GroundingValidator(1.0).check(prefix).passed is False  # 0/1

    inside = _response(1, answer="The sky [chunk:chunk_0] is blue.")
    assert GroundingValidator(1.0).check(inside).passed is True  # 1/1
    trailing = _response(1, answer="One sentence. [chunk:chunk_0]")
    assert GroundingValidator(1.0).check(trailing).passed is True  # 1/1

    # A marker abutting the next word (no space) is a prefix, so it cites the preceding sentence.
    abutting = _response(1, answer="Alpha [chunk:chunk_0]fact.  [chunk:chunk_0]Beta fact.")
    assert GroundingValidator(1.0).check(abutting).passed is False  # 1/2, both cite Alpha

    zero_ws = _response(1, answer="A fact.[chunk:chunk_0]B fact.")
    assert GroundingValidator(1.0).check(zero_ws).passed is False  # 1/2, the marker cites "A fact."

    lenient = GroundingValidator(0.0)
    assert lenient.check(_response(0, answer="")).passed is True
    assert lenient.check(_response(0, answer="ungrounded")).passed is False
    assert lenient.check(_response(1)).passed is True  # any resolved citation satisfies the 0.0 floor


def test_extra_validator_config_is_part_of_the_artifact_identity() -> None:
    # Two variants differing only in a custom validator's config must be distinct artifacts, so the
    # deployment identity guard cannot mistake one behavior for another that merely shares a class name.
    from dataclasses import dataclass

    from refaat.programs.program import ValidationOutcome

    @dataclass(frozen=True, slots=True)
    class ThresholdValidator:
        threshold: float
        name: str = "threshold"

        def check(self, response: QueryResponse) -> ValidationOutcome:
            return ValidationOutcome(validator=self.name, passed=True, detail="")

    base = GroundedRagProgram()
    lax = base.with_extra_validators((ThresholdValidator(0.5),))
    strict = base.with_extra_validators((ThresholdValidator(0.9),))
    assert lax.artifact().artifact_id != strict.artifact().artifact_id
    assert lax.artifact().artifact_id == base.with_extra_validators((ThresholdValidator(0.5),)).artifact().artifact_id
    assert lax.artifact().behavior_hash != strict.artifact().behavior_hash
    assert len(base.artifact().behavior_hash) == 64

    class PlainValidator:
        name = "plain"

        def check(self, response: QueryResponse) -> ValidationOutcome:
            return ValidationOutcome(validator=self.name, passed=True, detail="")

    # A non-dataclass validator folds by class identity alone and never crashes the definition hash.
    assert base.with_extra_validators((PlainValidator(),)).artifact().artifact_id


def test_validator_config_folds_through_a_definition_extra_override() -> None:
    # A subclass that overrides _definition_extra without merging super must still fold its extra
    # validators' config, so a config drift is refused on every program shape, not only the base.
    from dataclasses import dataclass

    from refaat.programs.program import ValidationOutcome

    @dataclass(frozen=True, slots=True)
    class ThresholdValidator:
        threshold: float
        name: str = "threshold"

        def check(self, response: QueryResponse) -> ValidationOutcome:
            return ValidationOutcome(validator=self.name, passed=True, detail="")

    class Overriding(GroundedRagProgram):
        name = "overriding"

        def _definition_extra(self) -> JsonObject:
            return {"policy": "x"}

    overriding = Overriding()
    lax = overriding.with_extra_validators((ThresholdValidator(0.5),))
    strict = overriding.with_extra_validators((ThresholdValidator(0.9),))
    assert lax.artifact().artifact_id != strict.artifact().artifact_id

    @dataclass(frozen=True, slots=True)
    class RulesValidator:
        rules: object
        name: str = "rules"

        def check(self, response: QueryResponse) -> ValidationOutcome:
            return ValidationOutcome(validator=self.name, passed=True, detail="")

    # An integer key and a string key must not collide once the config is folded.
    base = GroundedRagProgram()
    int_key = base.with_extra_validators((RulesValidator({1: "allow"}),))
    str_key = base.with_extra_validators((RulesValidator({"1": "allow"}),))
    assert int_key.artifact().artifact_id != str_key.artifact().artifact_id


def test_json_safe_coerces_every_shape_deterministically() -> None:
    import json as _json
    from dataclasses import dataclass

    from refaat.programs.program import _json_safe

    @dataclass(frozen=True, slots=True)
    class Nested:
        n: int

    class Weird:
        pass

    assert _json_safe(None) is None
    assert _json_safe("s") == "s"
    assert _json_safe(3) == 3
    assert _json_safe(True) is True
    assert _json_safe(int) == "builtins.int"
    assert _json_safe([1, "a"]) == [1, "a"]
    assert _json_safe((1, 2)) == [1, 2]
    assert _json_safe(Nested(5)) == {"n": 5}

    mapping = _json_safe({1: "a", "1": "b"})
    _json.dumps(mapping)
    assert [1, "a"] in mapping
    assert ["1", "b"] in mapping
    assert len(mapping) == 2

    unknown = _json_safe(Weird())
    assert unknown.endswith(".Weird")
    assert "0x" not in unknown


def test_program_result_to_json_includes_outputs() -> None:
    from refaat.programs.program import ProgramResult

    result = ProgramResult(
        program="p",
        artifact_id="sha256:x",
        answer="a",
        citations=(),
        routes=(),
        cached=False,
        corpus_version=1,
        trace_id="t",
        outputs={"summary": "structured", "react_steps": ["raw observation"]},
        validations=(),
    )
    serialized = result.to_json()["outputs"]
    assert serialized["summary"] == "structured"
    assert "react_steps" not in serialized
