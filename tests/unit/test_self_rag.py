from __future__ import annotations

from refaat.domain.models import GenerationRequest, GenerationResult
from refaat.errors import DependencyUnavailableError
from refaat.infrastructure.reflection import (
    DeterministicNecessityGate,
    ModelNecessityGate,
    is_useful_answer,
)
from refaat.programs import SelfRagProgram


class _RaisingGenerator:
    model_id = "stub-necessity"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        raise DependencyUnavailableError("necessity judge offline")


class _VerdictGenerator:
    model_id = "stub-necessity"

    def __init__(self, verdict: str) -> None:
        self._verdict = verdict

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return GenerationResult(answer=self._verdict, cited_chunk_ids=())


_CONTENT_RICH_QUERY = "Rollback restores the previously approved harness snapshot before deployment"
_TRIVIAL_QUERY = "ok thanks"


def test_deterministic_necessity_gate_is_a_live_discriminator() -> None:
    gate = DeterministicNecessityGate(min_content_terms=2)

    rich = gate.decide(_CONTENT_RICH_QUERY)
    trivial = gate.decide(_TRIVIAL_QUERY)

    assert rich is True
    assert trivial is False
    assert {rich, trivial} == {True, False}


def test_is_useful_heuristic_passes_a_grounded_answer_and_fails_an_empty_one() -> None:
    assert is_useful_answer("A grounded and sufficiently long answer.", grounded=True, min_length=1) is True
    assert is_useful_answer("", grounded=True, min_length=1) is False
    assert is_useful_answer("   ", grounded=True, min_length=1) is False
    assert is_useful_answer("A grounded answer.", grounded=False, min_length=1) is False


def test_model_gate_degrades_to_the_deterministic_decision_without_faking_it() -> None:
    deterministic = DeterministicNecessityGate(min_content_terms=2)
    model = ModelNecessityGate(_RaisingGenerator(), deterministic=deterministic)

    for query in (_CONTENT_RICH_QUERY, _TRIVIAL_QUERY):
        assert model.decide(query) == deterministic.decide(query)


def test_model_gate_uses_the_model_verdict_when_available() -> None:
    deterministic = DeterministicNecessityGate(min_content_terms=2)
    retrieve_model = ModelNecessityGate(_VerdictGenerator("retrieve"), deterministic=deterministic)
    skip_model = ModelNecessityGate(_VerdictGenerator("skip"), deterministic=deterministic)

    assert retrieve_model.decide(_TRIVIAL_QUERY) is True
    assert skip_model.decide(_CONTENT_RICH_QUERY) is False


def test_artifact_id_folds_the_reflection_policy_and_stays_stable_for_identical_config() -> None:
    base = SelfRagProgram(refine_attempt_cap=3, usefulness_min_answer_chars=1).artifact().artifact_id
    same = SelfRagProgram(refine_attempt_cap=3, usefulness_min_answer_chars=1).artifact().artifact_id
    other_cap = SelfRagProgram(refine_attempt_cap=5, usefulness_min_answer_chars=1).artifact().artifact_id
    other_threshold = SelfRagProgram(refine_attempt_cap=3, usefulness_min_answer_chars=20).artifact().artifact_id
    other_gate = (
        SelfRagProgram(
            necessity_gate=DeterministicNecessityGate(min_content_terms=5),
            refine_attempt_cap=3,
            usefulness_min_answer_chars=1,
        )
        .artifact()
        .artifact_id
    )

    assert base == same
    assert len({base, other_cap, other_threshold, other_gate}) == 4


def test_artifact_id_preserves_the_base_response_schema_pinning() -> None:
    plain = SelfRagProgram().artifact().artifact_id
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
    schemad = SelfRagProgram().with_response_schema(schema).artifact().artifact_id

    assert plain != schemad
