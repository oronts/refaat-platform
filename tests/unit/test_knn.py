from __future__ import annotations

import pytest

from refaat.core.guidance import guidance_digest
from refaat.domain.models import Demonstration
from refaat.errors import ValidationError
from refaat.infrastructure.embeddings import DeterministicEmbedder
from refaat.programs import (
    GroundedRagInput,
    GroundedRagOutput,
    GroundedRagProgram,
    KNNFewShotProgram,
)
from refaat.programs.program import Program, ProgramResult, ValidationOutcome

_POOL = (
    Demonstration(question="how does rollback restore the harness snapshot", answer="It restores the snapshot."),
    Demonstration(question="what is the version safe cache key", answer="The key folds in the corpus version."),
    Demonstration(
        question="explain the citation contract for chunk ids",
        answer="Every citation resolves to a real chunk [chunk:real_planted_id].",
    ),
    Demonstration(question="deployment approval gate for production", answer="Promotion needs an approval."),
)


class _CapturingProgram(Program):
    name = "capture"
    input_signature = GroundedRagInput
    output_signature = GroundedRagOutput

    def __init__(self, sink: list[tuple[Demonstration, ...]]) -> None:
        super().__init__()
        self._sink = sink

    def execute(self, project, *, inputs, user_id, session_id, principal=None) -> ProgramResult:
        self._sink.append(self.tunables.demonstrations)
        return ProgramResult(
            program=self.name,
            artifact_id=self.artifact().artifact_id,
            answer="ok",
            citations=(),
            routes=(),
            cached=False,
            corpus_version=1,
            trace_id="t",
            outputs={},
            validations=(ValidationOutcome(validator="grounding", passed=True, detail=""),),
        )


def _run(pool, query, *, k=2):
    sink: list[tuple[Demonstration, ...]] = []
    program = KNNFewShotProgram(_CapturingProgram(sink), pool, k=k, embedder=DeterministicEmbedder())
    result = program.execute(None, inputs={"question": query}, user_id=None, session_id=None)
    return result, sink[-1]


def test_selects_the_nearest_demonstration_first() -> None:
    _result, selected = _run(_POOL, "rollback harness snapshot restore")
    assert len(selected) == 2
    assert selected[0].question == "how does rollback restore the harness snapshot"


def test_selection_is_deterministic_for_a_fixed_query() -> None:
    _first_result, first = _run(_POOL, "rollback harness snapshot restore")
    _second_result, second = _run(_POOL, "rollback harness snapshot restore")
    assert first == second


def test_a_different_query_shifts_the_selection_and_cache_scope() -> None:
    _rollback_result, rollback = _run(_POOL, "rollback harness snapshot restore")
    _cache_result, cache = _run(_POOL, "version safe cache key corpus")
    assert cache[0].question == "what is the version safe cache key"
    assert guidance_digest(None, rollback) != guidance_digest(None, cache)


def test_selected_demonstrations_are_citation_sanitized() -> None:
    _result, selected = _run(_POOL, "citation contract chunk ids", k=1)
    assert selected[0].question == "explain the citation contract for chunk ids"
    assert "real_planted_id" not in selected[0].answer
    assert "[chunk:example]" in selected[0].answer


def test_does_not_mutate_the_inner_program() -> None:
    inner = _CapturingProgram([])
    KNNFewShotProgram(inner, _POOL, k=2, embedder=DeterministicEmbedder()).execute(
        None, inputs={"question": "rollback"}, user_id=None, session_id=None
    )
    assert inner.tunables.demonstrations == ()


def test_k_larger_than_the_pool_selects_all() -> None:
    _result, selected = _run(_POOL, "rollback", k=10)
    assert len(selected) == len(_POOL)


def test_outputs_record_the_selection() -> None:
    result, _selected = _run(_POOL, "rollback harness snapshot restore")
    assert result.outputs["knn"]["k"] == 2
    assert result.outputs["knn"]["pool_size"] == len(_POOL)
    assert len(result.outputs["knn"]["selected"]) == 2


def test_artifact_pins_pool_and_k() -> None:
    inner = GroundedRagProgram()
    other_pool = _POOL[:2]
    k2 = KNNFewShotProgram(inner, _POOL, k=2).artifact().artifact_id
    k3 = KNNFewShotProgram(inner, _POOL, k=3).artifact().artifact_id
    other = KNNFewShotProgram(inner, other_pool, k=2).artifact().artifact_id
    assert len({k2, k3, other, inner.artifact().artifact_id}) == 4


def test_empty_pool_is_rejected() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        KNNFewShotProgram(GroundedRagProgram(), [])


def test_non_positive_k_is_rejected() -> None:
    with pytest.raises(ValidationError, match="at least 1"):
        KNNFewShotProgram(GroundedRagProgram(), _POOL, k=0)


def test_nested_knn_folds_the_outer_selection_into_the_inner_request() -> None:
    sink: list[tuple[Demonstration, ...]] = []
    inner_pool = (Demonstration(question="inner nearest demonstration", answer="inner answer"),)
    outer_pool = (Demonstration(question="outer nearest demonstration", answer="outer answer"),)
    inner_knn = KNNFewShotProgram(_CapturingProgram(sink), inner_pool, k=1, embedder=DeterministicEmbedder())
    outer_knn = KNNFewShotProgram(inner_knn, outer_pool, k=1, embedder=DeterministicEmbedder())

    outer_knn.execute(None, inputs={"question": "which demonstration"}, user_id=None, session_id=None)

    captured = [demonstration.question for demonstration in sink[-1]]
    assert "outer nearest demonstration" in captured
    assert "inner nearest demonstration" in captured


def test_missing_question_is_rejected() -> None:
    program = KNNFewShotProgram(_CapturingProgram([]), _POOL, embedder=DeterministicEmbedder())
    with pytest.raises(ValidationError, match="question"):
        program.execute(None, inputs={}, user_id=None, session_id=None)
