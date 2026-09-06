from __future__ import annotations

from refaat.domain.models import Chunk, ContextPacket, GenerationRequest, GenerationResult, SearchHit
from refaat.errors import DependencyUnavailableError
from refaat.programs import (
    ConfidenceAssessment,
    CorrectiveRagProgram,
    DeterministicConfidenceEvaluator,
    ModelConfidenceEvaluator,
)


def _chunk(cid: str) -> Chunk:
    return Chunk(
        id=cid,
        corpus_id="corpus",
        document_id="doc",
        ordinal=0,
        text=f"text for {cid}",
        token_count=4,
        checksum="sum",
        metadata={},
        embedding=(0.0,),
    )


def _hit(cid: str, *, semantic_score: float, score: float | None = None) -> SearchHit:
    return SearchHit(
        chunk=_chunk(cid),
        score=score if score is not None else semantic_score,
        semantic_score=semantic_score,
    )


def _context(*hits: SearchHit) -> ContextPacket:
    return ContextPacket(
        query="q",
        hits=tuple(hits),
        graph_facts=(),
        session_summary=None,
        profile={},
        supplemental_harness={},
        token_count=0,
        routes=(),
    )


class _RaisingGenerator:
    model_id = "stub-judge"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        raise DependencyUnavailableError("judge offline")


class _VerdictGenerator:
    model_id = "stub-judge"

    def __init__(self, verdict: str) -> None:
        self._verdict = verdict

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return GenerationResult(answer=self._verdict, cited_chunk_ids=())


def test_gate_reads_non_normalized_semantic_score_not_the_saturated_fused_score() -> None:
    evaluator = DeterministicConfidenceEvaluator(top_semantic_threshold=0.35, semantic_floor=0.15)
    saturated_but_weak = _context(_hit("c1", semantic_score=0.10, score=1.0))
    saturated_and_strong = _context(_hit("c2", semantic_score=0.50, score=1.0))

    weak = evaluator.assess("q", saturated_but_weak)
    strong = evaluator.assess("q", saturated_and_strong)

    assert weak.label == "low"
    assert weak.top_semantic == 0.10
    assert strong.label == "correct"
    assert strong.top_semantic == 0.50


def test_deterministic_evaluator_is_a_live_discriminator() -> None:
    evaluator = DeterministicConfidenceEvaluator(top_semantic_threshold=0.35, semantic_floor=0.15)
    just_below = evaluator.assess("q", _context(_hit("c1", semantic_score=0.349)))
    at_threshold = evaluator.assess("q", _context(_hit("c2", semantic_score=0.35)))

    labels = {just_below.label, at_threshold.label}
    assert labels == {"low", "correct"}
    assert just_below.label == "low"
    assert at_threshold.label == "correct"


def test_min_grounded_chunks_gates_a_single_strong_hit_to_low() -> None:
    evaluator = DeterministicConfidenceEvaluator(
        top_semantic_threshold=0.35, semantic_floor=0.15, min_grounded_chunks=2
    )
    one_strong = _context(_hit("c1", semantic_score=0.90), _hit("c2", semantic_score=0.05))
    two_grounded = _context(_hit("c3", semantic_score=0.90), _hit("c4", semantic_score=0.20))

    assert evaluator.assess("q", one_strong).label == "low"
    assert evaluator.assess("q", two_grounded).label == "correct"


def test_semantic_floor_is_inclusive_when_counting_grounded_chunks() -> None:
    evaluator = DeterministicConfidenceEvaluator(semantic_floor=0.20)
    context = _context(
        _hit("c1", semantic_score=0.20),
        _hit("c2", semantic_score=0.19),
        _hit("c3", semantic_score=0.50),
    )
    assessment = evaluator.assess("q", context)
    assert assessment.grounded_chunks == 2


def test_empty_hits_is_low() -> None:
    assessment = DeterministicConfidenceEvaluator().assess("q", _context())
    assert assessment.label == "low"
    assert assessment.hit_count == 0
    assert assessment.top_semantic == 0.0


def test_artifact_id_is_distinct_across_thresholds_and_stable_for_identical_config() -> None:
    strict = (
        CorrectiveRagProgram(evaluator=DeterministicConfidenceEvaluator(top_semantic_threshold=0.50))
        .artifact()
        .artifact_id
    )
    loose = (
        CorrectiveRagProgram(evaluator=DeterministicConfidenceEvaluator(top_semantic_threshold=0.20))
        .artifact()
        .artifact_id
    )
    same_again = (
        CorrectiveRagProgram(evaluator=DeterministicConfidenceEvaluator(top_semantic_threshold=0.50))
        .artifact()
        .artifact_id
    )

    assert strict != loose
    assert strict == same_again


def test_artifact_id_is_distinct_across_corrective_breadth() -> None:
    narrow = CorrectiveRagProgram(corrective_max_query_variants=2).artifact().artifact_id
    wide = CorrectiveRagProgram(corrective_max_query_variants=6).artifact().artifact_id
    assert narrow != wide


def test_model_evaluator_degrades_to_the_deterministic_label_without_faking_a_score() -> None:
    deterministic = DeterministicConfidenceEvaluator(top_semantic_threshold=0.35, semantic_floor=0.15)
    context = _context(_hit("c1", semantic_score=0.50))
    baseline = deterministic.assess("q", context)
    model = ModelConfidenceEvaluator(_RaisingGenerator(), deterministic=deterministic)

    degraded = model.assess("q", context)

    assert degraded.label == baseline.label == "correct"
    assert degraded.score == baseline.score
    assert degraded.top_semantic == baseline.top_semantic
    assert isinstance(degraded, ConfidenceAssessment)


def test_model_evaluator_flips_only_the_label_and_keeps_the_measured_score() -> None:
    deterministic = DeterministicConfidenceEvaluator(top_semantic_threshold=0.35, semantic_floor=0.15)
    context = _context(_hit("c1", semantic_score=0.90))
    baseline = deterministic.assess("q", context)
    model = ModelConfidenceEvaluator(_VerdictGenerator("low"), deterministic=deterministic)

    verdict = model.assess("q", context)

    assert baseline.label == "correct"
    assert verdict.label == "low"
    assert verdict.score == baseline.score == 0.90
