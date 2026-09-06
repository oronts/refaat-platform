from __future__ import annotations

import pytest

from refaat.domain.models import Citation
from refaat.errors import ValidationError
from refaat.programs import default_result_score, select_best
from refaat.programs.program import ProgramResult, ValidationOutcome


def _result(
    *,
    grounded: bool,
    citations: int,
    answer: str = "answer",
    program: str = "p",
) -> ProgramResult:
    cites = tuple(
        Citation(chunk_id=f"c{index}", document_id="doc", label=f"[{index}]", quote="q") for index in range(citations)
    )
    return ProgramResult(
        program=program,
        artifact_id="sha256:x",
        answer=answer,
        citations=cites,
        routes=(),
        cached=False,
        corpus_version=1,
        trace_id="t",
        outputs={},
        validations=(ValidationOutcome(validator="grounding", passed=grounded, detail=""),),
    )


def test_grounding_dominates_citation_count() -> None:
    grounded = default_result_score(_result(grounded=True, citations=0))
    ungrounded = default_result_score(_result(grounded=False, citations=100))
    assert grounded > ungrounded


def test_more_citations_scores_higher_among_grounded() -> None:
    assert default_result_score(_result(grounded=True, citations=3)) > default_result_score(
        _result(grounded=True, citations=1)
    )


def test_length_is_a_bounded_tiebreak_below_a_citation() -> None:
    short = default_result_score(_result(grounded=True, citations=1, answer="a"))
    longer = default_result_score(_result(grounded=True, citations=1, answer="a much longer answer"))
    more_cited = default_result_score(_result(grounded=True, citations=2, answer="a"))
    assert short < longer < more_cited


def test_score_is_deterministic() -> None:
    result = _result(grounded=True, citations=2, answer="stable")
    assert default_result_score(result) == default_result_score(result)


def test_select_best_prefers_grounded_even_under_a_custom_score() -> None:
    ungrounded_rich = _result(grounded=False, citations=100, program="ungrounded")
    grounded_lean = _result(grounded=True, citations=1, program="grounded")
    chosen = select_best([ungrounded_rich, grounded_lean], score=lambda result: float(len(result.citations)))
    assert chosen.program == "grounded"


def test_select_best_breaks_ties_by_earliest_attempt() -> None:
    first = _result(grounded=True, citations=1, program="first")
    second = _result(grounded=True, citations=1, program="second")
    assert select_best([first, second]) is first


def test_select_best_rejects_empty() -> None:
    with pytest.raises(ValidationError, match="at least one"):
        select_best([])
