from __future__ import annotations

from refaat.core.text import content_terms, split_clauses
from refaat.infrastructure.query_transform import DeterministicQueryTransform

COMPOUND = "What is the capital of France and what is the population of Spain?"


def _transform() -> DeterministicQueryTransform:
    return DeterministicQueryTransform()


def test_single_clause_question_mark_query_yields_only_the_original() -> None:
    variants = _transform().expand("What is the capital of France?", 4)
    assert len(variants) == 1
    assert variants[0].kind == "original"
    assert variants[0].text == "What is the capital of France?"


def test_compound_query_decomposes_in_deterministic_order() -> None:
    variants = _transform().expand(COMPOUND, 4)
    assert [variant.kind for variant in variants] == ["original", "subquery", "subquery"]
    assert variants[0].text == COMPOUND
    assert variants[1].text == "What is the capital of France"
    assert variants[2].text == "what is the population of Spain"


def test_variants_are_capped_at_max_variants() -> None:
    variants = _transform().expand(COMPOUND, 2)
    assert len(variants) == 2
    assert [variant.kind for variant in variants] == ["original", "subquery"]


def test_max_variants_one_short_circuits_to_the_original() -> None:
    variants = _transform().expand(COMPOUND, 1)
    assert len(variants) == 1
    assert variants[0].kind == "original"


def test_clauses_with_the_same_content_terms_leave_only_the_original() -> None:
    variants = _transform().expand("Explain X, and explain X?", 4)
    assert [variant.kind for variant in variants] == ["original"]


def test_intra_token_punctuation_questions_yield_only_the_original() -> None:
    for question in (
        "What does the U.S. government do?",
        "What is 3.5 as a percentage?",
        "Is Python 3.11 faster?",
    ):
        variants = _transform().expand(question, 4)
        assert [variant.kind for variant in variants] == ["original"]
        assert variants[0].text == question


def test_intra_token_punctuation_is_never_split_into_clauses() -> None:
    assert split_clauses("What does the U.S. government do?") == ["What does the U.S. government do"]
    assert split_clauses("What is 3.5 as a percentage?") == ["What is 3.5 as a percentage"]
    assert split_clauses("Is Python 3.11 faster?") == ["Is Python 3.11 faster"]


def test_one_token_and_whitespace_queries_do_not_crash() -> None:
    assert [variant.kind for variant in _transform().expand("France", 4)] == ["original"]
    whitespace = _transform().expand("   ", 4)
    assert len(whitespace) == 1
    assert whitespace[0].kind == "original"


def test_split_clauses_and_content_terms_are_deterministic() -> None:
    assert split_clauses(COMPOUND) == [
        "What is the capital of France",
        "what is the population of Spain",
    ]
    assert content_terms(COMPOUND) == ["capital", "france", "population", "spain"]
    assert content_terms("the of and to") == []
    assert split_clauses("   ") == []
