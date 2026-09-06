from __future__ import annotations

from refaat.evaluation import Dataset
from refaat.evaluation.dataset import EvaluationCase
from refaat.optimization import propose_instructions, reflect_instruction
from refaat.optimization.instructions import (
    _MAX_REFLECTION_ROUNDS,
    _evolve_instructions,
    _salient_terms,
)


def _dataset() -> Dataset:
    return Dataset.from_records(
        "instructions",
        [
            {"id": "a", "query": "incident escalation path", "expected_terms": ["incident"]},
            {"id": "b", "query": "incident response window", "expected_terms": ["incident"]},
            {"id": "c", "query": "sev1 handling runbook", "expected_terms": ["runbook"]},
        ],
    )


def test_baseline_is_none_at_index_zero() -> None:
    proposals = propose_instructions(_dataset())
    assert proposals[0] is None
    assert all(instruction is not None for instruction in proposals[1:])


def test_proposals_are_distinct() -> None:
    proposals = propose_instructions(_dataset())
    guided = [instruction for instruction in proposals if instruction is not None]
    assert guided
    assert len(guided) == len(set(guided))


def test_proposals_are_reproducible_across_runs() -> None:
    assert propose_instructions(_dataset()) == propose_instructions(_dataset())


def test_proposals_are_dataset_signal_driven() -> None:
    guided = [instruction for instruction in propose_instructions(_dataset()) if instruction is not None]
    # "incident" is the most frequent stopword-filtered term, so it anchors the
    # focus clause of every generated candidate.
    assert all("incident" in instruction for instruction in guided)

    other = Dataset.from_records(
        "other",
        [
            {"id": "a", "query": "payroll deduction schedule", "expected_terms": ["payroll"]},
            {"id": "b", "query": "payroll tax filing", "expected_terms": ["payroll"]},
        ],
    )
    other_guided = [instruction for instruction in propose_instructions(other) if instruction is not None]
    assert all("payroll" in instruction for instruction in other_guided)
    assert set(other_guided).isdisjoint(guided)


def test_seed_instructions_are_offered_first_and_deduped() -> None:
    proposals = propose_instructions(_dataset(), seed_instructions=("Be terse.",), max_candidates=3)
    assert proposals[0] is None
    assert proposals[1] == "Be terse."
    assert len([instruction for instruction in proposals if instruction is not None]) == 3


def test_max_candidates_caps_the_non_none_members() -> None:
    proposals = propose_instructions(_dataset(), max_candidates=2)
    assert len([instruction for instruction in proposals if instruction is not None]) == 2


def test_salient_terms_frequency_then_alphabetical_tie_break() -> None:
    # "charlie" appears twice; "alpha" and "bravo" each appear once (a tie broken
    # alphabetically), so the order is frequency-descending then term-ascending.
    terms = _salient_terms(["charlie alpha", "charlie bravo"])
    assert terms == ["charlie", "alpha", "bravo"]


def test_salient_terms_drop_short_and_stopword_tokens() -> None:
    terms = _salient_terms(["the on of incident to sev1"])
    for stopword in ("the", "on", "of", "to"):
        assert stopword not in terms
    assert "incident" in terms
    assert "sev1" in terms


def _failing(*queries: str) -> tuple[EvaluationCase, ...]:
    return tuple(EvaluationCase(id=str(index), query=query) for index, query in enumerate(queries))


def test_reflect_is_deterministic_for_a_fixed_failure_set() -> None:
    failures = _failing("incident escalation", "runbook lookup")
    assert reflect_instruction("Answer plainly.", failures) == reflect_instruction("Answer plainly.", failures)


def test_reflect_appends_a_reminder_from_the_failures() -> None:
    mutated = reflect_instruction("Answer plainly.", _failing("incident escalation window"))
    assert mutated.startswith("Answer plainly.")
    assert "incident" in mutated
    assert mutated != "Answer plainly."


def test_reflect_is_idempotent() -> None:
    failures = _failing("incident escalation window")
    once = reflect_instruction("", failures)
    twice = reflect_instruction(once, failures)
    assert once == twice


def test_reflect_empty_failures_returns_instruction_unchanged() -> None:
    assert reflect_instruction("Answer plainly.", ()) == "Answer plainly."
    assert reflect_instruction(None, ()) == ""


def test_evolve_breeds_distinct_bounded_mutants() -> None:
    failures = _failing("incident escalation", "runbook lookup", "payroll filing")
    evolved = _evolve_instructions(failures, rounds=3)
    assert len(evolved) >= 1
    assert len(evolved) == len(set(evolved))
    assert len(evolved) <= 3


def test_evolve_depth_is_capped() -> None:
    failures = _failing(*[f"topic{index} distinct term{index}" for index in range(20)])
    evolved = _evolve_instructions(failures, rounds=100)
    assert len(evolved) <= _MAX_REFLECTION_ROUNDS


def test_evolve_single_failure_breeds_one_mutant() -> None:
    evolved = _evolve_instructions(_failing("incident escalation window"), rounds=3)
    assert len(evolved) == 1
