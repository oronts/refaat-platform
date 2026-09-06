from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from refaat.application.runtime import Refaat
from refaat.domain.models import Citation, Demonstration
from refaat.errors import ValidationError
from refaat.evaluation import Dataset
from refaat.evaluation.dataset import EvaluationCase
from refaat.optimization import (
    BootstrapFewShotOptimizer,
    BootstrapTrace,
    random_demonstration_subsets,
)
from refaat.optimization.bootstrap import _bootstrap_traces
from refaat.optimization.candidates import build_ranked_result as _shared_helper
from refaat.programs.program import GroundedRagProgram, ProgramResult, ValidationOutcome

# The unguided grounded-rag artifact id, pinned before this slice. Guidance is
# omitted from Tunables.as_json when empty, so a program with empty
# demonstrations must hash to exactly this value both before and after W3.3.
PRE_SLICE_UNGUIDED_ARTIFACT_ID = "sha256:b76d73f8d956bcd38ccc9d4c24372b07b387647a189f3a9e300cded186febbb4"


class _FakeTeacher:
    """A deterministic RunnableProgram double so the gate is unit-testable."""

    def __init__(self, answer: str, *, grounded: bool = True) -> None:
        self._answer = answer
        self._grounded = grounded

    @property
    def name(self) -> str:
        return "fake-teacher"

    def run(self, *, user_id: Any = None, session_id: Any = None, **inputs: Any) -> ProgramResult:
        citations = (Citation(chunk_id="real_777", document_id="doc-1", label="1", quote="quote"),)
        return ProgramResult(
            program="fake-teacher",
            artifact_id="sha256:fake",
            answer=self._answer,
            citations=citations if self._grounded else (),
            routes=(),
            cached=False,
            corpus_version=1,
            trace_id="trace",
            outputs={},
            validations=(ValidationOutcome(validator="grounding", passed=self._grounded, detail=""),),
            usage={},
        )


def test_bootstrap_traces_gate_and_sanitization() -> None:
    teacher = _FakeTeacher("the runbook says escalate ok [chunk:real_777]")
    cases = (
        EvaluationCase(id="a", query="qa", expected_terms=("runbook",)),
        EvaluationCase(id="b", query="qb", expected_terms=("missing",)),
        EvaluationCase(id="c", query="qc"),
        EvaluationCase(id="d", query="qd", reference_answer="ref"),
    )
    traces = _bootstrap_traces(teacher, cases, (), max_demos=4)

    # Case "d" is reference-only and its answer does not overlap the reference, so
    # the reference-aware gate now rejects it where AnswerContains passed vacuously.
    assert [trace.case_id for trace in traces] == ["a"]
    assert all(isinstance(trace, BootstrapTrace) for trace in traces)
    joined = " ".join(trace.demonstration.answer for trace in traces)
    assert "real_777" not in joined
    assert "[chunk:example]" in joined


def test_reference_only_case_requires_a_matching_answer() -> None:
    # A grounded but off-reference answer is rejected; a grounded answer that
    # overlaps the reference is kept. This closes the vacuous AnswerContains pass
    # that previously admitted any grounded answer for a reference-only case.
    off = _FakeTeacher("completely unrelated content here ok")
    matching = _FakeTeacher("the escalation runbook reference procedure ok")
    case = (EvaluationCase(id="r", query="qr", reference_answer="escalation runbook reference procedure"),)
    assert _bootstrap_traces(off, case, (), max_demos=4) == ()
    assert [trace.case_id for trace in _bootstrap_traces(matching, case, (), max_demos=4)] == ["r"]


def test_bootstrap_traces_skips_ungrounded() -> None:
    teacher = _FakeTeacher("runbook", grounded=False)
    cases = (EvaluationCase(id="a", query="qa", expected_terms=("runbook",)),)
    assert _bootstrap_traces(teacher, cases, (), max_demos=4) == ()


def test_bootstrap_traces_respects_max_demos() -> None:
    teacher = _FakeTeacher("runbook ok")
    cases = tuple(EvaluationCase(id=f"c{index}", query=f"q{index}", expected_terms=("runbook",)) for index in range(5))
    traces = _bootstrap_traces(teacher, cases, (), max_demos=2)
    assert [trace.case_id for trace in traces] == ["c0", "c1"]


def _pool(size: int) -> tuple[Demonstration, ...]:
    return tuple(Demonstration(question=f"q{index}", answer=f"a{index}") for index in range(size))


def test_random_subsets_deterministic_for_fixed_seed() -> None:
    pool = _pool(4)
    assert random_demonstration_subsets(pool, trials=8, seed=0) == random_demonstration_subsets(pool, trials=8, seed=0)


def test_random_subsets_dedup_subset_and_order() -> None:
    pool = _pool(4)
    subsets = random_demonstration_subsets(pool, trials=8, seed=0)
    assert len(subsets) == len(set(subsets))
    for subset in subsets:
        assert set(subset) <= set(pool)
        indices = [pool.index(demo) for demo in subset]
        assert indices == sorted(indices)
        assert len(set(subset)) == len(subset)


def test_random_subsets_seed_sensitivity_for_large_pool() -> None:
    pool = _pool(4)
    assert random_demonstration_subsets(pool, trials=8, seed=0) != random_demonstration_subsets(pool, trials=8, seed=1)


def test_random_subsets_tiny_pool_is_seed_insensitive() -> None:
    pool = _pool(1)
    first = random_demonstration_subsets(pool, trials=4, seed=0)
    second = random_demonstration_subsets(pool, trials=4, seed=99)
    assert first == second == (tuple(pool),)


def test_random_subsets_empty_pool() -> None:
    assert random_demonstration_subsets((), trials=8, seed=0) == ()


def test_unknown_mode_raises_before_runtime_use(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    dataset = Dataset.from_records("d", [{"id": "a", "query": "q", "expected_terms": ["x"]}])
    with pytest.raises(ValidationError):
        BootstrapFewShotOptimizer(project).optimize(GroundedRagProgram(), dataset, mode="nope")


def test_empty_demonstration_artifact_id_is_byte_identical() -> None:
    program = GroundedRagProgram()
    baseline_id = program.artifact().artifact_id
    empty_variant = program.with_tunables(replace(program.tunables, demonstrations=()))
    assert baseline_id == PRE_SLICE_UNGUIDED_ARTIFACT_ID
    assert empty_variant.artifact().artifact_id == PRE_SLICE_UNGUIDED_ARTIFACT_ID


def test_optimizer_shares_the_rank_helper() -> None:
    from refaat.optimization import bootstrap

    assert bootstrap.build_ranked_result is _shared_helper
