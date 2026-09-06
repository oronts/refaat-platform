from __future__ import annotations

from refaat.domain.models import Citation
from refaat.evaluation import (
    AnswerContains,
    CitationRecall,
    ContextPrecision,
    Dataset,
    EvaluationCase,
    Faithfulness,
    Grounded,
    ProgramEvaluator,
    SemanticSimilarity,
    TokenF1,
    default_bundle,
    extended_bundle,
)
from refaat.programs import ProgramResult, ValidationOutcome


def _result(answer: str, document_ids: tuple[str, ...] = (), *, grounded: bool = True) -> ProgramResult:
    citations = tuple(
        Citation(chunk_id=f"chunk_{index}", document_id=document_id, label=f"[{index}]", quote="quote")
        for index, document_id in enumerate(document_ids)
    )
    return ProgramResult(
        program="fake",
        artifact_id="sha256:fake",
        answer=answer,
        citations=citations,
        routes=("semantic",),
        cached=False,
        corpus_version=1,
        trace_id="trace",
        outputs={},
        validations=(ValidationOutcome("grounding", grounded, "detail"),),
    )


def test_extended_metrics_and_bundle() -> None:
    case = EvaluationCase(
        id="1",
        query="rollback",
        expected_terms=("rollback",),
        expected_document_ids=("doc",),
        reference_answer="rollback restores the prior artifact",
    )
    result = _result("rollback restores the prior artifact [chunk:chunk_0]", ("doc",))

    assert SemanticSimilarity().evaluate(case, result).score > 0.5
    assert ContextPrecision().evaluate(case, result).passed is True
    assert ContextPrecision().evaluate(case, _result("rollback", ("other",))).passed is False

    grounded = ProgramResult(
        program="fake",
        artifact_id="sha256:fake",
        answer="rollback restores the prior artifact [chunk:chunk_0]",
        citations=(
            Citation(
                chunk_id="chunk_0",
                document_id="doc",
                label="[1]",
                quote="rollback restores the prior artifact",
            ),
        ),
        routes=("semantic",),
        cached=False,
        corpus_version=1,
        trace_id="trace",
        outputs={},
        validations=(ValidationOutcome("grounding", True, "detail"),),
    )
    assert Faithfulness().evaluate(case, grounded).passed is True

    bundle = extended_bundle()
    assert bundle.name == "grounded-rag-extended"
    assert {"semantic_similarity", "faithfulness", "context_precision"} <= {metric.name for metric in bundle.metrics}


def test_metrics_compute_expected_scores() -> None:
    hit = EvaluationCase(id="1", query="q", expected_terms=("alpha",), expected_document_ids=("doc",))
    miss = EvaluationCase(id="2", query="q", expected_terms=("gamma",), expected_document_ids=("other",))
    result = _result("alpha beta", ("doc",))

    assert AnswerContains().evaluate(hit, result).passed is True
    assert AnswerContains().evaluate(miss, result).passed is False
    assert CitationRecall().evaluate(hit, result).passed is True
    assert CitationRecall().evaluate(miss, result).passed is False
    assert Grounded().evaluate(hit, result).passed is True
    assert Grounded().evaluate(hit, _result("x", grounded=False)).passed is False

    with_reference = EvaluationCase(id="3", query="q", reference_answer="alpha beta gamma")
    f1 = TokenF1(threshold=0.4).evaluate(with_reference, _result("alpha beta"))
    assert 0.0 < f1.score <= 1.0
    assert TokenF1().evaluate(EvaluationCase(id="4", query="q"), result).passed is True


def test_dataset_version_is_content_addressed() -> None:
    records = [{"query": "one", "expected_terms": ["a"]}, {"query": "two"}]
    first = Dataset.from_records("suite", records)
    second = Dataset.from_records("suite", records)
    changed = Dataset.from_records("suite", [{"query": "one", "expected_terms": ["b"]}, {"query": "two"}])
    assert first.version == second.version
    assert first.version != changed.version
    assert first.cases[0].id == "case-0"


class _FakeProgram:
    name = "fake-program"

    def __init__(self, result: ProgramResult) -> None:
        self._result = result

    def run(self, *, user_id: str | None = None, session_id: str | None = None, **inputs: object) -> ProgramResult:
        return self._result


def test_program_evaluator_aggregates_and_enforces_hard_constraints() -> None:
    dataset = Dataset.from_records(
        "suite",
        [
            {"query": "alpha", "expected_terms": ["alpha"], "expected_document_ids": ["doc"]},
            {"query": "beta", "expected_terms": ["beta"], "expected_document_ids": ["doc"]},
        ],
    )
    report = ProgramEvaluator(_FakeProgram(_result("alpha beta", ("doc",)))).evaluate(dataset, default_bundle())
    assert report.program == "fake-program"
    assert report.case_count == 2
    assert report.hard_constraints_passed is True
    assert 0.0 <= report.score <= 1.0
    assert set(report.aggregate) == {"grounded", "answer_contains", "citation_recall", "token_f1"}

    ungrounded = ProgramEvaluator(_FakeProgram(_result("alpha beta", grounded=False))).evaluate(dataset)
    assert ungrounded.hard_constraints_passed is False
