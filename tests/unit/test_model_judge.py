from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from refaat.core.citations import CITATION_PATTERN
from refaat.domain.models import Citation, GenerationRequest, GenerationResult, JudgeVerdict
from refaat.errors import DependencyUnavailableError
from refaat.evaluation import Faithfulness, ModelJudge, judge_bundle
from refaat.evaluation.dataset import EvaluationCase
from refaat.infrastructure.judging import GeneratorJudge
from refaat.programs import ProgramResult, ValidationOutcome


def _case() -> EvaluationCase:
    return EvaluationCase(id="1", query="rollback", expected_terms=("rollback",))


def _result(answer: str, quote: str) -> ProgramResult:
    citation = Citation(chunk_id="chunk_0", document_id="doc", label="[1]", quote=quote)
    return ProgramResult(
        program="fake",
        artifact_id="sha256:fake",
        answer=answer,
        citations=(citation,),
        routes=("semantic",),
        cached=False,
        corpus_version=1,
        trace_id="trace",
        outputs={},
        validations=(ValidationOutcome("grounding", True, "detail"),),
    )


@dataclass
class _StubJudge:
    verdict: JudgeVerdict
    model_id: str = "stub-judge"
    calls: list[tuple[str, str, tuple[str, ...]]] = field(default_factory=list)

    def assess(self, query: str, answer: str, evidence: Sequence[str]) -> JudgeVerdict:
        self.calls.append((query, answer, tuple(evidence)))
        return self.verdict


class _FailingJudge:
    model_id = "failing-judge"

    def assess(self, query: str, answer: str, evidence: Sequence[str]) -> JudgeVerdict:
        raise DependencyUnavailableError("judge unavailable")


@dataclass
class _StubGenerator:
    answer: str
    model_id: str = "stub-gen"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return GenerationResult(answer=self.answer, cited_chunk_ids=())

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        yield self.answer


def test_model_judge_conforms_to_metric_protocol() -> None:
    metric = ModelJudge()
    assert metric.name == "model_judge"
    outcome = metric.evaluate(_case(), _result("rollback [chunk:chunk_0]", "rollback"))
    assert outcome.name == "model_judge"
    assert isinstance(outcome.score, float)


def test_offline_proxy_equals_shipped_faithfulness() -> None:
    case = _case()
    result = _result(
        "rollback restores the prior artifact safely [chunk:chunk_0]", "rollback restores the prior artifact"
    )

    faith = Faithfulness().evaluate(case, result)
    proxy = ModelJudge(judge=None).evaluate(case, result)

    assert proxy.score == faith.score
    assert proxy.passed == faith.passed
    assert proxy.detail == faith.detail
    assert proxy.name == "model_judge"
    assert 0.0 < proxy.score < 1.0


def test_stub_judge_maps_verdict_into_metric_result() -> None:
    stub = _StubJudge(JudgeVerdict(score=0.42, passed=False, rationale="weak support"))
    result = _result("rollback restores the prior artifact [chunk:chunk_0]", "rollback restores the prior artifact")

    outcome = ModelJudge(judge=stub).evaluate(_case(), result)

    assert outcome.name == "model_judge"
    assert outcome.score == 0.42
    assert outcome.passed is False
    assert outcome.detail == "weak support"
    assert stub.calls == [("rollback", result.answer, ("rollback restores the prior artifact",))]


def test_dependency_unavailable_degrades_to_proxy_without_faking_a_score() -> None:
    case = _case()
    result = _result(
        "rollback restores the prior artifact safely [chunk:chunk_0]", "rollback restores the prior artifact"
    )

    faith = Faithfulness().evaluate(case, result)
    outcome = ModelJudge(judge=_FailingJudge()).evaluate(case, result)

    assert outcome.score == faith.score
    assert outcome.passed == faith.passed
    assert outcome.detail == faith.detail


def test_rationale_lives_in_detail_and_is_never_a_citation() -> None:
    stub = _StubJudge(JudgeVerdict(score=0.9, passed=True, rationale="supported [chunk:ghost]"))
    result = _result("rollback [chunk:chunk_0]", "rollback")

    outcome = ModelJudge(judge=stub).evaluate(_case(), result)

    assert outcome.detail == "supported [chunk:ghost]"
    assert not hasattr(outcome, "citations")
    assert result.citations == (Citation(chunk_id="chunk_0", document_id="doc", label="[1]", quote="rollback"),)
    cited_ids = {citation.chunk_id for citation in result.citations}
    assert "ghost" not in cited_ids
    assert CITATION_PATTERN.findall(outcome.detail) == ["ghost"]
    assert not any(chunk_id == "ghost" for chunk_id in cited_ids)


def test_judge_bundle_registers_model_judge_as_weighted_objective() -> None:
    bundle = judge_bundle()

    names = {metric.name for metric in bundle.metrics}
    assert "model_judge" in names
    assert bundle.name == "grounded-rag-judge"
    assert bundle.objectives["model_judge"] == 0.3
    assert "model_judge" not in bundle.hard_constraints
    judge_metric = next(metric for metric in bundle.metrics if metric.name == "model_judge")
    assert isinstance(judge_metric, ModelJudge)
    assert judge_metric.judge is None


def test_judge_bundle_carries_injected_judge() -> None:
    stub = _StubJudge(JudgeVerdict(score=1.0, passed=True, rationale="ok"))
    bundle = judge_bundle(stub)
    judge_metric = next(metric for metric in bundle.metrics if metric.name == "model_judge")
    assert isinstance(judge_metric, ModelJudge)
    assert judge_metric.judge is stub


def test_generator_judge_parses_bounded_score() -> None:
    judge = GeneratorJudge(_StubGenerator("score: 0.8\nWell supported by the evidence."))
    verdict = judge.assess("rollback", "rollback restores state", ("rollback restores state",))
    assert verdict.score == 0.8
    assert verdict.passed is True
    assert "Well supported" in verdict.rationale
    assert judge.model_id == "judge:stub-gen"


def test_generator_judge_score_flows_through_model_judge() -> None:
    judge = GeneratorJudge(_StubGenerator("score: 0.3 partial coverage only"))
    outcome = ModelJudge(judge=judge).evaluate(_case(), _result("rollback [chunk:chunk_0]", "rollback"))
    assert outcome.name == "model_judge"
    assert outcome.score == 0.3
    assert outcome.passed is False


def test_generator_judge_unparseable_score_raises_and_model_judge_falls_back() -> None:
    case = _case()
    result = _result(
        "rollback restores the prior artifact safely [chunk:chunk_0]", "rollback restores the prior artifact"
    )
    judge = GeneratorJudge(_StubGenerator("The answer looks reasonable overall."))

    try:
        judge.assess(case.query, result.answer, ("rollback restores the prior artifact",))
    except DependencyUnavailableError:
        pass
    else:  # pragma: no cover - the parser must reject a score-free generation
        raise AssertionError("expected DependencyUnavailableError for a score-free generation")

    faith = Faithfulness().evaluate(case, result)
    outcome = ModelJudge(judge=judge).evaluate(case, result)
    assert outcome.score == faith.score
    assert outcome.passed == faith.passed
