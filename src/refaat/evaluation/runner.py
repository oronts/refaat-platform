from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from refaat.domain.authz import Principal
from refaat.domain.models import JsonObject
from refaat.evaluation.bundle import MetricBundle, default_bundle
from refaat.evaluation.dataset import Dataset
from refaat.evaluation.metrics import MetricResult
from refaat.programs.program import ProgramResult


class RunnableProgram(Protocol):
    @property
    def name(self) -> str: ...

    def run(
        self,
        *,
        user_id: str | None = ...,
        session_id: str | None = ...,
        principal: Principal | None = ...,
        **inputs: object,
    ) -> ProgramResult: ...


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: str
    query: str
    passed: bool
    tokens: int
    metrics: tuple[MetricResult, ...]


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    program: str
    dataset: str
    dataset_version: str
    case_count: int
    hard_constraints_passed: bool
    score: float
    aggregate: dict[str, float]
    mean_tokens: float
    cases: tuple[CaseResult, ...]

    def to_json(self) -> JsonObject:
        return {
            "program": self.program,
            "dataset": self.dataset,
            "dataset_version": self.dataset_version,
            "case_count": self.case_count,
            "hard_constraints_passed": self.hard_constraints_passed,
            "score": self.score,
            "aggregate": self.aggregate,
            "mean_tokens": self.mean_tokens,
            "cases": [
                {
                    "case_id": case.case_id,
                    "query": case.query,
                    "passed": case.passed,
                    "tokens": case.tokens,
                    "metrics": [
                        {"name": metric.name, "score": metric.score, "passed": metric.passed, "detail": metric.detail}
                        for metric in case.metrics
                    ],
                }
                for case in self.cases
            ],
        }


class ProgramEvaluator:
    """Runs a program over a dataset and scores it with a metric bundle.

    Execution reuses the program's own runtime path, so evaluation exercises
    exactly the retrieval, generation, caching, and citation behavior that
    production uses.
    """

    def __init__(self, program: RunnableProgram) -> None:
        self._program = program

    def evaluate(self, dataset: Dataset, bundle: MetricBundle | None = None) -> EvaluationReport:
        active = bundle or default_bundle()
        scores: dict[str, list[float]] = {metric.name: [] for metric in active.metrics}
        passed_all: dict[str, bool] = {metric.name: True for metric in active.metrics}
        cases: list[CaseResult] = []
        token_costs: list[int] = []
        for case in dataset.cases:
            result = self._program.run(question=case.query)
            tokens = int(result.usage.get("context_tokens", 0))
            token_costs.append(tokens)
            outcomes = tuple(metric.evaluate(case, result) for metric in active.metrics)
            for outcome in outcomes:
                scores[outcome.name].append(outcome.score)
                passed_all[outcome.name] = passed_all[outcome.name] and outcome.passed
            case_passed = all(outcome.passed for outcome in outcomes if outcome.name in active.hard_constraints)
            cases.append(
                CaseResult(case_id=case.id, query=case.query, passed=case_passed, tokens=tokens, metrics=outcomes)
            )
        aggregate = {name: (sum(values) / len(values) if values else 0.0) for name, values in scores.items()}
        hard_passed = all(passed_all[name] for name in active.hard_constraints)
        total_weight = sum(active.objectives.values())
        score = (
            sum(active.weight_for(name) * aggregate.get(name, 0.0) for name in active.objectives) / total_weight
            if total_weight > 0
            else 0.0
        )
        return EvaluationReport(
            program=self._program.name,
            dataset=dataset.name,
            dataset_version=dataset.version,
            case_count=len(dataset.cases),
            hard_constraints_passed=hard_passed,
            score=score,
            aggregate=aggregate,
            mean_tokens=sum(token_costs) / len(token_costs) if token_costs else 0.0,
            cases=tuple(cases),
        )
