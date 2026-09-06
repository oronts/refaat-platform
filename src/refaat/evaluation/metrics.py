from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from refaat.core.citations import CITATION_PATTERN
from refaat.core.text import tokens
from refaat.domain.models import JsonObject
from refaat.errors import DependencyUnavailableError
from refaat.evaluation.dataset import EvaluationCase
from refaat.ports import JudgeModel
from refaat.programs.program import ProgramResult


def _react_steps(result: ProgramResult) -> list[JsonObject]:
    raw = result.outputs.get("react_steps", [])
    if not isinstance(raw, list):
        return []
    return [step for step in raw if isinstance(step, dict)]


def _budget_score(value: int, budget: int) -> float:
    """Monotonic budget score: 1.0 at or under budget, decaying past it.

    Shared by :class:`StepBudget` and :class:`TokenCost` so there is one scoring
    body. A value at the budget scores 1.0; a value one unit over scores below 1.0
    and the score decreases monotonically as the value grows.
    """
    if value <= 0:
        return 1.0
    return min(1.0, budget / value)


@dataclass(frozen=True, slots=True)
class MetricResult:
    name: str
    score: float
    passed: bool
    detail: str


class Metric(Protocol):
    @property
    def name(self) -> str: ...

    def evaluate(self, case: EvaluationCase, result: ProgramResult) -> MetricResult: ...


@dataclass(frozen=True, slots=True)
class AnswerContains:
    name: str = "answer_contains"

    def evaluate(self, case: EvaluationCase, result: ProgramResult) -> MetricResult:
        terms = [term.casefold() for term in case.expected_terms]
        if not terms:
            return MetricResult(self.name, 1.0, True, "no expected terms")
        answer = result.answer.casefold()
        hits = sum(1 for term in terms if term in answer)
        return MetricResult(self.name, hits / len(terms), hits == len(terms), f"{hits}/{len(terms)} terms present")


@dataclass(frozen=True, slots=True)
class CitationRecall:
    name: str = "citation_recall"

    def evaluate(self, case: EvaluationCase, result: ProgramResult) -> MetricResult:
        expected = {str(value) for value in case.expected_document_ids}
        if not expected:
            return MetricResult(self.name, 1.0, True, "no expected documents")
        cited = {citation.document_id for citation in result.citations}
        found = expected & cited
        return MetricResult(
            self.name, len(found) / len(expected), found == expected, f"{len(found)}/{len(expected)} documents cited"
        )


@dataclass(frozen=True, slots=True)
class Grounded:
    name: str = "grounded"

    def evaluate(self, case: EvaluationCase, result: ProgramResult) -> MetricResult:
        passed = result.grounded and bool(result.citations)
        return MetricResult(self.name, 1.0 if passed else 0.0, passed, "grounded" if passed else "not grounded")


@dataclass(frozen=True, slots=True)
class TokenF1:
    threshold: float = 0.5
    name: str = "token_f1"

    def evaluate(self, case: EvaluationCase, result: ProgramResult) -> MetricResult:
        if not case.reference_answer:
            return MetricResult(self.name, 1.0, True, "no reference answer")
        predicted = set(tokens(result.answer))
        expected = set(tokens(case.reference_answer))
        if not predicted or not expected:
            return MetricResult(self.name, 0.0, False, "empty tokens")
        overlap = len(predicted & expected)
        precision = overlap / len(predicted)
        recall = overlap / len(expected)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        return MetricResult(self.name, f1, f1 >= self.threshold, f"f1={f1:.2f}")


@dataclass(frozen=True, slots=True)
class SemanticSimilarity:
    """Deterministic term-frequency cosine similarity between the answer and the reference answer."""

    threshold: float = 0.5
    name: str = "semantic_similarity"

    def evaluate(self, case: EvaluationCase, result: ProgramResult) -> MetricResult:
        if not case.reference_answer:
            return MetricResult(self.name, 1.0, True, "no reference answer")
        predicted = Counter(tokens(result.answer))
        expected = Counter(tokens(case.reference_answer))
        if not predicted or not expected:
            return MetricResult(self.name, 0.0, False, "empty tokens")
        dot = sum(predicted[term] * expected[term] for term in set(predicted) & set(expected))
        norm = math.sqrt(sum(value * value for value in predicted.values())) * math.sqrt(
            sum(value * value for value in expected.values())
        )
        score = 0.0 if norm == 0 else dot / norm
        return MetricResult(self.name, score, score >= self.threshold, f"cosine={score:.2f}")


@dataclass(frozen=True, slots=True)
class Faithfulness:
    """Fraction of answer tokens supported by the cited evidence, a deterministic groundedness proxy."""

    threshold: float = 0.6
    name: str = "faithfulness"

    def evaluate(self, case: EvaluationCase, result: ProgramResult) -> MetricResult:
        answer_tokens = set(tokens(CITATION_PATTERN.sub(" ", result.answer)))
        if not answer_tokens:
            return MetricResult(self.name, 1.0, True, "empty answer")
        evidence: set[str] = set()
        for citation in result.citations:
            evidence |= set(tokens(citation.quote))
        if not evidence:
            return MetricResult(self.name, 0.0, False, "no cited evidence")
        supported = answer_tokens & evidence
        score = len(supported) / len(answer_tokens)
        return MetricResult(
            self.name, score, score >= self.threshold, f"{len(supported)}/{len(answer_tokens)} answer tokens supported"
        )


@dataclass(frozen=True, slots=True)
class ModelJudge:
    """LLM-as-judge quality metric that degrades to the Faithfulness proxy.

    With no ``judge`` the score is delegated to the shipped :class:`Faithfulness`
    computation, so the offline gate never calls a model and never fabricates a
    judge score. When a :class:`JudgeModel` is injected its verdict drives the
    score; the judge rationale lands in ``detail`` only and is never a citation.
    A judge that raises :class:`DependencyUnavailableError` degrades to the same
    deterministic proxy rather than guessing.
    """

    judge: JudgeModel | None = None
    threshold: float = 0.6
    name: str = "model_judge"

    def evaluate(self, case: EvaluationCase, result: ProgramResult) -> MetricResult:
        if self.judge is None:
            return self._proxy(case, result)
        evidence = tuple(citation.quote for citation in result.citations)
        try:
            verdict = self.judge.assess(case.query, result.answer, evidence)
        except DependencyUnavailableError:
            return self._proxy(case, result)
        return MetricResult(self.name, verdict.score, verdict.passed, verdict.rationale)

    def _proxy(self, case: EvaluationCase, result: ProgramResult) -> MetricResult:
        outcome = Faithfulness(threshold=self.threshold).evaluate(case, result)
        return MetricResult(self.name, outcome.score, outcome.passed, outcome.detail)


@dataclass(frozen=True, slots=True)
class ContextPrecision:
    """Fraction of cited documents that were expected, the precision counterpart to citation recall."""

    name: str = "context_precision"

    def evaluate(self, case: EvaluationCase, result: ProgramResult) -> MetricResult:
        expected = {str(value) for value in case.expected_document_ids}
        if not expected:
            return MetricResult(self.name, 1.0, True, "no expected documents")
        cited = {citation.document_id for citation in result.citations}
        if not cited:
            return MetricResult(self.name, 0.0, False, "no citations")
        relevant = cited & expected
        score = len(relevant) / len(cited)
        return MetricResult(
            self.name, score, relevant == cited, f"{len(relevant)}/{len(cited)} cited documents expected"
        )


@dataclass(frozen=True, slots=True)
class ToolSafety:
    """Hard structural gate: a denied governed step must carry no observation.

    In a real ``GovernedReActProgram`` trace a denied step is never executed, so
    it always has an empty observation and this constraint holds by construction.
    It is therefore a regression tripwire, not a live discriminator: it fails only
    if a future change lets an unauthorized tool leave an observation. The unit
    test exercises the failing branch with a hand-crafted denied-with-observation
    trace so the metric is not dead code.
    """

    name: str = "tool_safety"

    def evaluate(self, case: EvaluationCase, result: ProgramResult) -> MetricResult:
        del case
        steps = _react_steps(result)
        violations = [step for step in steps if not step.get("authorized", False) and step.get("observation")]
        passed = not violations
        detail = (
            "no denied step leaked an observation" if passed else f"{len(violations)} denied step(s) with observation"
        )
        return MetricResult(self.name, 1.0 if passed else 0.0, passed, detail)


@dataclass(frozen=True, slots=True)
class ToolAuthorization:
    """Plan coverage: the fraction of governed steps handled correctly.

    A step is handled correctly when it is authorized (and executed) or correctly
    denied (denied with no observation). A governed denial of an over-privileged
    tool is correct behavior, so it counts toward coverage rather than against it;
    a trajectory that correctly refuses an over-privileged tool is never ranked
    below one where every planned tool happened to be authorized. The score drops
    only when a denied step leaks an observation, i.e. a governance breach.
    """

    name: str = "tool_authorization"

    def evaluate(self, case: EvaluationCase, result: ProgramResult) -> MetricResult:
        del case
        steps = _react_steps(result)
        if not steps:
            return MetricResult(self.name, 1.0, True, "no governed steps")
        correct = sum(1 for step in steps if step.get("authorized", False) or not step.get("observation"))
        score = correct / len(steps)
        return MetricResult(self.name, score, correct == len(steps), f"{correct}/{len(steps)} steps handled correctly")


@dataclass(frozen=True, slots=True)
class StepBudget:
    """Reward a trajectory that answers within a step budget, via ``_budget_score``."""

    max_steps: int = 4
    name: str = "step_budget"

    def evaluate(self, case: EvaluationCase, result: ProgramResult) -> MetricResult:
        del case
        value = len(_react_steps(result))
        score = _budget_score(value, self.max_steps)
        return MetricResult(self.name, score, value <= self.max_steps, f"{value}/{self.max_steps} steps")


@dataclass(frozen=True, slots=True)
class TokenCost:
    """Reward a trajectory that stays within a context-token budget, via ``_budget_score``."""

    max_tokens: int = 900
    name: str = "token_cost"

    def evaluate(self, case: EvaluationCase, result: ProgramResult) -> MetricResult:
        del case
        value = int(result.usage.get("context_tokens", 0))
        score = _budget_score(value, self.max_tokens)
        return MetricResult(self.name, score, value <= self.max_tokens, f"{value}/{self.max_tokens} tokens")
