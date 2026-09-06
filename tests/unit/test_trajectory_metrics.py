from __future__ import annotations

import pytest

from refaat.domain.models import JsonObject
from refaat.evaluation.dataset import EvaluationCase
from refaat.evaluation.metrics import (
    StepBudget,
    TokenCost,
    ToolAuthorization,
    ToolSafety,
    _budget_score,
)
from refaat.programs.program import ProgramResult

CASE = EvaluationCase(id="c", query="q")


def _result(steps: list[JsonObject], *, tokens: int = 0) -> ProgramResult:
    return ProgramResult(
        program="governed-react",
        artifact_id="sha256:test",
        answer="answer",
        citations=(),
        routes=(),
        cached=False,
        corpus_version=1,
        trace_id="trace",
        outputs={"answer": "answer", "citations": [], "react_steps": steps},
        validations=(),
        usage={"context_tokens": tokens},
    )


def _step(tool: str, *, authorized: bool, observation: JsonObject | None = None) -> JsonObject:
    return {
        "tool": tool,
        "arguments": {},
        "authorized": authorized,
        "reason": "authorized" if authorized else "denied",
        "observation": observation or {},
    }


def test_tool_safety_passes_when_denied_steps_have_no_observation() -> None:
    steps = [_step("run", authorized=True, observation={"answer": "x"}), _step("pager", authorized=False)]
    outcome = ToolSafety().evaluate(CASE, _result(steps))
    assert outcome.passed
    assert outcome.score == 1.0


def test_tool_safety_fails_when_a_denied_step_carries_an_observation() -> None:
    steps = [_step("pager", authorized=False, observation={"sent": True})]
    outcome = ToolSafety().evaluate(CASE, _result(steps))
    assert not outcome.passed
    assert outcome.score == 0.0


def test_tool_safety_passes_vacuously_on_empty_steps() -> None:
    outcome = ToolSafety().evaluate(CASE, _result([]))
    assert outcome.passed
    assert outcome.score == 1.0


def test_tool_authorization_does_not_penalize_a_correct_governed_denial() -> None:
    all_authorized = [_step("a", authorized=True), _step("b", authorized=True)]
    with_denial = [_step("a", authorized=True), _step("pager", authorized=False)]
    score_authorized = ToolAuthorization().evaluate(CASE, _result(all_authorized)).score
    score_denial = ToolAuthorization().evaluate(CASE, _result(with_denial)).score
    assert score_authorized == 1.0
    assert score_denial == 1.0
    assert score_denial >= score_authorized


def test_tool_authorization_drops_on_a_governance_breach() -> None:
    steps = [_step("a", authorized=True), _step("pager", authorized=False, observation={"sent": True})]
    outcome = ToolAuthorization().evaluate(CASE, _result(steps))
    assert outcome.score == 0.5
    assert not outcome.passed


def test_tool_authorization_passes_vacuously_on_empty_steps() -> None:
    outcome = ToolAuthorization().evaluate(CASE, _result([]))
    assert outcome.passed
    assert outcome.score == 1.0


def test_step_budget_passes_at_max_and_fails_at_max_plus_one() -> None:
    at_max = [_step(f"t{index}", authorized=True) for index in range(3)]
    over = [_step(f"t{index}", authorized=True) for index in range(4)]
    metric = StepBudget(max_steps=3)
    at_outcome = metric.evaluate(CASE, _result(at_max))
    over_outcome = metric.evaluate(CASE, _result(over))
    assert at_outcome.passed
    assert at_outcome.score == 1.0
    assert not over_outcome.passed
    assert over_outcome.score < 1.0


def test_step_budget_is_monotonic_past_the_budget() -> None:
    metric = StepBudget(max_steps=2)
    scores = [
        metric.evaluate(CASE, _result([_step(f"t{i}", authorized=True) for i in range(count)])).score
        for count in (2, 3, 4, 5)
    ]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == 1.0
    assert all(scores[i] > scores[i + 1] for i in range(len(scores) - 2))


def test_token_cost_passes_and_fails_around_the_budget() -> None:
    metric = TokenCost(max_tokens=100)
    assert metric.evaluate(CASE, _result([], tokens=100)).passed
    over = metric.evaluate(CASE, _result([], tokens=101))
    assert not over.passed
    assert over.score < 1.0


def test_budget_score_boundaries() -> None:
    assert _budget_score(0, 10) == 1.0
    assert _budget_score(10, 10) == 1.0
    assert _budget_score(20, 10) == pytest.approx(0.5)
    assert _budget_score(11, 10) < 1.0
