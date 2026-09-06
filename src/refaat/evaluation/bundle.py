from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from refaat.evaluation.metrics import (
    AnswerContains,
    CitationRecall,
    ContextPrecision,
    Faithfulness,
    Grounded,
    Metric,
    ModelJudge,
    SemanticSimilarity,
    StepBudget,
    TokenCost,
    TokenF1,
    ToolAuthorization,
    ToolSafety,
)
from refaat.ports import JudgeModel


@dataclass(frozen=True, slots=True)
class MetricBundle:
    """A named set of metrics with hard constraints and weighted objectives.

    Hard constraints must pass on every case regardless of the average score;
    objectives are combined into a single weighted score for ranking.
    """

    name: str
    metrics: tuple[Metric, ...]
    hard_constraints: frozenset[str] = frozenset()
    objectives: Mapping[str, float] = field(default_factory=dict)

    def weight_for(self, metric_name: str) -> float:
        return float(self.objectives.get(metric_name, 0.0))


def default_bundle() -> MetricBundle:
    metrics: tuple[Metric, ...] = (Grounded(), AnswerContains(), CitationRecall(), TokenF1())
    return MetricBundle(
        name="grounded-rag",
        metrics=metrics,
        hard_constraints=frozenset({"grounded"}),
        objectives={"answer_contains": 0.5, "citation_recall": 0.3, "token_f1": 0.2},
    )


def extended_bundle() -> MetricBundle:
    metrics: tuple[Metric, ...] = (
        Grounded(),
        AnswerContains(),
        CitationRecall(),
        ContextPrecision(),
        TokenF1(),
        SemanticSimilarity(),
        Faithfulness(),
    )
    return MetricBundle(
        name="grounded-rag-extended",
        metrics=metrics,
        hard_constraints=frozenset({"grounded"}),
        objectives={
            "answer_contains": 0.25,
            "citation_recall": 0.2,
            "context_precision": 0.1,
            "token_f1": 0.15,
            "semantic_similarity": 0.15,
            "faithfulness": 0.15,
        },
    )


def judge_bundle(judge: JudgeModel | None = None) -> MetricBundle:
    """The grounded-rag metrics plus a model-judge objective.

    With no ``judge`` the model_judge metric is the deterministic Faithfulness
    proxy, so this bundle scores offline with no model and no fabricated score;
    injecting a JudgeModel activates the real judge. model_judge is a weighted
    objective, never a hard constraint, so a low judge score never fails a case
    on its own.
    """

    metrics: tuple[Metric, ...] = (
        Grounded(),
        AnswerContains(),
        CitationRecall(),
        TokenF1(),
        ModelJudge(judge=judge),
    )
    return MetricBundle(
        name="grounded-rag-judge",
        metrics=metrics,
        hard_constraints=frozenset({"grounded"}),
        objectives={
            "answer_contains": 0.3,
            "citation_recall": 0.2,
            "token_f1": 0.2,
            "model_judge": 0.3,
        },
    )


def trajectory_bundle(*, max_steps: int = 4, max_tokens: int = 900) -> MetricBundle:
    """A bundle for scoring a governed agent trajectory end to end.

    Composes the shipped grounding and citation metrics with the trajectory
    metrics. Grounding and tool safety are hard constraints, so a trajectory that
    leaves a denied step with an observation or fails to ground never passes,
    independent of the weighted objective score. The step and token budgets are
    bundle construction parameters, not new evaluation-case fields.
    """

    metrics: tuple[Metric, ...] = (
        Grounded(),
        CitationRecall(),
        ToolSafety(),
        ToolAuthorization(),
        StepBudget(max_steps=max_steps),
        TokenCost(max_tokens=max_tokens),
    )
    return MetricBundle(
        name="trajectory",
        metrics=metrics,
        hard_constraints=frozenset({"grounded", "tool_safety"}),
        objectives={
            "tool_authorization": 0.4,
            "citation_recall": 0.3,
            "step_budget": 0.15,
            "token_cost": 0.15,
        },
    )
