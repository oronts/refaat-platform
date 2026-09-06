from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from refaat.application.runtime import Refaat
from refaat.domain.models import DocumentInput, JudgeVerdict
from refaat.evaluation import Dataset, extended_bundle, judge_bundle

pytestmark = pytest.mark.integration


@dataclass
class _ConstantJudge:
    score: float
    model_id: str = "constant-judge"

    def assess(self, query: str, answer: str, evidence: Sequence[str]) -> JudgeVerdict:
        return JudgeVerdict(score=self.score, passed=self.score >= 0.6, rationale="constant judge")


def _seed(runtime: Refaat) -> Dataset:
    project = runtime.project(workspace="acme", corpus="handbook")
    runtime.ingest(
        project.workspace_id,
        project.corpus_id,
        DocumentInput(
            source_id="incident.txt",
            title="Incident",
            media_type="text/plain",
            content=b"Escalate a Sev1 incident to the on-call engineer within fifteen minutes of detection.",
        ),
    )
    return Dataset.from_records(
        "incident-regression",
        [
            {"query": "incident escalation", "expected_terms": ["incident"]},
            {"query": "on-call response time", "expected_terms": ["minutes"]},
        ],
    )


def test_judge_bundle_offline_uses_faithfulness_proxy(runtime: Refaat) -> None:
    dataset = _seed(runtime)
    program = runtime.project(workspace="acme", corpus="handbook").programs.get("grounded-rag")

    judged = program.evaluate(dataset, judge_bundle())
    extended = program.evaluate(dataset, extended_bundle())

    assert "model_judge" in judged.aggregate
    assert runtime.project(workspace="acme", corpus="handbook").judge_model is None
    assert judged.aggregate["model_judge"] == pytest.approx(extended.aggregate["faithfulness"])
    assert judged.score > 0.0


def test_injected_judge_score_flows_into_aggregate(runtime: Refaat) -> None:
    dataset = _seed(runtime)
    program = runtime.project(workspace="acme", corpus="handbook").programs.get("grounded-rag")

    proxy = program.evaluate(dataset, judge_bundle())
    judged = program.evaluate(dataset, judge_bundle(_ConstantJudge(0.9)))

    assert judged.aggregate["model_judge"] == pytest.approx(0.9)
    assert judged.aggregate["model_judge"] != pytest.approx(proxy.aggregate["model_judge"])
