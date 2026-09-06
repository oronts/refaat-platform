from __future__ import annotations

import pytest

from refaat.application.runtime import Refaat
from refaat.domain.models import DocumentInput
from refaat.evaluation import Dataset
from refaat.optimization import context_packing_search_space, retrieval_search_space

pytestmark = pytest.mark.integration


def test_retrieval_optimization_ranks_candidates(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    for index in range(4):
        runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id=f"doc{index}.txt",
                title=f"Doc {index}",
                media_type="text/plain",
                content=f"Runbook {index}: escalate a Sev1 incident to on-call within fifteen minutes.".encode(),
            ),
        )
    dataset = Dataset.from_records(
        "optimization",
        [
            {"query": "incident escalation", "expected_terms": ["incident"]},
            {"query": "on-call response time", "expected_terms": ["minutes"]},
        ],
    )

    result = project.programs.get("grounded-rag").optimize(
        dataset, search_space=retrieval_search_space(top_k=(2, 6, 12))
    )

    assert result.program == "grounded-rag"
    assert result.candidate_count == 3
    assert result.best.hard_constraints_passed is True
    scores = [(candidate.hard_constraints_passed, candidate.score) for candidate in result.candidates]
    assert scores == sorted(scores, reverse=True)
    assert result.best.artifact_id.startswith("sha256:")
    assert isinstance(result.improvement, float)
    assert len({candidate.artifact_id for candidate in result.candidates}) == 3
    assert result.best.mean_tokens >= 0.0

    packed = project.programs.get("grounded-rag").optimize(
        dataset, search_space=context_packing_search_space(top_k=(6,), token_budget=(128, 4000))
    )
    assert packed.best.hard_constraints_passed is True
    assert all(candidate.mean_tokens >= 0.0 for candidate in packed.candidates)
    assert packed.pareto_front
    assert all(candidate.hard_constraints_passed for candidate in packed.pareto_front)
