from __future__ import annotations

import pytest

from refaat.application.runtime import Refaat
from refaat.domain.models import DocumentInput
from refaat.evaluation import Dataset
from refaat.optimization import (
    RANDOM_SEARCH,
    BootstrapFewShotOptimizer,
    bootstrap_demonstrations,
    random_demonstration_subsets,
)
from refaat.programs.program import GroundedRagProgram

pytestmark = pytest.mark.integration


def _ingest(project) -> None:
    for index in range(4):
        project.runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id=f"doc{index}.txt",
                title=f"Doc {index}",
                media_type="text/plain",
                content=f"Runbook {index}: escalate a Sev1 incident to on-call within fifteen minutes.".encode(),
            ),
        )


def _dataset() -> Dataset:
    return Dataset.from_records(
        "bootstrap",
        [
            {"id": "a", "query": "incident escalation path", "expected_terms": ["incident"]},
            {"id": "b", "query": "on-call response window", "expected_terms": ["minutes"]},
            {"id": "c", "query": "sev1 handling", "expected_terms": ["sev1"]},
            {"id": "d", "query": "unicorn taxonomy", "expected_terms": ["unicorn"]},
            {"id": "e", "query": "runbook location", "expected_terms": ["runbook"]},
        ],
    )


def test_bootstrap_filters_sources_holds_out_and_distinct_artifact(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    _ingest(project)
    program = GroundedRagProgram()
    before = program.tunables

    traces = bootstrap_demonstrations(project, program, _dataset().cases)
    # The ungrounded 'unicorn' case fails the metric gate and never bootstraps.
    assert {trace.case_id for trace in traces} == {"a", "b", "c", "e"}

    result = BootstrapFewShotOptimizer(project).optimize(program, _dataset())

    assert result.candidate_count == 2
    baseline = result.baseline
    assert "demonstrations" not in baseline.tunables
    guided = next(candidate for candidate in result.candidates if "demonstrations" in candidate.tunables)
    assert guided.tunables["demonstrations"]
    assert guided.artifact_id != baseline.artifact_id
    # Honesty boundary: on the extractive stack the demonstrations do not change
    # the answer, so the guided candidate has the same objective score but a
    # strictly higher token cost, and the zero-shot baseline ranks first.
    assert guided.score == baseline.score
    assert guided.mean_tokens > baseline.mean_tokens
    assert result.best.artifact_id == baseline.artifact_id
    # Held-out excludes the source cases, so the eval version differs from the full set.
    assert result.dataset_version != _dataset().version
    # The active program is never mutated.
    assert program.tunables is before
    assert program.tunables.demonstrations == ()


def test_bootstrap_is_reproducible(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    _ingest(project)
    program = GroundedRagProgram()

    first = BootstrapFewShotOptimizer(project).optimize(program, _dataset())
    second = BootstrapFewShotOptimizer(project).optimize(program, _dataset())

    first_guided = next(candidate for candidate in first.candidates if "demonstrations" in candidate.tunables)
    second_guided = next(candidate for candidate in second.candidates if "demonstrations" in candidate.tunables)
    assert first_guided.artifact_id == second_guided.artifact_id
    assert first_guided.tunables == second_guided.tunables


def test_bootstrap_sanitizes_real_chunk_ids(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    _ingest(project)
    program = GroundedRagProgram()
    real_ids = {chunk.id for chunk in runtime.store.list_chunks(project.workspace_id, project.corpus_id)}
    assert real_ids

    traces = bootstrap_demonstrations(project, program, _dataset().cases)
    joined = " ".join(f"{trace.demonstration.question} {trace.demonstration.answer}" for trace in traces)

    assert "[chunk:example]" in joined
    for chunk_id in real_ids:
        assert chunk_id not in joined


def test_random_search_ranks_all_trials_via_shared_helper(runtime: Refaat) -> None:
    from refaat.optimization import bootstrap
    from refaat.optimization.candidates import build_ranked_result as shared_helper

    assert bootstrap.build_ranked_result is shared_helper

    project = runtime.project(workspace="acme", corpus="handbook")
    _ingest(project)
    program = GroundedRagProgram()

    traces = bootstrap_demonstrations(project, program, _dataset().cases)
    pool = tuple(trace.demonstration for trace in traces)
    subsets = random_demonstration_subsets(pool, trials=8, seed=0)

    result = BootstrapFewShotOptimizer(project).optimize(program, _dataset(), mode=RANDOM_SEARCH, trials=8, seed=0)

    assert result.candidate_count == 1 + len(subsets)
    guided = [candidate for candidate in result.candidates if "demonstrations" in candidate.tunables]
    assert len(guided) == len(subsets)
    assert program.tunables.demonstrations == ()
