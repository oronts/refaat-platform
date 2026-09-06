from __future__ import annotations

import pytest

from refaat.application.runtime import Refaat
from refaat.core.guidance import guidance_digest
from refaat.domain.models import DocumentInput
from refaat.evaluation import Dataset
from refaat.optimization import InstructionOptimizer
from refaat.programs.program import GroundedRagProgram

pytestmark = pytest.mark.integration

# The unguided grounded-rag artifact id, pinned before this slice. Instructions
# are omitted from Tunables.as_json when empty, so an unguided program must hash
# to exactly this value; a guided program must differ.
PRE_SLICE_UNGUIDED_ARTIFACT_ID = "sha256:b76d73f8d956bcd38ccc9d4c24372b07b387647a189f3a9e300cded186febbb4"


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
        "instructions",
        [
            {"id": "a", "query": "incident escalation path", "expected_terms": ["incident"]},
            {"id": "b", "query": "on-call response window", "expected_terms": ["minutes"]},
            {"id": "c", "query": "sev1 handling", "expected_terms": ["sev1"]},
            {"id": "d", "query": "runbook location", "expected_terms": ["runbook"]},
            {"id": "e", "query": "escalation timing", "expected_terms": ["fifteen"]},
            {"id": "f", "query": "on-call duty", "expected_terms": ["call"]},
        ],
    )


def test_mipro_enumerates_distinct_ranked_candidates(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    _ingest(project)
    program = GroundedRagProgram()
    before = program.tunables

    result = InstructionOptimizer(project).optimize_mipro(program, _dataset(), max_candidates=4)

    assert result.candidate_count == 5
    baseline = result.baseline
    assert "instructions" not in baseline.tunables
    guided = [candidate for candidate in result.candidates if "instructions" in candidate.tunables]
    assert len(guided) == 4
    assert len({candidate.artifact_id for candidate in guided}) == 4
    assert all(candidate.artifact_id != baseline.artifact_id for candidate in guided)
    # Held-out excludes the signal split, so the eval version differs from the full set.
    assert result.dataset_version != _dataset().version
    # The active program is never mutated.
    assert program.tunables is before
    assert program.tunables.instructions is None


def test_mipro_is_reproducible(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    _ingest(project)
    program = GroundedRagProgram()

    first = InstructionOptimizer(project).optimize_mipro(program, _dataset())
    second = InstructionOptimizer(project).optimize_mipro(program, _dataset())

    first_ids = [candidate.artifact_id for candidate in first.candidates]
    second_ids = [candidate.artifact_id for candidate in second.candidates]
    assert first_ids == second_ids


def test_gepa_breeds_at_least_one_ranked_mutant(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    _ingest(project)
    program = GroundedRagProgram()

    result = InstructionOptimizer(project).optimize_gepa(program, _dataset(), rounds=3)

    assert result.candidate_count >= 2
    guided = [candidate for candidate in result.candidates if "instructions" in candidate.tunables]
    assert guided
    assert len({candidate.artifact_id for candidate in guided}) == len(guided)
    assert program.tunables.instructions is None


def test_unguided_candidate_id_is_byte_identical_and_guided_differs(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    _ingest(project)
    program = GroundedRagProgram()

    result = InstructionOptimizer(project).optimize_mipro(program, _dataset())

    assert result.baseline.artifact_id == PRE_SLICE_UNGUIDED_ARTIFACT_ID
    guided = next(candidate for candidate in result.candidates if "instructions" in candidate.tunables)
    assert guided.artifact_id != PRE_SLICE_UNGUIDED_ARTIFACT_ID
    # The winning instruction folds into the shipped guidance digest.
    instruction = guided.tunables["instructions"]
    assert isinstance(instruction, str)
    assert guidance_digest(instruction, ()) is not None


def test_guided_instruction_folds_into_the_answer_cache_identity(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    _ingest(project)
    result = InstructionOptimizer(project).optimize_mipro(GroundedRagProgram(), _dataset())
    guided = next(candidate for candidate in result.candidates if "instructions" in candidate.tunables)
    instruction = guided.tunables["instructions"]
    assert isinstance(instruction, str)
    query = "How fast must a Sev1 incident be escalated?"

    unguided_first = project.ask(query)
    unguided_repeat = project.ask(query)
    guided_first = project.ask(query, instructions=instruction)
    guided_repeat = project.ask(query, instructions=instruction)

    assert not unguided_first.cached
    assert unguided_repeat.cached
    # A distinct instruction opens a distinct cache scope, then repeats hit it.
    assert not guided_first.cached
    assert guided_repeat.cached
