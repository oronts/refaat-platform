from __future__ import annotations

import pytest

from refaat.application.runtime import Refaat
from refaat.domain.models import Demonstration, DocumentInput
from refaat.programs.program import GroundedRagProgram, Tunables

pytestmark = pytest.mark.integration

# The unguided grounded-rag artifact id, pinned before this slice. Guidance is
# omitted from Tunables.as_json when empty, so an unguided program must hash to
# exactly this value; a guided program must differ.
PRE_SLICE_UNGUIDED_ARTIFACT_ID = "sha256:b76d73f8d956bcd38ccc9d4c24372b07b387647a189f3a9e300cded186febbb4"


def _ingest(project) -> None:
    documents = {
        "alpha.txt": "Escalate a Sev1 incident to on-call within fifteen minutes of detection.",
        "bravo.txt": "Vacation accrues at two days per calendar month for full-time staff.",
    }
    for source_id, content in documents.items():
        project.runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id=source_id,
                title=source_id.removesuffix(".txt").title(),
                media_type="text/plain",
                content=content.encode(),
            ),
        )


def test_guidance_changes_cache_scope_and_unguided_scope_is_preserved(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    _ingest(project)
    query = "How fast must a Sev1 incident be escalated?"

    first = project.ask(query)
    repeated = project.ask(query)
    guided_a = project.ask(query, instructions="Answer in one sentence.")
    guided_a_repeat = project.ask(query, instructions="Answer in one sentence.")
    guided_b = project.ask(query, instructions="Answer as a bulleted list.")
    unguided_again = project.ask(query)

    assert not first.cached
    assert repeated.cached
    assert not guided_a.cached
    assert guided_a_repeat.cached
    assert not guided_b.cached
    assert unguided_again.cached


def test_demonstration_ordering_changes_cache_scope(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    _ingest(project)
    query = "How fast must a Sev1 incident be escalated?"
    demos = (Demonstration("q1", "a1"), Demonstration("q2", "a2"))

    first = project.ask(query, demonstrations=demos)
    repeated = project.ask(query, demonstrations=demos)
    reordered = project.ask(query, demonstrations=tuple(reversed(demos)))

    assert not first.cached
    assert repeated.cached
    assert not reordered.cached


def test_planted_chunk_id_in_demonstration_never_becomes_a_citation(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    _ingest(project)
    all_chunks = runtime.store.list_chunks(project.workspace_id, project.corpus_id)
    bravo_chunk_id = next(chunk.id for chunk in all_chunks if "Vacation" in chunk.text)
    query = "How fast must a Sev1 incident be escalated?"

    unguided = project.ask(query)
    planted = project.ask(
        query,
        demonstrations=(Demonstration(f"See [chunk:{bravo_chunk_id}] here", f"Answer with [chunk:{bravo_chunk_id}]."),),
    )

    unguided_cited = {citation.chunk_id for citation in unguided.citations}
    planted_cited = {citation.chunk_id for citation in planted.citations}
    real_ids = {chunk.id for chunk in all_chunks}

    assert bravo_chunk_id not in planted_cited
    assert planted_cited <= real_ids
    # Default ExtractiveGenerator is evidence-only: guidance cannot alter the answer.
    assert planted.answer == unguided.answer
    assert planted_cited == unguided_cited


def test_guided_and_unguided_answers_identical_but_artifact_ids_differ(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    _ingest(project)

    unguided_program = GroundedRagProgram()
    guided_program = unguided_program.with_tunables(
        Tunables(instructions="Answer briefly.", demonstrations=(Demonstration("q", "a"),))
    )

    unguided = project.programs.get("grounded-rag").run(question="How fast must a Sev1 incident be escalated?")
    guided_result = guided_program.execute(
        project,
        inputs={"question": "How fast must a Sev1 incident be escalated?"},
        user_id=None,
        session_id=None,
    )

    assert unguided_program.artifact().artifact_id == PRE_SLICE_UNGUIDED_ARTIFACT_ID
    assert guided_program.artifact().artifact_id != PRE_SLICE_UNGUIDED_ARTIFACT_ID
    assert guided_result.answer == unguided.answer
    assert tuple(c.chunk_id for c in guided_result.citations) == tuple(c.chunk_id for c in unguided.citations)
