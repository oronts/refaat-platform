from __future__ import annotations

import pytest

from refaat.application.runtime import Refaat
from refaat.domain.models import Demonstration, DocumentInput
from refaat.programs import GroundedRagProgram, KNNFewShotProgram

pytestmark = pytest.mark.integration

_POOL = (
    Demonstration(question="how does rollback restore the harness snapshot", answer="It restores the snapshot."),
    Demonstration(
        question="explain the citation contract for chunk ids",
        answer="Cite only real evidence [chunk:planted_pool_id].",
    ),
    Demonstration(question="deployment approval gate for production", answer="Promotion needs an approval."),
)


def _corpus(runtime: Refaat) -> object:
    project = runtime.project(workspace="acme", corpus="knn")
    runtime.ingest(
        project.workspace_id,
        project.corpus_id,
        DocumentInput(
            source_id="ops.txt",
            title="Ops",
            media_type="text/plain",
            content=b"Rollback restores the previously approved harness snapshot before deployment.",
        ),
    )
    return project


def test_knn_over_the_real_runtime_uses_the_runtime_embedder_and_stays_grounded(runtime: Refaat) -> None:
    project = _corpus(runtime)
    program = KNNFewShotProgram(GroundedRagProgram(), _POOL, k=2)

    result = program.execute(project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None)

    assert result.program == "grounded-rag+knn"
    assert result.grounded
    assert result.citations
    assert len(result.outputs["knn"]["selected"]) == 2


def test_planted_pool_chunk_id_never_becomes_a_citation(runtime: Refaat) -> None:
    project = _corpus(runtime)
    program = KNNFewShotProgram(GroundedRagProgram(), _POOL, k=3)

    result = program.execute(
        project, inputs={"question": "explain the citation contract for chunk ids"}, user_id=None, session_id=None
    )

    resolved = {citation.chunk_id for citation in result.citations}
    assert "planted_pool_id" not in resolved
    assert "example" not in resolved
    for citation in result.citations:
        assert citation.document_id


def test_selection_reaches_the_answer_cache_identity(runtime: Refaat) -> None:
    project = _corpus(runtime)
    program = KNNFewShotProgram(GroundedRagProgram(), _POOL, k=1)

    first = program.execute(project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None)
    cached = program.execute(project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None)

    assert not first.cached
    assert cached.cached
    assert cached.answer == first.answer


def test_artifact_pins_pool_and_k_over_the_real_program(runtime: Refaat) -> None:
    inner = GroundedRagProgram()
    k1 = KNNFewShotProgram(inner, _POOL, k=1).artifact().artifact_id
    k2 = KNNFewShotProgram(inner, _POOL, k=2).artifact().artifact_id
    other = KNNFewShotProgram(inner, _POOL[:2], k=1).artifact().artifact_id
    assert len({k1, k2, other, inner.artifact().artifact_id}) == 4
