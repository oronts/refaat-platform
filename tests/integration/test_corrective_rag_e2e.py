from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from refaat.application.runtime import Refaat, RefaatProject
from refaat.config import Settings
from refaat.domain.models import DocumentInput
from refaat.programs import CorrectiveRagProgram, DeterministicConfidenceEvaluator

pytestmark = pytest.mark.integration

_STRONG_QUERY = "Rollback restores the previously approved harness snapshot before deployment"
_WEAK_QUERY = "how does rollback work"

_CORPUS = {
    "rollback.txt": (
        "Rollback restores the previously approved harness snapshot before deployment. "
        "The rollback procedure reverts the active program to the last known good artifact."
    ),
    "cache.txt": (
        "The answer cache key includes the corpus version and the retrieval policy. "
        "A replace or delete advances the corpus version and invalidates cached answers."
    ),
    "graph.txt": (
        "The co-occurrence graph extractor links entities that appear together in a document. "
        "Graph facts boost documents whose entities match the query terms."
    ),
    "ingest.txt": (
        "Ingestion chunks a document, embeds each chunk, and writes the search index atomically. "
        "A first-write upsert avoids a duplicate document race."
    ),
}


@pytest.fixture
def stepback_runtime(tmp_path: Path) -> Iterator[Refaat]:
    settings = Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path / 'refaat.db'}",
        auto_create_schema=True,
        blob_root=tmp_path / "blobs",
        connector_file_root=tmp_path / "connectors",
        skill_root=tmp_path / "skills",
        api_key_pepper="test-pepper-with-enough-entropy",
        bootstrap_token="bootstrap-test-token",
        query_transform_provider="step_back",
        query_expansion=True,
        max_query_variants=4,
        default_chunk_tokens=64,
        chunk_overlap_tokens=12,
        query_token_budget=600,
        log_json=False,
    )
    value = Refaat(settings)
    try:
        yield value
    finally:
        value.close()


def _project(runtime: Refaat) -> RefaatProject:
    project = runtime.project(workspace="acme", corpus="crag")
    for source_id, text in _CORPUS.items():
        runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(source_id=source_id, title=source_id, media_type="text/plain", content=text.encode()),
        )
    return project


def _tuned_evaluator(project: RefaatProject) -> DeterministicConfidenceEvaluator:
    """Set the gate threshold from the real measured narrow-retrieval semantic scores.

    Deriving the threshold from the live stack proves the gate reads the genuine
    per-hit semantic signal: the weak query lands below and the strong query above
    the same threshold, so the discriminator is non-vacuous on real retrieval.
    """
    weak = project.retrieve(_WEAK_QUERY, query_expansion=False, max_query_variants=1)
    strong = project.retrieve(_STRONG_QUERY, query_expansion=False, max_query_variants=1)
    weak_top = max((hit.semantic_score for hit in weak.hits), default=0.0)
    strong_top = max((hit.semantic_score for hit in strong.hits), default=0.0)
    assert weak_top < strong_top
    return DeterministicConfidenceEvaluator(
        top_semantic_threshold=(weak_top + strong_top) / 2,
        semantic_floor=strong_top / 2,
        min_grounded_chunks=1,
    )


def test_low_confidence_query_triggers_corrective_broadening(stepback_runtime: Refaat) -> None:
    project = _project(stepback_runtime)
    program = CorrectiveRagProgram(evaluator=_tuned_evaluator(project), corrective_max_query_variants=4)

    result = program.execute(project, inputs={"question": _WEAK_QUERY}, user_id=None, session_id=None)

    gate = result.outputs["corrective_rag"]
    assert gate["label"] == "low"
    assert gate["corrective_applied"] is True
    assert "expansion" in result.routes
    assert "rrf" in result.routes
    assert result.grounded
    assert result.citations

    cited_ids = [citation.chunk_id for citation in result.citations]
    stored = stepback_runtime.store.get_chunks(project.workspace_id, project.corpus_id, cited_ids)
    resolved = {chunk.id for chunk in stored}
    assert set(cited_ids) <= resolved
    assert gate["rationale"] not in {citation.quote for citation in result.citations}
    assert gate["rationale"] not in cited_ids


def test_high_confidence_query_stays_narrow(stepback_runtime: Refaat) -> None:
    project = _project(stepback_runtime)
    program = CorrectiveRagProgram(evaluator=_tuned_evaluator(project), corrective_max_query_variants=4)

    result = program.execute(project, inputs={"question": _STRONG_QUERY}, user_id=None, session_id=None)

    gate = result.outputs["corrective_rag"]
    assert gate["label"] == "correct"
    assert gate["corrective_applied"] is False
    assert "expansion" not in result.routes
    assert result.grounded


def test_corrective_answer_is_cached_on_re_run(stepback_runtime: Refaat) -> None:
    project = _project(stepback_runtime)
    program = CorrectiveRagProgram(evaluator=_tuned_evaluator(project), corrective_max_query_variants=4)

    first = program.execute(project, inputs={"question": _WEAK_QUERY}, user_id=None, session_id=None)
    second = program.execute(project, inputs={"question": _WEAK_QUERY}, user_id=None, session_id=None)

    assert first.cached is False
    assert second.cached is True
    assert second.outputs["corrective_rag"]["corrective_applied"] is True
