from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from refaat.application.runtime import Refaat, RefaatProject
from refaat.config import Settings
from refaat.domain.models import DocumentInput
from refaat.programs import SelfRagProgram

pytestmark = pytest.mark.integration

_GROUNDED_QUERY = "Rollback restores the previously approved harness snapshot before deployment"
_TRIVIAL_QUERY = "ok thanks"

_CORPUS = {
    "rollback.txt": (
        "Rollback restores the previously approved harness snapshot before deployment. "
        "The rollback procedure reverts the active program to the last known good artifact."
    ),
    "cache.txt": (
        "The answer cache key includes the corpus version and the retrieval policy. "
        "A replace or delete advances the corpus version and invalidates cached answers."
    ),
}


@pytest.fixture
def runtime(tmp_path: Path) -> Iterator[Refaat]:
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


def _populated(runtime: Refaat) -> RefaatProject:
    project = runtime.project(workspace="acme", corpus="selfrag")
    for source_id, text in _CORPUS.items():
        runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(source_id=source_id, title=source_id, media_type="text/plain", content=text.encode()),
        )
    return project


def test_grounded_query_is_supported_useful_and_every_citation_resolves(runtime: Refaat) -> None:
    project = _populated(runtime)
    program = SelfRagProgram()

    result = program.execute(project, inputs={"question": _GROUNDED_QUERY}, user_id=None, session_id=None)

    reflection = result.outputs["self_rag"]
    assert reflection["retrieve"] is True
    assert reflection["is_supported"] is True
    assert reflection["is_useful"] is True
    assert reflection["refine_attempts"] == 0
    assert result.grounded
    assert result.citations

    cited_ids = [citation.chunk_id for citation in result.citations]
    stored = runtime.store.get_chunks(project.workspace_id, project.corpus_id, cited_ids)
    resolved = {chunk.id for chunk in stored}
    assert set(cited_ids) <= resolved

    assert reflection["rationale"] not in cited_ids
    assert reflection["rationale"] not in {citation.quote for citation in result.citations}


def test_ungrounded_query_drives_the_refine_retry_and_stays_on_the_grounded_pipeline(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="empty")
    program = SelfRagProgram(refine_attempt_cap=3)

    result = program.execute(project, inputs={"question": _GROUNDED_QUERY}, user_id=None, session_id=None)

    reflection = result.outputs["self_rag"]
    assert reflection["is_supported"] is False
    assert reflection["is_useful"] is False
    assert reflection["refine_attempts"] == 3
    assert not result.grounded
    assert result.citations == ()
    assert result.program == "self-rag"


def test_no_retrieve_answer_still_flows_through_the_grounded_pipeline(runtime: Refaat) -> None:
    project = _populated(runtime)
    program = SelfRagProgram()

    result = program.execute(project, inputs={"question": _TRIVIAL_QUERY}, user_id=None, session_id=None)

    assert result.outputs["self_rag"]["retrieve"] is False
    assert result.program == "self-rag"
    assert "answer" in result.outputs


def test_active_program_and_corpus_version_are_unchanged_by_reflection(runtime: Refaat) -> None:
    project = _populated(runtime)
    before = project.retrieve(_GROUNDED_QUERY).routes
    program = SelfRagProgram()

    first = program.execute(project, inputs={"question": _GROUNDED_QUERY}, user_id=None, session_id=None)
    second = program.execute(project, inputs={"question": _GROUNDED_QUERY}, user_id=None, session_id=None)

    assert first.corpus_version == second.corpus_version
    assert project.retrieve(_GROUNDED_QUERY).routes == before
