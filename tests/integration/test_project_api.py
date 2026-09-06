from __future__ import annotations

from pathlib import Path

import pytest

from refaat import Refaat
from refaat.config import Settings
from refaat.errors import ValidationError

pytestmark = pytest.mark.integration


def test_high_level_project_ingests_directory_and_reuses_resources(tmp_path: Path) -> None:
    source = tmp_path / "knowledge"
    source.mkdir()
    (source / "release.md").write_text(
        "Production releases require peer review and a tested rollback plan.",
        encoding="utf-8",
    )
    (source / "ignored.bin").write_bytes(b"not supported")
    (source / ".env").write_text("SECRET=must-not-be-indexed", encoding="utf-8")
    nested = source / "operations"
    nested.mkdir()
    (nested / "incident.txt").write_text(
        "Every severe incident needs an incident commander.",
        encoding="utf-8",
    )
    settings = Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path / 'project.db'}",
        blob_root=tmp_path / "blobs",
        connector_file_root=tmp_path / "connectors",
        skill_root=tmp_path / "skills",
        api_key_pepper="project-test-pepper",
        log_json=False,
    )

    with Refaat(settings) as runtime:
        project = runtime.project(workspace="oronts", corpus="engineering")
        first = project.ingest(source)
        second = project.ingest(source)
        reused = runtime.project(workspace="oronts", corpus="engineering")
        answer = reused.ask("What is required for production releases?")
        cached = reused.ask("What is required for production releases?")

        assert len(first) == 2
        assert "must-not-be-indexed" not in answer.answer
        assert all(result.changed for result in first)
        assert not any(result.changed for result in second)
        assert reused.workspace_id == project.workspace_id
        assert reused.corpus_id == project.corpus_id
        assert answer.citations
        assert "rollback" in answer.answer.casefold()
        assert cached.cached


def test_project_rejects_missing_and_empty_sources(runtime, tmp_path: Path) -> None:
    project = runtime.project(workspace="project-errors", corpus="knowledge")
    with pytest.raises(ValidationError, match="does not exist"):
        project.ingest(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValidationError, match="no supported"):
        project.ingest(empty)


def test_project_rejects_session_from_another_corpus(runtime) -> None:
    workspace = "project-session-boundary"
    source = runtime.project(workspace=workspace, corpus="source")
    target = runtime.project(workspace=workspace, corpus="target")
    session_id = source.create_session(user_id="alice", goal="Review source knowledge")

    with pytest.raises(ValidationError, match="another corpus"):
        target.retrieve("What is in the target corpus?", session_id=session_id)

    with pytest.raises(ValidationError, match="another corpus"):
        target.ask("What is in the target corpus?", session_id=session_id)
