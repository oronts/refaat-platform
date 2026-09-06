from pathlib import Path

from refaat.application.runtime import RuntimeBuilder
from refaat.config import Settings
from refaat.domain.models import DocumentInput, QueryRequest


def test_state_survives_runtime_restart(tmp_path: Path) -> None:
    settings = Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path / 'durable.db'}",
        blob_root=tmp_path / "blobs",
        skill_root=tmp_path / "skills",
        api_key_pepper="durable-pepper",
        log_json=False,
    )
    first = RuntimeBuilder(settings).build()
    workspace = first.create_workspace("durable", "Durable")
    corpus = first.create_corpus(workspace, "knowledge")
    first.ingest(
        workspace,
        corpus,
        DocumentInput(
            source_id="durable.txt",
            title="Durable",
            media_type="text/plain",
            content=b"Durable sessions and documents survive process restarts.",
        ),
    )
    first.close()

    second = RuntimeBuilder(settings).build()
    try:
        response = second.query(
            QueryRequest(
                workspace_id=workspace,
                corpus_id=corpus,
                query="What survives process restarts?",
            )
        )
        assert response.citations
        assert "documents" in response.answer.casefold()
    finally:
        second.close()
