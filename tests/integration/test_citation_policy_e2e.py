from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.application.services import CITATION_REQUIRED_MESSAGE, CITATION_ROUTE
from refaat.config import Settings
from refaat.domain.models import DocumentInput, QueryRequest

pytestmark = pytest.mark.integration

GROUNDED_DOC = "The service database engine is PostgreSQL managed by the platform team."
GROUNDED_QUERY = GROUNDED_DOC
NO_EVIDENCE_QUERY = "What is the capital of Australia and how tall is its tallest building?"


def _settings(tmp_path: Path, policy: str, *, name: str = "refaat") -> Settings:
    return Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path / f'{name}.db'}",
        auto_create_schema=True,
        blob_root=tmp_path / f"{name}-blobs",
        connector_file_root=tmp_path / f"{name}-connectors",
        skill_root=tmp_path / f"{name}-skills",
        api_key_pepper="test-pepper-with-enough-entropy",
        bootstrap_token="bootstrap-test-token",
        citation_policy=policy,
        log_json=False,
    )


def _build(settings: Settings) -> Iterator[Refaat]:
    runtime = RuntimeBuilder(settings).build()
    try:
        yield runtime
    finally:
        runtime.close()


def _ingest(project) -> None:
    project.runtime.ingest(
        project.workspace_id,
        project.corpus_id,
        DocumentInput(source_id="engine.md", title="engine", media_type="text/markdown", content=GROUNDED_DOC.encode()),
    )


def test_optional_serves_a_zero_citation_answer(tmp_path: Path) -> None:
    for runtime in _build(_settings(tmp_path, "optional")):
        response = runtime.project(workspace="w", corpus="c").ask(NO_EVIDENCE_QUERY)
        assert CITATION_ROUTE not in response.routes
        assert response.answer != CITATION_REQUIRED_MESSAGE


def test_required_refuses_a_zero_citation_answer(tmp_path: Path) -> None:
    for runtime in _build(_settings(tmp_path, "required")):
        response = runtime.project(workspace="w", corpus="c").ask(NO_EVIDENCE_QUERY)
        assert CITATION_ROUTE in response.routes
        assert response.answer == CITATION_REQUIRED_MESSAGE
        assert response.citations == ()


def test_required_serves_a_grounded_answer(tmp_path: Path) -> None:
    for runtime in _build(_settings(tmp_path, "required")):
        project = runtime.project(workspace="w", corpus="c")
        _ingest(project)
        response = project.ask(GROUNDED_QUERY)
        assert CITATION_ROUTE not in response.routes
        assert response.citations


def test_required_when_evidence_serves_when_no_evidence_was_retrieved(tmp_path: Path) -> None:
    for runtime in _build(_settings(tmp_path, "required_when_evidence")):
        response = runtime.project(workspace="w", corpus="c").ask(NO_EVIDENCE_QUERY)
        # No evidence means nothing could be cited, so the answer is served.
        assert CITATION_ROUTE not in response.routes


def test_required_streaming_buffers_and_never_leaks_an_uncited_token(tmp_path: Path) -> None:
    for runtime in _build(_settings(tmp_path, "required")):
        project = runtime.project(workspace="w", corpus="c")
        request = QueryRequest(workspace_id=project.workspace_id, corpus_id=project.corpus_id, query=NO_EVIDENCE_QUERY)
        events = list(runtime.stream_query(request))

        finals = [event for event in events if event["type"] == "final"]
        assert len(finals) == 1
        assert CITATION_ROUTE in finals[0]["routes"]
        assert finals[0]["citations"] == []
        # The stream is buffered: no delta leaks the uncited answer, only the fixed refusal message is emitted.
        deltas = [event for event in events if event["type"] == "delta"]
        assert all(delta["text"] == CITATION_REQUIRED_MESSAGE for delta in deltas)


def test_policy_scopes_the_answer_cache(tmp_path: Path) -> None:
    """A zero-citation answer cached under OPTIONAL is never served under REQUIRED."""
    permissive = _settings(tmp_path, "optional", name="shared")
    for runtime in _build(permissive):
        served = runtime.project(workspace="w", corpus="c").ask(NO_EVIDENCE_QUERY)
        assert CITATION_ROUTE not in served.routes

    strict = _settings(tmp_path, "required", name="shared")
    for runtime in _build(strict):
        refused = runtime.project(workspace="w", corpus="c").ask(NO_EVIDENCE_QUERY)

    assert refused.cached is False
    assert CITATION_ROUTE in refused.routes
    assert refused.answer == CITATION_REQUIRED_MESSAGE
