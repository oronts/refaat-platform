from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.application.services import ABSTAIN_MESSAGE
from refaat.config import Settings
from refaat.domain.models import DocumentInput

pytestmark = pytest.mark.integration

DOCUMENTS = {
    "engine.md": "The service database engine is PostgreSQL managed by the platform team.",
    "backups.md": "Nightly backups are stored in cold object storage for disaster recovery.",
    "network.md": "The internal network uses a mesh topology with redundant links.",
}
STRONG_QUERY = "The service database engine is PostgreSQL managed by the platform team."
OUT_OF_CORPUS = "What is the capital of Australia and how tall is its tallest building?"


def _settings(tmp_path: Path, floor: float, *, name: str = "refaat") -> Settings:
    return Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path / f'{name}.db'}",
        auto_create_schema=True,
        blob_root=tmp_path / f"{name}-blobs",
        connector_file_root=tmp_path / f"{name}-connectors",
        skill_root=tmp_path / f"{name}-skills",
        api_key_pepper="test-pepper-with-enough-entropy",
        bootstrap_token="bootstrap-test-token",
        default_chunk_tokens=64,
        chunk_overlap_tokens=12,
        query_token_budget=600,
        abstain_below_score=floor,
        log_json=False,
    )


def _build(settings: Settings) -> Iterator[Refaat]:
    runtime = RuntimeBuilder(settings).build()
    try:
        yield runtime
    finally:
        runtime.close()


def _ingest(project) -> None:
    for source_id, content in DOCUMENTS.items():
        project.runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id=source_id,
                title=source_id.removesuffix(".md"),
                media_type="text/markdown",
                content=content.encode(),
            ),
        )


def test_abstains_when_best_evidence_is_below_the_floor(tmp_path: Path) -> None:
    for runtime in _build(_settings(tmp_path, 1.0)):
        project = runtime.project(workspace="abstain", corpus="handbook")
        _ingest(project)

        response = project.ask(OUT_OF_CORPUS)

        assert response.abstained is True
        assert response.citations == ()
        assert "abstain" in response.routes
        assert response.answer == ABSTAIN_MESSAGE
        # The extractive fallback is never reached, so no chunk text is stitched in.
        for content in DOCUMENTS.values():
            assert content not in response.answer


def test_abstention_is_cached_and_repeats_from_the_cache(tmp_path: Path) -> None:
    for runtime in _build(_settings(tmp_path, 1.0)):
        project = runtime.project(workspace="abstain-cache", corpus="handbook")
        _ingest(project)

        first = project.ask(OUT_OF_CORPUS)
        second = project.ask(OUT_OF_CORPUS)

        assert first.cached is False
        assert second.cached is True
        assert second.abstained is True
        assert second.citations == ()


def test_disabled_floor_takes_the_normal_grounded_flow(tmp_path: Path) -> None:
    for runtime in _build(_settings(tmp_path, 0.0)):
        project = runtime.project(workspace="abstain-off", corpus="handbook")
        _ingest(project)

        response = project.ask(OUT_OF_CORPUS)

        assert response.abstained is False
        assert "abstain" not in response.routes
        assert response.answer != ABSTAIN_MESSAGE


def test_strong_evidence_does_not_abstain_with_a_moderate_floor(tmp_path: Path) -> None:
    for runtime in _build(_settings(tmp_path, 0.02)):
        project = runtime.project(workspace="abstain-strong", corpus="handbook")
        _ingest(project)

        response = project.ask(STRONG_QUERY)

        assert response.abstained is False
        assert "abstain" not in response.routes
        assert response.citations


def test_abstain_decision_does_not_leak_across_a_floor_change(tmp_path: Path) -> None:
    """A cached abstain from an enabled floor never leaks into the disabled path.

    The floor is folded into the answer cache identity, so a decision made at one
    floor cannot be served at another. Concretely, an abstain cached at floor 1.0 is
    not returned when the floor is later set back to 0.0: the disabled run takes the
    normal grounded flow on a fresh cache scope, so it stays byte-identical to a run
    where the feature was never enabled."""
    enabled = _settings(tmp_path, 1.0, name="shared")
    for runtime in _build(enabled):
        project = runtime.project(workspace="abstain-scope", corpus="handbook")
        _ingest(project)
        abstained = project.ask(OUT_OF_CORPUS)
        assert abstained.abstained is True

    disabled = _settings(tmp_path, 0.0, name="shared")
    for runtime in _build(disabled):
        project = runtime.project(workspace="abstain-scope", corpus="handbook")
        grounded = project.ask(OUT_OF_CORPUS)

    assert grounded.cached is False
    assert grounded.abstained is False
    assert "abstain" not in grounded.routes
    assert grounded.answer != ABSTAIN_MESSAGE
