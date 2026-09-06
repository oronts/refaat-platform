from __future__ import annotations

from collections.abc import Sequence

import pytest

from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.config import Settings
from refaat.domain.models import DocumentInput

pytestmark = pytest.mark.integration

_CONTENT = b"Rollback restores the prior artifact. Peer review is required before a production deployment."


class _KeywordScorer:
    def __init__(self, keyword: str) -> None:
        self.keyword = keyword

    @property
    def model_id(self) -> str:
        return "test-cross-encoder-v1"

    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        return [1.0 if self.keyword in document.lower() else 0.0 for document in documents]


def test_reranker_route_present_when_enabled(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings.model_copy(update={"reranker_provider": "lexical"})).build()
    try:
        project = runtime.project(workspace="acme", corpus="handbook")
        runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(source_id="policy.txt", title="Policy", media_type="text/plain", content=_CONTENT),
        )
        answer = project.ask("how does rollback restore the prior artifact")
        assert "rerank" in answer.routes
        assert answer.citations
    finally:
        runtime.close()


def test_cross_encoder_reranker_reorders_real_candidates(settings: Settings) -> None:
    runtime = (
        RuntimeBuilder(settings.model_copy(update={"reranker_provider": "cross_encoder"}))
        .with_cross_encoder_scorer(_KeywordScorer("rollback"))
        .build()
    )
    try:
        project = runtime.project(workspace="acme", corpus="handbook")
        runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id="rollback.txt",
                title="Rollback",
                media_type="text/plain",
                content=b"Rollback restores the prior artifact atomically.",
            ),
        )
        runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id="review.txt",
                title="Review",
                media_type="text/plain",
                content=b"Peer review is required before a production deployment.",
            ),
        )
        answer = project.ask("how does rollback restore the prior artifact")
        assert "rerank" in answer.routes
        assert answer.citations
        stored = {chunk.id: chunk for chunk in runtime.store.list_chunks(project.workspace_id, project.corpus_id)}
        for citation in answer.citations:
            assert citation.chunk_id in stored
        assert "rollback" in stored[answer.citations[0].chunk_id].text.lower()
    finally:
        runtime.close()


def test_reranker_route_absent_by_default(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    runtime.ingest(
        project.workspace_id,
        project.corpus_id,
        DocumentInput(source_id="policy.txt", title="Policy", media_type="text/plain", content=_CONTENT),
    )
    answer = project.ask("how does rollback restore the prior artifact")
    assert "rerank" not in answer.routes
