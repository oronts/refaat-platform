from __future__ import annotations

import pytest

from refaat.domain.models import DocumentInput

pytestmark = pytest.mark.integration


def _ingest(project) -> None:
    documents = {
        "security.md": (
            "security",
            "High-risk production releases require Security Team approval and a threat review.",
        ),
        "operations.md": (
            "ops",
            "The Release Manager verifies release evidence and the Platform Team watches the canary.",
        ),
        "handbook.md": (
            "ops",
            "Production releases require peer review, automated tests, and a rollback plan.",
        ),
    }
    for source_id, (category, content) in documents.items():
        project.runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id=source_id,
                title=source_id.removesuffix(".md").title(),
                media_type="text/markdown",
                content=content.encode(),
                metadata={"category": category},
            ),
        )


def _activate_filter(project, spec: dict[str, object]) -> None:
    project.create_harness_snapshot(
        supplemental={"retrieval": {"filters": spec}},
        evidence={"reason": "metadata filter test"},
        activate=True,
    )


def test_filter_scopes_retrieval_before_scoring(runtime) -> None:
    project = runtime.project(workspace="filter-scope", corpus="handbook")
    _ingest(project)
    _activate_filter(project, {"category": "security"})
    query = "What controls apply to a production release?"

    context = project.retrieve(query)
    assert context.hits
    assert "filter" in context.routes
    assert all(hit.chunk.metadata["category"] == "security" for hit in context.hits)

    response = project.ask(query)
    assert response.citations
    cited = {citation.chunk_id for citation in response.citations}
    hit_ids = {hit.chunk.id for hit in context.hits}
    assert cited <= hit_ids


def test_distinct_filters_yield_distinct_answers_and_cache_scopes(runtime) -> None:
    project = runtime.project(workspace="filter-distinct", corpus="handbook")
    _ingest(project)
    query = "What controls apply to a production release?"

    _activate_filter(project, {"category": "security"})
    security = project.ask(query)
    security_repeat = project.ask(query)

    _activate_filter(project, {"category": "ops"})
    ops = project.ask(query)

    assert not security.cached
    assert security_repeat.cached
    assert not ops.cached
    assert security.answer != ops.answer


def test_flipping_bm25_b_recomputes_the_answer(runtime) -> None:
    project = runtime.project(workspace="filter-bm25", corpus="handbook")
    _ingest(project)
    query = "What controls apply to a production release?"

    first = project.ask(query)
    cached = project.ask(query)
    project.create_harness_snapshot(
        supplemental={"retrieval": {"bm25_b": 0.1}},
        evidence={"reason": "bm25 tuning"},
        activate=True,
    )
    retuned = project.ask(query)

    assert not first.cached
    assert cached.cached
    assert not retuned.cached


def test_over_filter_degrades_to_insufficient_evidence(runtime) -> None:
    project = runtime.project(workspace="filter-empty", corpus="handbook")
    _ingest(project)
    _activate_filter(project, {"category": "nonexistent"})
    query = "What controls apply to a production release?"

    context = project.retrieve(query)
    assert context.hits == ()

    response = project.ask(query)
    assert response.citations == ()
    assert response.answer
