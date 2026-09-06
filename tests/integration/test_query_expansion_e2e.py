from __future__ import annotations

import pytest

from refaat.domain.models import DocumentInput

pytestmark = pytest.mark.integration

QUERY = "Which database engine powers the service and where are nightly backups stored?"
DOCUMENTS = {
    "engine.md": "The service database engine is PostgreSQL managed by the platform team.",
    "backups.md": "Nightly backups are stored in cold object storage for disaster recovery.",
    "billing.md": "The billing service and the reporting service both emit invoices monthly.",
    "onboarding.md": "New employees complete onboarding paperwork during their first week.",
    "network.md": "The internal network uses a mesh topology with redundant links.",
}
TARGETS = {"engine.md", "backups.md"}


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


def _covered_targets(hits) -> set[str]:
    return {hit.chunk.metadata.get("source_id") for hit in hits} & TARGETS


def test_expansion_improves_recall_on_a_compound_query(runtime) -> None:
    project = runtime.project(workspace="expand", corpus="handbook")
    _ingest(project)

    expanded = project.retrieve(QUERY, top_k=2)
    baseline = project.retrieve(QUERY, top_k=2, query_expansion=False)

    assert "expansion" in expanded.routes
    assert "rrf" in expanded.routes
    assert "expansion" not in baseline.routes

    assert _covered_targets(expanded.hits) == TARGETS
    assert len(_covered_targets(expanded.hits)) > len(_covered_targets(baseline.hits))


def test_every_citation_resolves_to_a_real_corpus_chunk(runtime) -> None:
    project = runtime.project(workspace="expand-cite", corpus="handbook")
    _ingest(project)

    response = project.ask(QUERY, top_k=4)
    real_ids = {hit.chunk.id for hit in project.retrieve(QUERY, top_k=100).hits}

    assert response.citations
    for citation in response.citations:
        assert citation.chunk_id in real_ids


def test_toggling_expansion_and_variants_changes_the_cache_scope(runtime) -> None:
    project = runtime.project(workspace="expand-cache", corpus="handbook")
    _ingest(project)

    first = project.ask(QUERY)
    repeated = project.ask(QUERY)
    disabled = project.ask(QUERY, query_expansion=False)
    fewer_variants = project.ask(QUERY, max_query_variants=2)

    assert not first.cached
    assert repeated.cached
    assert not disabled.cached
    assert not fewer_variants.cached


def test_intra_token_punctuation_query_stays_on_the_single_variant_path(runtime) -> None:
    project = runtime.project(workspace="expand-intra", corpus="handbook")
    _ingest(project)

    question = "Which database engine runs on Python 3.11 in the U.S. region?"
    expanded = project.retrieve(question, top_k=3)
    disabled = project.retrieve(question, top_k=3, query_expansion=False)

    assert "expansion" not in expanded.routes
    assert expanded.routes.count("rrf") == disabled.routes.count("rrf")
    assert expanded.routes == disabled.routes
    assert [hit.chunk.id for hit in expanded.hits] == [hit.chunk.id for hit in disabled.hits]
    assert [hit.score for hit in expanded.hits] == [hit.score for hit in disabled.hits]


def test_disabled_expansion_matches_the_single_variant_baseline(runtime) -> None:
    project = runtime.project(workspace="expand-baseline", corpus="handbook")
    _ingest(project)

    disabled = project.retrieve(QUERY, top_k=3, query_expansion=False)
    single_variant = project.retrieve(QUERY, top_k=3, max_query_variants=1)

    assert [hit.chunk.id for hit in disabled.hits] == [hit.chunk.id for hit in single_variant.hits]
    assert "expansion" not in single_variant.routes
