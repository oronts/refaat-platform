from __future__ import annotations

import pytest

from refaat.application.runtime import Refaat
from refaat.domain.models import DocumentInput

pytestmark = pytest.mark.integration

# The vector path is language-neutral with a multilingual embedding model. Offline the
# deterministic hash embedder is language-agnostic too, so a metadata language tag plus the
# shipped metadata hard filter is the deterministic way to scope retrieval by language.
# Each document carries a shared distinctive token ("checkout") so lexical retrieval fires
# regardless of language on the deterministic offline stack; the language metadata tag is
# what scopes retrieval to one language through the shipped metadata hard filter.
DOCUMENTS = {
    "policy.en.md": ("en", "The checkout service production release requires an approved rollback plan."),
    "policy.de.md": ("de", "Die checkout Produktivfreigabe erfordert einen genehmigten Rollback-Plan."),
    "policy.fr.md": ("fr", "La mise en production du service checkout exige un plan de restauration approuve."),
}


def _ingest(project) -> None:
    for source_id, (language, content) in DOCUMENTS.items():
        project.runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id=source_id,
                title=source_id,
                media_type="text/markdown",
                content=content.encode(),
                metadata={"language": language},
            ),
        )


def _activate_language_filter(project, language: str) -> None:
    project.create_harness_snapshot(
        supplemental={"retrieval": {"filters": {"language": language}}},
        evidence={"reason": f"language filter {language}"},
        activate=True,
    )


def test_language_filter_scopes_retrieval_to_one_language(runtime: Refaat) -> None:
    project = runtime.project(workspace="multilang", corpus="policies")
    _ingest(project)
    _activate_language_filter(project, "de")
    query = "checkout release rollback plan"

    context = project.retrieve(query)
    assert context.hits
    assert "filter" in context.routes
    assert all(hit.chunk.metadata["language"] == "de" for hit in context.hits)

    response = project.ask(query)
    assert response.citations
    cited = {citation.chunk_id for citation in response.citations}
    hit_ids = {hit.chunk.id for hit in context.hits}
    assert cited <= hit_ids


def test_distinct_language_filters_return_distinct_evidence(runtime: Refaat) -> None:
    project = runtime.project(workspace="multilang-distinct", corpus="policies")
    _ingest(project)
    query = "checkout release rollback plan"

    _activate_language_filter(project, "en")
    english = project.retrieve(query)
    _activate_language_filter(project, "fr")
    french = project.retrieve(query)

    english_docs = {hit.chunk.metadata["language"] for hit in english.hits}
    french_docs = {hit.chunk.metadata["language"] for hit in french.hits}
    assert english_docs == {"en"}
    assert french_docs == {"fr"}
    assert {hit.chunk.id for hit in english.hits}.isdisjoint({hit.chunk.id for hit in french.hits})
