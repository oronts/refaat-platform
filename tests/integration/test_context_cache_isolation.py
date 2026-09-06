from __future__ import annotations

import pytest
from sqlalchemy import delete

from refaat.domain.models import DocumentInput
from refaat.infrastructure.database.schema import chunks

pytestmark = pytest.mark.integration


def ingest_documents(project) -> None:
    documents = {
        "release.md": "Production releases require peer review, automated tests, and a rollback plan.",
        "security.md": "High-risk production releases require Security Team approval and a threat review.",
        "operations.md": "The Release Manager verifies evidence and the Platform Team watches the canary.",
    }
    for source_id, content in documents.items():
        project.runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id=source_id,
                title=source_id.removesuffix(".md").title(),
                media_type="text/markdown",
                content=content.encode(),
            ),
        )


def test_cache_isolated_by_effective_retrieval_policy(runtime) -> None:
    project = runtime.project(workspace="cache-policy", corpus="handbook")
    ingest_documents(project)
    query = "What controls apply to a production release?"

    first = project.ask(query, top_k=1, token_budget=180)
    repeated = project.ask(query, top_k=1, token_budget=180)
    expanded = project.ask(query, top_k=3, token_budget=600)

    assert not first.cached
    assert repeated.cached
    assert repeated.routes[0] == "cag:exact"
    assert not expanded.cached
    assert int(first.usage["retrieved_chunks"]) == 1
    assert int(expanded.usage["retrieved_chunks"]) >= 2


def test_cache_isolated_by_fusion_strategy(runtime) -> None:
    project = runtime.project(workspace="cache-fusion", corpus="handbook")
    ingest_documents(project)
    query = "What controls apply to a production release?"

    default_rrf = project.ask(query)
    repeated = project.ask(query)
    linear = project.ask(query, fusion="linear")

    assert not default_rrf.cached
    assert repeated.cached
    assert not linear.cached


def test_harness_activation_changes_policy_and_invalidates_cache(runtime) -> None:
    project = runtime.project(workspace="cache-harness", corpus="handbook")
    ingest_documents(project)
    query = "What evidence and approval are needed for production?"

    initial = project.ask(query)
    cached = project.ask(query)
    snapshot = project.create_harness_snapshot(
        supplemental={
            "retrieval": {
                "top_k": 1,
                "semantic_weight": 0.6,
                "lexical_weight": 0.25,
                "graph_weight": 0.15,
            }
        },
        evidence={"reason": "regression test"},
        activate=True,
    )
    changed = project.ask(query)

    assert cached.cached
    assert snapshot.active
    assert not changed.cached
    assert "harness" in changed.routes
    assert int(changed.usage["retrieved_chunks"]) == 1
    hits = project.retrieve(query).hits
    assert hits
    assert len(hits) == 1
    assert initial.answer


def test_feedback_profile_change_invalidates_user_cache(runtime) -> None:
    project = runtime.project(workspace="cache-profile", corpus="handbook")
    ingest_documents(project)
    query = "Who verifies release evidence?"

    first = project.ask(query, user_id="alice")
    cached = project.ask(query, user_id="alice")
    selected = (first.citations[0].chunk_id,)
    project.submit_feedback(rating=1, user_id="alice", selected_chunk_ids=selected)
    personalized = project.ask(query, user_id="alice")
    anonymous = project.ask(query)

    assert cached.cached
    assert not personalized.cached
    assert "profile" in personalized.routes
    assert not anonymous.cached


def test_valid_cache_hit_preserves_answer_and_citations(runtime) -> None:
    project = runtime.project(workspace="cache-valid", corpus="handbook")
    ingest_documents(project)
    query = "What controls apply to a production release?"

    first = project.ask(query, top_k=1, token_budget=180)
    repeated = project.ask(query, top_k=1, token_budget=180)

    assert repeated.cached
    assert repeated.answer == first.answer
    assert [c.chunk_id for c in repeated.citations] == [c.chunk_id for c in first.citations]
    assert [c.quote for c in repeated.citations] == [c.quote for c in first.citations]


def test_cache_hit_drops_a_citation_whose_chunk_was_deleted(runtime) -> None:
    project = runtime.project(workspace="cache-dangling", corpus="handbook")
    ingest_documents(project)
    query = "What controls apply to a production release?"

    first = project.ask(query, top_k=1, token_budget=180)
    assert first.citations
    dangling = first.citations[0].chunk_id
    assert f"[chunk:{dangling}]" in first.answer

    # Delete the cited chunk row out of band WITHOUT advancing the corpus version, so the
    # version-scoped cache row is still an exact hit but its stored citation now dangles.
    with runtime.store.engine.begin() as conn:
        conn.execute(delete(chunks).where(chunks.c.id == dangling))

    repeated = project.ask(query, top_k=1, token_budget=180)

    # The dangling citation is dropped AND its now-invalid marker is stripped from the answer body,
    # so a served cached answer never cites or references evidence that no longer exists.
    assert repeated.cached
    assert all(citation.chunk_id != dangling for citation in repeated.citations)
    assert f"[chunk:{dangling}]" not in repeated.answer


def test_session_summary_change_invalidates_session_cache(runtime) -> None:
    project = runtime.project(workspace="cache-session", corpus="handbook")
    ingest_documents(project)
    session_id = project.create_session(user_id="alice", goal="Review production controls")
    query = "What controls apply to a production release?"

    first = project.ask(query, user_id="alice", session_id=session_id)
    cached = project.ask(query, user_id="alice", session_id=session_id)
    project.ask(
        "Who verifies the release evidence?",
        user_id="alice",
        session_id=session_id,
    )
    session = runtime.store.get_session(project.workspace_id, session_id)
    after_compaction = project.ask(query, user_id="alice", session_id=session_id)

    assert not first.cached
    assert cached.cached
    assert session["summary"]
    assert not after_compaction.cached
    assert "session" in after_compaction.routes
