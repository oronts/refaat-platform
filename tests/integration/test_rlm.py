from __future__ import annotations

import pytest

from refaat.application.worker import DurableWorker
from refaat.domain.models import DocumentInput, RlmLimits
from refaat.errors import ValidationError

pytestmark = pytest.mark.integration


def _ingest_recursive_fixture(runtime, workspace_id: str, corpus_id: str) -> None:
    topics = (
        ("release", "Production releases require peer review and a tested rollback plan."),
        ("incident", "Severe incidents require an incident commander and a written timeline."),
        ("security", "Security exceptions expire after thirty days and require owner approval."),
        ("data", "Customer records remain in the European region and are encrypted at rest."),
    )
    for name, fact in topics:
        filler = " ".join(f"{name} evidence detail {index}." for index in range(80))
        runtime.ingest(
            workspace_id,
            corpus_id,
            DocumentInput(
                source_id=f"{name}.md",
                title=f"{name.title()} policy",
                media_type="text/markdown",
                content=f"{fact}\n\n{filler}".encode(),
            ),
        )


def test_rlm_recurses_with_bounded_calls_and_canonical_citations(
    runtime,
    workspace_id: str,
    corpus_id: str,
) -> None:
    _ingest_recursive_fixture(runtime, workspace_id, corpus_id)
    session_id = runtime.store.create_session(
        workspace_id,
        corpus_id,
        user_id="alice",
        goal="Review operational controls",
    )
    limits = RlmLimits(max_depth=3, max_nodes=9, branch_factor=3, leaf_token_budget=64)

    result = runtime.run_rlm(
        workspace_id,
        corpus_id,
        "What controls govern production releases and severe incidents?",
        user_id="alice",
        session_id=session_id,
        limits=limits,
    )

    chunks = {chunk.id: chunk for chunk in runtime.store.list_chunks(workspace_id, corpus_id)}
    assert result.answer
    assert result.citations
    assert 1 < len(result.nodes) <= limits.max_nodes
    assert result.usage["model_calls"] == len(result.nodes)
    assert result.usage["model_calls"] <= limits.max_nodes
    assert result.usage["max_depth_reached"] <= limits.max_depth
    assert result.nodes[-1].parent_id is None
    assert result.nodes[-1].route == "synthesis"
    assert any(node.parent_id for node in result.nodes)
    assert all(citation.chunk_id in chunks for citation in result.citations)
    assert all(citation.quote == chunks[citation.chunk_id].text[:500].strip() for citation in result.citations)
    assert len(runtime.store.list_session_turns(workspace_id, session_id)) == 2


def test_project_rlm_and_durable_job(runtime, workspace_id: str, corpus_id: str) -> None:
    _ingest_recursive_fixture(runtime, workspace_id, corpus_id)
    project = runtime.project(workspace="acme", corpus="handbook")
    direct = project.rlm(
        "Which production control is required?",
        limits=RlmLimits(max_depth=1, max_nodes=4, branch_factor=2, leaf_token_budget=64),
    )
    assert direct.citations

    job_id = project.enqueue_rlm(
        "Who leads a severe incident?",
        limits=RlmLimits(max_depth=1, max_nodes=4, branch_factor=2, leaf_token_budget=64),
    )
    assert DurableWorker(runtime).run_once()
    job = runtime.store.get_job(workspace_id, job_id)
    assert str(job["status"]) == "succeeded"
    assert job["result"]["citations"]
    assert job["result"]["usage"]["model_calls"] <= 4


def test_rlm_validates_query_session_and_limits(runtime, workspace_id: str, corpus_id: str) -> None:
    with pytest.raises(ValidationError, match="query is empty"):
        runtime.run_rlm(workspace_id, corpus_id, "   ")
    with pytest.raises(ValueError, match="max_nodes"):
        RlmLimits(max_nodes=0)

    other_corpus = runtime.create_corpus(workspace_id, "other")
    session_id = runtime.store.create_session(workspace_id, other_corpus, user_id=None, goal=None)
    with pytest.raises(ValidationError, match="another corpus"):
        runtime.run_rlm(workspace_id, corpus_id, "question", session_id=session_id)


def test_rlm_empty_corpus_returns_explicit_grounding_message(
    runtime,
    workspace_id: str,
    corpus_id: str,
) -> None:
    result = runtime.run_rlm(workspace_id, corpus_id, "What is documented?")
    assert not result.citations
    assert not result.nodes
    assert "corpus is empty" in result.answer
    assert result.usage["model_calls"] == 0


# An opt-in ceiling fails the RLM closed before it materializes an over-large corpus.
def test_rlm_input_ceiling_fails_closed_before_loading_the_corpus(settings) -> None:  # type: ignore[no-untyped-def]
    from refaat.application.runtime import RuntimeBuilder

    runtime = RuntimeBuilder(settings.model_copy(update={"rlm_max_input_chunks": 1})).build()
    try:
        project = runtime.project(workspace="acme", corpus="rlm-ceiling")
        _ingest_recursive_fixture(runtime, project.workspace_id, project.corpus_id)
        assert len(runtime.store.list_chunks(project.workspace_id, project.corpus_id)) > 1
        with pytest.raises(ValidationError, match="exceeds the configured input-chunk limit"):
            project.rlm(
                "what controls govern releases",
                limits=RlmLimits(max_depth=1, max_nodes=4, branch_factor=2, leaf_token_budget=64),
            )
    finally:
        runtime.close()


def test_rlm_input_ceiling_within_bound_admits_the_run(settings) -> None:  # type: ignore[no-untyped-def]
    from refaat.application.runtime import RuntimeBuilder

    runtime = RuntimeBuilder(settings.model_copy(update={"rlm_max_input_chunks": 100000})).build()
    try:
        project = runtime.project(workspace="acme", corpus="rlm-within")
        _ingest_recursive_fixture(runtime, project.workspace_id, project.corpus_id)
        result = project.rlm(
            "what controls govern releases",
            limits=RlmLimits(max_depth=1, max_nodes=4, branch_factor=2, leaf_token_budget=64),
        )
        assert result.answer
    finally:
        runtime.close()


def test_rlm_ceiling_message_does_not_leak_the_corpus_size(settings) -> None:  # type: ignore[no-untyped-def]
    from refaat.application.runtime import RuntimeBuilder

    runtime = RuntimeBuilder(settings.model_copy(update={"rlm_max_input_chunks": 1})).build()
    try:
        project = runtime.project(workspace="acme", corpus="rlm-leak")
        _ingest_recursive_fixture(runtime, project.workspace_id, project.corpus_id)
        total = len(runtime.store.list_chunks(project.workspace_id, project.corpus_id))
        assert total > 1
        with pytest.raises(ValidationError) as exc:
            project.rlm(
                "what controls govern releases",
                limits=RlmLimits(max_depth=1, max_nodes=4, branch_factor=2, leaf_token_budget=64),
            )
        assert str(total) not in str(exc.value)
    finally:
        runtime.close()
