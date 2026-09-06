from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.config import Settings
from refaat.core.citations import validate_cited_chunks
from refaat.domain.models import DocumentInput, QueryRequest
from refaat.evaluation import Dataset
from refaat.optimization import ChunkingOptimizer

pytestmark = pytest.mark.integration

_SECTIONS = [
    "The alpha service caches grounded answers reliably. It reuses cached answers across repeated retrieval requests. "
    "Each cached answer stays stable for a long window.",
    "The beta service fuses semantic and lexical candidates. It ranks the fused candidates by relevance. "
    "The budgeter finally packs the answer context tightly.",
    "The gamma service escalates a severe incident to on-call quickly. It pages the responder within fifteen minutes. "
    "The responder acknowledges the page immediately.",
    "The delta service rotates encryption keys every quarter. It archives the retired keys securely. "
    "An auditor reviews the rotation log afterwards.",
    "The epsilon service replicates the primary database continuously. It fails over to a standby on error. "
    "The standby promotes itself within seconds.",
    "The zeta service throttles abusive clients at the edge. It counts requests inside a sliding window. "
    "A blocked client receives a clear retry hint.",
]


@contextmanager
def _hierarchical_runtime(settings: Settings, strategy: str, **extra: object) -> Iterator[Refaat]:
    with TemporaryDirectory(prefix="refaat-hier-") as directory:
        root = Path(directory)
        updated = settings.model_copy(
            update={
                "chunking_strategy": strategy,
                "database_url": f"sqlite:///{root / 'refaat.db'}",
                "blob_root": root / "blobs",
                "connector_file_root": root / "connectors",
                "skill_root": root / "skills",
                **extra,
            }
        )
        value = RuntimeBuilder(updated).build()
        try:
            yield value
        finally:
            value.close()


def _ingest_sections(runtime: Refaat) -> tuple[str, str]:
    workspace_id = runtime.create_workspace("acme", "Acme")
    corpus_id = runtime.create_corpus(workspace_id, "handbook")
    for index, body in enumerate(_SECTIONS):
        runtime.ingest(
            workspace_id,
            corpus_id,
            DocumentInput(
                source_id=f"section{index}.txt",
                title=f"Section {index}",
                media_type="text/plain",
                content=body.encode(),
            ),
        )
    return workspace_id, corpus_id


def test_chunking_optimizer_uses_shadow_and_leaves_active_corpus_untouched(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    for index in range(3):
        runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id=f"runbook{index}.txt",
                title=f"Runbook {index}",
                media_type="text/plain",
                content=(
                    f"Runbook {index}: escalate a Sev1 incident to on-call within fifteen minutes of detection."
                ).encode(),
            ),
        )
    before = len(project.list_documents())
    dataset = Dataset.from_records("chunking", [{"query": "incident escalation", "expected_terms": ["incident"]}])

    result = ChunkingOptimizer(project).optimize(dataset, chunk_sizes=(128, 512), overlaps=(32,))

    assert result.candidate_count == 2
    assert {candidate.chunk_size for candidate in result.candidates} == {128, 512}
    assert result.best.hard_constraints_passed is True
    assert all(candidate.mean_tokens >= 0.0 for candidate in result.candidates)
    assert result.to_json()["candidate_count"] == 2
    assert len(project.list_documents()) == before


def test_chunking_optimizer_sweeps_strategy_axis_on_shadow(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    for index in range(2):
        runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id=f"doc{index}.txt",
                title=f"Doc {index}",
                media_type="text/plain",
                content=(
                    b"The pipeline escalates a severe incident to on-call. "
                    b"It pages the responder within fifteen minutes."
                ),
            ),
        )
    before = len(project.list_documents())
    dataset = Dataset.from_records("strategy", [{"query": "incident escalation", "expected_terms": ["incident"]}])

    result = ChunkingOptimizer(project).optimize(
        dataset,
        chunk_sizes=(256,),
        overlaps=(32,),
        strategies=("flat", "parent_child"),
    )

    assert result.candidate_count == 2
    assert {candidate.strategy for candidate in result.candidates} == {"flat", "parent_child"}
    assert len(project.list_documents()) == before


def test_flat_ingest_stored_rows_omit_granularity(runtime: Refaat, workspace_id: str, corpus_id: str) -> None:
    runtime.ingest(
        workspace_id,
        corpus_id,
        DocumentInput(
            source_id="flat.txt",
            title="Flat",
            media_type="text/plain",
            content=_SECTIONS[0].encode(),
        ),
    )
    documents = runtime.store.list_documents(workspace_id, corpus_id)
    chunks = runtime.store.list_document_chunks(workspace_id, corpus_id, str(documents[0]["id"]))
    assert chunks
    for chunk in chunks:
        assert "granularity" not in chunk.metadata
        assert "parent_id" not in chunk.metadata


def test_parent_child_citations_resolve_to_real_parent_rows(settings: Settings) -> None:
    with _hierarchical_runtime(settings, "parent_child") as hierarchical:
        workspace_id, corpus_id = _ingest_sections(hierarchical)
        response = hierarchical.query(
            QueryRequest(workspace_id=workspace_id, corpus_id=corpus_id, query="budgeter packs the answer context")
        )
        assert "expand" in response.routes
        assert response.citations
        cited_ids = tuple(citation.chunk_id for citation in response.citations)
        stored = hierarchical.store.get_chunks(workspace_id, corpus_id, cited_ids)
        assert validate_cited_chunks(cited_ids, stored) == cited_ids
        stored_by_id = {chunk.id: chunk for chunk in stored}
        for citation in response.citations:
            chunk = stored_by_id[citation.chunk_id]
            assert chunk.metadata.get("granularity") == "parent"
            assert citation.quote
            assert citation.quote in chunk.text


def test_sentence_window_quotes_are_contiguous_substrings_of_parent(settings: Settings) -> None:
    with _hierarchical_runtime(settings, "sentence_window", sentence_window_radius=1) as hierarchical:
        workspace_id, corpus_id = _ingest_sections(hierarchical)
        response = hierarchical.query(
            QueryRequest(workspace_id=workspace_id, corpus_id=corpus_id, query="cached grounded answers reuse")
        )
        assert "expand" in response.routes
        assert response.citations
        for citation in response.citations:
            stored = hierarchical.store.get_chunks(workspace_id, corpus_id, (citation.chunk_id,))
            assert len(stored) == 1, "every cited id must be a real stored row"
            child = stored[0]
            assert child.metadata.get("granularity") == "child"
            parent_id = child.metadata["parent_id"]
            parent = hierarchical.store.get_chunks(workspace_id, corpus_id, (parent_id,))[0]
            assert child.text in parent.text
            assert citation.quote in parent.text


def test_parent_child_is_never_below_the_flat_baseline(settings: Settings) -> None:
    query = "fused candidates and the budgeter answer context"
    with _hierarchical_runtime(settings, "flat") as flat:
        flat_ws, flat_corpus = _ingest_sections(flat)
        flat_response = flat.query(QueryRequest(workspace_id=flat_ws, corpus_id=flat_corpus, query=query))
    with _hierarchical_runtime(settings, "parent_child") as hierarchical:
        hier_ws, hier_corpus = _ingest_sections(hierarchical)
        hier_response = hierarchical.query(QueryRequest(workspace_id=hier_ws, corpus_id=hier_corpus, query=query))
    assert hier_response.usage["retrieved_chunks"] >= flat_response.usage["retrieved_chunks"]
    assert hier_response.usage["retrieved_chunks"] >= 1
