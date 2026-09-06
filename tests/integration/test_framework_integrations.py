from __future__ import annotations

from typing import Any

import pytest

from refaat.application.runtime import Refaat
from refaat.domain.enums import SideEffectClass
from refaat.domain.models import ContextPacket, DocumentInput, JsonObject, ReActPolicy, ToolGrant
from refaat.integrations import (
    RefaatAnswerer,
    as_langgraph_node,
    build_grounded_graph,
    build_react_subgraph,
)
from refaat.programs import GovernedReActProgram, KeywordToolPlanner
from refaat.sdk.models import Answer, RetrievalResult, RetrievedDocument

pytestmark = pytest.mark.integration


class _FakeGroundedClient:
    def query(self, corpus_id: str, query: str, **options: object) -> Answer:
        return Answer(
            answer=f"grounded:{query}",
            citations=({"chunk_id": "chunk-1", "document_id": "doc-1", "label": "[1]", "quote": "q"},),
            routes=("vector",),
            cached=False,
            corpus_version=1,
            trace_id="trace-1",
            usage={},
        )


def test_refaat_answerer_returns_grounded_answer() -> None:
    answer = RefaatAnswerer(_FakeGroundedClient(), "corpus")("who verifies releases?")
    assert answer.answer == "grounded:who verifies releases?"
    assert answer.citations[0]["chunk_id"] == "chunk-1"


def test_langgraph_node_writes_answer_and_citations_into_state() -> None:
    node = as_langgraph_node(_FakeGroundedClient(), "corpus")
    updated = node({"question": "who verifies releases?"})
    assert updated["answer"] == "grounded:who verifies releases?"
    assert updated["citations"] == [{"chunk_id": "chunk-1", "document_id": "doc-1", "label": "[1]", "quote": "q"}]


def test_langgraph_node_honors_custom_state_keys() -> None:
    node = as_langgraph_node(_FakeGroundedClient(), "corpus", question_key="q", answer_key="a", citations_key="cites")
    updated = node({"q": "hello"})
    assert updated["a"] == "grounded:hello"
    assert "cites" in updated


class _GraphClient:
    def __init__(self, *, documents: tuple[RetrievedDocument, ...]) -> None:
        self._documents = documents

    def retrieve(self, corpus_id: str, query: str, **options: object) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            documents=self._documents,
            routes=("semantic",) if self._documents else (),
            graph_facts=(),
            token_count=0,
        )

    def query(self, corpus_id: str, query: str, **options: object) -> Answer:
        return Answer(
            answer=f"grounded:{query}",
            citations=({"chunk_id": "cited-c", "document_id": "doc-c", "label": "[1]", "quote": "q"},),
            routes=("vector",),
            cached=False,
            corpus_version=1,
            trace_id="trace-1",
            usage={},
        )


def test_build_grounded_graph_routes_to_reason_when_evidence_exists() -> None:
    # Distinct retrieved ids vs the query's citation id, so the compiled graph itself
    # proves retrieved evidence can never leak into a citation, not just the unit path.
    documents = (
        RetrievedDocument(chunk_id="retr-a", document_id="doc-a", text="x", score=0.9),
        RetrievedDocument(chunk_id="retr-b", document_id="doc-b", text="y", score=0.8),
    )
    graph = build_grounded_graph(_GraphClient(documents=documents), "corpus")

    final = graph.invoke({"question": "who verifies releases?"})

    assert final["answer"] == "grounded:who verifies releases?"
    assert [chunk["chunk_id"] for chunk in final["retrieved"]] == ["retr-a", "retr-b"]
    assert [citation["chunk_id"] for citation in final["citations"]] == ["cited-c"]


def test_build_grounded_graph_routes_to_abstain_when_no_evidence() -> None:
    graph = build_grounded_graph(_GraphClient(documents=()), "corpus")

    final = graph.invoke({"question": "unanswerable question"})

    assert final["answer"] == ""
    assert final["citations"] == []


def test_build_grounded_graph_rejects_non_positive_min_documents() -> None:
    with pytest.raises(ValueError, match="min_documents must be at least 1"):
        build_grounded_graph(_GraphClient(documents=()), "corpus", min_documents=0)


def test_build_grounded_graph_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = __import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> Any:
        if name.startswith("langgraph"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked_import)
    with pytest.raises(RuntimeError, match="Install langgraph for the LangGraph state-machine integration"):
        build_grounded_graph(_GraphClient(documents=()), "corpus")


class _LookupSkill:
    name = "lookup"
    description = "Read-only lookup over the retrieved context."

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        del arguments
        return {"documents": len({hit.chunk.document_id for hit in context.hits})}


def test_build_react_subgraph_runs_governed_program_over_runtime(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="react-graph")
    runtime.ingest(
        project.workspace_id,
        project.corpus_id,
        DocumentInput(
            source_id="ops.txt",
            title="Ops",
            media_type="text/plain",
            content=b"Rollback restores the previously approved harness snapshot before deployment.",
        ),
    )
    policy = ReActPolicy(
        max_iterations=4,
        granted_scopes=("read:corpus",),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        grants=(ToolGrant("lookup", required_scopes=("read:corpus",), side_effect=SideEffectClass.READ_ONLY),),
    )
    planner = KeywordToolPlanner({"lookup": ("rollback",)})
    program = GovernedReActProgram(policy=policy, planner=planner, skills={"lookup": _LookupSkill()})
    bound = project.programs.register(program)

    subgraph = build_react_subgraph(bound)
    final = subgraph.invoke({"question": "how does rollback work"})

    stored = {
        chunk.id
        for document in project.list_documents()
        for chunk in project.runtime.store.list_document_chunks(
            project.workspace_id, project.corpus_id, str(document["id"])
        )
    }
    assert final["react_steps"]
    assert any(step["authorized"] for step in final["react_steps"])
    assert final["citations"]
    assert all(citation["chunk_id"] in stored for citation in final["citations"])
    assert final["grounded"] is True
