"""The framework agent layer: LangChain, LangGraph, and CrewAI over refaat.

Every agent answers only through refaat. Retrieval and grounded answers come from the
project's own query pipeline, so an agent can never fabricate a citation: the answer
and its citations are produced by the grounded runtime, and the citation ids resolve to
real stored chunks. The shipped integrations take a small client Protocol, so the app
drives them in process over a :class:`RefaatProject` with no external service and no HTTP
server. LangChain, LangGraph, and CrewAI are optional: each path lazily imports its
package and degrades to a recorded status when the package is absent, so the offline gate
never requires them.
"""

from __future__ import annotations

from typing import Any

from refaat import RefaatProject
from refaat.integrations import (
    as_crewai_answer_tool,
    as_langchain_retriever,
    as_langchain_tool,
    as_langgraph_node,
    as_langgraph_program_node,
    build_grounded_graph,
    build_react_subgraph,
)
from refaat.programs.registry import BoundProgram
from refaat.sdk.models import Answer, RetrievalResult, RetrievedDocument

AGENT_QUESTION = "How do OGuardAI and PimTx fit into Oronts GmbH?"
REACT_AGENT_QUESTION = (
    "During a security incident, how does OGuardAI prevent sensitive data drift before it reaches an LLM?"
)
CORPUS = "in-process"


class InProcessGroundedClient:
    """A grounded retriever client that satisfies the integration Protocols in process.

    The shipped LangChain, LangGraph, and CrewAI helpers depend only on ``retrieve`` and
    ``query`` returning the SDK result models, so this adapter maps the project's own
    query pipeline into those models without an HTTP server. The corpus argument is
    ignored because the adapter is bound to one project.
    """

    def __init__(self, project: RefaatProject) -> None:
        self._project = project

    def retrieve(self, corpus_id: str, query: str, **options: Any) -> RetrievalResult:
        del corpus_id
        top_k = options.get("top_k")
        context = self._project.retrieve(query, top_k=top_k) if top_k else self._project.retrieve(query)
        documents = tuple(
            RetrievedDocument(
                chunk_id=hit.chunk.id,
                document_id=hit.chunk.document_id,
                text=hit.chunk.text,
                score=hit.score,
                metadata=dict(hit.chunk.metadata),
                reasons=tuple(hit.reasons),
                semantic_score=hit.semantic_score,
                lexical_score=hit.lexical_score,
                graph_score=hit.graph_score,
            )
            for hit in context.hits
        )
        return RetrievalResult(
            query=query,
            documents=documents,
            routes=tuple(context.routes),
            graph_facts=tuple(context.graph_facts),
            token_count=context.token_count,
            session_summary=context.session_summary,
        )

    def query(self, corpus_id: str, query: str, **options: Any) -> Answer:
        del corpus_id, options
        response = self._project.ask(query)
        citations = tuple(
            {"chunk_id": citation.chunk_id, "document_id": citation.document_id, "label": citation.label}
            for citation in response.citations
        )
        return Answer(
            answer=response.answer,
            citations=citations,
            routes=tuple(response.routes),
            cached=response.cached,
            corpus_version=response.corpus_version,
            trace_id=response.trace_id,
            usage=dict(response.usage),
        )


def _optional_dependency(exc: RuntimeError) -> bool:
    message = str(exc)
    return "Install" in message and ("langchain" in message or "langgraph" in message.lower())


def langchain_agent(project: RefaatProject, chunk_ids: set[str]) -> dict[str, Any]:
    """Answer through a grounded StructuredTool and a retriever, degrading if langchain is absent."""
    client = InProcessGroundedClient(project)
    try:
        retriever = as_langchain_retriever(client, CORPUS, top_k=4)
        tool = as_langchain_tool(client, CORPUS)
    except RuntimeError as exc:
        if not _optional_dependency(exc):
            raise
        return {"status": "optional_dependency_missing"}
    documents = retriever.invoke(AGENT_QUESTION)
    answer = str(tool.invoke({"question": AGENT_QUESTION}))
    resolved = client.query(CORPUS, AGENT_QUESTION)
    cited = [str(citation["chunk_id"]) for citation in resolved.citations]
    return {
        "status": "executed",
        "documents": len(documents),
        "answer_present": bool(answer.strip()),
        "grounded": bool(cited) and all(chunk_id in chunk_ids for chunk_id in cited),
    }


def langgraph_agent(project: RefaatProject, chunk_ids: set[str]) -> dict[str, Any]:
    """Build a grounded retrieve-then-reason-or-abstain StateGraph, degrading if langgraph is absent.

    The node primitives are dependency free, so a minimal install still produces a
    grounded answer. Where langgraph is installed the real branching graph runs: a normal
    question reaches the grounded reason branch, an unreachable evidence threshold reaches
    the honest empty-citation abstain branch.
    """
    client = InProcessGroundedClient(project)
    node = as_langgraph_node(client, CORPUS)
    try:
        graph: Any = build_grounded_graph(client, CORPUS)
    except RuntimeError as exc:
        if not _optional_dependency(exc):
            raise
        skipped = node({"question": AGENT_QUESTION})
        cited = [str(citation["chunk_id"]) for citation in skipped["citations"]]
        return {
            "status": "optional_dependency_missing",
            "grounded": bool(cited) and all(chunk_id in chunk_ids for chunk_id in cited),
        }
    reasoned = graph.invoke({"question": AGENT_QUESTION})
    abstained = build_grounded_graph(client, CORPUS, min_documents=10_000).invoke({"question": AGENT_QUESTION})
    cited = [str(citation["chunk_id"]) for citation in reasoned["citations"]]
    return {
        "status": "executed",
        "reason_branch_grounded": bool(cited) and all(chunk_id in chunk_ids for chunk_id in cited),
        "abstain_branch_empty": abstained["answer"] == "" and abstained["citations"] == [],
    }


def langgraph_react_subgraph(bound_react: BoundProgram, chunk_ids: set[str]) -> dict[str, Any]:
    """Compile the governed-ReAct program as a subgraph, degrading if langgraph is absent."""
    try:
        subgraph: Any = build_react_subgraph(bound_react)
        trajectory = subgraph.invoke({"question": REACT_AGENT_QUESTION})
        status = "executed"
    except RuntimeError as exc:
        if not _optional_dependency(exc):
            raise
        trajectory = as_langgraph_program_node(bound_react)({"question": REACT_AGENT_QUESTION})
        status = "optional_dependency_missing"
    cited = [str(citation["chunk_id"]) for citation in trajectory["citations"]]
    return {
        "status": status,
        "grounded": (
            bool(trajectory["react_steps"])
            and trajectory["grounded"] is True
            and bool(cited)
            and all(chunk_id in chunk_ids for chunk_id in cited)
        ),
    }


def crewai_agent(project: RefaatProject, chunk_ids: set[str]) -> dict[str, Any]:
    """A CrewAI answer tool over refaat, exercised directly through the grounded pipeline.

    The answer tool is a plain callable that always answers through refaat's grounded
    pipeline, so its answer carries real citations whether or not crewai is present. The
    tool is refaat's contribution to a crew, and it is exercised directly here; no crew is
    kicked off, because a crew kickoff drives its own agent model that is out of scope for
    this offline, deterministic demonstration. When crewai is importable the run is
    labelled ``tool_executed``; when the package is absent the same tool still runs and the
    missing dependency is recorded, so the offline gate never requires crewai.
    """
    client = InProcessGroundedClient(project)
    answer_tool = as_crewai_answer_tool(client, CORPUS)
    tool_output = answer_tool(AGENT_QUESTION)
    grounded = client.query(CORPUS, AGENT_QUESTION)
    cited = [str(citation["chunk_id"]) for citation in grounded.citations]
    tool_grounded = bool(cited) and all(chunk_id in chunk_ids for chunk_id in cited)
    try:
        import crewai  # noqa: F401
    except ImportError:
        status = "optional_dependency_missing"
    else:
        status = "tool_executed"
    return {"status": status, "tool_grounded": tool_grounded, "answer_present": bool(tool_output.strip())}


def run_agent_layer(project: RefaatProject, bound_react: BoundProgram, chunk_ids: set[str]) -> dict[str, Any]:
    return {
        "langchain": langchain_agent(project, chunk_ids),
        "langgraph": langgraph_agent(project, chunk_ids),
        "langgraph_react": langgraph_react_subgraph(bound_react, chunk_ids),
        "crewai": crewai_agent(project, chunk_ids),
    }


def agent_layer_grounded(report: dict[str, Any]) -> bool:
    langchain = report["langchain"]
    langgraph = report["langgraph"]
    react = report["langgraph_react"]
    crewai = report["crewai"]
    langchain_ok = langchain["status"] == "optional_dependency_missing" or langchain["grounded"] is True
    if langgraph["status"] == "optional_dependency_missing":
        langgraph_ok = langgraph["grounded"] is True
    else:
        langgraph_ok = langgraph["reason_branch_grounded"] is True and langgraph["abstain_branch_empty"] is True
    return bool(langchain_ok and langgraph_ok and react["grounded"] and crewai["tool_grounded"])


__all__ = [
    "AGENT_QUESTION",
    "InProcessGroundedClient",
    "agent_layer_grounded",
    "crewai_agent",
    "langchain_agent",
    "langgraph_agent",
    "langgraph_react_subgraph",
    "run_agent_layer",
]
