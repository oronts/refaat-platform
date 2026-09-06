from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from _shared import bootstrap_workspace, emit, live_api_server

from refaat.integrations import (
    RefaatAnswerer,
    RefaatRetriever,
    as_crewai_answer_tool,
    as_crewai_tool,
    as_langchain_retriever,
    as_langchain_tool,
    as_langgraph_node,
    as_llamaindex_retriever,
)
from refaat.sdk import RefaatClient


def optional_integrations(client: RefaatClient, corpus_id: str) -> dict[str, object]:
    status: dict[str, object] = {}
    try:
        retriever = as_langchain_retriever(client, corpus_id, top_k=3)
        tool = as_langchain_tool(client, corpus_id)
        answer = str(tool.invoke({"question": "Who verifies release evidence?"}))
        status["langchain"] = {
            "status": "executed",
            "documents": len(retriever.invoke("release evidence")),
            "tool_answer_chars": len(answer),
        }
    except RuntimeError as exc:
        status["langchain"] = {"status": "optional_dependency_missing", "message": str(exc)}

    try:
        retriever = as_llamaindex_retriever(client, corpus_id, top_k=3)
        status["llamaindex"] = {
            "status": "executed",
            "documents": len(retriever.retrieve("release evidence")),
        }
    except RuntimeError as exc:
        status["llamaindex"] = {"status": "optional_dependency_missing", "message": str(exc)}
    return status


def main() -> None:
    with TemporaryDirectory(prefix="refaat-frameworks-") as directory:
        root = Path(directory)
        with live_api_server(root) as (base_url, bootstrap_token):
            bootstrap = bootstrap_workspace(base_url, bootstrap_token)
            with RefaatClient(base_url, bootstrap["api_key"]) as client:
                corpus_id = str(client.create_corpus("framework-handbook")["id"])
                client.ingest_text(
                    corpus_id,
                    source_id="release.md",
                    title="Release policy",
                    content=(
                        "Production releases require peer review, automated tests, and a rollback plan. "
                        "The Release Manager verifies the evidence."
                    ),
                    media_type="text/markdown",
                )
                neutral = RefaatRetriever(client, corpus_id, top_k=3)("release evidence")
                crewai_tool = as_crewai_tool(client, corpus_id, top_k=3)
                crewai_context = crewai_tool("Who verifies release evidence?")
                grounded = RefaatAnswerer(client, corpus_id)("Who verifies release evidence?")
                answer_tool = as_crewai_answer_tool(client, corpus_id)
                answer_tool_text = answer_tool("Who verifies release evidence?")
                langgraph_state = as_langgraph_node(client, corpus_id)({"question": "Who verifies release evidence?"})
                optional = optional_integrations(client, corpus_id)

            assert neutral.documents
            assert "[chunk:" in crewai_context
            assert grounded.answer
            assert answer_tool_text
            assert langgraph_state["answer"]
            assert langgraph_state["citations"]

            emit(
                {
                    "corpus_id": corpus_id,
                    "framework_neutral_documents": len(neutral.documents),
                    "crewai_tool_name": crewai_tool.__name__,
                    "crewai_context_contains_evidence_ids": "[chunk:" in crewai_context,
                    "grounded_answer_present": bool(grounded.answer),
                    "grounded_citation_count": len(grounded.citations),
                    "crewai_answer_tool_name": answer_tool.__name__,
                    "langgraph_answer_present": bool(langgraph_state["answer"]),
                    "langgraph_citation_count": len(langgraph_state["citations"]),
                    "optional": optional,
                }
            )


if __name__ == "__main__":
    main()
