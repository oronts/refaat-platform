from __future__ import annotations

from collections.abc import Callable

from refaat.sdk.client import RefaatClient


def as_crewai_tool(
    client: RefaatClient,
    corpus_id: str,
    top_k: int = 8,
) -> Callable[[str], str]:
    def retrieve_knowledge(query: str) -> str:
        """Retrieve grounded knowledge from a Refaat corpus."""
        result = client.retrieve(corpus_id, query, top_k=top_k)
        return "\n\n".join(f"[chunk:{document.chunk_id}] {document.text}" for document in result.documents)

    retrieve_knowledge.__name__ = "retrieve_refaat_knowledge"
    return retrieve_knowledge


def as_crewai_answer_tool(client: RefaatClient, corpus_id: str) -> Callable[[str], str]:
    def answer_question(question: str) -> str:
        """Answer a question with grounded evidence from a Refaat corpus."""
        answer = client.query(corpus_id, question)
        cited = ", ".join(
            str(citation.get("chunk_id", "")) for citation in answer.citations if citation.get("chunk_id")
        )
        return f"{answer.answer}\n\nCitations: {cited}" if cited else answer.answer

    answer_question.__name__ = "answer_refaat_question"
    return answer_question
