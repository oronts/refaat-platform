from __future__ import annotations

from typing import Any

from refaat.sdk.client import RefaatClient


def as_langchain_retriever(client: RefaatClient, corpus_id: str, top_k: int = 8) -> Any:
    try:
        from langchain_core.documents import Document
        from langchain_core.retrievers import BaseRetriever
        from pydantic import ConfigDict, PrivateAttr
    except ImportError as exc:
        raise RuntimeError("Install refaat[integrations] for LangChain support") from exc

    class LangChainRefaatRetriever(BaseRetriever):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        _client: RefaatClient = PrivateAttr()
        _corpus_id: str = PrivateAttr()
        _top_k: int = PrivateAttr()

        def __init__(self) -> None:
            super().__init__()
            self._client = client
            self._corpus_id = corpus_id
            self._top_k = top_k

        def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Any]:
            del run_manager
            result = self._client.retrieve(self._corpus_id, query, top_k=self._top_k)
            return [
                Document(
                    page_content=value.text,
                    metadata={
                        **value.metadata,
                        "refaat_chunk_id": value.chunk_id,
                        "refaat_document_id": value.document_id,
                        "refaat_score": value.score,
                    },
                )
                for value in result.documents
            ]

    return LangChainRefaatRetriever()


def as_langchain_tool(
    client: RefaatClient,
    corpus_id: str,
    *,
    name: str = "refaat_grounded_answer",
    description: str = (
        "Answer a question from the Refaat knowledge corpus and return a grounded answer with canonical citation ids."
    ),
) -> Any:
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:
        raise RuntimeError("Install refaat[integrations] for LangChain support") from exc

    def grounded_answer(question: str) -> str:
        answer = client.query(corpus_id, question)
        cited = ", ".join(
            str(citation.get("chunk_id", "")) for citation in answer.citations if citation.get("chunk_id")
        )
        return f"{answer.answer}\n\nCitations: {cited}" if cited else answer.answer

    return StructuredTool.from_function(func=grounded_answer, name=name, description=description)
