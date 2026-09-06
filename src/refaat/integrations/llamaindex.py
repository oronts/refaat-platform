from __future__ import annotations

from typing import Any

from refaat.sdk.client import RefaatClient


def as_llamaindex_retriever(client: RefaatClient, corpus_id: str, top_k: int = 8) -> Any:
    try:
        from llama_index.core.base.base_retriever import BaseRetriever
        from llama_index.core.schema import NodeWithScore, TextNode
    except ImportError as exc:
        raise RuntimeError("Install refaat[integrations] for LlamaIndex support") from exc

    class LlamaIndexRefaatRetriever(BaseRetriever):
        def _retrieve(self, query_bundle: Any) -> list[Any]:
            result = client.retrieve(corpus_id, str(query_bundle.query_str), top_k=top_k)
            return [
                NodeWithScore(
                    node=TextNode(
                        text=value.text,
                        id_=value.chunk_id,
                        metadata={
                            **value.metadata,
                            "refaat_document_id": value.document_id,
                        },
                    ),
                    score=value.score,
                )
                for value in result.documents
            ]

    return LlamaIndexRefaatRetriever()
