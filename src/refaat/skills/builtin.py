from __future__ import annotations

from refaat.domain.models import ContextPacket, JsonObject


class CorpusHealthSkill:
    name = "corpus_health"
    description = "Summarize retrieval coverage and context composition for a query."

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        del arguments
        documents = {hit.chunk.document_id for hit in context.hits}
        return {
            "query": context.query,
            "routes": list(context.routes),
            "retrieved_chunks": len(context.hits),
            "retrieved_documents": len(documents),
            "graph_facts": len(context.graph_facts),
            "context_tokens": context.token_count,
            "scores": [
                {
                    "chunk_id": hit.chunk.id,
                    "score": hit.score,
                    "semantic": hit.semantic_score,
                    "lexical": hit.lexical_score,
                    "graph": hit.graph_score,
                    "reasons": list(hit.reasons),
                }
                for hit in context.hits
            ],
        }
