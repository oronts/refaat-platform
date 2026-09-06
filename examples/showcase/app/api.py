"""The HTTP surface over the showcase runtime.

Every endpoint answers only through refaat: the retrieval modes, the program layer,
and the agent layer are wrapped as REST routes, and every answer and citation comes
from the grounded pipeline. The citation contract holds over HTTP because each cited
id is checked against the stored-chunk universe before it is returned. There is no
auth: the app is a single self-contained demonstration on the loopback interface.

The service registers each program once at startup and reuses the bound handles, so a
repeated request never re-registers a program. The langgraph mode runs the real
StateGraph where langgraph is installed and degrades to the dependency-free grounded
node otherwise, exactly like the agent layer.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from refaat import RefaatProject
from refaat.domain.models import Citation, QueryRequest
from refaat.integrations import as_langgraph_node, build_grounded_graph
from refaat.programs import (
    CorrectiveRagProgram,
    GroundedRagProgram,
    SelfRagProgram,
    StructuredOutputProgram,
)

from .agents import CORPUS, InProcessGroundedClient, _optional_dependency
from .knowledge import load_documents
from .observability import SessionEconomics, capture_program_trace
from .programs import build_governed_react
from .retrieval import ShowcaseRetrieval

QUERY_MODES = ("rag", "graphrag", "crag", "self_rag", "react_agent", "langgraph", "structured")

FEATURE_CATALOG: tuple[dict[str, Any], ...] = (
    {"id": "rag", "name": "RAG with resolved citations", "model_lift": False},
    {"id": "graphrag", "name": "GraphRAG multi-hop facts", "model_lift": False},
    {"id": "cag", "name": "CAG exact answer cache", "model_lift": False},
    {"id": "kag", "name": "KAG graph-aware answer", "model_lift": False},
    {"id": "rrf_autocut", "name": "RRF fusion with autocut", "model_lift": False},
    {"id": "reranking", "name": "Lexical reranker stage", "model_lift": False},
    {"id": "hype", "name": "HyPE hypothetical prompt embeddings", "model_lift": False},
    {"id": "raptor", "name": "RAPTOR recursive summary tree", "model_lift": False},
    {"id": "contextual_retrieval", "name": "Contextual retrieval", "model_lift": False},
    {"id": "compression", "name": "Context compression", "model_lift": False},
    {"id": "metadata_filter", "name": "Metadata hard filter and BM25", "model_lift": False},
    {"id": "corrective_rag", "name": "Corrective RAG (CRAG)", "model_lift": False},
    {"id": "self_rag", "name": "Self-RAG reflection", "model_lift": False},
    {"id": "recursive_language_method", "name": "Bounded recursive method (RLM)", "model_lift": False},
    {"id": "governed_react", "name": "Governed ReAct agent", "model_lift": False},
    {"id": "structured_output", "name": "Structured typed output", "model_lift": False},
    {"id": "multi_agent", "name": "Multi-agent orchestrator", "model_lift": False},
    {"id": "immutable_registry", "name": "Immutable program registry", "model_lift": False},
    {"id": "langchain", "name": "LangChain grounded tool", "model_lift": False},
    {"id": "langgraph", "name": "LangGraph state machine", "model_lift": False},
    {"id": "mcp", "name": "MCP serve and consume", "model_lift": False},
    {"id": "query_rewrite", "name": "Model query rewriting", "model_lift": True},
    {"id": "multi_query", "name": "Multi-query expansion", "model_lift": True},
    {"id": "cross_encoder", "name": "Cross-encoder reranker", "model_lift": True},
    {"id": "model_judge", "name": "Model-as-judge metric", "model_lift": True},
)


# Each fired route maps to a short, human explanation of the stage it names. The UI
# turns the real ordered routes of a query into an ordered pipeline story with these
# captions, so every stage shown is one the runtime actually ran.
ROUTE_EXPLANATIONS: dict[str, tuple[str, str]] = {
    "semantic": ("semantic search", "dense vector similarity over the stored chunk embeddings"),
    "lexical": ("lexical search", "sparse keyword and BM25 match over the chunk text"),
    "filter": ("metadata filter", "a hard metadata constraint applied before ranking"),
    "graph": ("graph facts", "the co-occurrence entity graph contributed multi-hop facts"),
    "rrf": ("reciprocal rank fusion", "the dense and sparse candidate lists fused by reciprocal rank"),
    "fusion": ("fusion", "candidate lists merged into one ranking"),
    "expansion": ("query expansion", "extra query variants retrieved and merged in"),
    "hype": ("HyPE", "hypothetical prompt embeddings matched and mapped back to real chunks"),
    "raptor": ("RAPTOR", "the recursive summary tree matched and mapped back to leaf chunks"),
    "rerank": ("rerank", "the fused candidates re-scored by the reranker stage"),
    "autocut": ("autocut", "the low-similarity tail dropped from the candidate list"),
    "compress": ("compression", "redundant context compressed before packing"),
    "expand": ("hierarchical expand", "parent or neighbour chunks pulled in for context"),
    "session": ("session memory", "a prior session summary folded into the context"),
    "profile": ("learned profile", "a learned user profile signal applied to retrieval"),
    "harness": ("active harness", "an active harness snapshot policy applied"),
    "cag:exact": ("cache exact hit", "an identical prior query served from the answer cache, no model call"),
    "cag:semantic": ("cache semantic hit", "a semantically equivalent prior query served from the answer cache"),
    "abstain": ("abstain", "the best retrieved evidence was below the floor, so an honest no-answer was returned"),
}


def build_stages(routes: tuple[str, ...]) -> list[dict[str, str]]:
    stages = []
    for route in routes:
        label, explanation = ROUTE_EXPLANATIONS.get(route, (route, "retrieval stage"))
        stages.append({"route": route, "label": label, "explanation": explanation})
    return stages


class QueryBody(BaseModel):
    question: str = Field(min_length=1)
    mode: str = "rag"


@dataclass(frozen=True, slots=True)
class AnswerView:
    mode: str
    answer: str
    citations: tuple[Citation, ...]
    routes: tuple[str, ...]
    cached: bool
    usage: dict[str, Any]
    extra: dict[str, Any] = field(default_factory=dict)
    abstained: bool = False


class ShowcaseService:
    """Wraps one ingested project as the whole HTTP capability surface."""

    def __init__(
        self,
        project: RefaatProject,
        *,
        mode: str,
        online: bool = False,
        generation_model: str = "extractive-v1",
        embedding_model: str = "refaat-hash-v1",
    ) -> None:
        self.project = project
        self.mode = mode
        self.online = online
        self.generation_model = generation_model
        self.embedding_model = embedding_model
        self.client = InProcessGroundedClient(project)
        retrieval = ShowcaseRetrieval(project)
        self.chunk_ids = retrieval._chunk_ids
        self.doc_titles = {str(doc["id"]): str(doc["title"]) for doc in project.list_documents()}
        self.corrective = project.programs.register(CorrectiveRagProgram())
        self.self_rag = project.programs.register(SelfRagProgram())
        self.structured = project.programs.register(StructuredOutputProgram(GroundedRagProgram()))
        self.react = build_governed_react(project)
        self.settings = project.runtime.settings
        self.economics = SessionEconomics()

    def enrich(self, citations: tuple[Citation, ...]) -> list[dict[str, Any]]:
        return [
            {
                "chunk_id": citation.chunk_id,
                "quote": citation.quote,
                "label": citation.label,
                "document_id": citation.document_id,
                "document": self.doc_titles.get(citation.document_id, citation.document_id),
                "resolves": citation.chunk_id in self.chunk_ids,
            }
            for citation in citations
        ]

    def grounded(self, citations: tuple[Citation, ...]) -> bool:
        return bool(citations) and all(citation.chunk_id in self.chunk_ids for citation in citations)

    def answer(self, mode: str, question: str) -> AnswerView:
        if mode not in QUERY_MODES:
            raise HTTPException(status_code=422, detail=f"Unknown mode: {mode}")
        return getattr(self, f"_mode_{mode}")(question)

    def _mode_rag(self, question: str) -> AnswerView:
        response = self.project.ask(question, fusion="rrf")
        return AnswerView(
            "rag",
            response.answer,
            response.citations,
            response.routes,
            response.cached,
            response.usage,
            abstained=response.abstained,
        )

    def _mode_graphrag(self, question: str) -> AnswerView:
        response = self.project.ask(question)
        context = self.project.retrieve(question)
        extra = {"graph_facts": list(context.graph_facts[:8]), "graph_route": "graph" in response.routes}
        return AnswerView(
            "graphrag",
            response.answer,
            response.citations,
            response.routes,
            response.cached,
            response.usage,
            extra,
            abstained=response.abstained,
        )

    def _mode_crag(self, question: str) -> AnswerView:
        result = self.corrective.run(question=question)
        assessment = result.outputs.get("corrective_rag", {})
        extra = {"label": assessment.get("label"), "assessment_present": bool(assessment)}
        return AnswerView("crag", result.answer, result.citations, result.routes, result.cached, result.usage, extra)

    def _mode_self_rag(self, question: str) -> AnswerView:
        result = self.self_rag.run(question=question)
        reflection = result.outputs.get("self_rag", {})
        extra = {"is_supported": reflection.get("is_supported"), "reflection_present": bool(reflection)}
        return AnswerView(
            "self_rag", result.answer, result.citations, result.routes, result.cached, result.usage, extra
        )

    def _mode_react_agent(self, question: str) -> AnswerView:
        result = self.react.run(question=question)
        steps = result.outputs["react_steps"]
        trajectory = [{"tool": step["tool"], "authorized": step["authorized"]} for step in steps]
        extra = {
            "trajectory": trajectory,
            "authorized": sorted(step["tool"] for step in steps if step["authorized"]),
            "denied": sorted(step["tool"] for step in steps if not step["authorized"]),
        }
        return AnswerView(
            "react_agent", result.answer, result.citations, result.routes, result.cached, result.usage, extra
        )

    def _mode_structured(self, question: str) -> AnswerView:
        result = self.structured.run(question=question)
        return AnswerView(
            "structured",
            result.answer,
            result.citations,
            result.routes,
            result.cached,
            result.usage,
            {"schema_enforced": True},
        )

    def _mode_langgraph(self, question: str) -> AnswerView:
        response = self.project.ask(question)
        try:
            graph: Any = build_grounded_graph(self.client, CORPUS)
            state = graph.invoke({"question": question})
            status = "executed"
        except RuntimeError as exc:
            if not _optional_dependency(exc):
                raise
            state = as_langgraph_node(self.client, CORPUS)({"question": question})
            status = "optional_dependency_missing"
        branch = "reason" if state["citations"] else "abstain"
        extra = {"branch": branch, "langgraph_status": status}
        return AnswerView(
            "langgraph",
            response.answer,
            response.citations,
            response.routes,
            response.cached,
            response.usage,
            extra,
            abstained=response.abstained,
        )

    def graph_view(self, question: str) -> dict[str, Any]:
        context = self.project.retrieve(question)
        document_ids = {hit.chunk.document_id for hit in context.hits}
        entities, relations = self.project.runtime.store.list_graph(self.project.workspace_id, self.project.corpus_id)
        by_id = {entity.id: entity for entity in entities}
        scoped = sorted(
            (relation for relation in relations if relation.document_id in document_ids),
            key=lambda relation: relation.weight,
            reverse=True,
        )[:40]
        referenced = {relation.source_entity_id for relation in scoped} | {
            relation.target_entity_id for relation in scoped
        }
        nodes = [
            {"id": entity.id, "label": entity.name, "kind": entity.kind}
            for entity in entities
            if entity.id in referenced
        ]
        edges = [
            {
                "source": relation.source_entity_id,
                "target": relation.target_entity_id,
                "relation": relation.kind,
                "weight": round(relation.weight, 4),
            }
            for relation in scoped
            if relation.source_entity_id in by_id and relation.target_entity_id in by_id
        ]
        return {
            "query": question,
            "facts": list(context.graph_facts[:12]),
            "nodes": nodes,
            "edges": edges,
            # Aliases kept so any older consumer of the entity and relation keys still works.
            "entities": nodes,
            "relations": edges,
            "routes": list(context.routes),
        }

    def stats_view(self) -> dict[str, Any]:
        stats = self.project.stats()
        model_lift = sum(1 for feature in FEATURE_CATALOG if feature["model_lift"])
        return {
            "workspace": self.project.workspace_id,
            "corpus": self.project.corpus_id,
            "documents": stats.get("documents"),
            "chunks": stats.get("chunks"),
            "entities": stats.get("entities"),
            "relations": stats.get("relations"),
            "capabilities": list(FEATURE_CATALOG),
            "capability_counts": {
                "total": len(FEATURE_CATALOG),
                "model_lift": model_lift,
                "offline": len(FEATURE_CATALOG) - model_lift,
            },
            "economics": self.economics.snapshot(),
        }

    def record_economics(
        self, question: str, mode: str, view_cached: bool, usage: dict[str, Any], elapsed_ms: float
    ) -> dict[str, Any]:
        return self.economics.record(
            question=question,
            mode=mode,
            cached=view_cached,
            usage=usage,
            model=self.generation_model,
            prices=self.settings.model_prices,
            elapsed_ms=elapsed_ms,
        )

    def recent_events(self, limit: int = 12) -> list[dict[str, Any]]:
        """Recent real runtime records: the outbox rows written at ingestion.

        These are genuine ``document.indexed`` events the store persisted while the
        corpus was ingested, not a synthetic log. They stay pending because the
        showcase never runs the delivery worker, so the panel shows the real event
        ledger the runtime recorded.
        """
        rows: list[dict[str, Any]] = []
        for event in self.project.runtime.store.pending_outbox(limit=limit):
            payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
            rows.append(
                {
                    "topic": event.get("topic"),
                    "document_id": payload.get("document_id"),
                    "corpus_version": payload.get("corpus_version"),
                    "created_at": str(event.get("created_at")),
                }
            )
        return rows

    def observability_view(self) -> dict[str, Any]:
        captured = capture_program_trace(self.project, self.settings)
        return {
            "trace": captured["trace"],
            "budget": captured["budget"],
            "evaluation": captured["evaluation"],
            "economics": self.economics.snapshot(),
            "events": self.recent_events(),
            "online": self.online,
            "generation_model": self.generation_model,
        }

    def retrieval_snapshot(self, question: str) -> dict[str, Any]:
        """Real retrieval counts for the query, computed from the runtime, not the answer.

        The counts come from a real retrieve over the same question, so the per-query
        panel reports what the runtime actually surfaced (how many chunks were retrieved,
        whether the reranker fired, how many tokens were packed) rather than any estimate.
        """
        context = self.project.retrieve(question)
        return {
            "retrieved": len(context.hits),
            "reranked": "rerank" in context.routes,
            "packed_tokens": context.token_count,
        }

    def corpus_view(self) -> dict[str, Any]:
        documents = []
        for row in self.project.list_documents():
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            documents.append(
                {
                    "id": str(row["id"]),
                    "source_id": str(row.get("source_id", "")),
                    "title": str(row["title"]),
                    "media_type": str(row.get("media_type", "")),
                    "category": metadata.get("category"),
                    "tags": metadata.get("tags", []),
                }
            )
        stats = self.project.stats()
        return {
            "workspace": self.project.workspace_id,
            "corpus": self.project.corpus_id,
            "documents": documents,
            "document_count": len(documents),
            "chunk_count": stats.get("chunks"),
        }

    def stream(self, question: str) -> Iterator[str]:
        request = QueryRequest(
            workspace_id=self.project.workspace_id,
            corpus_id=self.project.corpus_id,
            query=question,
        )
        started = time.perf_counter()
        for event in self.project.runtime.stream_query(request):
            if event["type"] == "final":
                elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
                event = dict(event)
                event["citations_resolve"] = bool(event["citations"]) and all(
                    str(citation["chunk_id"]) in self.chunk_ids for citation in event["citations"]
                )
                event["elapsed_ms"] = elapsed_ms
                event["stages"] = build_stages(tuple(event["routes"]))
                event["retrieval"] = self.retrieval_snapshot(question)
                econ = self.record_economics(
                    question, "stream", bool(event["cached"]), dict(event.get("usage", {})), elapsed_ms
                )
                event["tokens"] = econ["tokens"]
                event["usd_cost"] = econ["usd_cost"]
                event["saved"] = econ["saved"]
                event["session_economics"] = self.economics.snapshot()
            yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"


def _answer_payload(service: ShowcaseService, view: AnswerView) -> dict[str, Any]:
    return {
        "mode": view.mode,
        "answer": view.answer,
        "citations": service.enrich(view.citations),
        "routes": list(view.routes),
        "grounded": service.grounded(view.citations),
        "abstained": view.abstained,
        "cached": view.cached,
        "usage": dict(view.usage),
        "stages": build_stages(view.routes),
        "extra": view.extra,
    }


def api_router(service: ShowcaseService) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.post("/query")
    def query(body: QueryBody) -> dict[str, Any]:
        started = time.perf_counter()
        view = service.answer(body.mode, body.question)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        payload = _answer_payload(service, view)
        payload["elapsed_ms"] = elapsed_ms
        payload["retrieval"] = service.retrieval_snapshot(body.question)
        econ = service.record_economics(body.question, view.mode, view.cached, dict(view.usage), elapsed_ms)
        payload["tokens"] = econ["tokens"]
        payload["usd_cost"] = econ["usd_cost"]
        payload["saved"] = econ["saved"]
        payload["session_economics"] = service.economics.snapshot()
        return payload

    @router.get("/stats")
    def stats() -> dict[str, Any]:
        return service.stats_view()

    @router.get("/observability")
    def observability() -> dict[str, Any]:
        return service.observability_view()

    @router.get("/query/stream")
    def query_stream(question: str = Query(min_length=1)) -> StreamingResponse:
        return StreamingResponse(service.stream(question), media_type="text/event-stream")

    @router.get("/corpus")
    def corpus() -> dict[str, Any]:
        return service.corpus_view()

    @router.get("/features")
    def features() -> dict[str, Any]:
        return {
            "mode": service.mode,
            "online": service.online,
            "generation_model": service.generation_model,
            "embedding_model": service.embedding_model,
            "modes": list(QUERY_MODES),
            "features": list(FEATURE_CATALOG),
        }

    @router.get("/graph")
    def graph(question: str = Query(min_length=1)) -> dict[str, Any]:
        return service.graph_view(question)

    @router.post("/program/{name}/run")
    def program_run(name: str, body: QueryBody) -> dict[str, Any]:
        if not service.project.programs.resolvable(name):
            raise HTTPException(status_code=404, detail=f"Program not found: {name}")
        result = service.project.programs.get(name).run(question=body.question)
        return {
            "program": result.program,
            "artifact_id": result.artifact_id,
            "answer": result.answer,
            "citations": service.enrich(result.citations),
            "routes": list(result.routes),
            "grounded": result.grounded and service.grounded(result.citations),
        }

    @router.get("/registry")
    def registry() -> dict[str, Any]:
        programs = [program["program"] for program in service.project.programs.list()]
        documents = len(load_documents())
        return {"programs": programs, "knowledge_documents": documents}

    return router


__all__ = [
    "FEATURE_CATALOG",
    "QUERY_MODES",
    "AnswerView",
    "QueryBody",
    "ShowcaseService",
    "api_router",
]
