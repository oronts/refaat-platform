from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RetrievedDocument:
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    graph_score: float = 0.0


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    query: str
    documents: tuple[RetrievedDocument, ...]
    routes: tuple[str, ...]
    graph_facts: tuple[str, ...]
    token_count: int
    session_summary: str | None = None


@dataclass(frozen=True, slots=True)
class Answer:
    answer: str
    citations: tuple[dict[str, Any], ...]
    routes: tuple[str, ...]
    cached: bool
    corpus_version: int
    trace_id: str
    usage: dict[str, Any]
    abstained: bool = False


@dataclass(frozen=True, slots=True)
class ProgramRunResult:
    program: str
    artifact_id: str
    answer: str
    citations: tuple[dict[str, Any], ...]
    routes: tuple[str, ...]
    cached: bool
    corpus_version: int
    trace_id: str
    grounded: bool
    validations: tuple[dict[str, Any], ...]
    usage: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RlmNode:
    id: str
    parent_id: str | None
    depth: int
    route: str
    input_chunk_count: int
    input_tokens: int
    selected_chunk_ids: tuple[str, ...]
    cited_chunk_ids: tuple[str, ...]
    answer_excerpt: str


@dataclass(frozen=True, slots=True)
class RlmAnswer:
    answer: str
    citations: tuple[dict[str, Any], ...]
    nodes: tuple[RlmNode, ...]
    corpus_version: int
    trace_id: str
    usage: dict[str, Any]


def retrieval_result_from_payload(payload: Mapping[str, Any]) -> RetrievalResult:
    session_summary = payload.get("session_summary")
    return RetrievalResult(
        query=str(payload["query"]),
        documents=tuple(
            RetrievedDocument(
                chunk_id=str(hit["chunk_id"]),
                document_id=str(hit["document_id"]),
                text=str(hit["text"]),
                score=float(hit["score"]),
                metadata=hit.get("metadata", {}),
                reasons=tuple(hit.get("reasons", [])),
                semantic_score=float(hit.get("semantic_score", 0.0)),
                lexical_score=float(hit.get("lexical_score", 0.0)),
                graph_score=float(hit.get("graph_score", 0.0)),
            )
            for hit in payload["hits"]
        ),
        routes=tuple(payload["routes"]),
        graph_facts=tuple(payload["graph_facts"]),
        token_count=int(payload["token_count"]),
        session_summary=str(session_summary) if session_summary is not None else None,
    )


def program_result_from_payload(payload: Mapping[str, Any]) -> ProgramRunResult:
    return ProgramRunResult(
        program=str(payload["program"]),
        artifact_id=str(payload["artifact_id"]),
        answer=str(payload["answer"]),
        citations=tuple(payload["citations"]),
        routes=tuple(payload["routes"]),
        cached=bool(payload["cached"]),
        corpus_version=int(payload["corpus_version"]),
        trace_id=str(payload["trace_id"]),
        grounded=bool(payload["grounded"]),
        validations=tuple(payload["validations"]),
        usage=dict(payload["usage"]),
    )


def answer_from_payload(payload: Mapping[str, Any]) -> Answer:
    return Answer(
        answer=str(payload["answer"]),
        citations=tuple(payload["citations"]),
        routes=tuple(payload["routes"]),
        cached=bool(payload["cached"]),
        corpus_version=int(payload["corpus_version"]),
        trace_id=str(payload["trace_id"]),
        usage=payload["usage"],
        abstained=bool(payload.get("abstained", False)),
    )


def rlm_answer_from_payload(payload: Mapping[str, Any]) -> RlmAnswer:
    return RlmAnswer(
        answer=str(payload["answer"]),
        citations=tuple(payload["citations"]),
        nodes=tuple(
            RlmNode(
                id=str(node["id"]),
                parent_id=str(node["parent_id"]) if node.get("parent_id") else None,
                depth=int(node["depth"]),
                route=str(node["route"]),
                input_chunk_count=int(node["input_chunk_count"]),
                input_tokens=int(node["input_tokens"]),
                selected_chunk_ids=tuple(node["selected_chunk_ids"]),
                cited_chunk_ids=tuple(node["cited_chunk_ids"]),
                answer_excerpt=str(node["answer_excerpt"]),
            )
            for node in payload["nodes"]
        ),
        corpus_version=int(payload["corpus_version"]),
        trace_id=str(payload["trace_id"]),
        usage=dict(payload["usage"]),
    )
