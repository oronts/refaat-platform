from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode


@dataclass(frozen=True, slots=True)
class RequestSpec:
    method: str
    path: str
    json: dict[str, Any] | None = None
    data: dict[str, Any] | None = None


def build_create_corpus(name: str, settings: dict[str, Any] | None = None) -> RequestSpec:
    return RequestSpec("POST", "/v1/corpora", json={"name": name, "settings": settings or {}})


def build_list_corpora() -> RequestSpec:
    return RequestSpec("GET", "/v1/corpora")


def build_list_derivations(corpus_id: str) -> RequestSpec:
    return RequestSpec("GET", f"/v1/corpora/{corpus_id}/derivations")


def build_verify_derivation(corpus_id: str) -> RequestSpec:
    return RequestSpec("GET", f"/v1/corpora/{corpus_id}/derivations/verify")


def build_compare_derivations(corpus_id: str, left: str, right: str) -> RequestSpec:
    return RequestSpec(
        "GET", f"/v1/corpora/{corpus_id}/derivations/compare?{urlencode({'left': left, 'right': right})}"
    )


def build_get_derivation(corpus_id: str, derivation_key: str) -> RequestSpec:
    return RequestSpec("GET", f"/v1/corpora/{corpus_id}/derivations/{derivation_key}")


def build_ingest_text(
    corpus_id: str,
    *,
    source_id: str,
    title: str,
    content: str,
    media_type: str = "text/plain",
    metadata: dict[str, Any] | None = None,
) -> RequestSpec:
    return RequestSpec(
        "POST",
        f"/v1/corpora/{corpus_id}/documents/text",
        json={
            "source_id": source_id,
            "title": title,
            "content": content,
            "media_type": media_type,
            "metadata": metadata or {},
        },
    )


def build_ingest_source(
    corpus_id: str,
    source: str,
    *,
    source_id: str | None = None,
    title: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> RequestSpec:
    return RequestSpec(
        "POST",
        f"/v1/corpora/{corpus_id}/documents/source",
        json={
            "source": source,
            "source_id": source_id,
            "title": title,
            "metadata": metadata or {},
        },
    )


def build_upload(
    corpus_id: str,
    filename: str,
    *,
    source_id: str | None = None,
    title: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> RequestSpec:
    return RequestSpec(
        "POST",
        f"/v1/corpora/{corpus_id}/documents/upload",
        data={
            "source_id": source_id or filename,
            "title": title or filename,
            "metadata_json": json.dumps(metadata or {}),
        },
    )


def build_erase_document(corpus_id: str, document_id: str) -> RequestSpec:
    return RequestSpec("POST", f"/v1/corpora/{corpus_id}/documents/{document_id}/erase")


def build_place_legal_hold(corpus_id: str, *, document_id: str | None = None, reason: str | None = None) -> RequestSpec:
    return RequestSpec(
        "POST",
        f"/v1/corpora/{corpus_id}/legal-holds",
        json={"document_id": document_id, "reason": reason},
    )


def build_list_legal_holds(corpus_id: str) -> RequestSpec:
    return RequestSpec("GET", f"/v1/corpora/{corpus_id}/legal-holds")


def build_lift_legal_hold(hold_id: str) -> RequestSpec:
    return RequestSpec("POST", f"/v1/legal-holds/{hold_id}/lift")


def build_retrieve(
    corpus_id: str,
    query: str,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    top_k: int | None = None,
    token_budget: int | None = None,
) -> RequestSpec:
    return RequestSpec(
        "POST",
        f"/v1/corpora/{corpus_id}/retrieve",
        json={
            "query": query,
            "user_id": user_id,
            "session_id": session_id,
            "top_k": top_k,
            "token_budget": token_budget,
        },
    )


def build_query(
    corpus_id: str,
    query: str,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    top_k: int | None = None,
    token_budget: int | None = None,
) -> RequestSpec:
    return RequestSpec(
        "POST",
        f"/v1/corpora/{corpus_id}/query",
        json={
            "query": query,
            "user_id": user_id,
            "session_id": session_id,
            "top_k": top_k,
            "token_budget": token_budget,
        },
    )


def build_stream_query(
    corpus_id: str,
    query: str,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    top_k: int | None = None,
    token_budget: int | None = None,
) -> RequestSpec:
    return RequestSpec(
        "POST",
        f"/v1/corpora/{corpus_id}/query/stream",
        json={
            "query": query,
            "user_id": user_id,
            "session_id": session_id,
            "top_k": top_k,
            "token_budget": token_budget,
        },
    )


def build_run_program(
    corpus_id: str,
    name: str,
    question: str,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
) -> RequestSpec:
    return RequestSpec(
        "POST",
        f"/v1/corpora/{corpus_id}/programs/{name}/run",
        json={"question": question, "user_id": user_id, "session_id": session_id},
    )


def build_stream_program(
    corpus_id: str,
    name: str,
    question: str,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
) -> RequestSpec:
    return RequestSpec(
        "POST",
        f"/v1/corpora/{corpus_id}/programs/{name}/run/stream",
        json={"question": question, "user_id": user_id, "session_id": session_id},
    )


def build_run_rlm(
    corpus_id: str,
    query: str,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    max_depth: int = 2,
    max_nodes: int = 16,
    branch_factor: int = 4,
    leaf_token_budget: int = 1_500,
) -> RequestSpec:
    return RequestSpec(
        "POST",
        f"/v1/corpora/{corpus_id}/rlm",
        json=_rlm_payload(
            query,
            user_id=user_id,
            session_id=session_id,
            max_depth=max_depth,
            max_nodes=max_nodes,
            branch_factor=branch_factor,
            leaf_token_budget=leaf_token_budget,
        ),
    )


def build_enqueue_rlm(
    corpus_id: str,
    query: str,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    max_depth: int = 2,
    max_nodes: int = 16,
    branch_factor: int = 4,
    leaf_token_budget: int = 1_500,
) -> RequestSpec:
    return RequestSpec(
        "POST",
        f"/v1/corpora/{corpus_id}/rlm/jobs",
        json=_rlm_payload(
            query,
            user_id=user_id,
            session_id=session_id,
            max_depth=max_depth,
            max_nodes=max_nodes,
            branch_factor=branch_factor,
            leaf_token_budget=leaf_token_budget,
        ),
    )


def build_get_job(job_id: str) -> RequestSpec:
    return RequestSpec("GET", f"/v1/jobs/{job_id}")


def build_create_session(
    corpus_id: str,
    *,
    user_id: str | None = None,
    goal: str | None = None,
) -> RequestSpec:
    return RequestSpec(
        "POST",
        "/v1/sessions",
        json={"corpus_id": corpus_id, "user_id": user_id, "goal": goal},
    )


def build_submit_feedback(
    corpus_id: str,
    *,
    rating: int,
    user_id: str | None = None,
    selected_chunk_ids: list[str] | None = None,
    note: str | None = None,
) -> RequestSpec:
    return RequestSpec(
        "POST",
        f"/v1/corpora/{corpus_id}/feedback",
        json={
            "rating": rating,
            "user_id": user_id,
            "selected_chunk_ids": selected_chunk_ids or [],
            "note": note,
        },
    )


def build_run_skill(
    name: str,
    corpus_id: str,
    query: str,
    arguments: dict[str, Any] | None = None,
) -> RequestSpec:
    return RequestSpec(
        "POST",
        f"/v1/skills/{name}/run",
        json={"corpus_id": corpus_id, "query": query, "arguments": arguments or {}},
    )


def _rlm_payload(
    query: str,
    *,
    user_id: str | None,
    session_id: str | None,
    max_depth: int,
    max_nodes: int,
    branch_factor: int,
    leaf_token_budget: int,
) -> dict[str, Any]:
    return {
        "query": query,
        "user_id": user_id,
        "session_id": session_id,
        "limits": {
            "max_depth": max_depth,
            "max_nodes": max_nodes,
            "branch_factor": branch_factor,
            "leaf_token_budget": leaf_token_budget,
        },
    }
