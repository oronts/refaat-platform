from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Self

import httpx

from refaat.sdk._requests import (
    build_compare_derivations,
    build_create_corpus,
    build_create_session,
    build_enqueue_rlm,
    build_erase_document,
    build_get_derivation,
    build_get_job,
    build_ingest_source,
    build_ingest_text,
    build_lift_legal_hold,
    build_list_corpora,
    build_list_derivations,
    build_list_legal_holds,
    build_place_legal_hold,
    build_query,
    build_retrieve,
    build_run_program,
    build_run_rlm,
    build_run_skill,
    build_stream_program,
    build_stream_query,
    build_submit_feedback,
    build_upload,
    build_verify_derivation,
)
from refaat.sdk.client import decode_error
from refaat.sdk.models import (
    Answer,
    ProgramRunResult,
    RetrievalResult,
    RlmAnswer,
    answer_from_payload,
    program_result_from_payload,
    retrieval_result_from_payload,
    rlm_answer_from_payload,
)


class AsyncRefaatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(timeout=timeout)
        self.headers = {"Authorization": f"Bearer {api_key}"}

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def create_corpus(self, name: str, settings: dict[str, Any] | None = None) -> dict[str, Any]:
        spec = build_create_corpus(name, settings)
        payload: dict[str, Any] = await self._request(spec.method, spec.path, json=spec.json)
        return payload

    async def list_corpora(self) -> list[dict[str, Any]]:
        spec = build_list_corpora()
        payload: list[dict[str, Any]] = await self._request(spec.method, spec.path)
        return payload

    async def list_derivations(self, corpus_id: str) -> list[dict[str, Any]]:
        spec = build_list_derivations(corpus_id)
        payload: list[dict[str, Any]] = await self._request(spec.method, spec.path)
        return payload

    async def get_derivation(self, corpus_id: str, derivation_key: str) -> dict[str, Any]:
        spec = build_get_derivation(corpus_id, derivation_key)
        payload: dict[str, Any] = await self._request(spec.method, spec.path)
        return payload

    async def verify_derivation(self, corpus_id: str) -> dict[str, Any]:
        spec = build_verify_derivation(corpus_id)
        payload: dict[str, Any] = await self._request(spec.method, spec.path)
        return payload

    async def compare_derivations(self, corpus_id: str, left: str, right: str) -> dict[str, Any]:
        spec = build_compare_derivations(corpus_id, left, right)
        payload: dict[str, Any] = await self._request(spec.method, spec.path)
        return payload

    async def ingest_text(
        self,
        corpus_id: str,
        *,
        source_id: str,
        title: str,
        content: str,
        media_type: str = "text/plain",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        spec = build_ingest_text(
            corpus_id,
            source_id=source_id,
            title=title,
            content=content,
            media_type=media_type,
            metadata=metadata,
        )
        payload: dict[str, Any] = await self._request(spec.method, spec.path, json=spec.json)
        return payload

    async def ingest_source(
        self,
        corpus_id: str,
        source: str,
        *,
        source_id: str | None = None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        spec = build_ingest_source(corpus_id, source, source_id=source_id, title=title, metadata=metadata)
        payload: dict[str, Any] = await self._request(spec.method, spec.path, json=spec.json)
        return payload

    async def upload(
        self,
        corpus_id: str,
        path: Path,
        *,
        source_id: str | None = None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        spec = build_upload(corpus_id, path.name, source_id=source_id, title=title, metadata=metadata)
        with path.open("rb") as handle:
            payload: dict[str, Any] = await self._request(
                spec.method,
                spec.path,
                data=spec.data,
                files={"file": (path.name, handle)},
            )
            return payload

    async def erase_document(self, corpus_id: str, document_id: str) -> dict[str, Any]:
        spec = build_erase_document(corpus_id, document_id)
        payload: dict[str, Any] = await self._request(spec.method, spec.path)
        return payload

    async def place_legal_hold(
        self,
        corpus_id: str,
        *,
        document_id: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        spec = build_place_legal_hold(corpus_id, document_id=document_id, reason=reason)
        payload: dict[str, Any] = await self._request(spec.method, spec.path, json=spec.json)
        return payload

    async def list_legal_holds(self, corpus_id: str) -> dict[str, Any]:
        spec = build_list_legal_holds(corpus_id)
        payload: dict[str, Any] = await self._request(spec.method, spec.path)
        return payload

    async def lift_legal_hold(self, hold_id: str) -> dict[str, Any]:
        spec = build_lift_legal_hold(hold_id)
        payload: dict[str, Any] = await self._request(spec.method, spec.path)
        return payload

    async def retrieve(
        self,
        corpus_id: str,
        query: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        top_k: int | None = None,
        token_budget: int | None = None,
    ) -> RetrievalResult:
        spec = build_retrieve(
            corpus_id,
            query,
            user_id=user_id,
            session_id=session_id,
            top_k=top_k,
            token_budget=token_budget,
        )
        payload = await self._request(spec.method, spec.path, json=spec.json)
        return retrieval_result_from_payload(payload)

    async def query(
        self,
        corpus_id: str,
        query: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        top_k: int | None = None,
        token_budget: int | None = None,
    ) -> Answer:
        spec = build_query(
            corpus_id,
            query,
            user_id=user_id,
            session_id=session_id,
            top_k=top_k,
            token_budget=token_budget,
        )
        payload = await self._request(spec.method, spec.path, json=spec.json)
        return answer_from_payload(payload)

    async def stream_query(
        self,
        corpus_id: str,
        query: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        top_k: int | None = None,
        token_budget: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        spec = build_stream_query(
            corpus_id,
            query,
            user_id=user_id,
            session_id=session_id,
            top_k=top_k,
            token_budget=token_budget,
        )
        async with self.client.stream(
            spec.method,
            f"{self.base_url}{spec.path}",
            headers=self.headers,
            json=spec.json,
        ) as response:
            if response.is_error:
                await response.aread()
                raise decode_error(response)
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    payload = line[len("data:") :].strip()
                    if payload:
                        yield json.loads(payload)

    async def run_program(
        self,
        corpus_id: str,
        name: str,
        question: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> ProgramRunResult:
        spec = build_run_program(corpus_id, name, question, user_id=user_id, session_id=session_id)
        payload = await self._request(spec.method, spec.path, json=spec.json)
        return program_result_from_payload(payload)

    async def stream_program(
        self,
        corpus_id: str,
        name: str,
        question: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        spec = build_stream_program(corpus_id, name, question, user_id=user_id, session_id=session_id)
        async with self.client.stream(
            spec.method,
            f"{self.base_url}{spec.path}",
            headers=self.headers,
            json=spec.json,
        ) as response:
            if response.is_error:
                await response.aread()
                raise decode_error(response)
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    payload = line[len("data:") :].strip()
                    if payload:
                        yield json.loads(payload)

    async def run_rlm(
        self,
        corpus_id: str,
        query: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        max_depth: int = 2,
        max_nodes: int = 16,
        branch_factor: int = 4,
        leaf_token_budget: int = 1_500,
    ) -> RlmAnswer:
        spec = build_run_rlm(
            corpus_id,
            query,
            user_id=user_id,
            session_id=session_id,
            max_depth=max_depth,
            max_nodes=max_nodes,
            branch_factor=branch_factor,
            leaf_token_budget=leaf_token_budget,
        )
        payload = await self._request(spec.method, spec.path, json=spec.json)
        return rlm_answer_from_payload(payload)

    async def enqueue_rlm(
        self,
        corpus_id: str,
        query: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        max_depth: int = 2,
        max_nodes: int = 16,
        branch_factor: int = 4,
        leaf_token_budget: int = 1_500,
    ) -> str:
        spec = build_enqueue_rlm(
            corpus_id,
            query,
            user_id=user_id,
            session_id=session_id,
            max_depth=max_depth,
            max_nodes=max_nodes,
            branch_factor=branch_factor,
            leaf_token_budget=leaf_token_budget,
        )
        payload = await self._request(spec.method, spec.path, json=spec.json)
        return str(payload["id"])

    async def get_job(self, job_id: str) -> dict[str, Any]:
        spec = build_get_job(job_id)
        payload: dict[str, Any] = await self._request(spec.method, spec.path)
        return payload

    async def enqueue_optimization(
        self,
        name: str,
        corpus_id: str,
        cases: list[dict[str, Any]],
        *,
        search_space: list[dict[str, Any]] | None = None,
        budget: int | None = None,
    ) -> str:
        body: dict[str, Any] = {"corpus_id": corpus_id, "cases": cases}
        if search_space is not None:
            body["search_space"] = search_space
        if budget is not None:
            body["budget"] = budget
        payload = await self._request("POST", f"/v1/programs/{name}/optimizations", json=body)
        return str(payload["id"])

    async def get_optimization_run(self, run_id: str) -> dict[str, Any]:
        payload: dict[str, Any] = await self._request("GET", f"/v1/optimizations/{run_id}")
        return payload

    async def set_canary(
        self,
        name: str,
        corpus_id: str,
        environment: str,
        candidate_artifact_id: str,
        *,
        percent: int = 0,
        compare: bool = True,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = await self._request(
            "POST",
            f"/v1/corpora/{corpus_id}/programs/{name}/canary",
            json={
                "environment": environment,
                "candidate_artifact_id": candidate_artifact_id,
                "percent": percent,
                "compare": compare,
            },
        )
        return payload

    async def clear_canary(self, name: str, corpus_id: str, environment: str) -> dict[str, Any]:
        payload: dict[str, Any] = await self._request(
            "DELETE",
            f"/v1/corpora/{corpus_id}/programs/{name}/canary",
            params={"environment": environment},
        )
        return payload

    async def canary_status(self, name: str, environment: str) -> dict[str, Any]:
        payload: dict[str, Any] = await self._request("GET", f"/v1/programs/{name}/canary/{environment}")
        return payload

    async def deploy_run(
        self,
        name: str,
        corpus_id: str,
        environment: str,
        question: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = await self._request(
            "POST",
            f"/v1/corpora/{corpus_id}/programs/{name}/deployments/{environment}/run",
            json={"question": question, "user_id": user_id, "session_id": session_id},
        )
        return payload

    async def create_session(
        self,
        corpus_id: str,
        *,
        user_id: str | None = None,
        goal: str | None = None,
    ) -> str:
        spec = build_create_session(corpus_id, user_id=user_id, goal=goal)
        payload = await self._request(spec.method, spec.path, json=spec.json)
        return str(payload["id"])

    async def submit_feedback(
        self,
        corpus_id: str,
        *,
        rating: int,
        user_id: str | None = None,
        selected_chunk_ids: list[str] | None = None,
        note: str | None = None,
    ) -> None:
        spec = build_submit_feedback(
            corpus_id,
            rating=rating,
            user_id=user_id,
            selected_chunk_ids=selected_chunk_ids,
            note=note,
        )
        await self._request(spec.method, spec.path, json=spec.json)

    async def run_skill(
        self,
        name: str,
        corpus_id: str,
        query: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        spec = build_run_skill(name, corpus_id, query, arguments)
        payload: dict[str, Any] = await self._request(spec.method, spec.path, json=spec.json)
        return payload

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = await self.client.request(
            method,
            f"{self.base_url}{path}",
            headers=self.headers,
            **kwargs,
        )
        if response.is_error:
            raise decode_error(response)
        return None if response.status_code == 204 else response.json()
