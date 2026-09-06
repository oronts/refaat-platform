from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from refaat.api.app import create_app
from refaat.domain.models import DocumentInput
from refaat.sdk import AsyncRefaatClient, RefaatClient

pytestmark = [pytest.mark.e2e, pytest.mark.asyncio]


async def test_async_sdk_queries_public_asgi_contract(
    runtime,
    settings,
    workspace_id: str,
    corpus_id: str,
    api_key: str,
) -> None:
    runtime.ingest(
        workspace_id,
        corpus_id,
        DocumentInput(
            source_id="async.txt",
            title="Async",
            media_type="text/plain",
            content=b"Async clients use the same grounded HTTP contract.",
        ),
    )
    transport = httpx.ASGITransport(app=create_app(settings, runtime))
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as client,
        AsyncRefaatClient("http://testserver", api_key, client=client) as sdk,
    ):
        retrieval = await sdk.retrieve(corpus_id, "What contract do async clients use?")
        answer = await sdk.query(corpus_id, "What contract do async clients use?")
    assert retrieval.documents
    assert answer.citations
    assert "grounded" in answer.answer.casefold()


async def test_async_sdk_runs_a_program(
    runtime,
    settings,
    workspace_id: str,
    corpus_id: str,
    api_key: str,
) -> None:
    runtime.ingest(
        workspace_id,
        corpus_id,
        DocumentInput(
            source_id="program.txt",
            title="Program",
            media_type="text/plain",
            content=b"Refund approval requires a signed authorization from finance.",
        ),
    )
    transport = httpx.ASGITransport(app=create_app(settings, runtime))
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as client,
        AsyncRefaatClient("http://testserver", api_key, client=client) as sdk,
    ):
        result = await sdk.run_program(corpus_id, "grounded-rag", "What does refund approval require?")
    assert result.program == "grounded-rag"
    assert result.citations
    assert result.grounded is True
    assert "authorization" in result.answer.casefold()


async def test_async_sdk_runs_and_enqueues_rlm(
    runtime,
    settings,
    workspace_id: str,
    corpus_id: str,
    api_key: str,
) -> None:
    for index in range(3):
        runtime.ingest(
            workspace_id,
            corpus_id,
            DocumentInput(
                source_id=f"async-rlm-{index}.txt",
                title=f"Async RLM {index}",
                media_type="text/plain",
                content=(
                    f"Async policy {index} requires canonical citations. "
                    + "Recursive context remains external to the model. " * 80
                ).encode(),
            ),
        )
    transport = httpx.ASGITransport(app=create_app(settings, runtime))
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as client,
        AsyncRefaatClient("http://testserver", api_key, client=client) as sdk,
    ):
        answer = await sdk.run_rlm(
            corpus_id,
            "What does the async policy require?",
            max_depth=2,
            max_nodes=7,
            branch_factor=3,
            leaf_token_budget=64,
        )
        job_id = await sdk.enqueue_rlm(
            corpus_id,
            "What remains external?",
            max_depth=1,
            max_nodes=4,
            branch_factor=2,
            leaf_token_budget=64,
        )
        job = await sdk.get_job(job_id)
    assert answer.citations
    assert len(answer.nodes) > 1
    assert job["status"] == "pending"


async def test_async_sdk_enqueues_and_reads_optimization(
    runtime,
    settings,
    workspace_id: str,
    corpus_id: str,
    api_key: str,
) -> None:
    from refaat.application.worker import DurableWorker

    for index in range(3):
        runtime.ingest(
            workspace_id,
            corpus_id,
            DocumentInput(
                source_id=f"async-opt-{index}.txt",
                title=f"Async Opt {index}",
                media_type="text/plain",
                content=b"Release approval requires two reviewers before deployment to production.",
            ),
        )
    transport = httpx.ASGITransport(app=create_app(settings, runtime))
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as client,
        AsyncRefaatClient("http://testserver", api_key, client=client) as sdk,
    ):
        run_id = await sdk.enqueue_optimization(
            "grounded-rag",
            corpus_id,
            [{"query": "release approval reviewers", "expected_terms": ["release"]}],
            search_space=[
                {"top_k": 2, "token_budget": None, "minimum_citation_coverage": 0.0},
                {"top_k": 6, "token_budget": None, "minimum_citation_coverage": 0.0},
            ],
        )
        assert DurableWorker(runtime).run_once() is True
        run = await sdk.get_optimization_run(run_id)

    assert run["status"] == "succeeded"
    assert run["candidate_total"] == 2
    assert run["best_artifact_id"]


async def test_async_sdk_drives_canary(
    runtime,
    settings,
    workspace_id: str,
    corpus_id: str,
    api_key: str,
) -> None:
    from refaat.programs import GroundedRagProgram, Tunables

    runtime.ingest(
        workspace_id,
        corpus_id,
        DocumentInput(
            source_id="async-canary.txt",
            title="Async Canary",
            media_type="text/plain",
            content=b"Release approval requires two reviewers before deployment to production.",
        ),
    )
    project = runtime.project(workspace="acme", corpus="handbook")
    base = GroundedRagProgram().artifact()
    candidate = GroundedRagProgram(tunables=Tunables(top_k=2)).artifact()
    project.registry.save(base)
    project.registry.save(candidate)
    project.registry.promote("grounded-rag", "staging", base.artifact_id)

    transport = httpx.ASGITransport(app=create_app(settings, runtime))
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as client,
        AsyncRefaatClient("http://testserver", api_key, client=client) as sdk,
    ):
        attached = await sdk.set_canary("grounded-rag", corpus_id, "staging", candidate.artifact_id, percent=100)
        assert attached["action"] == "set"
        served = await sdk.deploy_run("grounded-rag", corpus_id, "staging", "release approval reviewers", user_id="u1")
        status = await sdk.canary_status("grounded-rag", "staging")
        cleared = await sdk.clear_canary("grounded-rag", corpus_id, "staging")

    assert served["served"] == "candidate"
    assert served["result"]["citations"]
    assert status["canary"]["percent"] == 100
    assert cleared["action"] == "clear"


async def test_async_sdk_covers_full_sync_surface(
    runtime,
    settings,
    workspace_id: str,
    api_key: str,
    tmp_path: Path,
) -> None:
    upload_file = tmp_path / "upload.txt"
    upload_file.write_text("Uploaded evidence requires reviewer approval before release.")
    transport = httpx.ASGITransport(app=create_app(settings, runtime))
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as client,
        AsyncRefaatClient("http://testserver", api_key, client=client) as sdk,
    ):
        created = await sdk.create_corpus("async-parity")
        corpora = await sdk.list_corpora()
        await sdk.ingest_text(
            created["id"],
            source_id="policy.txt",
            title="Policy",
            content="Invoices require a purchase order and manager approval.",
        )
        uploaded = await sdk.upload(created["id"], upload_file)
        session_id = await sdk.create_session(created["id"], user_id="u1", goal="close the month")
        feedback = await sdk.submit_feedback(created["id"], rating=1, user_id="u1", note="clear")
        answer = await sdk.query(created["id"], "What do invoices require?")
        skill = await sdk.run_skill("corpus_health", created["id"], "manager approval")

    assert created["id"]
    assert any(entry["id"] == created["id"] for entry in corpora)
    assert uploaded["document_id"]
    assert session_id
    assert feedback is None
    assert answer.citations
    assert "approval" in answer.answer.casefold()
    assert skill["retrieved_chunks"] >= 1


async def test_async_query_matches_sync_query(
    runtime,
    settings,
    workspace_id: str,
    corpus_id: str,
    api_key: str,
) -> None:
    runtime.ingest(
        workspace_id,
        corpus_id,
        DocumentInput(
            source_id="parity.txt",
            title="Parity",
            media_type="text/plain",
            content=b"Refund approval requires a signed authorization from finance.",
        ),
    )
    question = "What does refund approval require?"
    transport = httpx.ASGITransport(app=create_app(settings, runtime))
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as async_httpx,
        AsyncRefaatClient("http://testserver", api_key, client=async_httpx) as async_sdk,
    ):
        async_answer = await async_sdk.query(corpus_id, question)
    with (
        TestClient(create_app(settings, runtime), raise_server_exceptions=False) as sync_httpx,
        RefaatClient("http://testserver", api_key, client=sync_httpx) as sync_sdk,
    ):
        sync_answer = sync_sdk.query(corpus_id, question)

    assert async_answer.answer == sync_answer.answer
    assert async_answer.citations == sync_answer.citations
