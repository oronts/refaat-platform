from __future__ import annotations

import pytest

from refaat.sdk import RefaatClient

pytestmark = pytest.mark.e2e


def test_sync_sdk_uses_public_http_contract(client) -> None:
    bootstrap = client.post(
        "/v1/bootstrap/workspaces",
        headers={"X-Bootstrap-Token": "bootstrap-test-token"},
        json={"slug": "sdk-acme", "name": "SDK Acme"},
    ).json()
    with RefaatClient("http://testserver", bootstrap["api_key"], client=client) as sdk:
        corpus = sdk.create_corpus("sdk-corpus")
        sdk.ingest_text(
            corpus["id"],
            source_id="policy.txt",
            title="Policy",
            content="Invoices require a purchase order and manager approval.",
        )
        retrieval = sdk.retrieve(corpus["id"], "What do invoices require?")
        answer = sdk.query(corpus["id"], "What do invoices require?")

    assert retrieval.documents
    assert retrieval.documents[0].document_id
    assert answer.citations
    assert "approval" in answer.answer.casefold()


def test_sync_sdk_runs_a_program(client) -> None:
    bootstrap = client.post(
        "/v1/bootstrap/workspaces",
        headers={"X-Bootstrap-Token": "bootstrap-test-token"},
        json={"slug": "sdk-prog", "name": "SDK Prog"},
    ).json()
    with RefaatClient("http://testserver", bootstrap["api_key"], client=client) as sdk:
        corpus = sdk.create_corpus("sdk-prog-corpus")
        sdk.ingest_text(
            corpus["id"],
            source_id="policy.txt",
            title="Policy",
            content="Invoices require a purchase order and manager approval.",
        )
        result = sdk.run_program(corpus["id"], "grounded-rag", "What do invoices require?")

    assert result.program == "grounded-rag"
    assert result.artifact_id
    assert result.citations
    assert result.grounded is True
    assert "approval" in result.answer.casefold()


def test_sync_sdk_runs_and_enqueues_rlm(client) -> None:
    bootstrap = client.post(
        "/v1/bootstrap/workspaces",
        headers={"X-Bootstrap-Token": "bootstrap-test-token"},
        json={"slug": "sdk-rlm", "name": "SDK RLM"},
    ).json()
    with RefaatClient("http://testserver", bootstrap["api_key"], client=client) as sdk:
        corpus = sdk.create_corpus("recursive")
        for index in range(4):
            sdk.ingest_text(
                corpus["id"],
                source_id=f"policy-{index}.txt",
                title=f"Policy {index}",
                content=(
                    f"Policy {index} requires verified evidence and owner approval. "
                    + "Supporting operational context. " * 80
                ),
            )
        answer = sdk.run_rlm(
            corpus["id"],
            "What do policies require?",
            max_depth=2,
            max_nodes=7,
            branch_factor=3,
            leaf_token_budget=64,
        )
        job_id = sdk.enqueue_rlm(
            corpus["id"],
            "Who approves the policy?",
            max_depth=1,
            max_nodes=4,
            branch_factor=2,
            leaf_token_budget=64,
        )

    assert answer.citations
    assert len(answer.nodes) > 1
    assert answer.usage["model_calls"] <= 7
    assert sdk.get_job(job_id)["status"] == "pending"


def test_sync_sdk_enqueues_and_reads_optimization(client, runtime) -> None:
    from refaat.application.worker import DurableWorker

    bootstrap = client.post(
        "/v1/bootstrap/workspaces",
        headers={"X-Bootstrap-Token": "bootstrap-test-token"},
        json={"slug": "sdk-opt", "name": "SDK Opt"},
    ).json()
    with RefaatClient("http://testserver", bootstrap["api_key"], client=client) as sdk:
        corpus = sdk.create_corpus("opt-corpus")
        for index in range(3):
            sdk.ingest_text(
                corpus["id"],
                source_id=f"policy-{index}.txt",
                title=f"Policy {index}",
                content="Release approval requires two reviewers before deployment to production.",
            )
        run_id = sdk.enqueue_optimization(
            "grounded-rag",
            corpus["id"],
            [{"query": "release approval reviewers", "expected_terms": ["release"]}],
            search_space=[
                {"top_k": 2, "token_budget": None, "minimum_citation_coverage": 0.0},
                {"top_k": 6, "token_budget": None, "minimum_citation_coverage": 0.0},
            ],
        )
        assert DurableWorker(runtime).run_once() is True
        run = sdk.get_optimization_run(run_id)

    assert run["status"] == "succeeded"
    assert run["candidate_total"] == 2
    assert run["best_artifact_id"]
