from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from refaat.sdk import AsyncRefaatClient, RefaatClient
from refaat.sdk._requests import (
    RequestSpec,
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
from refaat.sdk.models import (
    answer_from_payload,
    program_result_from_payload,
    retrieval_result_from_payload,
)


def test_derivation_request_builders() -> None:
    assert build_list_derivations("c") == RequestSpec("GET", "/v1/corpora/c/derivations")
    assert build_verify_derivation("c") == RequestSpec("GET", "/v1/corpora/c/derivations/verify")
    assert build_get_derivation("c", "abc") == RequestSpec("GET", "/v1/corpora/c/derivations/abc")
    assert build_compare_derivations("c", "L", "R") == RequestSpec(
        "GET", "/v1/corpora/c/derivations/compare?left=L&right=R"
    )


CORPUS = "corpus-1"
JOB = "job-1"
SKILL = "corpus_health"
PROGRAM = "grounded-rag"
UPLOAD_INDEX = 4


def _recording_transport(sink: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        request.read()
        sink.append(request)
        return httpx.Response(200, json={})

    return httpx.MockTransport(handler)


def _drive_sync(sdk: RefaatClient, upload_path: Path) -> None:
    calls = [
        lambda: sdk.create_corpus("mycorpus", {"lang": "en"}),
        lambda: sdk.list_corpora(),
        lambda: sdk.ingest_text(CORPUS, source_id="s1", title="T", content="body", metadata={"k": "v"}),
        lambda: sdk.ingest_source(CORPUS, "https://ex/a", source_id="s2", title="T2", metadata={"k": 1}),
        lambda: sdk.upload(CORPUS, upload_path, source_id="s3", title="T3", metadata={"m": 2}),
        lambda: sdk.erase_document(CORPUS, "doc-1"),
        lambda: sdk.place_legal_hold(CORPUS, document_id="doc-1", reason="litigation"),
        lambda: sdk.list_legal_holds(CORPUS),
        lambda: sdk.lift_legal_hold("hold-1"),
        lambda: sdk.retrieve(CORPUS, "q", user_id="u", session_id="sess", top_k=5, token_budget=100),
        lambda: sdk.query(CORPUS, "q", user_id="u", session_id="sess", top_k=5, token_budget=100),
        lambda: list(sdk.stream_query(CORPUS, "q", user_id="u", session_id="sess", top_k=5, token_budget=100)),
        lambda: list(sdk.stream_program(CORPUS, PROGRAM, "q", user_id="u", session_id="sess")),
        lambda: sdk.run_program(CORPUS, PROGRAM, "q", user_id="u", session_id="sess"),
        lambda: sdk.run_rlm(
            CORPUS,
            "q",
            user_id="u",
            session_id="sess",
            max_depth=3,
            max_nodes=9,
            branch_factor=2,
            leaf_token_budget=200,
        ),
        lambda: sdk.enqueue_rlm(
            CORPUS, "q", user_id="u", session_id="sess", max_depth=1, max_nodes=4, branch_factor=2, leaf_token_budget=64
        ),
        lambda: sdk.get_job(JOB),
        lambda: sdk.create_session(CORPUS, user_id="u", goal="g"),
        lambda: sdk.submit_feedback(CORPUS, rating=1, user_id="u", selected_chunk_ids=["a"], note="n"),
        lambda: sdk.run_skill(SKILL, CORPUS, "q", {"arg": 1}),
        lambda: sdk.enqueue_optimization(PROGRAM, CORPUS, [{"question": "q"}], search_space=[{"a": 1}], budget=2),
        lambda: sdk.get_optimization_run("run-1"),
        lambda: sdk.set_canary(PROGRAM, CORPUS, "production", "artifact-1", percent=10, compare=True),
        lambda: sdk.clear_canary(PROGRAM, CORPUS, "production"),
        lambda: sdk.canary_status(PROGRAM, "production"),
        lambda: sdk.deploy_run(PROGRAM, CORPUS, "production", "q", user_id="u", session_id="sess"),
        lambda: sdk.list_derivations(CORPUS),
        lambda: sdk.get_derivation(CORPUS, "dk-1"),
        lambda: sdk.verify_derivation(CORPUS),
        lambda: sdk.compare_derivations(CORPUS, "dk-1", "dk-2"),
    ]
    for call in calls:
        with contextlib.suppress(Exception):
            call()


async def _drive_async(sdk: AsyncRefaatClient, upload_path: Path) -> None:
    async def _stream() -> list[dict[str, Any]]:
        return [
            event
            async for event in sdk.stream_query(CORPUS, "q", user_id="u", session_id="sess", top_k=5, token_budget=100)
        ]

    async def _stream_program() -> list[dict[str, Any]]:
        return [event async for event in sdk.stream_program(CORPUS, PROGRAM, "q", user_id="u", session_id="sess")]

    calls = [
        lambda: sdk.create_corpus("mycorpus", {"lang": "en"}),
        lambda: sdk.list_corpora(),
        lambda: sdk.ingest_text(CORPUS, source_id="s1", title="T", content="body", metadata={"k": "v"}),
        lambda: sdk.ingest_source(CORPUS, "https://ex/a", source_id="s2", title="T2", metadata={"k": 1}),
        lambda: sdk.upload(CORPUS, upload_path, source_id="s3", title="T3", metadata={"m": 2}),
        lambda: sdk.erase_document(CORPUS, "doc-1"),
        lambda: sdk.place_legal_hold(CORPUS, document_id="doc-1", reason="litigation"),
        lambda: sdk.list_legal_holds(CORPUS),
        lambda: sdk.lift_legal_hold("hold-1"),
        lambda: sdk.retrieve(CORPUS, "q", user_id="u", session_id="sess", top_k=5, token_budget=100),
        lambda: sdk.query(CORPUS, "q", user_id="u", session_id="sess", top_k=5, token_budget=100),
        _stream,
        _stream_program,
        lambda: sdk.run_program(CORPUS, PROGRAM, "q", user_id="u", session_id="sess"),
        lambda: sdk.run_rlm(
            CORPUS,
            "q",
            user_id="u",
            session_id="sess",
            max_depth=3,
            max_nodes=9,
            branch_factor=2,
            leaf_token_budget=200,
        ),
        lambda: sdk.enqueue_rlm(
            CORPUS, "q", user_id="u", session_id="sess", max_depth=1, max_nodes=4, branch_factor=2, leaf_token_budget=64
        ),
        lambda: sdk.get_job(JOB),
        lambda: sdk.create_session(CORPUS, user_id="u", goal="g"),
        lambda: sdk.submit_feedback(CORPUS, rating=1, user_id="u", selected_chunk_ids=["a"], note="n"),
        lambda: sdk.run_skill(SKILL, CORPUS, "q", {"arg": 1}),
        lambda: sdk.enqueue_optimization(PROGRAM, CORPUS, [{"question": "q"}], search_space=[{"a": 1}], budget=2),
        lambda: sdk.get_optimization_run("run-1"),
        lambda: sdk.set_canary(PROGRAM, CORPUS, "production", "artifact-1", percent=10, compare=True),
        lambda: sdk.clear_canary(PROGRAM, CORPUS, "production"),
        lambda: sdk.canary_status(PROGRAM, "production"),
        lambda: sdk.deploy_run(PROGRAM, CORPUS, "production", "q", user_id="u", session_id="sess"),
        lambda: sdk.list_derivations(CORPUS),
        lambda: sdk.get_derivation(CORPUS, "dk-1"),
        lambda: sdk.verify_derivation(CORPUS),
        lambda: sdk.compare_derivations(CORPUS, "dk-1", "dk-2"),
    ]
    for call in calls:
        with contextlib.suppress(Exception):
            await call()


@pytest.mark.asyncio
async def test_sync_and_async_emit_byte_identical_requests(tmp_path: Path) -> None:
    upload_path = tmp_path / "upload.txt"
    upload_path.write_text("evidence")

    sync_requests: list[httpx.Request] = []
    async_requests: list[httpx.Request] = []

    with RefaatClient(
        "http://testserver", "key", client=httpx.Client(transport=_recording_transport(sync_requests))
    ) as sdk:
        _drive_sync(sdk, upload_path)
    async with AsyncRefaatClient(
        "http://testserver", "key", client=httpx.AsyncClient(transport=_recording_transport(async_requests))
    ) as sdk_async:
        await _drive_async(sdk_async, upload_path)

    # Every public request-emitting method must be driven, so a body divergence in any of
    # them fails the byte-for-byte comparison below. Tying the count to the client surface
    # (minus the lifecycle method) means a newly added method that is not driven fails here.
    expected = len(_public_methods(RefaatClient) - {"close"})
    assert len(sync_requests) == expected
    assert len(async_requests) == expected
    for index, (sync_req, async_req) in enumerate(zip(sync_requests, async_requests, strict=True)):
        assert sync_req.method == async_req.method
        assert sync_req.url.path == async_req.url.path
        if index == UPLOAD_INDEX:
            # multipart boundaries are randomized per httpx request; the shared
            # build_upload data dict is asserted byte-for-byte separately below.
            assert async_req.headers["content-type"].startswith("multipart/form-data")
            continue
        assert sync_req.content == async_req.content


def _public_methods(cls: type) -> set[str]:
    return {name for name, value in vars(cls).items() if callable(value) and not name.startswith("_")}


def test_sync_and_async_expose_the_same_public_methods() -> None:
    # A method added to one client but not the other (or renamed) diverges the surface;
    # run_program is included here, so the two clients cannot drift apart silently.
    assert _public_methods(RefaatClient) == _public_methods(AsyncRefaatClient)
    assert "run_program" in _public_methods(RefaatClient)


def test_request_specs_match_hand_written_expectations() -> None:
    assert build_create_corpus("c", {"a": 1}) == RequestSpec(
        "POST", "/v1/corpora", json={"name": "c", "settings": {"a": 1}}
    )
    assert build_list_corpora() == RequestSpec("GET", "/v1/corpora")
    assert build_ingest_text(CORPUS, source_id="s", title="t", content="b") == RequestSpec(
        "POST",
        f"/v1/corpora/{CORPUS}/documents/text",
        json={"source_id": "s", "title": "t", "content": "b", "media_type": "text/plain", "metadata": {}},
    )
    assert build_ingest_source(CORPUS, "src", source_id="s") == RequestSpec(
        "POST",
        f"/v1/corpora/{CORPUS}/documents/source",
        json={"source": "src", "source_id": "s", "title": None, "metadata": {}},
    )
    assert build_upload(CORPUS, "file.txt", metadata={"m": 1}) == RequestSpec(
        "POST",
        f"/v1/corpora/{CORPUS}/documents/upload",
        data={"source_id": "file.txt", "title": "file.txt", "metadata_json": json.dumps({"m": 1})},
    )
    assert build_erase_document(CORPUS, "doc-1") == RequestSpec("POST", f"/v1/corpora/{CORPUS}/documents/doc-1/erase")
    assert build_place_legal_hold(CORPUS, document_id="doc-1", reason="r") == RequestSpec(
        "POST", f"/v1/corpora/{CORPUS}/legal-holds", json={"document_id": "doc-1", "reason": "r"}
    )
    assert build_list_legal_holds(CORPUS) == RequestSpec("GET", f"/v1/corpora/{CORPUS}/legal-holds")
    assert build_lift_legal_hold("hold-1") == RequestSpec("POST", "/v1/legal-holds/hold-1/lift")
    assert build_retrieve(CORPUS, "q", top_k=5) == RequestSpec(
        "POST",
        f"/v1/corpora/{CORPUS}/retrieve",
        json={"query": "q", "user_id": None, "session_id": None, "top_k": 5, "token_budget": None},
    )
    assert build_query(CORPUS, "q") == RequestSpec(
        "POST",
        f"/v1/corpora/{CORPUS}/query",
        json={"query": "q", "user_id": None, "session_id": None, "top_k": None, "token_budget": None},
    )
    assert build_stream_query(CORPUS, "q") == RequestSpec(
        "POST",
        f"/v1/corpora/{CORPUS}/query/stream",
        json={"query": "q", "user_id": None, "session_id": None, "top_k": None, "token_budget": None},
    )
    assert build_run_program(CORPUS, PROGRAM, "q") == RequestSpec(
        "POST",
        f"/v1/corpora/{CORPUS}/programs/{PROGRAM}/run",
        json={"question": "q", "user_id": None, "session_id": None},
    )
    assert build_stream_program(CORPUS, PROGRAM, "q") == RequestSpec(
        "POST",
        f"/v1/corpora/{CORPUS}/programs/{PROGRAM}/run/stream",
        json={"question": "q", "user_id": None, "session_id": None},
    )
    assert build_run_rlm(CORPUS, "q") == RequestSpec(
        "POST",
        f"/v1/corpora/{CORPUS}/rlm",
        json={
            "query": "q",
            "user_id": None,
            "session_id": None,
            "limits": {"max_depth": 2, "max_nodes": 16, "branch_factor": 4, "leaf_token_budget": 1500},
        },
    )
    assert build_enqueue_rlm(CORPUS, "q") == RequestSpec(
        "POST",
        f"/v1/corpora/{CORPUS}/rlm/jobs",
        json={
            "query": "q",
            "user_id": None,
            "session_id": None,
            "limits": {"max_depth": 2, "max_nodes": 16, "branch_factor": 4, "leaf_token_budget": 1500},
        },
    )
    assert build_get_job(JOB) == RequestSpec("GET", f"/v1/jobs/{JOB}")
    assert build_create_session(CORPUS, user_id="u") == RequestSpec(
        "POST", "/v1/sessions", json={"corpus_id": CORPUS, "user_id": "u", "goal": None}
    )
    assert build_submit_feedback(CORPUS, rating=1) == RequestSpec(
        "POST",
        f"/v1/corpora/{CORPUS}/feedback",
        json={"rating": 1, "user_id": None, "selected_chunk_ids": [], "note": None},
    )
    assert build_run_skill(SKILL, CORPUS, "q") == RequestSpec(
        "POST", f"/v1/skills/{SKILL}/run", json={"corpus_id": CORPUS, "query": "q", "arguments": {}}
    )


def test_retrieval_result_from_payload_reconstructs_dataclass() -> None:
    payload = {
        "query": "q",
        "hits": [
            {
                "chunk_id": "c",
                "document_id": "d",
                "text": "t",
                "score": 0.5,
                "metadata": {"a": 1},
                "reasons": ["r"],
                "semantic_score": 0.4,
                "lexical_score": 0.3,
                "graph_score": 0.2,
            }
        ],
        "routes": ["vector"],
        "graph_facts": ["f"],
        "token_count": 3,
        "session_summary": "prior turn summary",
    }
    result = retrieval_result_from_payload(payload)
    assert result.query == "q"
    assert result.documents[0].chunk_id == "c"
    assert result.documents[0].document_id == "d"
    assert result.documents[0].reasons == ("r",)
    assert result.documents[0].semantic_score == 0.4
    assert result.documents[0].lexical_score == 0.3
    assert result.documents[0].graph_score == 0.2
    assert result.routes == ("vector",)
    assert result.graph_facts == ("f",)
    assert result.token_count == 3
    assert result.session_summary == "prior turn summary"


def test_retrieval_result_from_payload_defaults_missing_score_fields() -> None:
    payload = {
        "query": "q",
        "hits": [{"chunk_id": "c", "document_id": "d", "text": "t", "score": 0.5}],
        "routes": ["vector"],
        "graph_facts": [],
        "token_count": 1,
    }
    result = retrieval_result_from_payload(payload)
    assert result.session_summary is None
    assert result.documents[0].semantic_score == 0.0
    assert result.documents[0].lexical_score == 0.0
    assert result.documents[0].graph_score == 0.0


def test_program_result_from_payload_reconstructs_dataclass() -> None:
    payload = {
        "program": "grounded-rag",
        "artifact_id": "artifact-1",
        "answer": "grounded answer",
        "citations": [{"chunk_id": "c", "document_id": "d", "label": "L"}],
        "routes": ["vector", "rerank"],
        "cached": False,
        "corpus_version": 4,
        "trace_id": "tid",
        "grounded": True,
        "validations": [{"validator": "grounding", "passed": True, "detail": ""}],
        "usage": {"model_calls": 0},
    }
    result = program_result_from_payload(payload)
    assert result.program == "grounded-rag"
    assert result.artifact_id == "artifact-1"
    assert result.answer == "grounded answer"
    assert result.citations == ({"chunk_id": "c", "document_id": "d", "label": "L"},)
    assert result.routes == ("vector", "rerank")
    assert result.cached is False
    assert result.corpus_version == 4
    assert result.trace_id == "tid"
    assert result.grounded is True
    assert result.validations == ({"validator": "grounding", "passed": True, "detail": ""},)
    assert result.usage == {"model_calls": 0}


def test_answer_from_payload_reconstructs_dataclass() -> None:
    payload = {
        "answer": "a",
        "citations": [{"chunk_id": "c"}],
        "routes": ["cag:exact"],
        "cached": True,
        "corpus_version": 7,
        "trace_id": "tid",
        "usage": {"model_calls": 1},
    }
    answer = answer_from_payload(payload)
    assert answer.answer == "a"
    assert answer.citations == ({"chunk_id": "c"},)
    assert answer.routes == ("cag:exact",)
    assert answer.cached is True
    assert answer.corpus_version == 7
    assert answer.trace_id == "tid"
    assert answer.usage == {"model_calls": 1}
