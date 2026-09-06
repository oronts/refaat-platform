from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.e2e


def _parse_sse(text: str) -> list[dict]:
    events: list[dict] = []
    for block in text.strip().split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data:"):
                events.append(json.loads(line[len("data:") :].strip()))
    return events


def test_query_stream_emits_deltas_then_a_grounded_final(client: TestClient, api_key: str, corpus_id: str) -> None:
    headers = {"Authorization": f"Bearer {api_key}"}
    ingest = client.post(
        f"/v1/corpora/{corpus_id}/documents/text",
        headers=headers,
        json={
            "source_id": "release.md",
            "title": "Release policy",
            "content": "The Release Manager verifies the evidence before a production deployment is approved.",
        },
    )
    assert ingest.status_code in (200, 201), ingest.text

    response = client.post(
        f"/v1/corpora/{corpus_id}/query/stream",
        headers=headers,
        json={"query": "who verifies release evidence"},
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(response.text)
    assert any(event["type"] == "delta" for event in events)
    finals = [event for event in events if event["type"] == "final"]
    assert len(finals) == 1
    final = finals[0]
    assert final["answer"]
    assert final["citations"]
    assert final["cached"] is False
    assert final["replaced"] is False


def test_query_stream_requires_authentication(client: TestClient, corpus_id: str) -> None:
    response = client.post(f"/v1/corpora/{corpus_id}/query/stream", json={"query": "anything"})
    assert response.status_code == 401


def test_program_run_stream_emits_deltas_then_a_grounded_final(
    client: TestClient, api_key: str, corpus_id: str
) -> None:
    headers = {"Authorization": f"Bearer {api_key}"}
    ingest = client.post(
        f"/v1/corpora/{corpus_id}/documents/text",
        headers=headers,
        json={
            "source_id": "release.md",
            "title": "Release policy",
            "content": "The Release Manager verifies the evidence before a production deployment is approved.",
        },
    )
    assert ingest.status_code in (200, 201), ingest.text

    response = client.post(
        f"/v1/corpora/{corpus_id}/programs/grounded-rag/run/stream",
        headers=headers,
        json={"question": "who verifies release evidence"},
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(response.text)
    types = [event["type"] for event in events]
    assert "delta" in types
    finals = [event for event in events if event["type"] == "final"]
    assert len(finals) == 1
    assert finals[0]["answer"]
    assert finals[0]["citations"]


def test_program_run_stream_requires_authentication(client: TestClient, corpus_id: str) -> None:
    response = client.post(
        f"/v1/corpora/{corpus_id}/programs/grounded-rag/run/stream",
        json={"question": "anything"},
    )
    assert response.status_code == 401


def test_program_run_stream_surfaces_a_mid_stream_failure_as_a_terminal_error(
    client: TestClient, api_key: str, corpus_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from refaat.programs.program import GroundedRagProgram

    def _failing_stream(
        self: object,
        project: object,
        *,
        inputs: object,
        user_id: object,
        session_id: object,
        principal: object = None,
    ) -> object:
        yield {"type": "delta", "text": "partial"}
        raise RuntimeError("generator failed mid-stream")

    monkeypatch.setattr(GroundedRagProgram, "stream", _failing_stream)

    headers = {"Authorization": f"Bearer {api_key}"}
    response = client.post(
        f"/v1/corpora/{corpus_id}/programs/grounded-rag/run/stream",
        headers=headers,
        json={"question": "anything"},
    )
    # The status line is committed with the first flushed byte, so a mid-stream
    # failure cannot become a non-200; it must arrive as a terminal error event.
    assert response.status_code == 200, response.text

    blocks = [block for block in response.text.strip().split("\n\n") if block]
    assert blocks[0].startswith("event: delta")
    last = blocks[-1]
    assert last.startswith("event: error")
    data_line = next(line for line in last.splitlines() if line.startswith("data:"))
    payload = json.loads(data_line[len("data:") :].strip())
    assert payload["type"] == "error"


def test_program_run_stream_tool_timeout_emits_a_first_class_terminal_error(
    client: TestClient, api_key: str, corpus_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from refaat.errors import ToolTimeoutError
    from refaat.programs.program import GroundedRagProgram

    def _timeout_stream(
        self: object,
        project: object,
        *,
        inputs: object,
        user_id: object,
        session_id: object,
        principal: object = None,
    ) -> object:
        # A tool timeout raises inside the governed loop, before the answer stream opens, so only a
        # tool_step (never a delta/final answer byte) precedes it.
        yield {"type": "tool_step", "tool": "writer", "authorized": True}
        raise ToolTimeoutError(outcome="unknown", side_effect="write", tool="writer", execution_terminated=False)

    monkeypatch.setattr(GroundedRagProgram, "stream", _timeout_stream)

    response = client.post(
        f"/v1/corpora/{corpus_id}/programs/grounded-rag/run/stream",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"question": "anything"},
    )
    assert response.status_code == 200, response.text

    events = _parse_sse(response.text)
    assert not any(event["type"] in {"delta", "final"} for event in events)
    error = events[-1]
    assert error["code"] == "tool_timeout"
    assert error["outcome"] == "unknown"
    assert error["side_effect"] == "write"
    assert error["tool"] == "writer"
    assert error["execution_terminated"] is False
    assert error["retryable"] is False
    assert error["trace_id"]
    assert set(error) == {
        "type",
        "code",
        "message",
        "outcome",
        "side_effect",
        "tool",
        "execution_terminated",
        "retryable",
        "trace_id",
    }
