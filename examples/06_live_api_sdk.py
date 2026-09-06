from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from _shared import bootstrap_workspace, emit, live_api_server

from refaat.sdk import AsyncRefaatClient, RefaatClient


async def async_query(base_url: str, api_key: str, corpus_id: str) -> dict[str, object]:
    async with AsyncRefaatClient(base_url, api_key) as client:
        answer = await client.query(corpus_id, "Who verifies release evidence?")
        retrieval = await client.retrieve(corpus_id, "Release Manager and rollback plan")
        stream_events = [event async for event in client.stream_query(corpus_id, "Who verifies release evidence?")]
        stream_final = next(event for event in stream_events if event["type"] == "final")
        return {
            "answer": answer.answer,
            "citations": list(answer.citations),
            "retrieval_routes": list(retrieval.routes),
            "retrieved_documents": len(retrieval.documents),
            "stream_delta_count": sum(1 for event in stream_events if event["type"] == "delta"),
            "stream_answer_grounded": bool(stream_final["citations"]),
        }


def main() -> None:
    with TemporaryDirectory(prefix="refaat-live-api-") as directory:
        root = Path(directory)
        with live_api_server(root) as (base_url, bootstrap_token):
            bootstrap = bootstrap_workspace(base_url, bootstrap_token)
            with RefaatClient(base_url, bootstrap["api_key"]) as client:
                corpus = client.create_corpus("release-handbook")
                corpus_id = str(corpus["id"])
                client.ingest_text(
                    corpus_id,
                    source_id="release.md",
                    title="Release policy",
                    content=(
                        "A Production Release requires peer review, automated tests, and a rollback plan. "
                        "The Release Manager verifies the evidence before deployment."
                    ),
                    media_type="text/markdown",
                )
                first = client.query(corpus_id, "What does a Production Release require?")
                cached = client.query(corpus_id, "What does a Production Release require?")
                recursive = client.run_rlm(
                    corpus_id,
                    "Summarize every release control.",
                    max_depth=1,
                    max_nodes=4,
                    branch_factor=2,
                    leaf_token_budget=128,
                )
                session_id = client.create_session(
                    corpus_id,
                    user_id="api-user",
                    goal="Review releases",
                )
                session_answer = client.query(
                    corpus_id,
                    "Who verifies the evidence?",
                    user_id="api-user",
                    session_id=session_id,
                )
                stream_events = list(client.stream_query(corpus_id, "What does a Production Release require?"))
                stream_deltas = [event for event in stream_events if event["type"] == "delta"]
                stream_final = next(event for event in stream_events if event["type"] == "final")

            async_result = asyncio.run(async_query(base_url, bootstrap["api_key"], corpus_id))

            assert first.citations
            assert cached.cached
            assert recursive.citations
            assert session_answer.citations
            assert async_result["retrieved_documents"]
            assert async_result["stream_delta_count"]
            assert stream_deltas
            assert stream_final["citations"]

            emit(
                {
                    "base_url": base_url,
                    "workspace_id": bootstrap["workspace_id"],
                    "corpus_id": corpus_id,
                    "first_cached": first.cached,
                    "second_cached": cached.cached,
                    "cache_route": cached.routes[0],
                    "rlm_nodes": len(recursive.nodes),
                    "session_id": session_id,
                    "stream_delta_count": len(stream_deltas),
                    "stream_answer_grounded": bool(stream_final["citations"]),
                    "async": async_result,
                }
            )


if __name__ == "__main__":
    main()
