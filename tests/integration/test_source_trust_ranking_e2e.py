"""End-to-end proof of the bounded source-trust down-rank.

Low-trust evidence is penalized so it cannot dominate trusted evidence, but the penalty
is a soft re-rank: a low-trust chunk stays retrievable and citable. The default weight of
0.0 applies no penalty and keeps the ranking, routes, and answer-cache identity byte-identical.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.config import Settings
from refaat.domain.models import DocumentInput
from refaat.programs.program import GroundedRagProgram

pytestmark = pytest.mark.integration

PINNED_GROUNDED_RAG_ID = "sha256:b76d73f8d956bcd38ccc9d4c24372b07b387647a189f3a9e300cded186febbb4"

CONTENT = b"Rollback restores the prior artifact atomically after a failed release."
QUERY = "how does rollback restore the prior artifact"


def _settings(tmp_path: Path, weight: float, *, name: str = "refaat") -> Settings:
    return Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path / f'{name}.db'}",
        auto_create_schema=True,
        blob_root=tmp_path / f"{name}-blobs",
        connector_file_root=tmp_path / f"{name}-connectors",
        skill_root=tmp_path / f"{name}-skills",
        api_key_pepper="test-pepper-with-enough-entropy",
        bootstrap_token="bootstrap-test-token",
        default_chunk_tokens=64,
        chunk_overlap_tokens=12,
        query_token_budget=600,
        source_trust_weight=weight,
        log_json=False,
    )


def _build(settings: Settings) -> Iterator[Refaat]:
    runtime = RuntimeBuilder(settings).build()
    try:
        yield runtime
    finally:
        runtime.close()


def _ingest(runtime: Refaat, project, source_id: str, source_trust: str | None) -> None:
    metadata = {"source_trust": source_trust} if source_trust is not None else {}
    runtime.ingest(
        project.workspace_id,
        project.corpus_id,
        DocumentInput(
            source_id=source_id,
            title=source_id.removesuffix(".txt"),
            media_type="text/plain",
            content=CONTENT,
            metadata=metadata,
        ),
    )


def test_external_ranks_below_authoritative_when_weighted(tmp_path: Path) -> None:
    for runtime in _build(_settings(tmp_path, 0.3)):
        project = runtime.project(workspace="trust", corpus="handbook")
        _ingest(runtime, project, "trusted.txt", "AUTHORITATIVE")
        _ingest(runtime, project, "external.txt", "EXTERNAL")

        packet = project.retrieve(QUERY)
        order = [hit.chunk.metadata["source_id"] for hit in packet.hits]

        assert "source_trust" in packet.routes
        # Both are retrieved: the penalty reorders, it never excludes.
        assert "trusted.txt" in order
        assert "external.txt" in order
        assert order.index("trusted.txt") < order.index("external.txt")


def test_low_trust_chunk_alone_is_still_retrievable_and_citable(tmp_path: Path) -> None:
    for runtime in _build(_settings(tmp_path, 0.3)):
        project = runtime.project(workspace="trust-solo", corpus="handbook")
        _ingest(runtime, project, "external.txt", "EXTERNAL")

        packet = project.retrieve(QUERY)
        assert [hit.chunk.metadata["source_id"] for hit in packet.hits] == ["external.txt"]

        answer = project.ask(QUERY)
        assert answer.citations
        stored = {chunk.id: chunk for chunk in runtime.store.list_chunks(project.workspace_id, project.corpus_id)}
        for citation in answer.citations:
            assert citation.chunk_id in stored


def test_untagged_ranks_below_authoritative_when_weighted(tmp_path: Path) -> None:
    # An untagged document defaults to UNVERIFIED, not AUTHORITATIVE, so trust must be
    # asserted: a tagged-authoritative document outranks an untagged one when the weight is on.
    for runtime in _build(_settings(tmp_path, 0.3)):
        project = runtime.project(workspace="untagged-mixed", corpus="handbook")
        _ingest(runtime, project, "trusted.txt", "AUTHORITATIVE")
        _ingest(runtime, project, "untagged.txt", None)

        packet = project.retrieve(QUERY)
        order = [hit.chunk.metadata["source_id"] for hit in packet.hits]

        assert "source_trust" in packet.routes
        assert "trusted.txt" in order
        assert "untagged.txt" in order
        assert order.index("trusted.txt") < order.index("untagged.txt")


def test_disabled_weight_is_byte_identical(tmp_path: Path) -> None:
    assert GroundedRagProgram().artifact().artifact_id == PINNED_GROUNDED_RAG_ID
    for runtime in _build(_settings(tmp_path, 0.0)):
        project = runtime.project(workspace="trust-off", corpus="handbook")
        _ingest(runtime, project, "trusted.txt", "AUTHORITATIVE")
        _ingest(runtime, project, "external.txt", "EXTERNAL")

        packet = project.retrieve(QUERY)
        assert "source_trust" not in packet.routes

        first = project.ask(QUERY)
        second = project.ask(QUERY)
        assert first.cached is False
        assert second.cached is True


def test_enabling_the_weight_scopes_the_answer_cache(tmp_path: Path) -> None:
    """A weight change folds into the answer-cache identity, so it never serves a stale entry.

    An answer cached under the disabled weight is not returned once the weight is enabled: the
    weighted run takes a fresh cache scope, mirroring how the abstain floor scopes the cache."""
    for runtime in _build(_settings(tmp_path, 0.0, name="shared")):
        project = runtime.project(workspace="trust-scope", corpus="handbook")
        _ingest(runtime, project, "trusted.txt", "AUTHORITATIVE")
        _ingest(runtime, project, "external.txt", "EXTERNAL")
        assert project.ask(QUERY).cached is False
        assert project.ask(QUERY).cached is True

    for runtime in _build(_settings(tmp_path, 0.3, name="shared")):
        project = runtime.project(workspace="trust-scope", corpus="handbook")
        weighted = project.ask(QUERY)

    assert weighted.cached is False
