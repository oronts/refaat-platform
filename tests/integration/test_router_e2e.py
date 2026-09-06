from __future__ import annotations

import pytest

from refaat.application.runtime import Refaat, RefaatProject
from refaat.domain.models import DocumentInput, QueryRequest, QueryResponse
from refaat.evaluation import Dataset
from refaat.optimization import RouterOptimizer
from refaat.programs.program import Tunables
from refaat.programs.router import RouterProgram

pytestmark = pytest.mark.integration

_SIMPLE = "what is rollback"
_COMPLEX = "How does Kafka connect to Postgres, and how does Redis cache the results"


def _seed(runtime: Refaat) -> RefaatProject:
    project = runtime.project(workspace="acme", corpus="handbook")
    documents = {
        "rollback.txt": "Rollback restores the previously approved harness snapshot before deployment.",
        "kafka.txt": "Kafka connects to Postgres through a sink connector for durable storage.",
        "redis.txt": "Redis caches query results so repeated reads avoid recomputation.",
        "postgres.txt": "Postgres stores the canonical rows that Kafka streams from upstream topics.",
    }
    for source_id, text in documents.items():
        runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(source_id=source_id, title=source_id, media_type="text/plain", content=text.encode()),
        )
    return project


def _spy_query(runtime: Refaat) -> list[QueryRequest]:
    captured: list[QueryRequest] = []
    original = runtime.query

    def spy(request: QueryRequest) -> QueryResponse:
        captured.append(request)
        return original(request)

    runtime.query = spy  # type: ignore[method-assign]
    return captured


def test_router_routes_complex_and_simple_differently(runtime: Refaat) -> None:
    project = _seed(runtime)
    captured = _spy_query(runtime)
    program = RouterProgram()

    simple = program.execute(project, inputs={"question": _SIMPLE}, user_id=None, session_id=None)
    complex_result = program.execute(project, inputs={"question": _COMPLEX}, user_id=None, session_id=None)

    assert captured[0].fusion == "linear"
    assert captured[0].query_expansion is False
    assert captured[0].autocut_enabled is True
    assert captured[1].fusion == "rrf"
    assert captured[1].query_expansion is True
    assert captured[1].autocut_enabled is False
    assert captured[0].router_id is not None
    assert captured[0].router_id.startswith("router:")

    for result in (simple, complex_result):
        assert result.grounded
        assert result.citations
        stored = {
            chunk.id
            for chunk in runtime.store.get_chunks(
                project.workspace_id, project.corpus_id, [c.chunk_id for c in result.citations]
            )
        }
        assert {citation.chunk_id for citation in result.citations} <= stored


def test_router_forwards_full_inner_tunables(runtime: Refaat) -> None:
    project = _seed(runtime)
    captured = _spy_query(runtime)
    program = RouterProgram(tunables=Tunables(top_k=7, token_budget=321))

    program.execute(project, inputs={"question": _SIMPLE}, user_id=None, session_id=None)

    assert captured[0].top_k == 7
    assert captured[0].token_budget == 321


def test_router_optimizer_emits_distinct_candidates_without_mutation(runtime: Refaat) -> None:
    project = _seed(runtime)
    program = RouterProgram()
    dataset = Dataset.from_records(
        "router",
        [
            {"query": _SIMPLE, "expected_terms": ["rollback"]},
            {"query": _COMPLEX, "expected_terms": ["kafka"]},
        ],
    )

    result = RouterOptimizer(project).optimize(program, dataset)

    assert result.program == "router"
    assert result.candidate_count == 3
    assert len({candidate.artifact_id for candidate in result.candidates}) == 3
    assert all(candidate.artifact_id.startswith("sha256:") for candidate in result.candidates)
    assert all("router" in candidate.tunables for candidate in result.candidates)
    scores = [(candidate.hard_constraints_passed, candidate.score) for candidate in result.candidates]
    assert scores == sorted(scores, reverse=True)
    # The active program is never mutated: its router tunable stays unset.
    assert program.tunables.router is None


def test_router_identity_isolates_cache_scope(runtime: Refaat) -> None:
    project = _seed(runtime)
    program = RouterProgram()

    first = program.execute(project, inputs={"question": _SIMPLE}, user_id=None, session_id=None)
    second = program.execute(project, inputs={"question": _SIMPLE}, user_id=None, session_id=None)
    # The routed path caches under its own scope: a repeat is an exact cache hit.
    assert "cag:exact" not in first.routes
    assert "cag:exact" in second.routes

    # Same effective retrieval regime but no router_id: it must NOT read the
    # router's cache entry, proving the defensive router_id fold isolates scopes.
    plain = project.ask(_SIMPLE, fusion="linear", query_expansion=False, autocut_enabled=True)
    assert "cag:exact" not in plain.routes
