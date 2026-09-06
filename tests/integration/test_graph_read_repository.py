"""The GraphReadRepository seam: the store is the default, an injected repo is consulted on both hot
paths and scopes the answer cache, and the read contract holds for the store and a conformant stub.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.config import Settings
from refaat.conformance import GraphReadConformanceCase, graph_read_conformance, structural_conformance
from refaat.core.retrieval import AdaptiveRetriever
from refaat.core.text import tokens
from refaat.domain.models import DocumentInput, Entity, Relation
from refaat.infrastructure.search import LocalSearchIndex
from refaat.ports import GraphReadRepository

pytestmark = pytest.mark.integration

_DOCS = [
    DocumentInput(
        source_id="alpha.txt",
        title="Alpha",
        media_type="text/plain",
        content=(
            b"Northwind Logistics partnered with Berlin Freight to expand European shipping.\n\n"
            b"Northwind Logistics and Berlin Freight signed the shipping agreement."
        ),
    ),
    DocumentInput(
        source_id="beta.txt",
        title="Beta",
        media_type="text/plain",
        content=(
            b"Acme Corporation acquired Tokyo Robotics to enter the Japanese market.\n\n"
            b"Acme Corporation and Tokyo Robotics announced the merger."
        ),
    ),
]

_QUERY = "northwind logistics shipping partnership"


def _built(settings: Settings) -> Iterator[Refaat]:
    runtime = RuntimeBuilder(settings).build()
    try:
        yield runtime
    finally:
        runtime.close()


def _ingest(runtime: Refaat, ws: str, corpus: str) -> None:
    for doc in _DOCS:
        runtime.ingest(ws, corpus, doc)


class _RecordingRepo:
    """Delegates every read to an inner repo set after build, counting calls per method."""

    cache_id: str | None = None

    def __init__(self) -> None:
        self.inner: GraphReadRepository | None = None
        self.calls: dict[str, int] = {}

    def _hit(self, name: str) -> GraphReadRepository:
        self.calls[name] = self.calls.get(name, 0) + 1
        assert self.inner is not None
        return self.inner

    def reset(self) -> None:
        self.calls = {}

    def list_graph(self, workspace_id: str, corpus_id: str) -> tuple[list[Entity], list[Relation]]:
        return self._hit("list_graph").list_graph(workspace_id, corpus_id)

    def graph_neighborhood(
        self, workspace_id: str, corpus_id: str, *, term_candidates: frozenset[str], name_substrings: frozenset[str]
    ) -> tuple[list[Entity], list[Relation]]:
        return self._hit("graph_neighborhood").graph_neighborhood(
            workspace_id, corpus_id, term_candidates=term_candidates, name_substrings=name_substrings
        )

    def graph_index_present(self, workspace_id: str, corpus_id: str) -> bool:
        return self._hit("graph_index_present").graph_index_present(workspace_id, corpus_id)

    def max_entity_name_length(self, workspace_id: str, corpus_id: str) -> int:
        return self._hit("max_entity_name_length").max_entity_name_length(workspace_id, corpus_id)


class _CachedRepo(_RecordingRepo):
    cache_id = "gr:test"


def test_resolver_defaults_to_the_store(settings: Settings) -> None:
    builder = RuntimeBuilder(settings)
    sentinel = object()
    assert builder._graph_read_repository(sentinel) is sentinel  # type: ignore[arg-type]
    repo = _RecordingRepo()
    assert builder.with_graph_read_repository(repo)._graph_read_repository(sentinel) is repo  # type: ignore[arg-type]


def test_graph_cache_id_folds_only_for_an_injected_repo(settings: Settings) -> None:
    runtime = next(_built(settings))
    try:
        store = runtime.store
        index = LocalSearchIndex(store)
        default = AdaptiveRetriever(store, index, settings)
        assert default.graph_cache_id is None
        assert default._graph_repo is store
        injected = AdaptiveRetriever(store, index, settings, graph_read_repository=_CachedRepo())
        assert injected.graph_cache_id == "gr:test"
    finally:
        runtime.close()


def test_injected_repo_is_consulted_on_both_query_and_rlm(settings: Settings) -> None:
    recording = _RecordingRepo()
    runtime = RuntimeBuilder(settings).with_graph_read_repository(recording).build()
    try:
        recording.inner = runtime.store
        project = runtime.project(workspace="acme", corpus="graph")
        _ingest(runtime, project.workspace_id, project.corpus_id)
        stored_ids = {chunk.id for chunk in runtime.store.list_chunks(project.workspace_id, project.corpus_id)}

        recording.reset()
        answer = project.ask(_QUERY)
        assert recording.calls.get("list_graph", 0) > 0
        assert answer.answer
        assert all(citation.chunk_id in stored_ids for citation in answer.citations)

        recording.reset()
        project.rlm(_QUERY)
        assert recording.calls.get("list_graph", 0) > 0
    finally:
        runtime.close()


def test_graph_read_conformance_passes_for_the_store(settings: Settings) -> None:
    # Seed an INDEXED corpus (flag on) so graph_index_present is true and the token branch is exercised.
    enabled = settings.model_copy(update={"graph_term_index_enabled": True})
    runtime = RuntimeBuilder(enabled).build()
    try:
        project = runtime.project(workspace="acme", corpus="graph")
        _ingest(runtime, project.workspace_id, project.corpus_id)
        assert runtime.store.graph_index_present(project.workspace_id, project.corpus_id)
        entities, _ = runtime.store.list_graph(project.workspace_id, project.corpus_id)
        named = next(entity.normalized_name for entity in entities if entity.normalized_name)
        case = GraphReadConformanceCase(
            repo=runtime.store, workspace_id=project.workspace_id, corpus_id=project.corpus_id, query=named
        )
        assert graph_read_conformance(case) == []
    finally:
        runtime.close()


def _stub_graph() -> tuple[list[Entity], list[Relation]]:
    # e2 is a token-only match with no relation to it, so a name-only backend's drop is not frontier-masked.
    entities = [
        Entity(id="e1", corpus_id="c", document_id="d", name="Platform", normalized_name="platform", kind="TOPIC"),
        Entity(
            id="e2",
            corpus_id="c",
            document_id="d",
            name="Platform Engineering",
            normalized_name="platform engineering",
            kind="TOPIC",
        ),
        Entity(id="e3", corpus_id="c", document_id="d", name="Berlin", normalized_name="berlin freight", kind="ORG"),
    ]
    relations = [
        Relation(
            id="r1", corpus_id="c", document_id="d", source_entity_id="e1", target_entity_id="e3", kind="co", weight=1.0
        )
    ]
    return entities, relations


class _StubRepo:
    def __init__(self, *, mode: str = "conformant") -> None:
        self._entities, self._relations = _stub_graph()
        self._mode = mode  # "conformant" (both branches), "name_only" (ignores tokens), or "empty"

    def list_graph(self, workspace_id: str, corpus_id: str) -> tuple[list[Entity], list[Relation]]:
        return sorted(self._entities, key=lambda e: e.id), sorted(self._relations, key=lambda r: r.id)

    def graph_index_present(self, workspace_id: str, corpus_id: str) -> bool:
        return True

    def max_entity_name_length(self, workspace_id: str, corpus_id: str) -> int:
        return max(len(entity.normalized_name) for entity in self._entities)

    def graph_neighborhood(
        self, workspace_id: str, corpus_id: str, *, term_candidates: frozenset[str], name_substrings: frozenset[str]
    ) -> tuple[list[Entity], list[Relation]]:
        if self._mode == "empty":
            return [], []
        matched = {e.id for e in self._entities if e.normalized_name in name_substrings}
        if self._mode == "conformant":
            matched |= {e.id for e in self._entities if set(tokens(e.normalized_name)) & term_candidates}
        rels = [r for r in self._relations if r.source_entity_id in matched or r.target_entity_id in matched]
        keep = matched | {r.source_entity_id for r in rels} | {r.target_entity_id for r in rels}
        entities = sorted((e for e in self._entities if e.id in keep), key=lambda e: e.id)
        return entities, sorted(rels, key=lambda r: r.id)


def test_conformant_stub_passes_and_a_token_ignoring_or_empty_stub_fails() -> None:
    assert structural_conformance(_StubRepo(), "GraphReadRepository") == []
    good = GraphReadConformanceCase(
        repo=_StubRepo(mode="conformant"), workspace_id="w", corpus_id="c", query="platform"
    )
    assert graph_read_conformance(good) == []

    # A backend honoring only the name-substring branch under-returns the token-only match (e2), and an
    # empty neighborhood likewise; the contract must catch both.
    for mode in ("name_only", "empty"):
        case = GraphReadConformanceCase(repo=_StubRepo(mode=mode), workspace_id="w", corpus_id="c", query="platform")
        assert any("under-returned" in v.problem for v in graph_read_conformance(case)), mode
