from __future__ import annotations

import pytest
from sqlalchemy import func, select

from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.config import Settings
from refaat.core.math import cosine_similarity
from refaat.core.raptor import RAPTOR_ID_PREFIX, ClusterItem, cluster_items, raptor_node_id
from refaat.domain.models import DocumentInput, GenerationResult, QueryRequest
from refaat.errors import DependencyUnavailableError
from refaat.infrastructure.database.schema import raptor_nodes
from refaat.infrastructure.summarization import DeterministicExtractiveSummarizer, ModelSummarizer

# Three themed clusters. The ledger cluster's shared subject is clearest at the summary
# level: a query about the ledger module matches the cluster summary more than the lone
# "records every accounting transaction" leaf, which is how the tree rescues that leaf.
_CORPUS = [
    ("a1", "Alpha module reads the ledger."),
    ("a2", "Beta module writes the ledger."),
    ("a3", "Gamma module audits the ledger."),
    ("a4", "The ledger records every accounting transaction."),
    ("b1", "Zebras roam the open savanna grassland."),
    ("b2", "Lions hunt across the savanna at dusk."),
    ("b3", "Elephants migrate through the dry savanna."),
    ("c1", "Quantum bits enable new computing hardware."),
    ("c2", "Photonic chips route light for computing."),
]
_QUERY = "module ledger audit records"
_TARGET_TEXT = "ledger records every accounting"


def _raptor_settings(settings: Settings) -> Settings:
    return settings.model_copy(update={"raptor_enabled": True})


def _count_nodes(runtime: Refaat) -> int:
    with runtime.store.engine.connect() as conn:
        return int(conn.execute(select(func.count()).select_from(raptor_nodes)).scalar_one())


def _node_rows(runtime: Refaat) -> list[dict[str, object]]:
    with runtime.store.engine.connect() as conn:
        return [dict(row) for row in conn.execute(select(raptor_nodes)).mappings()]


def _ingest(runtime: Refaat) -> tuple[str, str]:
    project = runtime.project(workspace="acme", corpus="handbook")
    for source_id, text in _CORPUS:
        runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id=f"{source_id}.txt", title=source_id, media_type="text/plain", content=text.encode()
            ),
        )
    return project.workspace_id, project.corpus_id


def test_cluster_items_is_deterministic_and_order_independent() -> None:
    items = [
        ClusterItem(key="a", embedding=(1.0, 0.0, 0.0)),
        ClusterItem(key="b", embedding=(0.9, 0.1, 0.0)),
        ClusterItem(key="c", embedding=(0.0, 0.0, 1.0)),
        ClusterItem(key="d", embedding=(0.0, 0.1, 0.9)),
    ]
    forward = cluster_items(items, threshold=0.5, max_cluster_size=2)
    backward = cluster_items(list(reversed(items)), threshold=0.5, max_cluster_size=2)

    def keys(clusters: list[tuple[ClusterItem, ...]]) -> list[tuple[str, ...]]:
        return sorted(tuple(item.key for item in cluster) for cluster in clusters)

    assert keys(forward) == keys(backward)
    assert keys(forward) == [("a", "b"), ("c", "d")]
    assert {item.key for cluster in forward for item in cluster} == {"a", "b", "c", "d"}


def test_raptor_node_id_is_uncitable_namespace_and_collision_free() -> None:
    first = raptor_node_id("corpus", 1, ["chunk_b", "chunk_a"])
    second = raptor_node_id("corpus", 1, ["chunk_a", "chunk_b"])
    assert first == second, "node id must be independent of leaf order"
    assert first.startswith(RAPTOR_ID_PREFIX)
    assert not first.startswith("chunk_"), "a node id must never live in the chunk namespace"
    assert raptor_node_id("corpus", 2, ["chunk_a", "chunk_b"]) != first, "level must change the id"


def test_extractive_summarizer_is_stable_and_content_derived() -> None:
    summarizer = DeterministicExtractiveSummarizer(max_sentences=2)
    texts = ["Photosynthesis converts sunlight. Plants store the energy.", "Chlorophyll absorbs light."]
    assert summarizer.summarize(texts) == summarizer.summarize(texts)
    assert summarizer.summarize(texts)
    assert summarizer.summarize([]) == ""


def test_model_summarizer_summarizes_and_falls_back() -> None:
    fallback = DeterministicExtractiveSummarizer()
    texts = ["Photosynthesis converts sunlight into energy."]

    good = ModelSummarizer(_StubGenerator("A concise model summary."), fallback)
    assert good.summarize(texts) == "A concise model summary."

    blank = ModelSummarizer(_StubGenerator("   "), fallback)
    assert blank.summarize(texts) == fallback.summarize(texts)

    failing = ModelSummarizer(_FailingGenerator(), fallback)
    assert failing.summarize(texts) == fallback.summarize(texts)


def test_disabled_path_writes_no_nodes_and_keeps_identity(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings).build()
    try:
        workspace_id, corpus_id = _ingest(runtime)
        assert _count_nodes(runtime) == 0
        assert runtime.store.list_raptor_nodes(workspace_id, corpus_id) == []
        assert runtime.queries.retriever.raptor_enabled is False
        assert runtime.queries.retriever.raptor_cache_id is None
        response = runtime.project(workspace="acme", corpus="handbook").ask(_QUERY)
        assert "raptor" not in response.routes
    finally:
        runtime.close()


def test_enabled_ingestion_builds_deterministic_embedded_nodes(settings: Settings) -> None:
    runtime = RuntimeBuilder(_raptor_settings(settings)).build()
    try:
        workspace_id, corpus_id = _ingest(runtime)
        stored_ids = {chunk.id for chunk in runtime.store.list_chunks(workspace_id, corpus_id)}
        rows = _node_rows(runtime)
        assert rows, "expected raptor nodes when raptor is enabled"
        for row in rows:
            assert str(row["id"]).startswith(RAPTOR_ID_PREFIX)
            expected = list(runtime.ingestion.embedder.embed([str(row["summary"])])[0])
            stored = [float(value) for value in row["embedding"]]  # type: ignore[union-attr]
            assert stored == pytest.approx(expected)
            leaves = [str(chunk_id) for chunk_id in row["leaf_chunk_ids"]]  # type: ignore[union-attr]
            assert leaves
            assert all(leaf in stored_ids for leaf in leaves)
    finally:
        runtime.close()


def test_raptor_hit_maps_back_to_real_chunk(settings: Settings) -> None:
    runtime = RuntimeBuilder(_raptor_settings(settings)).build()
    try:
        workspace_id, corpus_id = _ingest(runtime)
        stored_ids = {chunk.id for chunk in runtime.store.list_chunks(workspace_id, corpus_id)}
        retriever = runtime.queries.retriever
        request = QueryRequest(workspace_id=workspace_id, corpus_id=corpus_id, query=_QUERY)
        embedding = retriever.embedder.embed([_QUERY])[0]  # type: ignore[union-attr]
        policy = retriever.resolve_policy(request, {})
        hits = retriever._raptor_hits(request, embedding, policy)
        assert hits, "expected a raptor match for a concept-shaped query"
        for hit in hits:
            assert hit.chunk.id in stored_ids
            assert not hit.chunk.id.startswith(RAPTOR_ID_PREFIX)
            assert hit.reasons == ("raptor",)
    finally:
        runtime.close()


def test_summary_id_is_uncitable_and_dropped_by_the_store(settings: Settings) -> None:
    runtime = RuntimeBuilder(_raptor_settings(settings)).build()
    try:
        workspace_id, corpus_id = _ingest(runtime)
        node_ids = [str(row["id"]) for row in _node_rows(runtime)]
        assert node_ids
        stored_ids = {chunk.id for chunk in runtime.store.list_chunks(workspace_id, corpus_id)}
        for node_id in node_ids:
            assert node_id not in stored_ids
        assert runtime.store.get_chunks(workspace_id, corpus_id, node_ids) == [], (
            "a summary-node id must resolve to no stored chunk"
        )
    finally:
        runtime.close()


def test_citation_never_comes_from_summary_text(settings: Settings) -> None:
    runtime = RuntimeBuilder(_raptor_settings(settings)).build()
    try:
        workspace_id, corpus_id = _ingest(runtime)
        stored = {chunk.id: chunk for chunk in runtime.store.list_chunks(workspace_id, corpus_id)}
        summaries = {str(row["summary"]) for row in _node_rows(runtime)}
        response = runtime.project(workspace="acme", corpus="handbook").ask(_QUERY)
        assert "raptor" in response.routes
        assert response.citations
        for citation in response.citations:
            assert citation.chunk_id in stored
            assert not citation.chunk_id.startswith(RAPTOR_ID_PREFIX)
            assert citation.quote in stored[citation.chunk_id].text
            assert citation.quote not in summaries
    finally:
        runtime.close()


def test_reingestion_is_idempotent(settings: Settings) -> None:
    runtime = RuntimeBuilder(_raptor_settings(settings)).build()
    try:
        _ingest(runtime)
        first = {str(row["id"]) for row in _node_rows(runtime)}
        first_count = _count_nodes(runtime)
        _ingest(runtime)
        again = {str(row["id"]) for row in _node_rows(runtime)}
        assert _count_nodes(runtime) == first_count
        assert again == first
    finally:
        runtime.close()


def test_raptor_lifts_a_chunk_above_its_direct_query_similarity(settings: Settings) -> None:
    """The raptor stage surfaces a real leaf chunk with a score derived from its

    cluster summary that is strictly greater than the chunk's own direct similarity to
    the query, so the summary tree is a stronger query-to-chunk bridge than direct
    matching. The scores derive from the text through the deterministic summary and
    embeddings, so this holds run to run without depending on tie-breaking, and the
    lifted hit is still a real stored chunk. A regression to a zero or direct-only
    raptor score fails here.
    """
    runtime = RuntimeBuilder(_raptor_settings(settings)).build()
    try:
        workspace_id, corpus_id = _ingest(runtime)
        target = next(
            chunk for chunk in runtime.store.list_chunks(workspace_id, corpus_id) if _TARGET_TEXT in chunk.text
        )
        retriever = runtime.queries.retriever
        request = QueryRequest(workspace_id=workspace_id, corpus_id=corpus_id, query=_QUERY)
        query_embedding = retriever.embedder.embed([_QUERY])[0]  # type: ignore[union-attr]
        policy = retriever.resolve_policy(request, {})
        hits = {hit.chunk.id: hit for hit in retriever._raptor_hits(request, query_embedding, policy)}

        assert target.id in hits, "the raptor stage must surface the target chunk"
        direct = cosine_similarity(query_embedding, target.embedding)
        assert hits[target.id].score > direct, "the summary bridge must beat direct similarity"
        assert hits[target.id].chunk.id.startswith("chunk_"), "the lifted hit is a real stored chunk"
    finally:
        runtime.close()


class _StubGenerator:
    model_id = "stub-generator-v1"

    def __init__(self, answer: str) -> None:
        self._answer = answer

    def generate(self, request: object) -> GenerationResult:
        return GenerationResult(answer=self._answer, cited_chunk_ids=())

    def stream(self, request: object) -> object:  # pragma: no cover - not exercised
        raise NotImplementedError


class _FailingGenerator:
    model_id = "failing-generator-v1"

    def generate(self, request: object) -> GenerationResult:
        raise DependencyUnavailableError("no model configured")

    def stream(self, request: object) -> object:  # pragma: no cover - not exercised
        raise NotImplementedError
