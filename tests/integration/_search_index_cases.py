"""Shared fixtures for the SearchIndex conformance contract: a case builder and a fake OpenSearch client.

The case builder seeds a backend through the normal ingestion path, so the same builder works for the
Local, pgvector, and OpenSearch backends. It also seeds a second corpus and a second workspace so the
contract's scope-isolation checks have foreign data to catch. The fake client is an in-memory stand-in
that evaluates the knn clause (semantic), the bool filter, and the match clause the adapter builds, so
the OpenSearch adapter runs the full contract offline without the opensearch-py dependency. The fake is
faithful to the adapter's query shapes, not a substitute for a real cluster's ANN semantics.
"""

from __future__ import annotations

from typing import Any

from refaat.application.runtime import Refaat
from refaat.conformance.search_index import SearchIndexConformanceCase
from refaat.core.math import cosine_similarity
from refaat.domain.models import DocumentInput, MetadataFilter

_ALPHA = b"The alpha release checklist requires zephyrine sign-off before the production deployment ships."
_BRAVO = b"The bravo runbook lists quicksilver production rollback steps for the staging environment incident."
_FOREIGN = b"The gamma production ledger records unrelated tungsten accounting entries for a different corpus."


def _chunk_ids(runtime: Refaat, ws: str, corpus: str) -> frozenset[str]:
    return frozenset(chunk.id for chunk in runtime.store.list_chunks(ws, corpus))


def build_search_index_case(
    runtime: Refaat, *, workspace: str = "acme", corpus: str = "handbook"
) -> SearchIndexConformanceCase:
    ws = runtime.create_workspace(workspace, workspace.title())
    cp = runtime.create_corpus(ws, corpus)
    runtime.ingest(
        ws,
        cp,
        DocumentInput(
            source_id="alpha", title="alpha", media_type="text/plain", content=_ALPHA, metadata={"team": "alpha"}
        ),
    )
    runtime.ingest(
        ws,
        cp,
        DocumentInput(
            source_id="bravo", title="bravo", media_type="text/plain", content=_BRAVO, metadata={"team": "bravo"}
        ),
    )

    # A second corpus in the same workspace and a second workspace, so a backend that ignores
    # workspace_id or corpus_id returns their chunks and the scope-isolation check catches it.
    other_corpus = runtime.create_corpus(ws, f"{corpus}-other")
    runtime.ingest(
        ws,
        other_corpus,
        DocumentInput(
            source_id="gamma", title="gamma", media_type="text/plain", content=_FOREIGN, metadata={"team": "alpha"}
        ),
    )
    other_ws = runtime.create_workspace(f"{workspace}-other", "Other")
    other_ws_corpus = runtime.create_corpus(other_ws, corpus)
    runtime.ingest(
        other_ws,
        other_ws_corpus,
        DocumentInput(
            source_id="delta", title="delta", media_type="text/plain", content=_FOREIGN, metadata={"team": "alpha"}
        ),
    )
    foreign = _chunk_ids(runtime, ws, other_corpus) | _chunk_ids(runtime, other_ws, other_ws_corpus)

    chunks = runtime.store.list_chunks(ws, cp)
    alpha_ids = frozenset(chunk.id for chunk in chunks if chunk.metadata.get("team") == "alpha")
    alpha_doc = next(chunk.document_id for chunk in chunks if chunk.metadata.get("team") == "alpha")
    return SearchIndexConformanceCase(
        index=runtime.queries.retriever.index,
        workspace_id=ws,
        corpus_id=cp,
        query_embedding=runtime.queries.embedder.embed([_ALPHA.decode()])[0],
        all_chunk_ids=frozenset(chunk.id for chunk in chunks),
        document_id=alpha_doc,
        document_chunk_ids=alpha_ids,
        lexical_term="production",
        metadata_filter=MetadataFilter.from_spec({"team": "alpha"}),
        metadata_expected_ids=alpha_ids,
        foreign_scope_ids=foreign,
    )


def _apply_filter(docs: dict[str, dict[str, Any]], clauses: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    kept = docs
    for clause in clauses:
        if "term" in clause:
            field, value = next(iter(clause["term"].items()))
            kept = {k: v for k, v in kept.items() if v.get(field) == value}
        elif "terms" in clause:
            field, values = next(iter(clause["terms"].items()))
            allowed = set(values)
            kept = {k: v for k, v in kept.items() if v.get(field) in allowed}
    return kept


class _FakeIndices:
    def __init__(self, parent: FakeOpenSearch) -> None:
        self._parent = parent

    def exists(self, *, index: str) -> bool:
        return index in self._parent.created

    def create(self, *, index: str, body: dict[str, Any]) -> None:
        self._parent.created.add(index)
        self._parent.mappings[index] = body.get("mappings", {})

    def get_mapping(self, *, index: str) -> dict[str, Any]:
        return {index: {"mappings": self._parent.mappings.get(index, {})}}

    def refresh(self, *, index: str) -> None:
        del index


class FakeOpenSearch:
    """An in-memory OpenSearch stand-in that evaluates the adapter's knn, bool-filter, and match queries."""

    def __init__(self) -> None:
        self.created: set[str] = set()
        self.mappings: dict[str, dict[str, Any]] = {}
        self.docs: dict[str, dict[str, Any]] = {}

    @property
    def indices(self) -> _FakeIndices:
        return _FakeIndices(self)

    def index(self, *, index: str, id: str, body: dict[str, Any]) -> None:
        del index
        self.docs[id] = dict(body)

    def delete_by_query(self, *, index: str, body: dict[str, Any], refresh: bool = False) -> None:
        del index, refresh
        document_id = body["query"]["term"]["document_id"]
        self.docs = {key: value for key, value in self.docs.items() if value.get("document_id") != document_id}

    def search(self, *, index: str, body: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        del index, kwargs
        size = int(body.get("size", 10))
        query = body["query"]
        if "knn" in query:
            spec = query["knn"]["embedding"]
            vector = spec["vector"]
            clauses = spec.get("filter", {}).get("bool", {}).get("filter", [])
            candidates = _apply_filter(self.docs, clauses)
            ranked = sorted(
                candidates.values(),
                key=lambda doc: (-cosine_similarity(vector, doc["embedding"]), doc["chunk_id"]),
            )
            rows = ranked[: min(size, int(spec.get("k", size)))]
        else:
            bool_query = query["bool"]
            candidates = _apply_filter(self.docs, bool_query.get("filter", []))
            rows = []
            for doc in candidates.values():
                if _matches_text(doc, bool_query.get("must", [])):
                    rows.append(doc)
                if len(rows) >= size:
                    break
        return {
            "timed_out": False,
            "_shards": {"total": 1, "successful": 1, "failed": 0},
            "hits": {"hits": [{"_id": doc["chunk_id"], "_source": {"chunk_id": doc["chunk_id"]}} for doc in rows]},
        }

    def close(self) -> None:
        pass


class FaultyOpenSearch(FakeOpenSearch):
    """Writes and provisioning succeed, but every search raises, to exercise the fail-closed path."""

    def search(self, *, index: str, body: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("cluster unreachable")


class PartialShardOpenSearch(FakeOpenSearch):
    """Returns an HTTP 200 with a failed shard, the silent-degradation case a backend must reject."""

    def search(self, *, index: str, body: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        response = super().search(index=index, body=body, **kwargs)
        response["_shards"] = {"total": 3, "successful": 2, "failed": 1}
        return response


def _matches_text(doc: dict[str, Any], must: list[dict[str, Any]]) -> bool:
    for clause in must:
        if "match" in clause:
            field, text = next(iter(clause["match"].items()))
            doc_terms = set(str(doc.get(field, "")).lower().split())
            if not (set(str(text).lower().split()) & doc_terms):
                return False
    return True
