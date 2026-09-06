"""with_search_index: inject a search backend (object or factory), byte-identical when unused."""

from __future__ import annotations

from pathlib import Path

import pytest

from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.config import Settings
from refaat.domain.models import DocumentInput
from refaat.infrastructure.search import LocalSearchIndex

pytestmark = pytest.mark.integration

_BODY = b"The production release checklist requires reviewer approval before deployment ships."


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path / 'refaat.db'}",
        auto_create_schema=True,
        blob_root=tmp_path / "blobs",
        connector_file_root=tmp_path / "connectors",
        skill_root=tmp_path / "skills",
        api_key_pepper="test-pepper-with-enough-entropy",
        bootstrap_token="bootstrap-test-token",
        log_json=False,
    )


class _TaggedLocalIndex:
    """A store-dependent SearchIndex that declares a cache_id so the answer cache scopes to it."""

    def __init__(self, store: object, tag: str) -> None:
        self._inner = LocalSearchIndex(store)  # type: ignore[arg-type]
        self._tag = tag

    @property
    def cache_id(self) -> str:
        return self._tag

    def replace_document(self, *args: object, **kwargs: object) -> object:
        return self._inner.replace_document(*args, **kwargs)  # type: ignore[arg-type]

    def delete_document(self, *args: object, **kwargs: object) -> object:
        return self._inner.delete_document(*args, **kwargs)  # type: ignore[arg-type]

    def semantic_search(self, *args: object, **kwargs: object) -> object:
        return self._inner.semantic_search(*args, **kwargs)  # type: ignore[arg-type]

    def lexical_search(self, *args: object, **kwargs: object) -> object:
        return self._inner.lexical_search(*args, **kwargs)  # type: ignore[arg-type]


class _IndexFactory:
    def __init__(self, tag: str) -> None:
        self._tag = tag
        self.built_with_store = False

    def build(self, store: object, embedder: object) -> _TaggedLocalIndex:
        self.built_with_store = store is not None and embedder is not None
        return _TaggedLocalIndex(store, self._tag)


def _ingest(runtime: Refaat, ws: str, corpus: str) -> None:
    runtime.ingest(
        ws,
        corpus,
        DocumentInput(source_id="doc", title="doc", media_type="text/plain", content=_BODY, metadata={}),
    )


def test_default_runtime_uses_the_built_in_local_index(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_settings(tmp_path)).build()
    try:
        assert isinstance(runtime.queries.retriever.index, LocalSearchIndex)
    finally:
        runtime.close()


def test_injected_factory_receives_the_runtime_store_and_drives_retrieval(tmp_path: Path) -> None:
    factory = _IndexFactory("es-v1")
    runtime = RuntimeBuilder(_settings(tmp_path)).with_search_index(factory).build()
    try:
        assert factory.built_with_store is True
        assert isinstance(runtime.queries.retriever.index, _TaggedLocalIndex)
        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "handbook")
        _ingest(runtime, ws, corpus)
        response = runtime.project(workspace="acme", corpus="handbook").ask("What does a release require?")
        assert response.citations
        stored = {c.id for c in runtime.store.list_chunks(ws, corpus)}
        assert all(cit.chunk_id in stored for cit in response.citations)
    finally:
        runtime.close()


def _context_hash(runtime: Refaat, *, search_backend: str | None) -> str:
    from refaat.domain.models import QueryRequest

    q = runtime.queries
    policy = q.retriever.resolve_policy(QueryRequest(workspace_id="w", corpus_id="c", query="x"), {})
    return q._context_hash(
        policy=policy,
        profile={},
        supplemental={},
        session_summary=None,
        embedding_model=q.embedder.model_id,
        generation_model=q.generator.model_id,
        reranker="none",
        query_transform="none",
        guidance=None,
        response_schema=None,
        search_backend=search_backend,
    )


def test_backend_cache_id_scopes_the_answer_cache(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_settings(tmp_path)).build()
    try:
        assert _context_hash(runtime, search_backend=None) != _context_hash(runtime, search_backend="es-v1")
    finally:
        runtime.close()
