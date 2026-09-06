"""The SearchIndex behavioral conformance contract: Local and the OpenSearch adapter both pass it."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
from tests.integration._search_index_cases import (
    FakeOpenSearch,
    FaultyOpenSearch,
    PartialShardOpenSearch,
    build_search_index_case,
)

from refaat.application.runtime import RuntimeBuilder
from refaat.config import Settings
from refaat.conformance.search_index import search_index_conformance
from refaat.domain.models import DocumentInput
from refaat.domain.search import SearchIndexCapabilities
from refaat.errors import CapabilityNotSupportedError, DependencyUnavailableError
from refaat.infrastructure.opensearch_index import OpenSearchIndexFactory
from refaat.infrastructure.search import LocalSearchIndex

pytestmark = pytest.mark.integration


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


def test_local_index_passes_the_conformance_contract(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_settings(tmp_path)).build()
    try:
        assert search_index_conformance(build_search_index_case(runtime)) == []
    finally:
        runtime.close()


def test_opensearch_adapter_passes_the_conformance_contract(tmp_path: Path) -> None:
    factory = OpenSearchIndexFactory(index_name="refaat_test", client=FakeOpenSearch())
    runtime = RuntimeBuilder(_settings(tmp_path)).with_search_index(factory).build()
    try:
        assert search_index_conformance(build_search_index_case(runtime)) == []
    finally:
        runtime.close()


def _with_delete_probe(runtime: Any, case: Any) -> Any:
    chunks = runtime.store.list_chunks(case.workspace_id, case.corpus_id)
    probe_ids = frozenset(c.id for c in chunks if c.metadata.get("team") == "bravo")
    probe_doc = next(c.document_id for c in chunks if c.metadata.get("team") == "bravo")
    return dataclasses.replace(case, delete_probe_document_id=probe_doc, delete_probe_chunk_ids=probe_ids)


def test_opensearch_adapter_deletes_documents_from_the_index(tmp_path: Path) -> None:
    factory = OpenSearchIndexFactory(index_name="refaat_test", client=FakeOpenSearch())
    runtime = RuntimeBuilder(_settings(tmp_path)).with_search_index(factory).build()
    try:
        assert search_index_conformance(_with_delete_probe(runtime, build_search_index_case(runtime))) == []
    finally:
        runtime.close()


def test_opensearch_adapter_fails_closed_and_never_falls_back_to_a_store_scan(tmp_path: Path) -> None:
    factory = OpenSearchIndexFactory(index_name="refaat_test", client=FaultyOpenSearch())
    runtime = RuntimeBuilder(_settings(tmp_path)).with_search_index(factory).build()
    try:
        case = build_search_index_case(runtime)
        with pytest.raises(DependencyUnavailableError):
            case.index.semantic_search(
                workspace_id=case.workspace_id, corpus_id=case.corpus_id, query_embedding=case.query_embedding, limit=5
            )
        assert runtime.store.list_chunks(case.workspace_id, case.corpus_id)
    finally:
        runtime.close()


class _SemanticOnlyIndex:
    """Honestly declares no lexical support and raises for it, delegating semantic to the wrapped index."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    @property
    def capabilities(self) -> SearchIndexCapabilities:
        return SearchIndexCapabilities(supports_lexical=False)

    def replace_document(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.replace_document(*args, **kwargs)

    def delete_document(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.delete_document(*args, **kwargs)

    def semantic_search(self, **kwargs: Any) -> Any:
        return self._inner.semantic_search(**kwargs)

    def lexical_search(self, **kwargs: Any) -> Any:
        raise CapabilityNotSupportedError("lexical search is not supported")


class _DishonestIndex(_SemanticOnlyIndex):
    """Declares no lexical support but silently returns an empty result instead of raising."""

    def lexical_search(self, **kwargs: Any) -> Any:
        return []


def test_a_backend_declaring_a_mode_unsupported_must_raise_for_it(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_settings(tmp_path)).build()
    try:
        base = build_search_index_case(runtime)
        honest = dataclasses.replace(base, index=_SemanticOnlyIndex(base.index))
        assert search_index_conformance(honest) == []
        dishonest = dataclasses.replace(base, index=_DishonestIndex(base.index))
        violations = search_index_conformance(dishonest)
        assert any(v.member == "lexical_search" for v in violations)
    finally:
        runtime.close()


def test_opensearch_adapter_rejects_a_partial_shard_result(tmp_path: Path) -> None:
    factory = OpenSearchIndexFactory(index_name="refaat_test", client=PartialShardOpenSearch())
    runtime = RuntimeBuilder(_settings(tmp_path)).with_search_index(factory).build()
    try:
        case = build_search_index_case(runtime)
        with pytest.raises(DependencyUnavailableError):
            case.index.semantic_search(
                workspace_id=case.workspace_id, corpus_id=case.corpus_id, query_embedding=case.query_embedding, limit=5
            )
    finally:
        runtime.close()


class _PassThrough:
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def replace_document(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.replace_document(*args, **kwargs)

    def delete_document(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.delete_document(*args, **kwargs)

    def semantic_search(self, **kwargs: Any) -> Any:
        return self._inner.semantic_search(**kwargs)

    def lexical_search(self, **kwargs: Any) -> Any:
        return self._inner.lexical_search(**kwargs)


class _IgnoresDelete(_PassThrough):
    def delete_document(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


def test_delete_verification_catches_a_backend_that_does_not_delete(tmp_path: Path) -> None:
    # Over a backend that owns a separate copy (OpenSearch), a delete that silently no-ops must be caught.
    factory = OpenSearchIndexFactory(index_name="refaat_test", client=FakeOpenSearch())
    runtime = RuntimeBuilder(_settings(tmp_path)).with_search_index(factory).build()
    try:
        case = _with_delete_probe(runtime, build_search_index_case(runtime))
        broken = dataclasses.replace(case, index=_IgnoresDelete(case.index))
        assert any(v.member == "delete_document" for v in search_index_conformance(broken))
    finally:
        runtime.close()


class _LexicalIgnoresAcl(_PassThrough):
    def lexical_search(self, *, allowed_document_ids: Any = None, **kwargs: Any) -> Any:
        return self._inner.lexical_search(**kwargs)


class _LexicalIgnoresMetadata(_PassThrough):
    def lexical_search(self, *, metadata_filter: Any = None, **kwargs: Any) -> Any:
        return self._inner.lexical_search(**kwargs)


class _SemanticLeaksEmptyAcl(_PassThrough):
    def semantic_search(self, *, allowed_document_ids: Any = None, **kwargs: Any) -> Any:
        return self._inner.semantic_search(**kwargs)


class _ReversedOrder(_PassThrough):
    def semantic_search(self, **kwargs: Any) -> Any:
        return list(reversed(self._inner.semantic_search(**kwargs)))


class _SemanticEmptyAclOnly(_PassThrough):
    def semantic_search(self, *, allowed_document_ids: Any = None, **kwargs: Any) -> Any:
        if allowed_document_ids == frozenset():
            allowed_document_ids = None
        return self._inner.semantic_search(allowed_document_ids=allowed_document_ids, **kwargs)


def test_contract_is_non_vacuous_each_broken_backend_is_caught(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_settings(tmp_path)).build()
    try:
        base = build_search_index_case(runtime)
        for wrapper, member in (
            (_LexicalIgnoresAcl, "lexical_search"),
            (_LexicalIgnoresMetadata, "lexical_search"),
            (_SemanticLeaksEmptyAcl, "semantic_search"),
            (_SemanticEmptyAclOnly, "semantic_search"),
            (_ReversedOrder, "semantic_search"),
        ):
            case = dataclasses.replace(base, index=wrapper(base.index))
            violations = search_index_conformance(case)
            assert any(v.member == member for v in violations), f"{wrapper.__name__} was not caught"
    finally:
        runtime.close()


class _SemanticOnlyFactory:
    def build(self, store: Any, embedder: Any) -> _SemanticOnlyIndex:
        return _SemanticOnlyIndex(LocalSearchIndex(store))


def test_retriever_skips_a_declared_unsupported_lexical_mode(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_settings(tmp_path)).with_search_index(_SemanticOnlyFactory()).build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        cp = runtime.create_corpus(ws, "handbook")
        runtime.ingest(
            ws,
            cp,
            DocumentInput(
                source_id="d",
                title="d",
                media_type="text/plain",
                content=b"The production release checklist requires reviewer sign-off before deployment.",
                metadata={},
            ),
        )
        context = runtime.project(workspace="acme", corpus="handbook").retrieve("what does a release require?")
        assert context.hits
        assert "lexical" not in context.routes
        assert "semantic" in context.routes
    finally:
        runtime.close()
