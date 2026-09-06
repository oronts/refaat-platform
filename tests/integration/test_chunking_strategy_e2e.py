"""ChunkingStrategy port: byte-identical built-ins, custom injection, plugin discovery, composition."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from refaat.application.runtime import _PLUGIN_PIN_GROUPS, Refaat, RuntimeBuilder
from refaat.config import Settings
from refaat.core.chunking import (
    BUILTIN_CHUNKING_STRATEGIES,
    CHUNKING_STRATEGIES,
    AdaptiveChunker,
    TextChunk,
)
from refaat.core.text import estimate_tokens, normalize_text, split_sentences
from refaat.domain.models import DocumentInput, JsonObject
from refaat.errors import ConfigurationError, ValidationError
from refaat.infrastructure.plugins import PluginRegistry

pytestmark = pytest.mark.integration

_FLAT_TEXT = (
    "The alpha subsystem caches grounded answers reliably. "
    "It reuses those answers across repeated retrieval requests. "
    "Every cached answer stays stable for a long retention window. "
    "The beta subsystem fuses semantic and lexical candidates before packing them."
)
_SECTIONED_TEXT = "\n\n".join(
    [
        "## Overview\nThe platform indexes documents and answers grounded questions.",
        "## Retrieval\nSemantic and lexical candidates are fused. The budgeter packs the context.",
        "## Governance\nEvery answer cites a real chunk. A held document refuses a destructive overwrite.",
    ]
)
_FIXTURES = (_FLAT_TEXT, _SECTIONED_TEXT)


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {
        "environment": "test",
        "database_url": f"sqlite:///{tmp_path / 'refaat.db'}",
        "auto_create_schema": True,
        "blob_root": tmp_path / "blobs",
        "connector_file_root": tmp_path / "connectors",
        "skill_root": tmp_path / "skills",
        "api_key_pepper": "test-pepper-with-enough-entropy",
        "bootstrap_token": "bootstrap-test-token",
        "log_json": False,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _ingest(runtime: Refaat, ws: str, corpus: str, source_id: str, body: bytes) -> str:
    return runtime.ingest(
        ws,
        corpus,
        DocumentInput(source_id=source_id, title=source_id, media_type="text/plain", content=body, metadata={}),
    ).document_id


def _shape(chunk: TextChunk) -> tuple[int, str, int, str, str, int | None, int | None, int | None]:
    return (
        chunk.ordinal,
        chunk.text,
        chunk.token_count,
        chunk.heading,
        chunk.granularity,
        chunk.parent_ordinal,
        chunk.sent_start,
        chunk.sent_end,
    )


def assert_conforms(strategy: object, settings: Settings, fixtures: Sequence[str]) -> None:
    """The chunker conformance contract every ChunkingStrategy must satisfy."""
    assert strategy.strategy_id  # a stable non-empty identity
    for text in fixtures:
        chunks = list(strategy.chunk(text, metadata={}, settings=settings))  # type: ignore[attr-defined]
        assert chunks, "a non-empty document must produce at least one chunk"
        assert [c.ordinal for c in chunks] == list(range(len(chunks))), "ordinals must be contiguous from 0"
        for chunk in chunks:
            assert chunk.text.strip(), "chunk text must be non-empty"
            assert chunk.token_count > 0, "token_count must be a positive estimate"
            if chunk.granularity == "child":
                assert chunk.parent_ordinal is not None
                parent = chunks[chunk.parent_ordinal]
                assert parent.granularity == "parent"
                start = chunk.sent_start
                end = chunk.sent_end
                assert start is not None
                assert end is not None
                assert start >= 0
                assert start <= end
                assert end <= len(parent.text)
        again = list(strategy.chunk(text, metadata={}, settings=settings))  # type: ignore[attr-defined]
        assert [_shape(c) for c in chunks] == [_shape(c) for c in again], "chunking must be deterministic"


class _SentenceStrategy:
    """A trivial custom strategy: one chunk per sentence. Ignores metadata and settings."""

    strategy_id = "one-per-sentence"

    def chunk(self, text: str, *, metadata: JsonObject, settings: Settings) -> list[TextChunk]:
        sentences = [s for s in split_sentences(normalize_text(text)) if s.strip()]
        return [TextChunk(ordinal=i, text=s, token_count=estimate_tokens(s)) for i, s in enumerate(sentences)]


def test_builtin_strategies_are_byte_identical_to_the_direct_chunker(tmp_path: Path) -> None:
    for name in CHUNKING_STRATEGIES:
        settings = _settings(tmp_path, chunking_strategy=name, default_chunk_tokens=64, chunk_overlap_tokens=8)
        strategy = BUILTIN_CHUNKING_STRATEGIES[name](settings)
        assert strategy.strategy_id == name
        for text in _FIXTURES:
            direct = AdaptiveChunker(target_tokens=64, overlap_tokens=8, strategy=name).chunk(text)
            via_port = list(strategy.chunk(text, metadata={"language": "en"}, settings=settings))
            assert [_shape(c) for c in via_port] == [_shape(c) for c in direct]


def test_builtin_strategies_conform(tmp_path: Path) -> None:
    for name in CHUNKING_STRATEGIES:
        settings = _settings(tmp_path, chunking_strategy=name, default_chunk_tokens=48, chunk_overlap_tokens=6)
        assert_conforms(BUILTIN_CHUNKING_STRATEGIES[name](settings), settings, _FIXTURES)


def test_custom_strategy_conforms(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert_conforms(_SentenceStrategy(), settings, _FIXTURES)


def test_unknown_strategy_fails_startup(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="Unknown chunking strategy"):
        RuntimeBuilder(_settings(tmp_path, chunking_strategy="does-not-exist")).build()


def test_injected_strategy_drives_ingestion(tmp_path: Path) -> None:
    body = _SECTIONED_TEXT.encode()
    runtime = RuntimeBuilder(_settings(tmp_path)).with_chunking_strategy(_SentenceStrategy()).build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "handbook")
        _ingest(runtime, ws, corpus, "doc", body)
        expected = [s for s in split_sentences(normalize_text(body.decode())) if s.strip()]
        stored = runtime.store.list_chunks(ws, corpus)
        assert len(stored) == len(expected)
    finally:
        runtime.close()


def test_custom_strategy_composes_with_contextual_hype_and_raptor(tmp_path: Path) -> None:
    body = _SECTIONED_TEXT.encode()
    runtime = (
        RuntimeBuilder(_settings(tmp_path, contextual_retrieval=True, hype_enabled=True, raptor_enabled=True))
        .with_chunking_strategy(_SentenceStrategy())
        .build()
    )
    try:
        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "handbook")
        _ingest(runtime, ws, corpus, "doc", body)
        response = runtime.project(workspace="acme", corpus="handbook").ask("How are candidates fused?")
        assert response.answer.strip()
        assert response.citations
        stored_ids = {chunk.id for chunk in runtime.store.list_chunks(ws, corpus)}
        assert all(citation.chunk_id in stored_ids for citation in response.citations)
    finally:
        runtime.close()


def test_chunker_plugin_discovery_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    class EntryPoint:
        name = "one-per-sentence"
        group = "refaat.chunkers"

        def load(self) -> type[_SentenceStrategy]:
            return _SentenceStrategy

    monkeypatch.setattr(
        "refaat.infrastructure.plugins.entry_points",
        lambda group: [EntryPoint()] if group == "refaat.chunkers" else [],
    )
    discovered = PluginRegistry().discover("refaat.chunkers")
    assert discovered["one-per-sentence"] is _SentenceStrategy
    assert PluginRegistry(set()).discover("refaat.chunkers") == {}
    assert PluginRegistry({"other"}).discover("refaat.chunkers") == {}
    assert PluginRegistry({"one-per-sentence"}).discover("refaat.chunkers")["one-per-sentence"] is _SentenceStrategy


def test_discovered_chunker_plugin_drives_ingestion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class EntryPoint:
        name = "one-per-sentence"
        group = "refaat.chunkers"

        def load(self) -> type[_SentenceStrategy]:
            return _SentenceStrategy

    monkeypatch.setattr(
        "refaat.infrastructure.plugins.entry_points",
        lambda group: [EntryPoint()] if group == "refaat.chunkers" else [],
    )
    body = _SECTIONED_TEXT.encode()
    settings = _settings(tmp_path, chunking_strategy="one-per-sentence", plugin_allowlist=["one-per-sentence"])
    runtime = RuntimeBuilder(settings).build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "handbook")
        _ingest(runtime, ws, corpus, "doc", body)
        expected = [s for s in split_sentences(normalize_text(body.decode())) if s.strip()]
        assert len(runtime.store.list_chunks(ws, corpus)) == len(expected)
    finally:
        runtime.close()


class _ParentChildStrategy:
    """A custom strategy that emits a parent chunk plus one child per sentence and declares its
    retrieval shape, so the retriever expands its structure like the built-in parent_child."""

    strategy_id = "custom-parent-child"
    expansion_mode = "parent_child"

    def chunk(self, text: str, *, metadata: JsonObject, settings: Settings) -> list[TextChunk]:
        body = normalize_text(text)
        sentences = [s for s in split_sentences(body) if s.strip()]
        chunks = [TextChunk(ordinal=0, text=body, token_count=estimate_tokens(body), granularity="parent")]
        cursor = 0
        for offset, sentence in enumerate(sentences, start=1):
            start = body.find(sentence, cursor)
            end = start + len(sentence)
            cursor = end
            chunks.append(
                TextChunk(
                    ordinal=offset,
                    text=sentence,
                    token_count=estimate_tokens(sentence),
                    granularity="child",
                    parent_ordinal=0,
                    sent_start=start,
                    sent_end=end,
                )
            )
        return chunks


class _BrokenOrdinalStrategy:
    strategy_id = "broken-ordinals"

    def chunk(self, text: str, *, metadata: JsonObject, settings: Settings) -> list[TextChunk]:
        return [
            TextChunk(ordinal=0, text="first", token_count=1),
            TextChunk(ordinal=5, text="second", token_count=1),
        ]


def test_refaat_chunkers_is_a_pinnable_plugin_group() -> None:
    assert "refaat.chunkers" in _PLUGIN_PIN_GROUPS


def test_default_retriever_expansion_mode_matches_the_config_strategy(tmp_path: Path) -> None:
    for name in CHUNKING_STRATEGIES:
        root = tmp_path / name
        root.mkdir()
        runtime = RuntimeBuilder(_settings(root, chunking_strategy=name)).build()
        try:
            assert runtime.queries.retriever._chunking_expansion_mode == name
        finally:
            runtime.close()


def test_injected_strategy_threads_its_expansion_mode_into_retrieval(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_settings(tmp_path)).with_chunking_strategy(_ParentChildStrategy()).build()
    try:
        assert runtime.queries.retriever._chunking_expansion_mode == "parent_child"
    finally:
        runtime.close()


def test_a_strategy_with_bad_output_fails_ingestion_with_a_named_error(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_settings(tmp_path)).with_chunking_strategy(_BrokenOrdinalStrategy()).build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "handbook")
        with pytest.raises(ValidationError, match="broken-ordinals"):
            _ingest(runtime, ws, corpus, "doc", _SECTIONED_TEXT.encode())
    finally:
        runtime.close()
