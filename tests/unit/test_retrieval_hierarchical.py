from __future__ import annotations

from typing import Any, cast

from refaat.application.services import QueryService
from refaat.config import Settings
from refaat.core.retrieval import AdaptiveRetriever
from refaat.core.text import sentence_spans
from refaat.domain.models import Chunk, QueryRequest, SearchHit

_PARENT_TEXT = (
    "The alpha subsystem caches grounded answers reliably. "
    "It reuses those answers across repeated retrieval requests. "
    "The budgeter finally packs the answer context tightly."
)


def _retriever(settings: Settings) -> AdaptiveRetriever:
    return AdaptiveRetriever(cast(Any, None), cast(Any, None), settings)


def _request(**overrides: Any) -> QueryRequest:
    return QueryRequest(workspace_id="w", corpus_id="c", query="q", **overrides)


def _chunk(chunk_id: str, text: str, metadata: dict[str, Any]) -> Chunk:
    return Chunk(
        id=chunk_id,
        corpus_id="c",
        document_id="d",
        ordinal=0,
        text=text,
        token_count=len(text.split()),
        checksum="x",
        metadata=metadata,
        embedding=(),
    )


def _hit(chunk: Chunk, score: float = 1.0) -> SearchHit:
    return SearchHit(chunk=chunk, score=score)


def _context_hash(policy: Any) -> str:
    return QueryService._context_hash(
        policy=policy,
        profile={},
        supplemental={},
        session_summary=None,
        embedding_model="e",
        generation_model="g",
        reranker="none",
        query_transform="none",
        guidance=None,
        response_schema=None,
    )


def test_children_only_drops_parents_keeps_flat_and_children() -> None:
    parent = _hit(_chunk("p1", "parent", {"granularity": "parent"}))
    child = _hit(_chunk("c1", "child", {"granularity": "child", "parent_id": "p1"}))
    flat = _hit(_chunk("f1", "flat", {"title": "doc"}))
    kept = AdaptiveRetriever._children_only([parent, child, flat])
    assert [hit.chunk.id for hit in kept] == ["c1", "f1"]


def test_expand_to_parents_replaces_and_dedups(settings: Settings) -> None:
    parent = _chunk("p1", _PARENT_TEXT, {"granularity": "parent"})
    spans = sentence_spans(_PARENT_TEXT)
    child_a = _hit(
        _chunk(
            "c1",
            _PARENT_TEXT[spans[0][0] : spans[0][1]],
            {"granularity": "child", "parent_id": "p1", "sent_start": spans[0][0], "sent_end": spans[0][1]},
        )
    )
    child_b = _hit(
        _chunk(
            "c2",
            _PARENT_TEXT[spans[1][0] : spans[1][1]],
            {"granularity": "child", "parent_id": "p1", "sent_start": spans[1][0], "sent_end": spans[1][1]},
        )
    )
    orphan = _hit(_chunk("c3", "orphan child", {"granularity": "child", "parent_id": "missing"}))
    expanded = AdaptiveRetriever._expand_to_parents([child_a, child_b, orphan], {"p1": parent})
    assert [hit.chunk.id for hit in expanded] == ["p1", "c3"]
    assert expanded[0].chunk.text == _PARENT_TEXT


def test_expand_windows_keeps_child_id_and_widens_within_parent(settings: Settings) -> None:
    parent = _chunk("p1", _PARENT_TEXT, {"granularity": "parent"})
    spans = sentence_spans(_PARENT_TEXT)
    child = _hit(
        _chunk(
            "c2",
            _PARENT_TEXT[spans[1][0] : spans[1][1]],
            {"granularity": "child", "parent_id": "p1", "sent_start": spans[1][0], "sent_end": spans[1][1]},
        )
    )
    retriever = _retriever(settings)
    widened = retriever._expand_windows([child], {"p1": parent}, window_radius=1)[0]
    assert widened.chunk.id == "c2"
    assert widened.chunk.text in _PARENT_TEXT
    assert len(widened.chunk.text) > len(child.chunk.text)

    exact = retriever._expand_windows([child], {"p1": parent}, window_radius=0)[0]
    assert exact.chunk.text == child.chunk.text


def test_window_text_is_bounded_and_contiguous() -> None:
    spans = sentence_spans(_PARENT_TEXT)
    middle = AdaptiveRetriever._window_text(_PARENT_TEXT, spans[1][0], spans[1][1], 1)
    assert middle == _PARENT_TEXT
    assert middle in _PARENT_TEXT


def test_as_json_carries_expansion_mode_and_window_radius(settings: Settings) -> None:
    data = _retriever(settings).resolve_policy(_request(), {}).as_json()
    assert data["expansion_mode"] == "flat"
    assert data["window_radius"] == settings.sentence_window_radius


def test_expansion_mode_follows_settings_not_harness() -> None:
    settings = Settings(chunking_strategy="parent_child", api_key_pepper="test-pepper")
    policy = _retriever(settings).resolve_policy(_request(), {"retrieval": {"window_radius": 4}})
    assert policy.expansion_mode == "parent_child"
    assert policy.window_radius == 4


def test_window_radius_harness_override_clamps_to_settings_default() -> None:
    settings = Settings(sentence_window_radius=2, api_key_pepper="test-pepper")
    invalid = _retriever(settings).resolve_policy(_request(), {"retrieval": {"window_radius": 9}})
    assert invalid.window_radius == 2


def test_context_hash_changes_with_settings_window_radius() -> None:
    narrow = Settings(sentence_window_radius=1, api_key_pepper="test-pepper")
    wide = Settings(sentence_window_radius=3, api_key_pepper="test-pepper")
    narrow_policy = _retriever(narrow).resolve_policy(_request(), {})
    wide_policy = _retriever(wide).resolve_policy(_request(), {})
    assert narrow_policy.window_radius == 1
    assert wide_policy.window_radius == 3
    assert _context_hash(narrow_policy) != _context_hash(wide_policy)
