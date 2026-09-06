from __future__ import annotations

from refaat.core.compression import ContextCompressor, _rewrite_hit_text
from refaat.core.text import estimate_tokens, overlap_ratio
from refaat.domain.models import Chunk, SearchHit


def _hit(chunk_id: str, text: str, score: float = 1.0) -> SearchHit:
    chunk = Chunk(
        id=chunk_id,
        corpus_id="c",
        document_id="d",
        ordinal=0,
        text=text,
        token_count=estimate_tokens(text),
        checksum="x",
        metadata={},
        embedding=(),
    )
    return SearchHit(chunk=chunk, score=score)


def test_overlap_ratio_scores_query_term_density() -> None:
    assert overlap_ratio({"cache", "identity"}, "The cache identity is stable.") > 0.0
    assert overlap_ratio({"cache"}, "Birds fly south for winter.") == 0.0
    assert overlap_ratio(set(), "Anything at all here.") == 0.0
    assert overlap_ratio({"cache"}, "") == 0.0


def test_rewrite_hit_text_recomputes_or_pins_token_count() -> None:
    hit = _hit("chunk_a", "Alpha covers retrieval. Beta covers caching.")
    rewritten = _rewrite_hit_text(hit, "Alpha covers retrieval.")
    assert rewritten.chunk.id == "chunk_a"
    assert rewritten.chunk.text == "Alpha covers retrieval."
    assert rewritten.chunk.token_count == estimate_tokens("Alpha covers retrieval.")
    pinned = _rewrite_hit_text(hit, "Alpha covers retrieval.", token_count=2)
    assert pinned.chunk.token_count == 2


def test_disabled_returns_input_unchanged() -> None:
    hits = (_hit("chunk_a", "Alpha covers retrieval. Beta covers caching."),)
    assert ContextCompressor().compress("caching", hits, enabled=False, min_salience=0.0) == hits


def test_cross_chunk_duplicate_sentence_dropped_from_later_chunk() -> None:
    first = _hit("chunk_a", "Alpha covers the retrieval policy. Beta covers the cache identity.")
    second = _hit("chunk_b", "Beta covers the cache identity. Gamma covers the token budget.")
    compressed = ContextCompressor().compress(
        "cache identity",
        (first, second),
        enabled=True,
        min_salience=0.0,
    )
    assert [hit.chunk.id for hit in compressed] == ["chunk_a", "chunk_b"]
    assert "Beta covers the cache identity." in compressed[0].chunk.text
    assert "Beta covers the cache identity." not in compressed[1].chunk.text
    assert "Gamma covers the token budget." in compressed[1].chunk.text


def test_fully_duplicate_chunk_is_dropped() -> None:
    first = _hit("chunk_a", "Alpha covers retrieval. Beta covers caching.")
    duplicate = _hit("chunk_b", "Alpha covers retrieval. Beta covers caching.")
    compressed = ContextCompressor().compress(
        "caching",
        (first, duplicate),
        enabled=True,
        min_salience=0.0,
    )
    assert [hit.chunk.id for hit in compressed] == ["chunk_a"]


def test_min_salience_drops_off_topic_but_keeps_top_sentence() -> None:
    hit = _hit(
        "chunk_a",
        "The cache identity stays stable. Weather is pleasant today. Birds migrate south.",
    )
    compressed = ContextCompressor().compress(
        "cache identity",
        (hit,),
        enabled=True,
        min_salience=0.2,
    )
    assert len(compressed) == 1
    text = compressed[0].chunk.text
    assert "cache identity" in text
    assert "Weather is pleasant today." not in text
    assert "Birds migrate south." not in text


def test_all_below_threshold_still_retains_the_single_top_sentence() -> None:
    hit = _hit("chunk_a", "Weather is pleasant today. Birds migrate south. Rivers run east.")
    compressed = ContextCompressor().compress("zzz", (hit,), enabled=True, min_salience=0.5)
    assert len(compressed) == 1
    assert compressed[0].chunk.text == "Weather is pleasant today."


def test_compressed_chunk_preserves_id_and_recomputes_token_count() -> None:
    first = _hit("chunk_a", "Beta covers the cache identity.")
    second = _hit("chunk_b", "Beta covers the cache identity. Gamma covers the token budget.")
    compressed = ContextCompressor().compress(
        "token budget",
        (first, second),
        enabled=True,
        min_salience=0.0,
    )
    kept = compressed[1]
    assert kept.chunk.id == "chunk_b"
    assert kept.chunk.text == "Gamma covers the token budget."
    assert kept.chunk.token_count == estimate_tokens(kept.chunk.text)


def test_compression_is_deterministic_and_byte_identical_across_runs() -> None:
    hits = (
        _hit("chunk_a", "Alpha covers the retrieval policy. Beta covers the cache identity."),
        _hit("chunk_b", "Beta covers the cache identity. Gamma covers the token budget."),
    )
    first = ContextCompressor().compress("cache identity", hits, enabled=True, min_salience=0.0)
    second = ContextCompressor().compress("cache identity", hits, enabled=True, min_salience=0.0)
    assert [(h.chunk.id, h.chunk.text, h.chunk.token_count) for h in first] == [
        (h.chunk.id, h.chunk.text, h.chunk.token_count) for h in second
    ]
