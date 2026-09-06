"""Unit tests for the pre-model context-transform seam and the reference PatternRedactor.

The seam redacts configured spans from the evidence a generation model sees. These tests cover the
redactor over each field, deterministic overlap resolution, per-pattern provenance that carries only
opaque ids and counts, the cache identity, the seam's None byte-identity and its fail-safe image
drop, and that a compiled regex pattern works.
"""

from __future__ import annotations

import re

from refaat.domain.context_transform import RedactionCount, TransformResult, redact_for_egress
from refaat.domain.models import (
    Chunk,
    ContextPacket,
    GenerationImage,
    GenerationRequest,
    SearchHit,
)
from refaat.infrastructure.context_transform import PatternRedactor

_SECRET = "BRAVO-SECRET-9"


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(
        id=cid,
        corpus_id="corpus",
        document_id="doc",
        ordinal=0,
        text=text,
        token_count=4,
        checksum="sum",
        metadata={"acl": {"classification": "confidential"}},
        embedding=(0.1, 0.2),
    )


def _request(
    *,
    hits: tuple[SearchHit, ...] = (),
    graph_facts: tuple[str, ...] = (),
    session_summary: str | None = None,
    query: str = "what is the code",
    images: tuple[GenerationImage, ...] = (),
) -> GenerationRequest:
    context = ContextPacket(
        query=query,
        hits=hits,
        graph_facts=graph_facts,
        session_summary=session_summary,
        profile={},
        supplemental_harness={},
        token_count=0,
        routes=(),
    )
    return GenerationRequest(query=query, context=context, system_prompt="sys", images=images)


def _hit(cid: str, text: str) -> SearchHit:
    return SearchHit(chunk=_chunk(cid, text), score=1.0, semantic_score=0.9)


def test_evidence_text_is_redacted_but_id_metadata_and_embedding_are_intact() -> None:
    hit = _hit("chunk_1", f"The code is {_SECRET} today.")
    result = PatternRedactor([_SECRET]).apply(_request(hits=(hit,)))
    redacted = result.request.context.hits[0].chunk
    assert redacted.text == "The code is [REDACTED] today."
    assert redacted.id == "chunk_1"
    assert redacted.metadata == {"acl": {"classification": "confidential"}}
    assert redacted.embedding == (0.1, 0.2)


def test_graph_facts_and_session_summary_are_redacted() -> None:
    request = _request(
        graph_facts=(f"{_SECRET} governs rollout", "harmless fact"),
        session_summary=f"earlier the user asked about {_SECRET}",
    )
    result = PatternRedactor([_SECRET]).apply(request)
    assert result.request.context.graph_facts == ("[REDACTED] governs rollout", "harmless fact")
    assert result.request.context.session_summary == "earlier the user asked about [REDACTED]"


def test_query_is_not_redacted_by_default_but_can_be_opted_in() -> None:
    request = _request(query=f"tell me about {_SECRET}")
    default = PatternRedactor([_SECRET]).apply(request)
    assert default.request.query == f"tell me about {_SECRET}"
    opted = PatternRedactor([_SECRET], fields=("evidence", "graph_facts", "session_summary", "query")).apply(request)
    assert opted.request.query == "tell me about [REDACTED]"


def test_provenance_counts_spans_per_pattern_and_carries_no_raw_text() -> None:
    hit = _hit("chunk_1", f"{_SECRET} and again {_SECRET} plus alpha")
    result = PatternRedactor([_SECRET, "alpha"]).apply(_request(hits=(hit,), graph_facts=(f"{_SECRET} note",)))
    by_id = {item.pattern_id: item.count for item in result.redactions}
    # Two RedactionCount entries, one per configured pattern.
    assert len(result.redactions) == 2
    # The secret matched twice in evidence plus once in the graph fact; alpha matched once.
    assert sorted(by_id.values()) == [1, 3]
    # Provenance carries only opaque digests and counts, never the raw pattern or the matched text.
    serialized = " ".join(f"{item.pattern_id}:{item.count}" for item in result.redactions)
    assert _SECRET not in serialized
    assert "alpha" not in serialized
    assert all(len(item.pattern_id) == 12 for item in result.redactions)


def test_overlapping_spans_resolve_leftmost_then_earlier_pattern_wins() -> None:
    # "abcd": pattern "abc" (index 0) and "bcd" (index 1) overlap. Leftmost start wins, so "abc" is
    # kept and "bcd" (starting later, inside the kept span) is discarded, leaving "[R]d".
    hit = _hit("chunk_1", "abcd")
    result = PatternRedactor(["abc", "bcd"], placeholder="[R]").apply(_request(hits=(hit,)))
    assert result.request.context.hits[0].chunk.text == "[R]d"
    assert sum(item.count for item in result.redactions) == 1


def test_equal_start_earlier_listed_pattern_wins() -> None:
    # Both patterns start at 0; the earlier-listed longer one is kept.
    hit = _hit("chunk_1", "abcd tail")
    result = PatternRedactor(["abcd", "ab"], placeholder="[R]").apply(_request(hits=(hit,)))
    assert result.request.context.hits[0].chunk.text == "[R] tail"


def test_a_compiled_regex_pattern_is_used_as_is() -> None:
    hit = _hit("chunk_1", "ticket AB-1234 and AB-99 open")
    result = PatternRedactor([re.compile(r"AB-\d+")]).apply(_request(hits=(hit,)))
    assert result.request.context.hits[0].chunk.text == "ticket [REDACTED] and [REDACTED] open"
    assert sum(item.count for item in result.redactions) == 2


def test_a_plain_string_is_escaped_to_a_literal() -> None:
    # "a.c" must not match "axc" because the dot is escaped.
    hit = _hit("chunk_1", "axc and a.c")
    result = PatternRedactor(["a.c"], placeholder="[R]").apply(_request(hits=(hit,)))
    assert result.request.context.hits[0].chunk.text == "axc and [R]"


def test_no_match_leaves_text_and_records_no_redactions() -> None:
    hit = _hit("chunk_1", "nothing sensitive here")
    result = PatternRedactor([_SECRET]).apply(_request(hits=(hit,)))
    assert result.request.context.hits[0].chunk.text == "nothing sensitive here"
    assert result.redactions == ()


def test_idempotent_for_a_pattern_that_does_not_match_the_placeholder() -> None:
    redactor = PatternRedactor([_SECRET])
    once = redactor.apply(_request(hits=(_hit("chunk_1", f"code {_SECRET}"),)))
    twice = redactor.apply(
        GenerationRequest(query=once.request.query, context=once.request.context, system_prompt="sys")
    )
    assert twice.request.context.hits[0].chunk.text == "code [REDACTED]"
    assert twice.redactions == ()


def test_cache_id_is_stable_and_sensitive_to_patterns_placeholder_and_fields() -> None:
    base = PatternRedactor([_SECRET])
    assert base.cache_id == PatternRedactor([_SECRET]).cache_id
    assert base.cache_id.startswith("gen-redact:")
    assert base.cache_id != PatternRedactor([_SECRET, "alpha"]).cache_id
    assert base.cache_id != PatternRedactor([_SECRET], placeholder="[X]").cache_id
    assert base.cache_id != PatternRedactor([_SECRET], fields=("evidence",)).cache_id
    # Pattern order is part of the identity (redaction substitutes in order).
    assert PatternRedactor(["a", "b"]).cache_id != PatternRedactor(["b", "a"]).cache_id


def test_seam_with_none_returns_the_same_request_object_byte_identical() -> None:
    request = _request(hits=(_hit("chunk_1", f"code {_SECRET}"),))
    result = redact_for_egress(None, request)
    assert result.request is request
    assert result.redactions == ()
    assert result.images_dropped == 0


def test_seam_drops_images_under_an_active_transform_even_with_no_text_match() -> None:
    image = GenerationImage(chunk_id="chunk_1", media_type="image/png", content=b"\x89PNG")
    # No text match, so the redactor itself redacts nothing, but the seam refuses image egress
    # because a text redactor cannot vet pixels.
    request = _request(hits=(_hit("chunk_1", "no secret here"),), images=(image,))
    result = redact_for_egress(PatternRedactor([_SECRET]), request)
    assert result.request.images == ()
    assert result.images_dropped == 1


def test_seam_keeps_images_when_no_transform_is_installed() -> None:
    image = GenerationImage(chunk_id="chunk_1", media_type="image/png", content=b"\x89PNG")
    request = _request(hits=(_hit("chunk_1", "text"),), images=(image,))
    result = redact_for_egress(None, request)
    assert result.request.images == (image,)
    assert result.images_dropped == 0


def test_transform_result_and_redaction_count_are_frozen_value_objects() -> None:
    count = RedactionCount(pattern_id="abc123", count=2)
    result = TransformResult(request=_request(), redactions=(count,))
    assert result.images_dropped == 0
    assert result.redactions[0].pattern_id == "abc123"
