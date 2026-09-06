from __future__ import annotations

from refaat.core.chunking import AdaptiveChunker, contextualize
from refaat.core.text import sha256_text, stable_digest

_INTRO = "Intro text appears before any heading and is long enough to stand alone as its own chunk here today."
_H1 = "## Section One"
_B1 = "The caching subsystem keeps grounded answers stable across many repeated retrieval requests over a long window."
_H2 = "## Section Two"
_B2 = "The retrieval stage fuses semantic and lexical candidates before the budgeter finally packs the answer context."
_FIXTURE = "\n\n".join([_INTRO, f"{_H1}\n{_B1}", f"{_H2}\n{_B2}"])


def _chunks() -> list:
    return AdaptiveChunker(target_tokens=32, overlap_tokens=0).chunk(_FIXTURE)


def test_heading_is_stamped_and_inherited_with_none_before_first_heading() -> None:
    chunks = _chunks()
    assert [chunk.heading for chunk in chunks] == ["", "Section One", "Section Two"]


def test_heading_lines_stay_inline_and_text_matches_pre_slice_output() -> None:
    chunks = _chunks()
    assert [chunk.ordinal for chunk in chunks] == [0, 1, 2]
    assert [chunk.text for chunk in chunks] == [_INTRO, f"{_H1}\n{_B1}", f"{_H2}\n{_B2}"]


def test_chunk_ids_and_checksums_are_independent_of_contextualization() -> None:
    chunks = _chunks()
    for chunk in chunks:
        raw_checksum = sha256_text(chunk.text)
        contextual_input = contextualize("Handbook", chunk.heading, chunk.text)
        assert sha256_text(chunk.text) == raw_checksum
        assert contextual_input != chunk.text


def test_inline_number_sign_in_prose_is_not_a_heading() -> None:
    body = "First body paragraph without any heading that is reasonably long to stand on its own here today."
    prose = "Body with a #tag and a #5 cost appearing inline here in this second longer paragraph now today."
    chunks = AdaptiveChunker(target_tokens=32, overlap_tokens=0).chunk(f"{body}\n\n{prose}")
    assert all(chunk.heading == "" for chunk in chunks)


def test_contextualize_builds_title_and_heading_prefix() -> None:
    assert contextualize("Doc Title", "Section One", "Body text.") == "Doc Title\nSection One\nBody text."


def test_contextualize_returns_raw_text_when_title_and_heading_empty() -> None:
    assert contextualize("", "", "Body text.") == "Body text."
    assert contextualize("Doc Title", "", "Body text.") == "Doc Title\nBody text."


def test_contextualize_does_not_duplicate_a_heading_already_inline() -> None:
    result = contextualize("Doc", "Section One", "## Section One\nBody.")
    assert result == "Doc\n## Section One\nBody."
    assert result.count("Section One") == 1


def test_contextualize_does_not_duplicate_a_closed_atx_heading_already_inline() -> None:
    result = contextualize("Doc", "Section One", "## Section One ##\nBody.")
    assert result == "Doc\n## Section One ##\nBody."
    assert result.count("Section One") == 1


def test_heading_bearing_chunk_id_is_byte_stable_against_pre_slice_output() -> None:
    heading_chunk = next(chunk for chunk in _chunks() if chunk.heading == "Section One")
    checksum = sha256_text(heading_chunk.text)
    chunk_id = f"chunk_{stable_digest(('doc-1', checksum, str(heading_chunk.ordinal)), length=48)}"
    assert chunk_id == "chunk_7d0aa99794be4e4b9d5c9c9b7100e97935c7a9f093c00a91"
