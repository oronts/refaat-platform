import pytest

from refaat.core.chunking import AdaptiveChunker

_SECTIONS = "\n\n".join(
    [
        (
            "The alpha subsystem caches grounded answers reliably. "
            "It reuses those answers across repeated retrieval requests. "
            "Every cached answer stays stable for a long window."
        ),
        (
            "The beta subsystem fuses semantic and lexical candidates. "
            "It ranks the fused candidates before packing them. "
            "The budgeter finally packs the answer context tightly."
        ),
    ]
)


def test_chunker_preserves_content_and_overlap() -> None:
    text = "\n\n".join(
        [
            "Alpha is the first section with several meaningful words for testing.",
            "Beta is the second section and contains a separate operational rule.",
            "Gamma is the third section and closes the document with another rule.",
        ]
    )
    chunks = AdaptiveChunker(target_tokens=32, overlap_tokens=5).chunk(text)

    assert len(chunks) >= 2
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.text.strip() for chunk in chunks)
    assert "Alpha" in chunks[0].text
    assert "Gamma" in chunks[-1].text


def test_chunker_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError, match="at least 32"):
        AdaptiveChunker(target_tokens=20)


def test_chunker_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="strategy"):
        AdaptiveChunker(target_tokens=64, overlap_tokens=0, strategy="bogus")


def test_flat_output_omits_structural_fields() -> None:
    chunks = AdaptiveChunker(target_tokens=48, overlap_tokens=0).chunk(_SECTIONS)
    assert chunks, "flat chunking must produce chunks"
    for chunk in chunks:
        assert chunk.granularity == ""
        assert chunk.parent_ordinal is None
        assert chunk.sent_start is None
        assert chunk.sent_end is None


def test_hierarchical_emits_parents_and_children_with_spans() -> None:
    chunker = AdaptiveChunker(target_tokens=48, overlap_tokens=0, strategy="parent_child", child_target_tokens=32)
    flat = AdaptiveChunker(target_tokens=48, overlap_tokens=0).chunk(_SECTIONS)
    chunks = chunker.chunk(_SECTIONS)

    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    parents = [chunk for chunk in chunks if chunk.granularity == "parent"]
    children = [chunk for chunk in chunks if chunk.granularity == "child"]
    assert len(parents) == len(flat)
    assert [parent.text for parent in parents] == [chunk.text for chunk in flat]
    assert len(children) > len(parents)

    parent_ordinals = {parent.ordinal for parent in parents}
    parent_text_by_ordinal = {parent.ordinal: parent.text for parent in parents}
    for child in children:
        assert child.parent_ordinal in parent_ordinals
        assert isinstance(child.sent_start, int)
        assert isinstance(child.sent_end, int)
        assert child.token_count <= 48
        parent_text = parent_text_by_ordinal[child.parent_ordinal]
        assert parent_text[child.sent_start : child.sent_end] == child.text
        assert child.text in parent_text


def test_hierarchical_output_is_deterministic() -> None:
    chunker = AdaptiveChunker(target_tokens=48, overlap_tokens=0, strategy="sentence_window")
    assert chunker.chunk(_SECTIONS) == chunker.chunk(_SECTIONS)


def test_split_large_packs_all_sentences_in_order() -> None:
    paragraph = " ".join(f"Sentence number {index} carries several meaningful tokens here." for index in range(1, 7))
    chunks = AdaptiveChunker(target_tokens=32, overlap_tokens=0).chunk(paragraph)
    rejoined = " ".join(chunk.text for chunk in chunks)
    assert len(chunks) >= 2
    for index in range(1, 7):
        assert f"Sentence number {index}" in rejoined
