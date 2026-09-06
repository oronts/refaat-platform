import pytest

from refaat.core.citations import (
    build_citations,
    filter_cited_chunks,
    strip_uncited_markers,
    validate_cited_chunks,
)
from refaat.domain.models import Chunk
from refaat.errors import ValidationError


def test_strip_uncited_markers_removes_only_uncited_markers() -> None:
    text = "valid [chunk:real] invented [chunk:missing] and [chunk:gone]."
    assert strip_uncited_markers(text, ["real"]) == "valid [chunk:real] invented and."


def test_strip_uncited_markers_is_byte_identical_when_every_marker_is_cited() -> None:
    text = "a [chunk:one] b [chunk:two] c"
    assert strip_uncited_markers(text, ["one", "two"]) == text
    assert strip_uncited_markers("no markers here", ["one"]) == "no markers here"
    assert strip_uncited_markers("", []) == ""


def test_citations_are_deduplicated_and_bound_to_context() -> None:
    chunk = Chunk(
        id="chunk_1",
        corpus_id="corpus",
        document_id="doc",
        ordinal=0,
        text="Evidence text",
        token_count=2,
        checksum="checksum",
        metadata={},
        embedding=(1.0,),
    )
    ids = validate_cited_chunks(["chunk_1", "chunk_1"], [chunk])
    citations = build_citations(ids, [chunk])

    assert ids == ("chunk_1",)
    assert citations[0].document_id == "doc"
    with pytest.raises(ValidationError):
        validate_cited_chunks(["chunk_unknown"], [chunk])


def test_filter_cited_chunks_drops_unknown_and_dedupes() -> None:
    chunk = Chunk(
        id="chunk_1",
        corpus_id="corpus",
        document_id="doc",
        ordinal=0,
        text="Evidence text",
        token_count=2,
        checksum="checksum",
        metadata={},
        embedding=(1.0,),
    )
    assert filter_cited_chunks(["chunk_1", "chunk_unknown", "chunk_1"], [chunk]) == ("chunk_1",)
    assert filter_cited_chunks(["chunk_unknown"], [chunk]) == ()
