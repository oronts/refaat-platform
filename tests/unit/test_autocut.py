from __future__ import annotations

from refaat.core.retrieval import autocut
from refaat.domain.models import Chunk, SearchHit


def _hit(chunk_id: str, score: float) -> SearchHit:
    chunk = Chunk(
        id=chunk_id,
        corpus_id="c",
        document_id="d",
        ordinal=0,
        text=chunk_id,
        token_count=1,
        checksum="x",
        metadata={},
        embedding=(0.0,),
    )
    return SearchHit(chunk=chunk, score=score)


def _hits(scores: list[float]) -> list[SearchHit]:
    return [_hit(f"c{index}", score) for index, score in enumerate(scores)]


def test_autocut_trims_at_the_largest_drop() -> None:
    trimmed, dropped = autocut(_hits([1.0, 0.95, 0.9, 0.2, 0.15]), 0.5)
    assert dropped is True
    assert [hit.chunk.id for hit in trimmed] == ["c0", "c1", "c2"]


def test_autocut_uniform_scores_do_not_cut() -> None:
    trimmed, dropped = autocut(_hits([0.5, 0.5, 0.5]), 0.5)
    assert dropped is False
    assert len(trimmed) == 3


def test_autocut_high_threshold_does_not_cut() -> None:
    trimmed, dropped = autocut(_hits([1.0, 0.95, 0.9, 0.2, 0.15]), 0.99)
    assert dropped is False
    assert len(trimmed) == 5


def test_autocut_breaks_ties_at_the_earliest_gap() -> None:
    trimmed, dropped = autocut(_hits([1.0, 0.5, 0.5, 0.0]), 0.3)
    assert dropped is True
    assert [hit.chunk.id for hit in trimmed] == ["c0"]


def test_autocut_single_and_empty_are_unchanged() -> None:
    single = _hits([0.9])
    assert autocut(single, 0.5) == (tuple(single), False)
    assert autocut([], 0.5) == ((), False)


def test_autocut_never_empties_a_nonempty_list() -> None:
    trimmed, _dropped = autocut(_hits([1.0, 0.0]), 0.1)
    assert len(trimmed) >= 1
