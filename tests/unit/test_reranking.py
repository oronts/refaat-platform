from __future__ import annotations

from refaat.application.runtime import RuntimeBuilder
from refaat.config import Settings
from refaat.domain.models import Chunk, SearchHit
from refaat.infrastructure.reranking import LexicalReranker


def _hit(chunk_id: str, text: str, score: float) -> SearchHit:
    chunk = Chunk(
        id=chunk_id,
        corpus_id="c",
        document_id="d",
        ordinal=0,
        text=text,
        token_count=len(text.split()),
        checksum="x",
        metadata={},
        embedding=(0.0,),
    )
    return SearchHit(chunk=chunk, score=score)


def test_lexical_reranker_promotes_higher_query_overlap() -> None:
    reranker = LexicalReranker(weight=1.0)
    hits = [
        _hit("a", "unrelated filler content", 0.5),
        _hit("b", "rollback restores the prior artifact", 0.4),
    ]
    ranked = reranker.rerank("how does rollback work", hits)
    assert ranked[0].chunk.id == "b"


def test_lexical_reranker_passes_through_on_empty_query() -> None:
    reranker = LexicalReranker()
    hits = (_hit("a", "text one", 0.9), _hit("b", "text two", 0.1))
    assert reranker.rerank("", hits) == hits


def test_builder_reranker_factory_and_injection() -> None:
    assert RuntimeBuilder(Settings())._reranker() is None
    assert isinstance(RuntimeBuilder(Settings(reranker_provider="lexical"))._reranker(), LexicalReranker)
    custom = LexicalReranker(weight=0.2)
    assert RuntimeBuilder(Settings()).with_reranker(custom)._reranker() is custom
