from __future__ import annotations

# Custom chunking strategy: inject a domain splitter, prove a built-in stays byte-identical, and
# show contextual retrieval composes on the custom output. Offline and deterministic.
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

from _shared import demo_settings, emit

from refaat import RuntimeBuilder
from refaat.config import Settings
from refaat.core.chunking import BUILTIN_CHUNKING_STRATEGIES, AdaptiveChunker, TextChunk
from refaat.core.text import estimate_tokens, normalize_text, split_sentences
from refaat.domain.models import DocumentInput, JsonObject
from refaat.errors import ConfigurationError

_BODY = (
    "The alpha subsystem caches grounded answers reliably. "
    "It reuses those answers across repeated retrieval requests. "
    "The beta subsystem fuses semantic and lexical candidates before packing them. "
    "The budgeter packs the answer context tightly and cites every chunk it uses."
)


class SentenceStrategy:
    """A domain splitter that emits one chunk per sentence. Ignores metadata and settings."""

    strategy_id = "one-per-sentence"

    def chunk(self, text: str, *, metadata: JsonObject, settings: Settings) -> Sequence[TextChunk]:
        sentences = [s for s in split_sentences(normalize_text(text)) if s.strip()]
        return [TextChunk(ordinal=i, text=s, token_count=estimate_tokens(s)) for i, s in enumerate(sentences)]


def main() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ("custom", "identity", "compose", "unknown"):
            (root / name).mkdir()
        document = DocumentInput(
            source_id="handbook", title="handbook", media_type="text/plain", content=_BODY.encode(), metadata={}
        )
        expected_sentences = len([s for s in split_sentences(normalize_text(_BODY)) if s.strip()])

        # 1. Inject the custom strategy and confirm it drives ingestion.
        with RuntimeBuilder(demo_settings(root / "custom")).with_chunking_strategy(SentenceStrategy()).build() as rt:
            ws = rt.create_workspace("chunkdemo", "Chunk Demo")
            corpus = rt.create_corpus(ws, "handbook")
            rt.ingest(ws, corpus, document)
            custom_chunks = rt.store.list_chunks(ws, corpus)
            custom_chunk_count = len(custom_chunks)

        # 2. Prove a built-in strategy is byte-identical to the direct chunker.
        settings = demo_settings(root / "identity")
        builtin = BUILTIN_CHUNKING_STRATEGIES["flat"](settings)
        direct = AdaptiveChunker(
            target_tokens=settings.default_chunk_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
            strategy="flat",
        ).chunk(_BODY)
        via_port = list(builtin.chunk(_BODY, metadata={}, settings=settings))
        builtin_byte_identical = [(c.ordinal, c.text, c.token_count) for c in via_port] == [
            (c.ordinal, c.text, c.token_count) for c in direct
        ]

        # 3. Contextual retrieval composes on the custom strategy: a query still grounds and cites a real chunk.
        composed_settings = demo_settings(root / "compose").model_copy(update={"contextual_retrieval": True})
        with RuntimeBuilder(composed_settings).with_chunking_strategy(SentenceStrategy()).build() as rt:
            ws = rt.create_workspace("chunkdemo", "Chunk Demo")
            corpus = rt.create_corpus(ws, "handbook")
            rt.ingest(ws, corpus, document)
            project = rt.project(workspace="chunkdemo", corpus="handbook")
            response = project.ask("How are candidates fused?")
            stored_ids = {chunk.id for chunk in rt.store.list_chunks(ws, corpus)}
            composed_grounded = bool(response.citations) and all(
                citation.chunk_id in stored_ids for citation in response.citations
            )

        # 4. A misconfigured strategy fails startup rather than silently falling back to flat.
        unknown_settings = demo_settings(root / "unknown").model_copy(update={"chunking_strategy": "does-not-exist"})
        try:
            RuntimeBuilder(unknown_settings).build()
            unknown_refused = False
        except ConfigurationError:
            unknown_refused = True

        emit(
            {
                "custom_strategy_id": SentenceStrategy.strategy_id,
                "custom_chunk_count": custom_chunk_count,
                "custom_matches_sentences": custom_chunk_count == expected_sentences,
                "builtin_flat_byte_identical": builtin_byte_identical,
                "composed_answer_grounded": composed_grounded,
                "unknown_strategy_refused": unknown_refused,
            }
        )


if __name__ == "__main__":
    main()
