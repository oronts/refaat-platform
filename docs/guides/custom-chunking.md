# Custom chunking strategies

Chunking is the one ingestion decision that shapes every downstream retrieval result. Refaat exposes it as a typed `ChunkingStrategy` port so a domain splitter (legal clauses, spreadsheets, source code, OpenAPI specs, meeting transcripts, OCR layouts, CJK text) plugs in exactly like an embedder or a reranker. The three built-in strategies stay the default and are byte-identical to the pre-port chunker, and contextual embedding, HyPE, and RAPTOR compose on any conforming strategy without change.

## The port

```python
from collections.abc import Sequence
from refaat.config import Settings
from refaat.core.chunking import TextChunk
from refaat.domain.models import JsonObject


class ChunkingStrategy(Protocol):
    @property
    def strategy_id(self) -> str: ...

    def chunk(self, text: str, *, metadata: JsonObject, settings: Settings) -> Sequence[TextChunk]: ...
```

- `text` is the normalized document body, the same string the built-in chunker receives.
- `metadata` is the merged document metadata already present at ingestion (`{**document.metadata, **extracted_metadata}`), so a strategy can read `language`, `media_type`, and structural hints without a second parameter.
- `settings` carries the runtime configuration (`default_chunk_tokens`, `chunk_overlap_tokens`, and the rest), so a strategy reads its knobs from one place.
- `strategy_id` is a stable, non-empty identity for the strategy.

## The output contract

A strategy proposes text, ordinal, and structure only. It returns a sequence of `TextChunk`:

```python
@dataclass(frozen=True, slots=True)
class TextChunk:
    ordinal: int  # contiguous from 0 in emission order
    text: str  # non-empty
    token_count: int  # a positive, honest estimate
    heading: str = ""  # section heading, used by contextual embedding
    granularity: str = ""  # "" for flat, else "parent" or "child"
    parent_ordinal: int | None = None  # a child points at its parent's ordinal
    sent_start: int | None = None  # a child's character span into the parent
    sent_end: int | None = None
```

Rules the ingestion pipeline relies on:

- ordinals are contiguous from zero in emission order;
- `text` is non-empty and `token_count` is positive (use `refaat.core.text.estimate_tokens`);
- `heading` is a string (contextual embedding prefixes it before the body);
- for a hierarchical strategy, each `granularity == "child"` chunk carries a `parent_ordinal` that points at an emitted `"parent"` chunk and a `[sent_start, sent_end)` span into that parent's text.

Safety property: the chunk `id` and `checksum` are minted by the pipeline from `(document_id, checksum, ordinal)`, never by the strategy, so a plugin can never forge a chunk id or collide with the reserved summary and hypothetical namespaces.

## A minimal strategy

A fixed-size character-window splitter, the smallest complete example:

```python
from collections.abc import Sequence

from refaat.config import Settings
from refaat.core.chunking import TextChunk
from refaat.core.text import estimate_tokens, normalize_text
from refaat.domain.models import JsonObject


class FixedWindowStrategy:
    strategy_id = "fixed-window"

    def __init__(self, window_chars: int = 800) -> None:
        self._window = window_chars

    def chunk(self, text: str, *, metadata: JsonObject, settings: Settings) -> Sequence[TextChunk]:
        body = normalize_text(text)
        pieces = [body[i : i + self._window] for i in range(0, len(body), self._window)] or [body]
        return [
            TextChunk(ordinal=i, text=piece, token_count=estimate_tokens(piece))
            for i, piece in enumerate(pieces)
            if piece.strip()
        ]
```

## A domain strategy

A Markdown-heading splitter that keeps each section together and records the heading so contextual embedding can use it:

```python
from collections.abc import Sequence

from refaat.config import Settings
from refaat.core.chunking import TextChunk
from refaat.core.text import estimate_tokens, normalize_text
from refaat.domain.models import JsonObject


class MarkdownSectionStrategy:
    strategy_id = "markdown-section"

    def chunk(self, text: str, *, metadata: JsonObject, settings: Settings) -> Sequence[TextChunk]:
        heading = ""
        buffer: list[str] = []
        chunks: list[TextChunk] = []

        def flush() -> None:
            body = "\n".join(buffer).strip()
            if body:
                chunks.append(
                    TextChunk(ordinal=len(chunks), text=body, token_count=estimate_tokens(body), heading=heading)
                )

        for line in normalize_text(text).split("\n"):
            if line.startswith("#"):
                flush()
                heading = line.lstrip("#").strip()
                buffer = []
            else:
                buffer.append(line)
        flush()
        return chunks or [TextChunk(ordinal=0, text=normalize_text(text), token_count=estimate_tokens(text))]
```

## Injecting a strategy

For an application-owned strategy, inject the instance through the builder:

```python
from refaat.application.runtime import RuntimeBuilder

runtime = RuntimeBuilder(settings).with_chunking_strategy(MarkdownSectionStrategy()).build()
```

The injected strategy takes precedence over `REFAAT_CHUNKING_STRATEGY`. Everything downstream is unchanged: the same ingestion pipeline mints ids, embeds, builds the graph, and enforces the citation contract.

## Shipping a strategy as a plugin

A separately packaged strategy is discovered through the `refaat.chunkers` entry-point group and selected by name, gated by the same allowlist and code-hash pin as every other plugin group:

```toml
# in the plugin package's pyproject.toml
[project.entry-points."refaat.chunkers"]
markdown_section = "acme_refaat.chunking:MarkdownSectionStrategy"
```

```bash
export REFAAT_CHUNKING_STRATEGY=markdown_section
export REFAAT_PLUGIN_ALLOWLIST=markdown_section   # only reviewed names are discovered
```

Resolution order at build time:

1. an instance injected with `with_chunking_strategy`;
2. a built-in name (`flat`, `parent_child`, `sentence_window`);
3. a `refaat.chunkers` entry point whose name matches `REFAAT_CHUNKING_STRATEGY` and is allowlisted;
4. otherwise the runtime refuses to build.

A configured strategy that cannot be resolved (an unknown name, an import failure, a non-allowlisted plugin) fails startup rather than silently falling back to `flat`. A misconfigured splitter must never quietly change how a corpus is indexed.

## Composition: contextual embedding, HyPE, and RAPTOR

These three are downstream derivations of the base chunk output and compose on any conforming strategy:

- contextual embedding re-embeds `contextualize(title, heading, text)`, so it depends only on `heading` and `text`;
- HyPE derives hypothetical questions from the stored `chunk.text`;
- RAPTOR builds a summary tree over the stored leaf chunks.

None of them branches on the strategy name. A strategy that honors the output contract (contiguous ordinals, populated `heading`, real `text`) gets all three for free. A strategy that emits gaps or empty headings degrades those derivations, which is exactly what the conformance check guards against.

## The conformance contract

Before trusting a strategy, assert the invariants the pipeline relies on:

```python
def assert_conforms(strategy, settings, fixtures) -> None:
    assert strategy.strategy_id
    for text in fixtures:
        chunks = list(strategy.chunk(text, metadata={}, settings=settings))
        assert chunks
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))  # contiguous from 0
        for chunk in chunks:
            assert chunk.text.strip()  # non-empty text
            assert chunk.token_count > 0  # a positive estimate
            if chunk.granularity == "child":
                parent = chunks[chunk.parent_ordinal]
                assert parent.granularity == "parent"
                assert 0 <= chunk.sent_start <= chunk.sent_end <= len(parent.text)
        # determinism: two runs produce the identical chunk sequence
        again = list(strategy.chunk(text, metadata={}, settings=settings))
        assert [c.text for c in chunks] == [c.text for c in again]
```

## Byte-identity of the built-ins

The default configuration (`flat`, no injection, no plugin) routes through the built-in adapter over the unchanged chunker and produces the identical `TextChunk` sequence, so chunk ids, checksums, and structural metadata do not move. Chunking is not part of the answer-cache identity or any program artifact id: a different chunker changes stored rows only under a new corpus version, which the cache key already tracks, and the pinned program artifact ids are unaffected.

## See also

- `examples/29_custom_chunking_strategy.py` for a runnable, offline-deterministic walkthrough (inject, ingest, and confirm a built-in stays byte-identical).
- [Custom connectors, extractors, skills, and event sinks](custom-extensions.md) for the general plugin flow.
- [Plugins and executable skills](../plugins-and-skills.md) for the trust boundary, allowlist, and pinning.
