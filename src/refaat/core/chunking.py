from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from refaat.core.text import estimate_tokens, normalize_text, sentence_spans, split_sentences
from refaat.domain.models import JsonObject

if TYPE_CHECKING:
    from refaat.config import Settings

CHUNKING_STRATEGIES = ("flat", "parent_child", "sentence_window")

# ATX heading detection is per line (re.MULTILINE via match on each line) so a
# heading that opens a paragraph without a blank line before its body is still
# recognized. Only a leading run of 1 to 6 number signs followed by whitespace
# and heading text counts; an inline number sign inside prose is left untouched.
ATX_HEADING_PATTERN = re.compile(r"^#{1,6}[ \t]+(\S.*?)[ \t]*#*[ \t]*$")


def _heading_text(line: str) -> str | None:
    match = ATX_HEADING_PATTERN.match(line)
    if match is None:
        return None
    return match.group(1).strip() or None


def _heading_inline(heading: str, text: str) -> bool:
    target = heading.casefold()
    return any((_heading_text(line) or line.lstrip("#").strip()).casefold() == target for line in text.split("\n"))


def contextualize(title: str, heading: str, text: str) -> str:
    """Build the embedding-time input by prefixing the document title and section heading.

    The stored and cited chunk text is never changed; this prefix only steers the
    embedding. A heading already present inline in the chunk body is not repeated,
    and when both title and heading are empty the raw text is returned unchanged.
    """
    parts: list[str] = []
    title_clean = (title or "").strip()
    heading_clean = (heading or "").strip()
    if title_clean:
        parts.append(title_clean)
    if heading_clean and not _heading_inline(heading_clean, text):
        parts.append(heading_clean)
    if not parts:
        return text
    return "\n".join([*parts, text])


@dataclass(frozen=True, slots=True)
class TextChunk:
    ordinal: int
    text: str
    token_count: int
    heading: str = ""
    # Structural fields default to the flat shape so flat output is byte-identical:
    # granularity stays "" (the ingestion layer omits the metadata key entirely)
    # and the parent linkage and sentence span stay absent. Hierarchical chunking
    # sets granularity to "parent" or "child"; a child also carries its parent's
    # ordinal and the character span [sent_start, sent_end) into the parent text.
    granularity: str = ""
    parent_ordinal: int | None = None
    sent_start: int | None = None
    sent_end: int | None = None


@dataclass(frozen=True, slots=True)
class _ChildPiece:
    text: str
    token_count: int
    sent_start: int
    sent_end: int


class AdaptiveChunker:
    def __init__(
        self,
        target_tokens: int = 420,
        overlap_tokens: int = 64,
        *,
        strategy: str = "flat",
        child_target_tokens: int | None = None,
    ) -> None:
        if target_tokens < 32:
            raise ValueError("target_tokens must be at least 32")
        if overlap_tokens < 0 or overlap_tokens >= target_tokens:
            raise ValueError("overlap_tokens must be non-negative and smaller than target_tokens")
        if strategy not in CHUNKING_STRATEGIES:
            raise ValueError(f"strategy must be one of {CHUNKING_STRATEGIES}")
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.strategy = strategy
        self.child_target_tokens = max(32, target_tokens // 3) if child_target_tokens is None else child_target_tokens

    def chunk(self, text: str) -> list[TextChunk]:
        if self.strategy == "flat":
            return self._chunk_flat(text)
        return self._chunk_hierarchical(text)

    def _chunk_hierarchical(self, text: str) -> list[TextChunk]:
        parents = self._chunk_flat(text)
        result: list[TextChunk] = []
        ordinal = 0
        for parent in parents:
            parent_ordinal = ordinal
            result.append(
                TextChunk(
                    ordinal=ordinal,
                    text=parent.text,
                    token_count=parent.token_count,
                    heading=parent.heading,
                    granularity="parent",
                )
            )
            ordinal += 1
            for child in self._split_children(parent.text):
                result.append(
                    TextChunk(
                        ordinal=ordinal,
                        text=child.text,
                        token_count=child.token_count,
                        heading=parent.heading,
                        granularity="child",
                        parent_ordinal=parent_ordinal,
                        sent_start=child.sent_start,
                        sent_end=child.sent_end,
                    )
                )
                ordinal += 1
        return result

    def _split_children(self, parent_text: str) -> list[_ChildPiece]:
        spans = sentence_spans(parent_text)
        if len(spans) <= 1:
            return [
                _ChildPiece(
                    text=parent_text,
                    token_count=estimate_tokens(parent_text),
                    sent_start=0,
                    sent_end=len(parent_text),
                )
            ]
        token_counts = [estimate_tokens(parent_text[start:end]) for start, end in spans]
        pieces: list[_ChildPiece] = []
        for group_start, group_end in self._pack(token_counts, self.child_target_tokens):
            char_start = spans[group_start][0]
            char_end = spans[group_end - 1][1]
            body = parent_text[char_start:char_end]
            pieces.append(
                _ChildPiece(
                    text=body,
                    token_count=estimate_tokens(body),
                    sent_start=char_start,
                    sent_end=char_end,
                )
            )
        return pieces

    def _chunk_flat(self, text: str) -> list[TextChunk]:
        paragraphs = [normalize_text(value) for value in text.split("\n\n")]
        units: list[tuple[str, str]] = []
        current_heading = ""
        for paragraph in paragraphs:
            if not paragraph:
                continue
            stamped_heading, current_heading = self._paragraph_headings(paragraph, current_heading)
            if estimate_tokens(paragraph) <= self.target_tokens:
                units.append((paragraph, stamped_heading))
            else:
                units.extend((piece, stamped_heading) for piece in self._split_large(paragraph))

        chunks: list[TextChunk] = []
        current: list[tuple[str, str]] = []
        token_count = 0
        for unit in units:
            unit_tokens = estimate_tokens(unit[0])
            if current and token_count + unit_tokens > self.target_tokens:
                self._append_chunk(chunks, current)
                current = self._overlap_tail(current)
                token_count = sum(estimate_tokens(value[0]) for value in current)
            current.append(unit)
            token_count += unit_tokens
        if current:
            self._append_chunk(chunks, current)
        return chunks

    @staticmethod
    def _paragraph_headings(paragraph: str, current_heading: str) -> tuple[str, str]:
        lines = paragraph.split("\n")
        found = [heading for heading in (_heading_text(line) for line in lines) if heading is not None]
        first = _heading_text(lines[0])
        stamped = first if first is not None else current_heading
        next_heading = found[-1] if found else current_heading
        return stamped, next_heading

    @staticmethod
    def _append_chunk(chunks: list[TextChunk], units: list[tuple[str, str]]) -> None:
        value = "\n\n".join(text for text, _ in units)
        if chunks and value == chunks[-1].text:
            return
        chunks.append(TextChunk(len(chunks), value, estimate_tokens(value), units[-1][1]))

    def _split_large(self, paragraph: str) -> list[str]:
        sentences = split_sentences(paragraph)
        if len(sentences) <= 1:
            words = paragraph.split()
            return [
                " ".join(words[start : start + self.target_tokens])
                for start in range(0, len(words), self.target_tokens)
            ]
        token_counts = [estimate_tokens(sentence) for sentence in sentences]
        return [
            " ".join(sentences[group_start:group_end])
            for group_start, group_end in self._pack(token_counts, self.target_tokens)
        ]

    @staticmethod
    def _pack(token_counts: Sequence[int], target: int) -> list[tuple[int, int]]:
        """Greedily group consecutive units into packs of at most ``target`` tokens.

        Returns each pack as a half-open ``[start, end)`` index range into the
        input. A single oversized unit still forms its own pack. Shared by the flat
        paragraph splitter (which joins the grouped sentences into strings) and the
        child splitter (which maps the ranges to character spans), so the packing
        rule lives in exactly one place.
        """
        ranges: list[tuple[int, int]] = []
        start = 0
        running = 0
        for index, count in enumerate(token_counts):
            if index > start and running + count > target:
                ranges.append((start, index))
                start = index
                running = 0
            running += count
        if start < len(token_counts):
            ranges.append((start, len(token_counts)))
        return ranges

    def _overlap_tail(self, units: list[tuple[str, str]]) -> list[tuple[str, str]]:
        if self.overlap_tokens == 0:
            return []
        tail: list[tuple[str, str]] = []
        token_count = 0
        for unit in reversed(units):
            tail.insert(0, unit)
            token_count += estimate_tokens(unit[0])
            if token_count >= self.overlap_tokens:
                break
        return tail


class AdaptiveChunkingStrategy:
    """The default ChunkingStrategy: a thin adapter over an AdaptiveChunker.

    Ignores metadata and settings and delegates to the wrapped chunker, so the three
    built-in strategies produce output byte-identical to the pre-port direct call.
    """

    def __init__(self, strategy: str, chunker: AdaptiveChunker) -> None:
        self._strategy = strategy
        self._chunker = chunker

    @property
    def strategy_id(self) -> str:
        return self._strategy

    @property
    def expansion_mode(self) -> str:
        return self._strategy

    def chunk(self, text: str, *, metadata: JsonObject, settings: Settings) -> list[TextChunk]:
        return self._chunker.chunk(text)


def _adaptive_factory(strategy: str) -> Callable[[Settings], AdaptiveChunkingStrategy]:
    def factory(settings: Settings) -> AdaptiveChunkingStrategy:
        return AdaptiveChunkingStrategy(
            strategy,
            AdaptiveChunker(
                target_tokens=settings.default_chunk_tokens,
                overlap_tokens=settings.chunk_overlap_tokens,
                strategy=strategy,
            ),
        )

    return factory


BUILTIN_CHUNKING_STRATEGIES: dict[str, Callable[[Settings], AdaptiveChunkingStrategy]] = {
    name: _adaptive_factory(name) for name in CHUNKING_STRATEGIES
}
