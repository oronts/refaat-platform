from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from refaat.core.text import estimate_tokens, normalized_key, overlap_ratio, split_sentences, tokens
from refaat.domain.models import SearchHit


def _rewrite_hit_text(hit: SearchHit, text: str, token_count: int | None = None) -> SearchHit:
    """Return a copy of ``hit`` whose chunk carries ``text`` and a matching token count.

    Shared by the context compressor and the budgeter's single-hit truncation so
    there is one place that rewrites a hit's chunk body. Both pass, or default to,
    the ``estimate_tokens`` count of the rewritten text, so the recorded token count
    always matches the text actually carried.
    """
    resolved = estimate_tokens(text) if token_count is None else token_count
    chunk = replace(hit.chunk, text=text, token_count=resolved)
    return replace(hit, chunk=chunk)


class ContextCompressor:
    """Deterministic, model-free post-retrieval context compression.

    Within each retrieved chunk it keeps salient, non-redundant sentences and
    drops sentences already emitted by a higher-ranked chunk. A global seen set of
    normalized kept sentences removes cross-chunk redundancy. Every kept chunk
    retains at least its single highest-salience non-duplicate sentence so it stays
    citable, and a chunk is dropped only when all of its sentences are global
    duplicates. ``chunk.id`` is preserved; ``token_count`` is recomputed.
    """

    def compress(
        self,
        query: str,
        hits: Sequence[SearchHit],
        *,
        enabled: bool,
        min_salience: float,
    ) -> tuple[SearchHit, ...]:
        if not enabled:
            return tuple(hits)
        query_terms = set(tokens(query))
        seen: set[str] = set()
        result: list[SearchHit] = []
        for hit in hits:
            rewritten = self._compress_hit(hit, query_terms, min_salience, seen)
            if rewritten is not None:
                result.append(rewritten)
        return tuple(result)

    @staticmethod
    def _compress_hit(
        hit: SearchHit,
        query_terms: set[str],
        min_salience: float,
        seen: set[str],
    ) -> SearchHit | None:
        sentences = split_sentences(hit.chunk.text)
        if not sentences:
            return hit
        candidates: list[tuple[str, str, float]] = []
        local_keys: set[str] = set()
        for sentence in sentences:
            key = normalized_key(sentence)
            if key in seen or key in local_keys:
                continue
            local_keys.add(key)
            candidates.append((sentence, key, overlap_ratio(query_terms, sentence)))
        if not candidates:
            return None
        best_index = max(range(len(candidates)), key=lambda index: (candidates[index][2], -index))
        kept: list[tuple[str, str]] = [
            (sentence, key)
            for index, (sentence, key, salience) in enumerate(candidates)
            if salience >= min_salience or index == best_index
        ]
        for _, key in kept:
            seen.add(key)
        kept_sentences = [sentence for sentence, _ in kept]
        if kept_sentences == sentences:
            return hit
        return _rewrite_hit_text(hit, " ".join(kept_sentences))
