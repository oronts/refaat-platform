"""Reference ContextTransform adapters.

Infrastructure, so it may import a digest helper from core. The shipped reference transform is a
deterministic pattern redactor: a bounded blast radius on KNOWN operator strings appearing in the
evidence a generation model would see, honestly not a semantic DLP filter.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import replace

from refaat.core.text import stable_digest
from refaat.domain.context_transform import RedactionCount, TransformResult
from refaat.domain.models import GenerationRequest, SearchHit

_DEFAULT_FIELDS: tuple[str, ...] = ("evidence", "graph_facts", "session_summary")


class PatternRedactor:
    """Replace configured literal or regex spans in the evidence sent to a generation model.

    This rewrites the retrieved chunk bodies (``evidence``), the graph facts, and the session
    summary a generation request carries, replacing each configured span with ``placeholder``
    before the request reaches a real generator. A plain string is escaped to a literal so
    redacting ``a.b`` does not accidentally match ``axb``; a caller who wants a real regex passes a
    compiled ``re.Pattern``. Patterns are trusted operator configuration, never client input.

    It catches ONLY a configured span in the configured fields. It does not detect reconstructed or
    paraphrased PII, an unconfigured secret, or a secret that leaves through a different provider
    egress (embeddings, fine-tuning, ingestion-time HyPE or RAPTOR, evaluation-time judging). The
    query is not in the default field set because it is the user's own input, not corpus evidence;
    an operator may add ``"query"`` to redact it too.

    Idempotency holds for patterns that do not themselves match the placeholder: re-running the
    redactor over already-redacted text redacts nothing new. A pattern that matches the placeholder
    text is an operator concern and is not guarded here.
    """

    def __init__(
        self,
        patterns: Sequence[str | re.Pattern[str]],
        *,
        placeholder: str = "[REDACTED]",
        fields: Sequence[str] = _DEFAULT_FIELDS,
    ) -> None:
        self._patterns: tuple[re.Pattern[str], ...] = tuple(
            re.compile(re.escape(pattern)) if isinstance(pattern, str) else pattern for pattern in patterns
        )
        self._placeholder = placeholder
        self._fields = tuple(fields)
        # A pattern id is a digest of the compiled pattern, so provenance and any log line carry only
        # opaque ids and counts, never the raw pattern string or a matched span.
        self._pattern_ids: tuple[str, ...] = tuple(
            stable_digest((repr(pattern),), length=12) for pattern in self._patterns
        )
        self._reprs: tuple[str, ...] = tuple(repr(pattern) for pattern in self._patterns)

    @property
    def cache_id(self) -> str:
        # The placeholder, the patterns in the supplied order, and the redacted fields all shape the
        # output the model sees, so all three scope the answer cache: a differently-configured
        # redactor with the same patterns must not serve this one's stale output.
        identity = (self._placeholder, *self._reprs, *self._fields)
        return "gen-redact:" + stable_digest(identity, length=16)

    def apply(self, request: GenerationRequest) -> TransformResult:
        context = request.context
        counts: dict[str, int] = {}

        hits = context.hits
        if "evidence" in self._fields and hits:
            hits = tuple(self._redact_hit(hit, counts) for hit in hits)

        graph_facts = context.graph_facts
        if "graph_facts" in self._fields and graph_facts:
            graph_facts = tuple(self._redact_and_count(fact, counts) for fact in graph_facts)

        session_summary = context.session_summary
        if "session_summary" in self._fields and session_summary:
            session_summary = self._redact_and_count(session_summary, counts)

        query = request.query
        if "query" in self._fields and query:
            query = self._redact_and_count(query, counts)

        new_context = replace(context, hits=hits, graph_facts=graph_facts, session_summary=session_summary)
        new_request = replace(request, context=new_context, query=query)
        redactions = tuple(
            RedactionCount(pattern_id=pattern_id, count=count) for pattern_id, count in sorted(counts.items())
        )
        return TransformResult(request=new_request, redactions=redactions, images_dropped=0)

    def _redact_hit(self, hit: SearchHit, counts: dict[str, int]) -> SearchHit:
        redacted, hit_counts = self._redact_text(hit.chunk.text)
        if not hit_counts:
            return hit
        self._accumulate(counts, hit_counts)
        # The id, metadata, and embedding stay intact, so a citation still resolves to the stored
        # chunk; only the text the model reads is rewritten.
        return replace(hit, chunk=replace(hit.chunk, text=redacted))

    def _redact_and_count(self, text: str, counts: dict[str, int]) -> str:
        redacted, text_counts = self._redact_text(text)
        self._accumulate(counts, text_counts)
        return redacted

    def _redact_text(self, text: str) -> tuple[str, dict[str, int]]:
        spans: list[tuple[int, int, int]] = []
        for index, pattern in enumerate(self._patterns):
            for match in pattern.finditer(text):
                spans.append((match.start(), match.end(), index))
        if not spans:
            return text, {}
        # Deterministic overlap resolution: leftmost span wins, and at an equal start the
        # earlier-listed pattern wins (lower index); a span overlapping an already-kept one is
        # discarded. Counting is over the surviving original spans.
        spans.sort(key=lambda span: (span[0], span[2]))
        kept: list[tuple[int, int, int]] = []
        cursor = 0
        for start, end, index in spans:
            if start < cursor:
                continue
            kept.append((start, end, index))
            cursor = end
        counts: dict[str, int] = {}
        for _start, _end, index in kept:
            pattern_id = self._pattern_ids[index]
            counts[pattern_id] = counts.get(pattern_id, 0) + 1
        parts: list[str] = []
        previous = 0
        for start, end, _index in kept:
            parts.append(text[previous:start])
            parts.append(self._placeholder)
            previous = end
        parts.append(text[previous:])
        return "".join(parts), counts

    @staticmethod
    def _accumulate(total: dict[str, int], part: dict[str, int]) -> None:
        for pattern_id, count in part.items():
            total[pattern_id] = total.get(pattern_id, 0) + count


__all__ = ["PatternRedactor"]
