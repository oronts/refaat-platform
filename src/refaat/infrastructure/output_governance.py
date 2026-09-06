"""Reference OutputGovernor adapters.

Infrastructure, so it may import a digest helper from core. The shipped reference governor
is a deterministic pattern denylist: a bounded blast radius on a KNOWN string, honestly not
a semantic output filter.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from refaat.core.text import stable_digest
from refaat.domain.output_governance import OutputContext, OutputDecision


class PatternDenylistGovernor:
    """Refuse or redact a produced answer that contains a configured literal or regex.

    This catches ONLY a configured literal string or regular expression appearing in the
    final answer text. It does NOT detect reconstructed or paraphrased PII, prompt
    injection, an unconfigured secret, or unsafe generated URL, HTML, or SQL. The promise
    is a bounded blast radius on a KNOWN string, not injection prevention or a semantic
    output filter; compose it with a real DLP or classification governor for those.

    In ``refuse`` mode a match denies the whole output. In ``redact`` mode a match is
    allowed with every pattern occurrence replaced by ``placeholder``.
    """

    def __init__(
        self,
        patterns: Sequence[str | re.Pattern[str]] = (),
        *,
        mode: str = "refuse",
        placeholder: str = "[redacted]",
    ) -> None:
        if mode not in ("refuse", "redact"):
            raise ValueError("mode must be 'refuse' or 'redact'")
        # Patterns are trusted operator configuration, never client input, so an expensive
        # regex is an operator concern and not a client-reachable ReDoS vector. A plain
        # string is escaped to a literal so denylisting "a.b" does not accidentally match
        # "axb"; a caller who wants a real regex passes a compiled re.Pattern.
        self._patterns: tuple[re.Pattern[str], ...] = tuple(
            re.compile(re.escape(pattern)) if isinstance(pattern, str) else pattern for pattern in patterns
        )
        self._mode = mode
        self._placeholder = placeholder
        self._reprs: tuple[str, ...] = tuple(repr(pattern) for pattern in self._patterns)

    @property
    def cache_id(self) -> str:
        # Refuse is a pure any-match, so pattern order and the placeholder are irrelevant and a sorted
        # identity keeps the cache stable across a reorder. Redact substitutes in the supplied order into
        # the placeholder, so both shape the output and must scope the cache, or a differently-redacting
        # governor with the same patterns would serve this one's stale output.
        if self._mode == "redact":
            identity: tuple[str, ...] = (self._placeholder, *self._reprs)
        else:
            identity = tuple(sorted(self._reprs))
        return f"denylist:{self._mode}:{stable_digest(identity, length=16)}"

    def review(self, output: OutputContext) -> OutputDecision:
        answer = output.answer
        if not any(pattern.search(answer) for pattern in self._patterns):
            return OutputDecision(allowed=True, reason="clean")
        if self._mode == "redact":
            redacted = answer
            for pattern in self._patterns:
                redacted = pattern.sub(self._placeholder, redacted)
            return OutputDecision(
                allowed=True,
                reason="output_redacted_denied_pattern",
                redacted_answer=redacted,
            )
        return OutputDecision(allowed=False, reason="output_matched_denied_pattern")


__all__ = ["PatternDenylistGovernor"]
