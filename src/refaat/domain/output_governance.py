"""Output-governance seam: review a produced answer and its citations after generation.

Model output is untrusted. This pure, model-independent seam lets a deployment review a
generated answer plus its already-resolved citations after generation and before the
response is returned or cached, and refuse or redact it. The verdict is deterministic and
does no I/O; this module imports only domain value objects and stdlib, never core or
infrastructure, so the domain layer stays leaf. When no governor is installed the query
pipeline is byte-identical to the ungoverned default; this is an opt-in seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from refaat.domain.authz import Principal
from refaat.domain.models import Citation

# A shape check is not enough: a short answer word can look like a code, so only a known
# application reason is recorded and anything else (including a leaked answer) becomes generic.
_KNOWN_REASONS = frozenset({"output_denied", "output_matched_denied_pattern", "output_governor_error"})


def bounded_reason(reason: object) -> str:
    return reason if reason in _KNOWN_REASONS else "output_denied"


@dataclass(frozen=True, slots=True)
class OutputContext:
    """The produced answer plus its resolved citations, handed to a governor for review.

    ``citations`` are already resolved against stored corpus chunks, so a governor
    inspects the final grounded output rather than a raw model span. ``principal`` is the
    request principal, None when no policy is active. ``workspace_id`` and ``corpus_id``
    scope the output for a tenant-aware governor.
    """

    answer: str
    citations: tuple[Citation, ...]
    principal: Principal | None
    workspace_id: str
    corpus_id: str


@dataclass(frozen=True, slots=True)
class OutputDecision:
    """A governor's verdict on one produced output.

    ``reason`` is a bounded machine code, never raw offending text, so it is safe to record
    in an audit trail. When ``allowed`` is False the pipeline returns a governed no-output
    response and never caches it. When ``allowed`` is True and ``redacted_answer`` is set,
    the pipeline serves the redacted text in place of the original.
    """

    allowed: bool
    reason: str
    redacted_answer: str | None = None


class OutputGovernor(Protocol):
    """Review a produced answer and its citations after generation, before return or cache."""

    @property
    def cache_id(self) -> str:
        """A stable identity for this governor's verdict function.

        Folded into the answer-cache key so an answer produced under one governor (or none)
        is never served after the governor is installed, removed, or changed. Because a
        cache hit returns before the governor runs, the governor must be a pure function of
        the answer plus whatever it folds into this id.
        """
        ...

    def review(self, output: OutputContext) -> OutputDecision: ...


__all__ = ["OutputContext", "OutputDecision", "OutputGovernor", "bounded_reason"]
