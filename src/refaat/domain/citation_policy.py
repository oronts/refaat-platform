from __future__ import annotations

from enum import StrEnum


class CitationPolicy(StrEnum):
    """How a produced answer that resolves no citations is handled.

    OPTIONAL is the default and preserves the historical behavior: a zero-citation answer is served.
    The stricter modes refuse it instead, so an ungrounded answer is never served where the deployment
    requires grounding.
    """

    OPTIONAL = "optional"
    REQUIRED_WHEN_EVIDENCE = "required_when_evidence"
    REQUIRED = "required"


def citation_policy_refuses(policy: CitationPolicy, *, has_evidence: bool, has_citations: bool) -> bool:
    """Whether the policy refuses an answer given its evidence and resolved citations.

    OPTIONAL never refuses. REQUIRED_WHEN_EVIDENCE refuses a zero-citation answer only when evidence was
    retrieved (so the answer could and should have been grounded). REQUIRED refuses any zero-citation
    answer. An answer that resolved at least one citation is never refused.
    """
    if has_citations or policy is CitationPolicy.OPTIONAL:
        return False
    if policy is CitationPolicy.REQUIRED_WHEN_EVIDENCE:
        return has_evidence
    return True
