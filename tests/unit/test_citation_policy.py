from __future__ import annotations

import pytest

from refaat.domain.citation_policy import CitationPolicy, citation_policy_refuses


@pytest.mark.parametrize(
    ("policy", "has_evidence", "has_citations", "expected"),
    [
        # OPTIONAL never refuses.
        (CitationPolicy.OPTIONAL, True, False, False),
        (CitationPolicy.OPTIONAL, False, False, False),
        (CitationPolicy.OPTIONAL, True, True, False),
        # A cited answer is never refused, whatever the policy.
        (CitationPolicy.REQUIRED, True, True, False),
        (CitationPolicy.REQUIRED_WHEN_EVIDENCE, True, True, False),
        # REQUIRED refuses any zero-citation answer.
        (CitationPolicy.REQUIRED, True, False, True),
        (CitationPolicy.REQUIRED, False, False, True),
        # REQUIRED_WHEN_EVIDENCE refuses a zero-citation answer only when evidence was retrieved.
        (CitationPolicy.REQUIRED_WHEN_EVIDENCE, True, False, True),
        (CitationPolicy.REQUIRED_WHEN_EVIDENCE, False, False, False),
    ],
)
def test_citation_policy_refuses(
    policy: CitationPolicy, has_evidence: bool, has_citations: bool, expected: bool
) -> None:
    assert citation_policy_refuses(policy, has_evidence=has_evidence, has_citations=has_citations) is expected
