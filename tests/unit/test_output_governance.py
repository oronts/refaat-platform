"""The reference output governor and the conditional cache-identity fold.

PatternDenylistGovernor catches only a configured literal or regex in the final answer:
it allows a clean answer, refuses or redacts a match, never leaks the offending text into
its bounded reason, and its cache_id is stable for an equal denylist yet distinct for a
changed denylist or mode. The QueryService._context_hash fold is proven conditional so an
unconfigured runtime is byte-identical.
"""

from __future__ import annotations

import re

import pytest

from refaat.application.services import QueryService
from refaat.core.retrieval import RetrievalPolicy
from refaat.domain.output_governance import OutputContext, bounded_reason
from refaat.infrastructure.output_governance import PatternDenylistGovernor


def _ctx(answer: str) -> OutputContext:
    return OutputContext(answer=answer, citations=(), principal=None, workspace_id="ws", corpus_id="corpus")


def test_clean_answer_is_allowed() -> None:
    governor = PatternDenylistGovernor(["Bluejay"])
    decision = governor.review(_ctx("The release plan is approved."))
    assert decision.allowed is True
    assert decision.reason == "clean"
    assert decision.redacted_answer is None


def test_matching_answer_is_refused() -> None:
    governor = PatternDenylistGovernor(["Bluejay"])
    decision = governor.review(_ctx("Project Bluejay ships next week."))
    assert decision.allowed is False
    assert decision.reason == "output_matched_denied_pattern"
    assert decision.redacted_answer is None


def test_matching_answer_is_redacted_in_redact_mode() -> None:
    governor = PatternDenylistGovernor(["Bluejay"], mode="redact", placeholder="[secret]")
    decision = governor.review(_ctx("Project Bluejay ships; Bluejay is the codename."))
    assert decision.allowed is True
    assert decision.reason == "output_redacted_denied_pattern"
    assert decision.redacted_answer == "Project [secret] ships; [secret] is the codename."
    assert "Bluejay" not in (decision.redacted_answer or "")


def test_offending_text_is_absent_from_the_reason() -> None:
    governor = PatternDenylistGovernor(["Bluejay"])
    decision = governor.review(_ctx("Project Bluejay is a secret."))
    assert "Bluejay" not in decision.reason


def test_a_plain_string_is_matched_as_a_literal_not_a_regex() -> None:
    governor = PatternDenylistGovernor(["a.c"])
    assert governor.review(_ctx("the value axc is fine")).allowed is True
    assert governor.review(_ctx("the value a.c is denied")).allowed is False


def test_a_compiled_pattern_is_used_as_a_regex() -> None:
    governor = PatternDenylistGovernor([re.compile(r"case-\d+")])
    assert governor.review(_ctx("see case-42 for details")).allowed is False
    assert governor.review(_ctx("see the case notes")).allowed is True


def test_empty_denylist_always_allows() -> None:
    governor = PatternDenylistGovernor([])
    assert governor.review(_ctx("anything at all, secret or not")).allowed is True


def test_an_invalid_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        PatternDenylistGovernor(["Bluejay"], mode="bogus")


def test_cache_id_is_stable_for_an_equal_denylist_regardless_of_order() -> None:
    a = PatternDenylistGovernor(["alpha", "beta"])
    b = PatternDenylistGovernor(["beta", "alpha"])
    assert a.cache_id == b.cache_id


def test_cache_id_changes_with_the_denylist() -> None:
    a = PatternDenylistGovernor(["alpha"])
    b = PatternDenylistGovernor(["alpha", "beta"])
    assert a.cache_id != b.cache_id


def test_cache_id_changes_with_the_mode() -> None:
    a = PatternDenylistGovernor(["alpha"], mode="refuse")
    b = PatternDenylistGovernor(["alpha"], mode="redact")
    assert a.cache_id != b.cache_id


def test_redact_cache_id_depends_on_order_and_placeholder() -> None:
    # In redact mode the substitution order and the placeholder shape the served text, so a cache
    # scoped only on the sorted patterns would serve a differently-redacting governor's stale output.
    order_a = PatternDenylistGovernor(["a", "b"], mode="redact")
    order_b = PatternDenylistGovernor(["b", "a"], mode="redact")
    assert order_a.cache_id != order_b.cache_id
    ph_a = PatternDenylistGovernor(["a"], mode="redact", placeholder="[A]")
    ph_b = PatternDenylistGovernor(["a"], mode="redact", placeholder="[B]")
    assert ph_a.cache_id != ph_b.cache_id


def test_bounded_reason_only_passes_known_codes() -> None:
    assert bounded_reason("output_matched_denied_pattern") == "output_matched_denied_pattern"
    assert bounded_reason("output_governor_error") == "output_governor_error"
    # An identifier-shaped answer word is not a known code, so it is coerced, not leaked.
    assert bounded_reason("secret") == "output_denied"
    assert bounded_reason("secret\n") == "output_denied"
    assert bounded_reason("Project Bluejay is the secret codename for the release.") == "output_denied"
    assert bounded_reason("") == "output_denied"


def _policy() -> RetrievalPolicy:
    return RetrievalPolicy(
        top_k=8,
        candidate_limit=40,
        token_budget=600,
        semantic_weight=0.55,
        lexical_weight=0.30,
        graph_weight=0.15,
        fusion="rrf",
        rrf_k=60,
        autocut_enabled=False,
        autocut_threshold=0.5,
        metadata_filter=None,
        bm25_k1=1.2,
        bm25_b=0.75,
        query_expansion=True,
        max_query_variants=4,
        compression_enabled=False,
        compression_min_salience=0.0,
        expansion_mode="sub_query",
        window_radius=1,
    )


def test_context_hash_omits_the_governor_key_when_unset() -> None:
    base: dict[str, object] = {
        "policy": _policy(),
        "profile": {},
        "supplemental": {},
        "session_summary": None,
        "embedding_model": "emb",
        "generation_model": "gen",
        "reranker": "none",
        "query_transform": "none",
        "guidance": None,
        "response_schema": None,
    }
    # A None governor folds nothing, so the digest equals omitting the argument entirely;
    # a real governor identity changes the digest. This is the byte-identity contract.
    assert QueryService._context_hash(**base) == QueryService._context_hash(**base, output_governor=None)
    assert QueryService._context_hash(**base) != QueryService._context_hash(
        **base, output_governor="denylist:refuse:abc123"
    )
