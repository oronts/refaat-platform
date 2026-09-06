"""Token-measure truncation bounds content a whitespace split would leave whole."""

from __future__ import annotations

from refaat.core.text import estimate_tokens, truncate_to_tokens


def test_truncate_to_tokens_bounds_non_whitespace_delimited_content() -> None:
    # A whitespace split sees one giant word here and would keep the whole chunk; truncate_to_tokens cuts
    # at the token boundary so the token count is genuinely bounded.
    text = ",".join(f"term{i}" for i in range(200))
    assert estimate_tokens(text) >= 200
    assert len(text.split()) == 1
    truncated = truncate_to_tokens(text, 16)
    assert estimate_tokens(truncated) <= 16
    assert truncated != ""
    assert text.startswith(truncated)


def test_truncate_to_tokens_keeps_a_prefix_of_normal_text() -> None:
    text = "the quick brown fox jumps over the lazy dog again and again"
    truncated = truncate_to_tokens(text, 4)
    assert estimate_tokens(truncated) <= 4
    assert text.startswith(truncated)


def test_truncate_to_tokens_zero_or_negative_limit_is_empty() -> None:
    assert truncate_to_tokens("anything at all here", 0) == ""
    assert truncate_to_tokens("anything at all here", -3) == ""


def test_truncate_to_tokens_returns_all_when_limit_exceeds_tokens() -> None:
    text = "only a few tokens"
    assert truncate_to_tokens(text, 100) == text
