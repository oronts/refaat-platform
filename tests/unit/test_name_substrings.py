"""name_substrings enumerates exactly the entity names graph_context's substring branch can match."""

from __future__ import annotations

from refaat.core.text import name_substrings, normalized_key


def test_contains_iff_substring_for_nonempty_names() -> None:
    key = normalized_key("what is the northwind logistics shipping policy")
    for name in ["northwind", "logistics", "northwind logistics", "policy", "ship", "absent", "wha"]:
        assert (name in name_substrings(key, len(name))) is (name in key), name


def test_substring_only_case_a_token_index_would_miss() -> None:
    # 'ai' is a substring of 'chait' but not one of its tokens, so only the substring branch matches.
    key = normalized_key("chait")
    assert "ai" in name_substrings(key, 2)
    assert "ai" not in set(key.split())


def test_excludes_empty_string_and_is_deterministic() -> None:
    key = normalized_key("alpha")
    subs = name_substrings(key, 16)
    assert "" not in subs
    assert subs == name_substrings(key, 16)
    assert {"a", "alp", "alpha"} <= subs


def test_length_cap_bounds_enumeration_without_under_returning_within_the_cap() -> None:
    key = normalized_key("abcdef")
    capped = name_substrings(key, 3)
    assert "abc" in capped
    assert "abcd" not in capped
    # A name longer than the cap is only missed when max_len < len(name); the caller passes
    # max_len = the corpus max normalized_name length, so a real name is always within the cap.


def test_empty_query_key_returns_empty() -> None:
    assert name_substrings("", 10) == frozenset()
    assert name_substrings("x", 0) == frozenset()
