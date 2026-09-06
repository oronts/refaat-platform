"""Parsing of the top-level source_trust tag that drives the retrieval down-rank.

These pin the pure helper: a case-insensitive enum name resolves, and any absent,
malformed, or non-name value defaults to UNVERIFIED (the minimum), so trust must be
asserted and untagged or broken content can never evade a down-rank or claim maximum trust.
"""

from __future__ import annotations

import pytest

from refaat.domain.authz import SourceTrust, source_trust_from_metadata


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("unverified", SourceTrust.UNVERIFIED),
        ("EXTERNAL", SourceTrust.EXTERNAL),
        ("User_Upload", SourceTrust.USER_UPLOAD),
        ("  internal  ", SourceTrust.INTERNAL),
        ("authoritative", SourceTrust.AUTHORITATIVE),
    ],
)
def test_case_insensitive_enum_name_parses(raw: str, expected: SourceTrust) -> None:
    assert source_trust_from_metadata({"source_trust": raw}) is expected


def test_a_source_trust_instance_passes_through() -> None:
    assert source_trust_from_metadata({"source_trust": SourceTrust.EXTERNAL}) is SourceTrust.EXTERNAL


def test_absent_tag_defaults_to_unverified() -> None:
    assert source_trust_from_metadata({"title": "handbook"}) is SourceTrust.UNVERIFIED


def test_none_metadata_defaults_to_unverified() -> None:
    assert source_trust_from_metadata(None) is SourceTrust.UNVERIFIED


def test_unknown_name_defaults_to_unverified() -> None:
    assert source_trust_from_metadata({"source_trust": "gossip"}) is SourceTrust.UNVERIFIED


def test_non_string_value_defaults_to_unverified() -> None:
    assert source_trust_from_metadata({"source_trust": 1}) is SourceTrust.UNVERIFIED
    assert source_trust_from_metadata({"source_trust": {"level": "external"}}) is SourceTrust.UNVERIFIED


def test_trust_is_read_from_the_top_level_not_the_acl_block() -> None:
    # Trust is independent of confidentiality: an acl block never carries it, and a
    # top-level tag is honored regardless of any acl block on the same metadata.
    assert source_trust_from_metadata({"acl": {"source_trust": "external"}}) is SourceTrust.UNVERIFIED
    metadata = {"source_trust": "external", "acl": {"classification": "restricted"}}
    assert source_trust_from_metadata(metadata) is SourceTrust.EXTERNAL
