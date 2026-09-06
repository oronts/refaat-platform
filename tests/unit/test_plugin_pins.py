"""The pure plugin pinning decision core.

These prove the fail-closed rules of decide_plugin, the canonical manifest's determinism and
sensitivity, PEP 503 distribution-name canonicalization, and the constant-time digest compare.
The enforcement path over real temporary distributions lives in
tests/integration/test_plugin_pin_enforcement.py.
"""

from __future__ import annotations

import hashlib

import pytest

from refaat.domain.plugins import (
    REASON_ALLOWED,
    REASON_CODE_HASH_MISMATCH,
    REASON_DISTRIBUTION_ABSENT,
    REASON_DISTRIBUTION_NAME_MISMATCH,
    REASON_ORIGIN_UNBOUND,
    REASON_PREIMPORT_MISMATCH,
    REASON_RECORD_MISSING,
    REASON_SOURCE_UNLISTED,
    REASON_UNLISTED_FILE_PRESENT,
    REASON_VALUE_MISMATCH,
    REASON_VERSION_MISMATCH,
    ObservedPlugin,
    PluginDescriptor,
    canonical_manifest,
    canonicalize_distribution_name,
    decide_plugin,
    digests_match,
    manifest_digest,
)

PIN = PluginDescriptor(
    group="refaat.skills",
    name="demo",
    value="demo_pkg.tool:DemoSkill",
    dist_name="demo-pkg",
    dist_version="1.2.3",
    code_hash="a" * 64,
)


def _observed(**overrides: object) -> ObservedPlugin:
    base: dict[str, object] = {
        "group": "refaat.skills",
        "name": "demo",
        "value": "demo_pkg.tool:DemoSkill",
        "distribution_present": True,
        "dist_name": "demo_pkg",
        "dist_version": "1.2.3",
        "record_present": True,
        "source_file_listed": True,
        "unlisted_importable_present": False,
        "code_hash": "a" * 64,
        "origin_bound": True,
        "preimport_bound": True,
    }
    base.update(overrides)
    return ObservedPlugin(**base)  # type: ignore[arg-type]


def test_a_fully_matching_observation_is_admitted() -> None:
    decision = decide_plugin(_observed(), PIN)
    assert decision.allowed
    assert decision.reason == REASON_ALLOWED


def test_pep503_name_variants_still_match() -> None:
    # underscore, dot, and case variants of the same project canonicalize to one identity
    for observed_name in ("demo_pkg", "Demo.Pkg", "demo__pkg", "DEMO-PKG"):
        decision = decide_plugin(_observed(dist_name=observed_name), PIN)
        assert decision.allowed, observed_name


def test_value_mismatch_refuses() -> None:
    decision = decide_plugin(_observed(value="demo_pkg.tool:Other"), PIN)
    assert not decision.allowed
    assert decision.reason == REASON_VALUE_MISMATCH


def test_absent_distribution_refuses() -> None:
    decision = decide_plugin(_observed(distribution_present=False, dist_name="", dist_version=""), PIN)
    assert decision.reason == REASON_DISTRIBUTION_ABSENT


def test_wrong_distribution_name_refuses() -> None:
    decision = decide_plugin(_observed(dist_name="squatter"), PIN)
    assert decision.reason == REASON_DISTRIBUTION_NAME_MISMATCH


def test_version_swap_refuses() -> None:
    decision = decide_plugin(_observed(dist_version="9.9.9"), PIN)
    assert decision.reason == REASON_VERSION_MISMATCH


def test_missing_record_refuses() -> None:
    decision = decide_plugin(_observed(record_present=False, code_hash=None), PIN)
    assert decision.reason == REASON_RECORD_MISSING


def test_source_not_listed_refuses() -> None:
    decision = decide_plugin(_observed(source_file_listed=False), PIN)
    assert decision.reason == REASON_SOURCE_UNLISTED


def test_unlisted_importable_file_refuses() -> None:
    decision = decide_plugin(_observed(unlisted_importable_present=True), PIN)
    assert decision.reason == REASON_UNLISTED_FILE_PRESENT


def test_code_hash_mismatch_refuses() -> None:
    decision = decide_plugin(_observed(code_hash="b" * 64), PIN)
    assert decision.reason == REASON_CODE_HASH_MISMATCH


def test_uncomputable_code_hash_refuses() -> None:
    decision = decide_plugin(_observed(code_hash=None), PIN)
    assert decision.reason == REASON_CODE_HASH_MISMATCH


def test_unbound_origin_refuses() -> None:
    decision = decide_plugin(_observed(origin_bound=False), PIN)
    assert decision.reason == REASON_ORIGIN_UNBOUND


def test_preimport_mismatch_refuses() -> None:
    decision = decide_plugin(_observed(preimport_bound=False), PIN)
    assert decision.reason == REASON_PREIMPORT_MISMATCH


def test_decision_order_reports_the_first_drift() -> None:
    # value is checked before distribution, so a simultaneous value and version drift reports value
    decision = decide_plugin(_observed(value="x:y", dist_version="9.9.9"), PIN)
    assert decision.reason == REASON_VALUE_MISMATCH


def test_descriptor_rejects_a_non_sha256_code_hash() -> None:
    with pytest.raises(ValueError, match="code_hash"):
        PluginDescriptor(
            group="refaat.skills",
            name="demo",
            value="m:C",
            dist_name="d",
            dist_version="1",
            code_hash="not-a-digest",
        )


def test_descriptor_rejects_empty_fields() -> None:
    for field in ("value", "dist_name", "dist_version"):
        payload = {"value": "m:C", "dist_name": "d", "dist_version": "1", "code_hash": "a" * 64}
        payload[field] = ""
        with pytest.raises(ValueError, match="non-empty"):
            PluginDescriptor(group="refaat.skills", name="demo", **payload)


def test_canonicalize_distribution_name_matches_pep503() -> None:
    assert canonicalize_distribution_name("Demo.Pkg") == "demo-pkg"
    assert canonicalize_distribution_name("demo__pkg") == "demo-pkg"
    assert canonicalize_distribution_name("demo-_-pkg") == "demo-pkg"


def test_canonical_manifest_is_order_independent() -> None:
    a = [("a.py", 3, "1" * 64), ("b.py", 5, "2" * 64)]
    assert manifest_digest(a) == manifest_digest(list(reversed(a)))


def test_canonical_manifest_is_sensitive_to_every_field() -> None:
    base = [("pkg/a.py", 10, "1" * 64)]
    baseline = manifest_digest(base)
    assert manifest_digest([("pkg/b.py", 10, "1" * 64)]) != baseline
    assert manifest_digest([("pkg/a.py", 11, "1" * 64)]) != baseline
    assert manifest_digest([("pkg/a.py", 10, "2" * 64)]) != baseline


def test_canonical_manifest_binds_path_to_content_unambiguously() -> None:
    # length-delimiting the path means a split boundary cannot be moved to forge an equal manifest
    first = [("ab", 1, "c" * 64)]
    second = [("a", 1, "b" + "c" * 63)]
    assert manifest_digest(first) != manifest_digest(second)


def test_manifest_digest_is_sha256_of_the_canonical_bytes() -> None:
    entries = [("pkg/a.py", 4, "d" * 64)]
    assert manifest_digest(entries) == hashlib.sha256(canonical_manifest(entries)).hexdigest()


def test_digests_match_is_constant_time_equality() -> None:
    assert digests_match("f" * 64, "f" * 64)
    assert not digests_match("f" * 64, "e" * 64)
