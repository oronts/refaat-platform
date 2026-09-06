"""Residency-zone egress gating.

A residency rule that gates zones admits only evidence whose declared zones are all in the allowed
set, denying untagged and malformed evidence fail-closed. The zone rides the existing ACL metadata and
folds into the permission fingerprint and the router cache id only when present, so a zone-free corpus
is byte-identical to the pre-feature runtime.
"""

from __future__ import annotations

from typing import Any

import pytest

from refaat.application.runtime import _parse_residency_rule
from refaat.domain.authz import (
    MALFORMED_ZONE,
    UNTAGGED_ZONE,
    Classification,
    permission_fingerprint,
    residency_zones,
)
from refaat.domain.routing import (
    ClassificationCeilingRouter,
    CompositeModelRouter,
    ProviderResidency,
    ResidencyPolicy,
    ResidencyRouter,
    ResidencyRule,
    RetentionTerm,
)


def _acl(**extra: Any) -> dict[str, Any]:
    return {"acl": {"classification": "CONFIDENTIAL", "tenant": "acme", **extra}}


def test_residency_zones_helper_untagged_malformed_and_declared() -> None:
    assert residency_zones([{}]) == frozenset({UNTAGGED_ZONE})
    assert residency_zones([_acl()]) == frozenset({UNTAGGED_ZONE})
    assert residency_zones([{"acl": "broken"}]) == frozenset({MALFORMED_ZONE})
    assert residency_zones([_acl(residency_zone="eu")]) == frozenset({"eu"})
    assert residency_zones([_acl(residency_zone="")]) == frozenset({MALFORMED_ZONE})
    assert residency_zones([_acl(residency_zone=UNTAGGED_ZONE)]) == frozenset({MALFORMED_ZONE})
    assert residency_zones([_acl(residency_zone="eu\x01west")]) == frozenset({MALFORMED_ZONE})
    assert residency_zones([_acl(residency_zone="\teu")]) == frozenset({MALFORMED_ZONE})
    assert residency_zones([_acl(residency_zone="eu\n")]) == frozenset({MALFORMED_ZONE})
    assert residency_zones([_acl(residency_zone="eu ")]) == frozenset({"eu"})


def test_permission_fingerprint_is_byte_identical_without_a_zone() -> None:
    # A pinned digest that locks the canonical serialization; a zone-free ACL omits the rz key.
    assert permission_fingerprint(_acl()) == "23c847e491c124ca13b5e40b404cf1c493b89e0ddd7077bbb11f1b0102cd884f"
    assert permission_fingerprint(_acl(residency_zone="eu")) != permission_fingerprint(_acl())


def _policy(*, zones: frozenset[str] | None = None) -> ResidencyPolicy:
    return ResidencyPolicy(
        {Classification.CONFIDENTIAL: ResidencyRule(frozenset({"eu"}), RetentionTerm.ZERO, allowed_zones=zones)}
    )


def test_cache_id_is_byte_identical_without_gated_zones() -> None:
    # A pinned pre-feature digest: a policy that gates no zones keeps the exact cache id.
    assert _policy().cache_id == "residency-policy:d0d9556805ef6dbc5800f4ee2409107d4ce302ecdd01fead5babf94f87db9831"
    assert _policy(zones=frozenset({"eu"})).cache_id != _policy().cache_id


def _router(*, zones: frozenset[str] | None) -> ResidencyRouter:
    return ResidencyRouter(_policy(zones=zones), ProviderResidency("openai", "eu", RetentionTerm.ZERO))


def test_gated_rule_denies_untagged_malformed_and_out_of_zone() -> None:
    r = _router(zones=frozenset({"eu"}))
    assert r.route(Classification.CONFIDENTIAL, None, residency_zones=frozenset({"eu"})).allowed is True
    assert r.route(Classification.CONFIDENTIAL, None, residency_zones=frozenset({"us"})).allowed is False
    assert r.route(Classification.CONFIDENTIAL, None, residency_zones=frozenset({UNTAGGED_ZONE})).allowed is False
    assert r.route(Classification.CONFIDENTIAL, None, residency_zones=frozenset({MALFORMED_ZONE})).allowed is False
    assert r.route(Classification.CONFIDENTIAL, None, residency_zones=frozenset({"eu", "us"})).allowed is False


def test_ungated_rule_ignores_zones() -> None:
    r = _router(zones=None)
    for zones in (frozenset[str](), frozenset({"us"}), frozenset({UNTAGGED_ZONE}), frozenset({MALFORMED_ZONE})):
        assert r.route(Classification.CONFIDENTIAL, None, residency_zones=zones).allowed is True


def test_config_parser_drops_a_non_printable_zone() -> None:
    rule = _parse_residency_rule(
        Classification.CONFIDENTIAL,
        {"allowed_regions": ["eu"], "max_retention": "zero", "allowed_zones": ["eu", "eu\x01west", "\tus", "de\n"]},
    )
    assert rule.allowed_zones == frozenset({"eu"})


def test_config_parser_reads_each_item_str_once() -> None:
    class _Flip:
        def __init__(self) -> None:
            self.n = 0

        def __str__(self) -> str:
            self.n += 1
            return "eu" if self.n < 3 else MALFORMED_ZONE

    flip = _Flip()
    rule = _parse_residency_rule(
        Classification.CONFIDENTIAL,
        {"allowed_regions": ["eu"], "max_retention": "zero", "allowed_zones": [flip]},
    )
    assert flip.n == 1
    assert MALFORMED_ZONE not in rule.allowed_zones
    assert rule.allowed_zones == frozenset({"eu"})


class _PoisonText(str):
    def isprintable(self) -> bool:
        return True

    def strip(self, *args: object) -> str:
        return MALFORMED_ZONE


def test_config_parser_defeats_a_str_subclass_override() -> None:
    class _Holder:
        def __str__(self) -> str:
            return _PoisonText("eu")

    rule = _parse_residency_rule(
        Classification.CONFIDENTIAL,
        {"allowed_regions": ["eu"], "max_retention": "zero", "allowed_zones": [_Holder()]},
    )
    assert MALFORMED_ZONE not in rule.allowed_zones
    assert rule.allowed_zones == frozenset({"eu"})


def test_evidence_parse_defeats_a_str_subclass_override() -> None:
    zones = residency_zones([{"acl": {"classification": "confidential", "residency_zone": _PoisonText("eu")}}])
    assert MALFORMED_ZONE not in zones
    assert zones == frozenset({"eu"})


@pytest.mark.parametrize("bad", [MALFORMED_ZONE, "\teu", "eu\n", "eu\x01west"])
def test_residency_rule_rejects_a_non_printable_allowed_zone(bad: str) -> None:
    with pytest.raises(ValueError, match="printable"):
        ResidencyRule(frozenset({"eu"}), RetentionTerm.ZERO, allowed_zones=frozenset({bad}))


def test_residency_rule_normalizes_a_hash_eq_overriding_subclass() -> None:
    class _Alias(str):
        def __hash__(self) -> int:
            return hash(MALFORMED_ZONE)

        def __eq__(self, other: object) -> bool:
            return other == MALFORMED_ZONE

    rule = ResidencyRule(frozenset({"eu"}), RetentionTerm.ZERO, allowed_zones=frozenset({_Alias("eu")}))
    assert all(type(zone) is str for zone in rule.allowed_zones)
    router = ResidencyRouter(
        ResidencyPolicy({Classification.CONFIDENTIAL: rule}), ProviderResidency("p", "eu", RetentionTerm.ZERO)
    )
    assert router.route(Classification.CONFIDENTIAL, None, residency_zones=frozenset({MALFORMED_ZONE})).allowed is False


def test_refusal_reason_never_carries_the_zone_value() -> None:
    verdict = _router(zones=frozenset({"eu"})).route(
        Classification.CONFIDENTIAL, None, residency_zones=frozenset({"secret-region"})
    )
    assert verdict.allowed is False
    assert verdict.reason == "zone_not_allowed"
    assert "secret-region" not in verdict.reason


def test_composite_forwards_zones_to_members() -> None:
    composite = CompositeModelRouter(
        [ClassificationCeilingRouter(Classification.RESTRICTED), _router(zones=frozenset({"eu"}))]
    )
    assert composite.route(Classification.CONFIDENTIAL, None, residency_zones=frozenset({"eu"})).allowed is True
    assert composite.route(Classification.CONFIDENTIAL, None, residency_zones=frozenset({"us"})).allowed is False
