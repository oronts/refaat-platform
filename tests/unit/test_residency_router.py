from __future__ import annotations

import pytest

from refaat.domain.authz import Classification
from refaat.domain.routing import (
    ClassificationCeilingRouter,
    CompositeModelRouter,
    ProviderResidency,
    ResidencyPolicy,
    ResidencyRouter,
    ResidencyRule,
    RetentionTerm,
)


def _confidential_eu_policy() -> ResidencyPolicy:
    return ResidencyPolicy(
        {
            Classification.PUBLIC: ResidencyRule(
                allowed_regions=frozenset({"eu", "us"}), max_retention=RetentionTerm.RETAINED
            ),
            Classification.CONFIDENTIAL: ResidencyRule(
                allowed_regions=frozenset({"eu"}), max_retention=RetentionTerm.TRANSIENT
            ),
        }
    )


def _provider(
    region: str = "eu",
    retention: RetentionTerm | None = RetentionTerm.ZERO,
    purpose: str = "",
) -> ProviderResidency:
    return ProviderResidency(provider="gen", region=region, retention=retention, purpose=purpose)


def test_confidential_eu_policy_refuses_us_provider() -> None:
    router = ResidencyRouter(_confidential_eu_policy(), _provider(region="us"))
    assert router.route(Classification.CONFIDENTIAL, None).allowed is False


def test_confidential_eu_policy_admits_eu_provider() -> None:
    router = ResidencyRouter(_confidential_eu_policy(), _provider(region="eu"))
    assert router.route(Classification.CONFIDENTIAL, None).allowed is True


def test_untagged_provider_fails_closed() -> None:
    empty_region = ResidencyRouter(_confidential_eu_policy(), _provider(region=""))
    assert empty_region.route(Classification.CONFIDENTIAL, None).allowed is False
    no_retention = ResidencyRouter(_confidential_eu_policy(), ProviderResidency(provider="gen", region="eu"))
    assert no_retention.route(Classification.CONFIDENTIAL, None).allowed is False


def test_missing_policy_entry_fails_closed() -> None:
    # No RESTRICTED rule in the policy, so RESTRICTED evidence is refused by design.
    router = ResidencyRouter(_confidential_eu_policy(), _provider(region="eu"))
    assert router.route(Classification.RESTRICTED, None).allowed is False


def test_retention_exceeding_max_fails_closed() -> None:
    # CONFIDENTIAL admits at most TRANSIENT retention; LIMITED exceeds it.
    router = ResidencyRouter(_confidential_eu_policy(), _provider(region="eu", retention=RetentionTerm.LIMITED))
    assert router.route(Classification.CONFIDENTIAL, None).allowed is False


def test_purpose_gating_when_allowed_purposes_set() -> None:
    policy = ResidencyPolicy(
        {
            Classification.CONFIDENTIAL: ResidencyRule(
                allowed_regions=frozenset({"eu"}),
                max_retention=RetentionTerm.TRANSIENT,
                allowed_purposes=frozenset({"inference"}),
            )
        }
    )
    allowed = ResidencyRouter(policy, _provider(region="eu", purpose="inference"))
    denied = ResidencyRouter(policy, _provider(region="eu", purpose="training"))
    assert allowed.route(Classification.CONFIDENTIAL, None).allowed is True
    assert denied.route(Classification.CONFIDENTIAL, None).allowed is False


def test_cache_id_changes_with_policy() -> None:
    base = ResidencyRouter(_confidential_eu_policy(), _provider(region="eu"))
    laxer_policy = ResidencyPolicy(
        {
            Classification.CONFIDENTIAL: ResidencyRule(
                allowed_regions=frozenset({"eu"}), max_retention=RetentionTerm.RETAINED
            )
        }
    )
    other = ResidencyRouter(laxer_policy, _provider(region="eu"))
    assert base.cache_id != other.cache_id
    assert base.cache_id == ResidencyRouter(_confidential_eu_policy(), _provider(region="eu")).cache_id


def test_cache_id_is_collision_proof_for_region_sets() -> None:
    # The single element "eu,us" must not digest to the same identity as the two-element
    # set "eu","us": a delimiter-joined identity would collide, a structured one does not.
    joined = ResidencyPolicy(
        {
            Classification.CONFIDENTIAL: ResidencyRule(
                allowed_regions=frozenset({"eu,us"}), max_retention=RetentionTerm.TRANSIENT
            )
        }
    ).cache_id
    split = ResidencyPolicy(
        {
            Classification.CONFIDENTIAL: ResidencyRule(
                allowed_regions=frozenset({"eu", "us"}), max_retention=RetentionTerm.TRANSIENT
            )
        }
    ).cache_id
    assert joined != split


def test_composite_returns_first_deny() -> None:
    deny_ceiling = ClassificationCeilingRouter(ceiling=Classification.PUBLIC)
    allow_residency = ResidencyRouter(_confidential_eu_policy(), _provider(region="eu"))
    composite = CompositeModelRouter((deny_ceiling, allow_residency))
    verdict = composite.route(Classification.CONFIDENTIAL, None)
    assert verdict.allowed is False
    assert "ceiling" in verdict.reason


def test_composite_allows_only_when_all_allow() -> None:
    allow_ceiling = ClassificationCeilingRouter(ceiling=Classification.CONFIDENTIAL)
    allow_residency = ResidencyRouter(_confidential_eu_policy(), _provider(region="eu"))
    assert CompositeModelRouter((allow_ceiling, allow_residency)).route(Classification.CONFIDENTIAL, None).allowed
    # A residency deny (us provider) flips the composite even though the ceiling allows.
    deny_residency = ResidencyRouter(_confidential_eu_policy(), _provider(region="us"))
    flipped = CompositeModelRouter((allow_ceiling, deny_residency))
    assert flipped.route(Classification.CONFIDENTIAL, None).allowed is False


def test_composite_cache_id_folds_members() -> None:
    ceiling = ClassificationCeilingRouter(ceiling=Classification.CONFIDENTIAL)
    residency = ResidencyRouter(_confidential_eu_policy(), _provider(region="eu"))
    composite = CompositeModelRouter((ceiling, residency))
    assert composite.cache_id != ceiling.cache_id
    assert composite.cache_id != residency.cache_id
    assert composite.cache_id == CompositeModelRouter((ceiling, residency)).cache_id


def test_composite_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one router"):
        CompositeModelRouter(())
