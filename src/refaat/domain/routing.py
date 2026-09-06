"""Data-classification model-routing governance seam.

Decides whether the retrieved evidence may be sent to a model, given the data
classification derived deterministically from the evidence and the request
principal. The classification is never chosen by a model: it comes from the ACL
tags on the retrieved chunks (see ``domain.authz.access_tag_from_metadata``). The
router only maps a classification plus a principal to an allow or deny verdict.

Pure and deterministic, no I/O. When no router is installed the query pipeline is
byte-identical to the unrouted default; this is an opt-in governance seam.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol

from refaat.domain.authz import Classification, Principal


@dataclass(frozen=True, slots=True)
class ModelRoute:
    """A routing verdict: whether generation is allowed, why, and an optional model id.

    ``model_id`` lets a router steer an admitted classification to a specific model
    (for example a locally hosted model for CONFIDENTIAL evidence); the query
    pipeline may use it or ignore it. A denial carries a clear machine-readable
    ``reason`` and no ``model_id``.
    """

    allowed: bool
    reason: str
    model_id: str | None = None


class ModelRouter(Protocol):
    """Map a data classification and principal to a model-routing verdict."""

    @property
    def cache_id(self) -> str:
        """A stable identity for this router's verdict function.

        Folded into the answer-cache key so a cached answer produced under one router
        (or none) is never served after the router is installed, removed, or changed.
        """
        ...

    def route(
        self,
        classification: Classification,
        principal: Principal | None,
        *,
        residency_zones: frozenset[str] = frozenset(),
    ) -> ModelRoute: ...


class ClassificationCeilingRouter:
    """Allow generation only up to a fixed classification ceiling.

    Evidence at or below ``ceiling`` is admitted; anything above it is denied so
    over-classified evidence is never sent to the model. Pure and stateless, so a
    verdict is reproducible. The optional ``model_id`` is attached to an allow verdict.
    """

    def __init__(self, ceiling: Classification, model_id: str = "") -> None:
        self._ceiling = ceiling
        self._model_id = model_id

    @property
    def cache_id(self) -> str:
        return f"ceiling:{self._ceiling.name}:{self._model_id}"

    def route(
        self,
        classification: Classification,
        principal: Principal | None,
        *,
        residency_zones: frozenset[str] = frozenset(),
    ) -> ModelRoute:
        del residency_zones  # a sensitivity ceiling is orthogonal to residency
        if classification <= self._ceiling:
            return ModelRoute(
                allowed=True,
                reason="within_ceiling",
                model_id=self._model_id or None,
            )
        return ModelRoute(
            allowed=False,
            reason=f"classification_{classification.name}_exceeds_ceiling_{self._ceiling.name}",
        )


class RetentionTerm(IntEnum):
    """Ordered data-retention terms. Higher is more retained, so worse for residency.

    A provider is admitted only when its retention is at or below a rule's maximum, so the
    admission test is ``provider.retention <= rule.max_retention``.
    """

    ZERO = 0
    TRANSIENT = 1
    LIMITED = 2
    RETAINED = 3


@dataclass(frozen=True, slots=True)
class ProviderResidency:
    """The operator-declared residency tags of the active generation endpoint.

    These are trusted operator declarations, not verified facts about the provider. An empty
    ``region`` or a ``None`` retention is untagged and fails closed at the router.
    """

    provider: str
    region: str = ""
    retention: RetentionTerm | None = None
    purpose: str = ""


@dataclass(frozen=True, slots=True)
class ResidencyRule:
    """The residency admission rule for one classification level.

    ``allowed_purposes`` of ``None`` leaves purpose ungated; a frozenset gates it to the
    listed purposes. ``allowed_zones`` of ``None`` leaves the residency zone ungated; a frozenset
    requires every declared evidence zone to be in the set, denying untagged and malformed evidence.
    """

    allowed_regions: frozenset[str]
    max_retention: RetentionTerm
    allowed_purposes: frozenset[str] | None = None
    allowed_zones: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if self.allowed_zones is None:
            return
        # Coerce every zone to a genuine base str so a str subclass cannot override __eq__ or __hash__
        # to satisfy the subset gate, and refuse a non-printable zone (any control char, checked before
        # trimming so a leading or trailing one also fails, or a sentinel). This closes the direct
        # construction path the config parser already guards for untrusted input; a caller that
        # subclasses this rule to skip the check, or hands the router a hostile evidence set directly,
        # is operator-controlled trusted code outside the untrusted boundary (see the threat model).
        normalized: set[str] = set()
        for zone in self.allowed_zones:
            text = str(zone)
            if not str.isprintable(text):
                raise ValueError("allowed_zones must be printable strings with no control characters")
            normalized.add(str.strip(text))
        object.__setattr__(self, "allowed_zones", frozenset(normalized))


def _structured_digest(payload: object) -> str:
    """A collision-proof digest over a structured payload.

    Serializing a JSON structure with sorted keys and sorted member lists, rather than
    delimiter-joining operator strings, means an operator value that contains the delimiter
    cannot collide with a set boundary, so the identity is injection-proof. json and hashlib
    are stdlib, so this keeps the domain layer leaf with no core import.
    """
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _rule_entry(rule: ResidencyRule) -> dict[str, object]:
    entry: dict[str, object] = {
        "regions": sorted(rule.allowed_regions),
        "max_retention": int(rule.max_retention),
        "purposes": None if rule.allowed_purposes is None else sorted(rule.allowed_purposes),
    }
    # Added only when the rule gates zones, so a zone-free policy keeps the exact pre-feature cache id.
    if rule.allowed_zones is not None:
        entry["zones"] = sorted(rule.allowed_zones)
    return entry


@dataclass(frozen=True, slots=True)
class ResidencyPolicy:
    """A company residency policy: one admission rule per classification level."""

    rules: Mapping[Classification, ResidencyRule]

    def lookup(self, classification: Classification) -> ResidencyRule | None:
        return self.rules.get(classification)

    @property
    def cache_id(self) -> str:
        serialized = {classification.name: _rule_entry(rule) for classification, rule in self.rules.items()}
        return "residency-policy:" + _structured_digest(serialized)


class ResidencyRouter:
    """Gate generation on the active provider's region and retention against a residency policy.

    The classification is derived deterministically from the evidence, never by a model; this
    router maps it to the company residency rule and admits generation only when the operator
    declared provider region is allowed, its retention is within the rule maximum, and (when the
    rule gates purpose) its purpose is permitted. Fail-closed: a missing rule or an untagged
    provider denies. It trusts the operator declared tags and does not verify a provider actual
    hosting region or retention.
    """

    def __init__(self, policy: ResidencyPolicy, provider: ProviderResidency, model_id: str = "") -> None:
        self._policy = policy
        self._provider = provider
        self._model_id = model_id

    @property
    def cache_id(self) -> str:
        payload = {
            "policy": self._policy.cache_id,
            "provider": self._provider.provider,
            "region": self._provider.region,
            "retention": None if self._provider.retention is None else int(self._provider.retention),
            "purpose": self._provider.purpose,
            "model_id": self._model_id,
        }
        return "residency:" + _structured_digest(payload)

    def route(
        self,
        classification: Classification,
        principal: Principal | None,
        *,
        residency_zones: frozenset[str] = frozenset(),
    ) -> ModelRoute:
        rule = self._policy.lookup(classification)
        if rule is None:
            return ModelRoute(allowed=False, reason=f"no_residency_rule_{classification.name}")
        provider = self._provider
        # Guard the None retention before the comparison, so an untagged provider fails closed
        # rather than raising on a None-to-RetentionTerm comparison.
        if not provider.region or provider.retention is None:
            return ModelRoute(allowed=False, reason="provider_untagged")
        if provider.region not in rule.allowed_regions:
            return ModelRoute(allowed=False, reason="region_not_allowed")
        if provider.retention > rule.max_retention:
            return ModelRoute(allowed=False, reason="retention_exceeds_max")
        if rule.allowed_purposes is not None and provider.purpose not in rule.allowed_purposes:
            return ModelRoute(allowed=False, reason="purpose_not_allowed")
        # A gated rule requires every declared evidence zone in the allowed set; the untagged and
        # malformed sentinels are never in an operator set, so unzoned or broken evidence fails closed.
        if rule.allowed_zones is not None and not (residency_zones <= rule.allowed_zones):
            return ModelRoute(allowed=False, reason="zone_not_allowed")
        return ModelRoute(allowed=True, reason="residency_ok", model_id=self._model_id or None)


class CompositeModelRouter:
    """Run several model routers together: deny on the first denial, allow only when all allow.

    Lets a residency router run alongside the classification ceiling router. Pure and stateless,
    so a verdict is reproducible; the cache identity folds the members so a change to any member
    reroutes.
    """

    def __init__(self, routers: Sequence[ModelRouter]) -> None:
        members = tuple(routers)
        if not members:
            raise ValueError("CompositeModelRouter requires at least one router")
        self._routers = members

    @property
    def cache_id(self) -> str:
        return "composite:" + _structured_digest([router.cache_id for router in self._routers])

    def route(
        self,
        classification: Classification,
        principal: Principal | None,
        *,
        residency_zones: frozenset[str] = frozenset(),
    ) -> ModelRoute:
        # Deny on the first member that denies; otherwise the last member's verdict decides.
        # Non-empty by construction, so the final return is provably a ModelRoute, no Optional.
        members = self._routers
        for router in members[:-1]:
            verdict = router.route(classification, principal, residency_zones=residency_zones)
            if not verdict.allowed:
                return verdict
        return members[-1].route(classification, principal, residency_zones=residency_zones)


__all__ = [
    "ClassificationCeilingRouter",
    "CompositeModelRouter",
    "ModelRoute",
    "ModelRouter",
    "ProviderResidency",
    "ResidencyPolicy",
    "ResidencyRouter",
    "ResidencyRule",
    "RetentionTerm",
]
