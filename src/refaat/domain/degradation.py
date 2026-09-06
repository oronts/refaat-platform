"""The canonical, opt-in degradation decision core.

A generation provider can fail at a request seam (the query answer or an RLM node) with a hard
outage. Today an unconfigured runtime lets that outage propagate, so the request refuses with a
503. This module makes that behavior a first-class, governed choice: an installed DegradationPolicy
decides whether a served answer may be degraded to a grounded, cited fallback, abstain, or refuse.

The module is pure: it imports only the standard library, so it carries no dependency on the
query pipeline, the RLM, or any adapter. resolve_degradation centralizes the one
rule that a None policy means the legacy refuse, so no seam re-encodes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class DependencyClass(StrEnum):
    """The governed request seam a dependency failure occurred at. Extensible; only these two
    are wired today (the query generation call and an RLM synthesis node's generation call)."""

    GENERATION = "generation"
    RLM_SYNTHESIS = "rlm_synthesis"


class DegradationMode(StrEnum):
    """What a policy decided to do with a request whose dependency failed.

    SERVE_DEGRADED serves a grounded, cited fallback answer marked as degraded. ABSTAIN returns an
    honest no-answer. REFUSE lets the outage surface as an error (the legacy behavior).
    """

    SERVE_DEGRADED = "serve_degraded"
    ABSTAIN = "abstain"
    REFUSE = "refuse"


@dataclass(frozen=True, slots=True)
class DependencyFailure:
    """A described dependency outage handed to a policy.

    ``grounding_survives`` means a grounded, cited, honest answer still exists on the degraded
    path (for example retrieved chunks are present so the extractive fallback can produce a cited
    answer). A policy must never serve a degraded answer when grounding does not survive.
    """

    dependency_class: DependencyClass
    grounding_survives: bool
    provider_id: str | None = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class DegradationDecision:
    """A policy's verdict for one dependency failure.

    ``marked`` is True when the served answer must carry the degraded flag; it is only meaningful
    for SERVE_DEGRADED. ``reason`` is a machine reason code, never a human message.
    """

    mode: DegradationMode
    dependency_class: DependencyClass
    marked: bool
    reason: str

    @classmethod
    def serve_degraded(cls, dependency_class: DependencyClass, *, marked: bool, reason: str) -> DegradationDecision:
        return cls(mode=DegradationMode.SERVE_DEGRADED, dependency_class=dependency_class, marked=marked, reason=reason)

    @classmethod
    def abstain(cls, dependency_class: DependencyClass, reason: str) -> DegradationDecision:
        return cls(mode=DegradationMode.ABSTAIN, dependency_class=dependency_class, marked=False, reason=reason)

    @classmethod
    def refuse(cls, dependency_class: DependencyClass, reason: str) -> DegradationDecision:
        return cls(mode=DegradationMode.REFUSE, dependency_class=dependency_class, marked=False, reason=reason)


class DegradationPolicy(Protocol):
    """Decides how to handle a dependency failure at a governed request seam."""

    cache_id: str

    def decide(self, failure: DependencyFailure) -> DegradationDecision: ...


class RefuseAllDegradationPolicy:
    """Fail-closed reference policy: every dependency failure refuses, so an outage surfaces as an
    error rather than a degraded answer. This is the strictest posture."""

    cache_id = "refuse-all"

    def decide(self, failure: DependencyFailure) -> DegradationDecision:
        return DegradationDecision.refuse(failure.dependency_class, "refuse_all")


class SafeDefaultsDegradationPolicy:
    """Reference policy that serves a grounded, cited fallback when one exists and abstains
    otherwise, so it never serves an ungrounded answer on a provider outage."""

    cache_id = "safe-defaults"

    def decide(self, failure: DependencyFailure) -> DegradationDecision:
        if failure.grounding_survives:
            return DegradationDecision.serve_degraded(failure.dependency_class, marked=True, reason="grounded_fallback")
        return DegradationDecision.abstain(failure.dependency_class, "no_grounding")


def resolve_degradation(policy: DegradationPolicy | None, failure: DependencyFailure) -> DegradationDecision:
    """Resolve one dependency failure under the installed policy, or the legacy refuse when none.

    A None policy is the legacy behavior: today an outage propagates uncaught, which is a refuse,
    so an unconfigured runtime keeps refusing. The reason on the None path is internal only.
    """
    if policy is None:
        return DegradationDecision.refuse(failure.dependency_class, "legacy_refuse")
    return policy.decide(failure)


__all__ = [
    "DegradationDecision",
    "DegradationMode",
    "DegradationPolicy",
    "DependencyClass",
    "DependencyFailure",
    "RefuseAllDegradationPolicy",
    "SafeDefaultsDegradationPolicy",
    "resolve_degradation",
]
