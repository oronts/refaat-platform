"""Identity-freshness decision core for re-resolving a principal before a risky action.

Pure, model-free, fail-closed decisions for whether a captured identity snapshot is still
fresh enough to authorize an action of a given risk class, and whether the current identity
still holds. A durable job or a governed approval consume captures an :class:`IdentitySnapshot`
at enqueue or grant time, then re-resolves the principal at execution time and asks whether the
snapshot may still be trusted. The enforcement that acts on the decision lives in the worker and
the approval service, so this module is inert until wired: importing it changes nothing.

The snapshot is redacted by construction. It carries only ids, a scope digest, the policy id
and version, and a timestamp, never a full principal, raw groups, roles, entitlements, or a
secret, so it is safe to persist alongside a durable job.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import IntEnum
from typing import Any

from refaat.domain.authz import Principal, auth_scope_hash


class RiskClass(IntEnum):
    """Ordered risk of the action the re-resolved identity would authorize. Higher is riskier."""

    READ = 0
    WRITE = 1
    SENSITIVE_WRITE = 2
    IRREVERSIBLE = 3


@dataclass(frozen=True, slots=True)
class IdentitySnapshot:
    """A redacted capture of an authenticated identity, taken when a durable action is queued.

    Carries only the principal id, the tenant, the authorization scope digest, the deciding
    policy id and version, and the capture time. It never carries a full principal, raw
    entitlements, or a secret, so it is safe to persist. ``captured_at`` must be timezone-aware
    UTC; a naive capture time is treated as invalid at decision time.
    """

    principal_id: str
    tenant: str | None
    auth_scope_hash: str
    policy_id: str
    policy_version: str
    captured_at: datetime


@dataclass(frozen=True, slots=True)
class FreshnessPolicy:
    """How long a snapshot stays fresh per risk class, and which risk classes need approval.

    ``max_age_seconds`` maps a risk class to its maximum snapshot age in seconds;
    ``require_approval`` names the risk classes that always require a human approval alongside a
    fresh identity.
    """

    max_age_seconds: Mapping[RiskClass, int]
    require_approval: frozenset[RiskClass]

    def max_age_for(self, risk_class: RiskClass) -> int:
        """Configured maximum age for a risk class, failing closed to zero on a missing one.

        An unconfigured risk class gets no freshness window (zero, so any positive age denies), so a
        gap in the policy never grants a window the operator did not explicitly configure. This keeps
        the highest-risk classes safe: a policy that omits IRREVERSIBLE still denies any positive age.
        """
        return self.max_age_seconds.get(risk_class, 0)


DEFAULT_FRESHNESS_POLICY = FreshnessPolicy(
    max_age_seconds={
        RiskClass.READ: 86_400,
        RiskClass.WRITE: 3_600,
        RiskClass.SENSITIVE_WRITE: 300,
        RiskClass.IRREVERSIBLE: 0,
    },
    require_approval=frozenset({RiskClass.IRREVERSIBLE}),
)


@dataclass(frozen=True, slots=True)
class FreshnessDecision:
    allowed: bool
    reason: str
    require_approval: bool = False


def snapshot_from_principal(principal: Principal, policy: Any, now: datetime) -> IdentitySnapshot:
    """Capture a redacted identity snapshot for a principal under an access policy.

    The scope digest is computed with :func:`refaat.domain.authz.auth_scope_hash`, and the policy
    id and version are read defensively so a policy object that omits either still yields a stable
    string. ``now`` must be timezone-aware UTC; naivety is caught at decision time.
    """
    return IdentitySnapshot(
        principal_id=principal.id,
        tenant=principal.tenant,
        auth_scope_hash=auth_scope_hash(principal, policy),
        policy_id=str(getattr(policy, "name", type(policy).__name__)),
        policy_version=str(getattr(policy, "version", "")),
        captured_at=now,
    )


def snapshot_to_payload(snapshot: IdentitySnapshot) -> dict[str, str]:
    """Serialize a snapshot to a JSON-safe string mapping with an ISO timestamp.

    An absent tenant serializes to an empty string, the faithful inverse of
    :func:`snapshot_from_payload`, because a tenant is never itself an empty string.
    """
    return {
        "principal_id": snapshot.principal_id,
        "tenant": snapshot.tenant or "",
        "auth_scope_hash": snapshot.auth_scope_hash,
        "policy_id": snapshot.policy_id,
        "policy_version": snapshot.policy_version,
        "captured_at": snapshot.captured_at.isoformat(),
    }


def snapshot_from_payload(payload: Mapping[str, Any]) -> IdentitySnapshot | None:
    """Parse a snapshot back from a payload, failing closed to None on anything malformed.

    Returns None for a non-mapping, a missing or wrong-typed field, or a capture time that does
    not parse or is not timezone-aware. Never raises, so a corrupt persisted snapshot degrades to
    no-snapshot rather than crashing the caller.
    """
    if not isinstance(payload, Mapping):
        return None
    principal_id = payload.get("principal_id")
    tenant_raw = payload.get("tenant")
    scope = payload.get("auth_scope_hash")
    policy_id = payload.get("policy_id")
    policy_version = payload.get("policy_version")
    captured_raw = payload.get("captured_at")
    if not (
        isinstance(principal_id, str)
        and isinstance(tenant_raw, str)
        and isinstance(scope, str)
        and isinstance(policy_id, str)
        and isinstance(policy_version, str)
        and isinstance(captured_raw, str)
    ):
        return None
    try:
        captured_at = datetime.fromisoformat(captured_raw)
    except (TypeError, ValueError):
        return None
    if captured_at.utcoffset() is None:
        return None
    return IdentitySnapshot(
        principal_id=principal_id,
        tenant=(tenant_raw or None),
        auth_scope_hash=scope,
        policy_id=policy_id,
        policy_version=policy_version,
        captured_at=captured_at,
    )


def evaluate_reresolution(
    snapshot: IdentitySnapshot | None,
    refreshed: Principal | None,
    refreshed_scope_hash: str | None,
    risk_class: RiskClass,
    policy: FreshnessPolicy,
    now: datetime,
) -> FreshnessDecision:
    """Decide whether a captured snapshot may still authorize an action, fail-closed throughout.

    The checks apply in order, each with a distinct reason so an operator can tell them apart:
    an invalid clock, a missing snapshot, an invalid or future capture time, a snapshot older than
    the risk class allows, a revoked identity, a re-resolved principal whose id does not match the
    snapshot (the directory returned the wrong record), a changed tenant, and an indeterminate current
    scope all deny. Otherwise the action
    proceeds; the reason is "scope_changed" when the re-resolved scope differs from the captured
    one (the job adopts the current scope) and "fresh" when it matches. Whether the risk class
    also requires a human approval rides on the allow paths and on the staleness denial, so a
    caller re-approving a stale irreversible action learns it still needs approval.
    """
    needs_approval = risk_class in policy.require_approval
    if now.utcoffset() is None:
        return FreshnessDecision(False, "invalid_now")
    if snapshot is None:
        return FreshnessDecision(False, "no_snapshot")
    if snapshot.captured_at.utcoffset() is None or snapshot.captured_at > now:
        return FreshnessDecision(False, "invalid_snapshot_time")
    if now - snapshot.captured_at > timedelta(seconds=policy.max_age_for(risk_class)):
        return FreshnessDecision(False, "snapshot_too_old", require_approval=needs_approval)
    if refreshed is None:
        return FreshnessDecision(False, "identity_revoked")
    if refreshed.id != snapshot.principal_id:
        return FreshnessDecision(False, "principal_mismatch")
    if refreshed.tenant != snapshot.tenant:
        return FreshnessDecision(False, "tenant_changed")
    if refreshed_scope_hash is None:
        return FreshnessDecision(False, "scope_unknown")
    if refreshed_scope_hash != snapshot.auth_scope_hash:
        return FreshnessDecision(True, "scope_changed", require_approval=needs_approval)
    return FreshnessDecision(True, "fresh", require_approval=needs_approval)


__all__ = [
    "DEFAULT_FRESHNESS_POLICY",
    "FreshnessDecision",
    "FreshnessPolicy",
    "IdentitySnapshot",
    "RiskClass",
    "evaluate_reresolution",
    "snapshot_from_payload",
    "snapshot_from_principal",
    "snapshot_to_payload",
]
