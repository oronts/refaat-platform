"""Tool governance and human-approval decision core.

Pure, model-free, fail-closed decisions for whether a governed tool may run and whether a
required human approval is satisfied. A :class:`ToolCapability` is the authoritative registered
declaration of a tool's governance; the model can never influence it, because these functions
take the registered capability and the resolved principal, never model output. The approval
checks (approval_satisfied) are applied by the governed ReAct loop and the approval service.
capability_permits, the pure admission decision over scopes, side-effect ceiling, classification,
payload, network, namespace, and secret, is applied in the governed ReAct admission loop for the
measurable dimensions (scopes, the side-effect ceiling, the evidence-classification ceiling, and
the payload bound). The namespace and secret fields stay declared-only at admission, authoritative and
model-immutable; allowed_network is now enforced on the governed outbound path (the OutboundClient a
wants_outbound skill uses), not at admission. Importing this module changes nothing.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from refaat.canonical import canonical_digest
from refaat.domain.authz import Classification, Principal
from refaat.domain.enums import AuditLevel, SideEffectClass

_MAX_APPROVAL_TTL_SECONDS = 315_360_000  # ten years, an upper bound that also avoids timedelta overflow
_MAX_NETWORK_DESTINATIONS = 64
_MAX_REQUIRED_SECRETS = 64
_MAX_ENTRY_LENGTH = 256
_MAX_NAMESPACE_LENGTH = 512

# Assigned Default_Ignorable_Code_Point members that are not category Cc, Cf, or Cn (variation
# selectors, the combining grapheme joiner, Mongolian free variation selectors, Hangul fillers, tag
# characters). Unassigned (Cn) members such as U+2065 are covered by the category filter below, so
# together they strip the whole Default_Ignorable property the RFC 8264 IdentifierClass rejects.
_DEFAULT_IGNORABLE = frozenset(
    {0x034F, 0x115F, 0x1160, 0x17B4, 0x17B5, 0x3164, 0xFFA0}
    | set(range(0x180B, 0x1810))
    | set(range(0xFE00, 0xFE10))
    | set(range(0xE0000, 0xE1000))
)


def _normalize_id(value: str) -> str:
    """Canonical form for identity comparison, following the PRECIS UsernameCaseMapped recipe
    (RFC 8264/8265): drop control, format, unassigned, and default-ignorable characters, then NFKC,
    casefold, and NFKC again, so a case, whitespace, compatibility, or invisible-character variant
    of the same id cannot self-approve or pad a four-eyes count. Dropping the unassigned (Cn)
    category also future-proofs against any newly reserved default-ignorable code point. This is
    defense in depth: the caller must still supply the identity provider's canonical, stable
    principal id, not a display name, since genuine confusables and locale-dependent case are the
    identity provider's responsibility."""
    cleaned = "".join(
        ch for ch in value if unicodedata.category(ch) not in ("Cc", "Cf", "Cn") and ord(ch) not in _DEFAULT_IGNORABLE
    )
    folded = unicodedata.normalize("NFKC", cleaned).casefold()
    return unicodedata.normalize("NFKC", folded).strip()


def _valid_identifier_entry(entry: object, *, max_length: int) -> bool:
    return isinstance(entry, str) and bool(entry) and not any(c.isspace() for c in entry) and len(entry) <= max_length


@dataclass(frozen=True, slots=True)
class ToolCapability:
    """The authoritative, model-independent governance declaration for one tool.

    Every field is set at registration and is authoritative: no model output and no request
    body can widen it. ``requires_approval`` gates execution behind a human approval bound to an
    immutable action fingerprint, ``min_approvals`` above one is a four-eyes control, and
    ``approver_role`` restricts who may approve. The ``escalation_*`` group widens who may approve
    on the normal path after a bounded, clock-driven delay: it only adds a second acceptable role,
    never lowering ``min_approvals`` or the ttl and never bypassing separation of duties. The
    ``break_glass_*`` group opts a capability into an explicit, attributable alternate approval path
    that is stricter than the normal one (a bound role and a dual-approval floor), never a hidden
    super-admin flag and never a duties bypass.
    """

    name: str
    required_scopes: frozenset[str] = frozenset()
    side_effect: SideEffectClass = SideEffectClass.READ_ONLY
    allowed_classification: Classification = Classification.PUBLIC
    max_payload_bytes: int | None = None
    requires_approval: bool = False
    min_approvals: int = 1
    approver_role: str | None = None
    approval_ttl_seconds: int | None = None
    escalation_role: str | None = None
    escalation_after_seconds: int | None = None
    break_glass_enabled: bool = False
    break_glass_approver_role: str | None = None
    break_glass_min_approvals: int = 2
    break_glass_ttl_seconds: int | None = None
    # Empty allowed_network means not declared (no network metadata), never deny-all; deny-all egress is
    # the deployment default. Entries in allowed_network and required_secrets are logical destinations and
    # secret handles or names, never secret values, so they are safe to fold into an artifact or an audit event.
    allowed_network: frozenset[str] = frozenset()
    resource_namespace: str | None = None
    required_secrets: frozenset[str] = frozenset()
    # STANDARD is the default so an unset capability is byte-identical; ENHANCED opts a governed
    # step into a redacted-by-construction observability record (span plus structured log).
    audit_level: AuditLevel = AuditLevel.STANDARD

    def __post_init__(self) -> None:
        # isinstance, not a membership test: a StrEnum member equals its string value, so a raw
        # "standard" would pass ``in AuditLevel`` yet break the deterministic str() fold.
        if not isinstance(self.audit_level, AuditLevel):
            raise ValueError("audit_level must be an AuditLevel")
        if self.min_approvals < 1:
            raise ValueError("min_approvals must be at least 1")
        if self.approval_ttl_seconds is not None and not 0 < self.approval_ttl_seconds <= _MAX_APPROVAL_TTL_SECONDS:
            raise ValueError("approval_ttl_seconds must be positive and at most ten years")
        if self.escalation_role is not None or self.escalation_after_seconds is not None:
            # Escalation only widens the normal path's acceptable roles after a delay, so both the
            # widened role and the delay are mandatory together and it needs a base role to widen.
            if self.escalation_role is None or self.escalation_after_seconds is None:
                raise ValueError("escalation_role and escalation_after_seconds must be set together")
            if not (isinstance(self.escalation_role, str) and self.escalation_role):
                raise ValueError("escalation_role must be a non-empty string when escalation is declared")
            if not 0 < self.escalation_after_seconds <= _MAX_APPROVAL_TTL_SECONDS:
                raise ValueError("escalation_after_seconds must be positive and at most ten years")
            if self.approver_role is None:
                raise ValueError("escalation requires an approver_role to widen")
            if self.escalation_role == self.approver_role:
                raise ValueError("escalation_role must differ from approver_role")
        if self.break_glass_enabled:
            # Break-glass is stricter than the normal path, never a bypass: a bound approver role and
            # a dual-approval floor are mandatory the moment it is enabled.
            if not (isinstance(self.break_glass_approver_role, str) and self.break_glass_approver_role):
                raise ValueError("break_glass_approver_role must be a non-empty string when break-glass is enabled")
            if self.break_glass_min_approvals < 2:
                raise ValueError("break_glass_min_approvals must be at least 2")
            if (
                self.break_glass_ttl_seconds is not None
                and not 0 < self.break_glass_ttl_seconds <= _MAX_APPROVAL_TTL_SECONDS
            ):
                raise ValueError("break_glass_ttl_seconds must be positive and at most ten years")
        if len(self.allowed_network) > _MAX_NETWORK_DESTINATIONS or not all(
            _valid_identifier_entry(entry, max_length=_MAX_ENTRY_LENGTH) for entry in self.allowed_network
        ):
            raise ValueError("allowed_network entries must be non-empty whitespace-free strings within the bounds")
        if self.resource_namespace is not None and not _valid_identifier_entry(
            self.resource_namespace, max_length=_MAX_NAMESPACE_LENGTH
        ):
            raise ValueError("resource_namespace must be a non-empty whitespace-free string within the length bound")
        if len(self.required_secrets) > _MAX_REQUIRED_SECRETS or not all(
            _valid_identifier_entry(entry, max_length=_MAX_ENTRY_LENGTH) for entry in self.required_secrets
        ):
            raise ValueError("required_secrets entries must be non-empty whitespace-free strings within the bounds")


@dataclass(frozen=True, slots=True)
class DirectSkillPolicy:
    """The authoritative, model-independent admission policy for the direct skill-run surface.

    Mirrors the measurable-dimension half of a :class:`ReActPolicy` for skills invoked directly,
    outside a governed ReAct loop: a policy-level granted-scope set and side-effect ceiling, plus a
    ``ToolCapability`` per governed skill carrying its own required scopes, side-effect class,
    allowed classification, and payload bound. A skill without a declared capability runs unchanged.

    It admits a capability against the four measurable dimensions (scopes, the side-effect ceiling,
    the evidence-classification ceiling, and the payload bound), the same subset the governed loop
    enforces; the namespace and secret fields stay declared-only here, and allowed_network is not
    checked at admission but is enforced on the governed outbound path. A capability that
    requires approval is refused outright on the direct path, which derives no action fingerprint,
    so a four-eyes control lives only on the governed program surface. This policy has no
    content-addressed artifact and never enters any program identity.

    Deriving the evidence classification soundly needs principal-scoped evidence, because the Policy
    Enforcement Point filters before the top-k truncation. A capability that declares a real
    classification ceiling therefore fails closed on a principal-less direct call while an access
    policy is active; principal-precise ACL retrieval on the HTTP and MCP direct surfaces is a
    separate principal-threading follow-up, not introduced here.
    """

    granted_scopes: frozenset[str] = frozenset()
    side_effect_ceiling: SideEffectClass = SideEffectClass.READ_ONLY
    capabilities: tuple[ToolCapability, ...] = ()

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for capability in self.capabilities:
            if capability.name in seen:
                raise ValueError(f"duplicate capability for skill {capability.name!r}")
            seen.add(capability.name)


@dataclass(frozen=True, slots=True)
class CapabilityDecision:
    allowed: bool
    reason: str


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
    satisfied: bool
    reason: str
    consumed_ids: tuple[str, ...] = ()
    break_glass: bool = False


@dataclass(frozen=True, slots=True)
class ApprovalRecord:
    """A single recorded human approval, bound to an immutable action fingerprint.

    ``approver_roles`` is captured at grant time; ``granted_at`` is required for a capability
    that expires approvals, and an approval without it never counts under such a capability.
    ``kind`` partitions a normal approval from a break-glass approval so neither path can borrow
    the other's records; it is runtime approval state and never part of a content-addressed artifact.
    """

    fingerprint: str
    approver_id: str
    approver_roles: frozenset[str] = frozenset()
    granted_at: datetime | None = None
    record_id: str | None = None
    kind: str = "normal"


def approval_fingerprint(
    *,
    capability_name: str,
    arguments: Mapping[str, Any],
    principal: Principal,
    policy_id: str,
    policy_version: str,
    action_version: str,
) -> str:
    """Immutable digest binding tool, arguments, principal, policy, and action version.

    An approval is valid only for this exact fingerprint, so a change to the arguments, the
    initiating principal, the policy, or the action version invalidates a prior approval and
    forces re-approval.
    """
    return canonical_digest(
        {
            "capability": capability_name,
            "arguments": dict(arguments),
            "principal": principal.id,
            "principal_kind": principal.kind.value,
            "tenant": principal.tenant,
            "policy_id": policy_id,
            "policy_version": policy_version,
            "action_version": action_version,
        }
    )


def capability_permits(
    capability: ToolCapability,
    *,
    granted_scopes: Iterable[str],
    side_effect_ceiling: SideEffectClass,
    evidence_classification: Classification,
    payload_bytes: int | None = None,
    requested_destinations: Iterable[str] | None = None,
    requested_namespace: str | None = None,
    available_secrets: frozenset[str] | None = None,
) -> CapabilityDecision:
    """Model-independent admission of a tool against its authoritative capability, fail-closed."""
    granted = frozenset(granted_scopes)
    missing = tuple(sorted(scope for scope in capability.required_scopes if scope not in granted))
    if missing:
        return CapabilityDecision(False, "missing_scopes:" + ",".join(missing))
    if capability.side_effect.rank > side_effect_ceiling.rank:
        return CapabilityDecision(False, "side_effect_exceeds_ceiling")
    if evidence_classification > capability.allowed_classification:
        return CapabilityDecision(False, "classification_exceeds_capability")
    if capability.max_payload_bytes is not None:
        # A declared cap that cannot be measured denies rather than admits an unmeasured payload.
        if payload_bytes is None:
            return CapabilityDecision(False, "payload_size_unknown")
        if payload_bytes > capability.max_payload_bytes:
            return CapabilityDecision(False, "payload_exceeds_max")
    # A destination, namespace, or secret set is skill-internal, not generically measurable at the gate,
    # so each dimension is checked only when the caller supplies the concrete value; None means unchecked.
    if capability.allowed_network and requested_destinations is not None:
        denied = tuple(sorted(dest for dest in requested_destinations if dest not in capability.allowed_network))
        if denied:
            return CapabilityDecision(False, "network_destination_denied:" + ",".join(denied))
    if (
        capability.resource_namespace is not None
        and requested_namespace is not None
        and requested_namespace != capability.resource_namespace
    ):
        return CapabilityDecision(False, "resource_namespace_denied")
    if capability.required_secrets and available_secrets is not None:
        unavailable = tuple(sorted(secret for secret in capability.required_secrets if secret not in available_secrets))
        if unavailable:
            return CapabilityDecision(False, "required_secret_unavailable:" + ",".join(unavailable))
    return CapabilityDecision(True, "permitted")


def _escalation_active(capability: ToolCapability, *, request_anchor: datetime | None, now: datetime) -> bool:
    """Whether the normal path's acceptable roles are widened for this now and anchor.

    Escalation activates only when the capability declares it, a request anchor is supplied, and
    ``now`` is past ``escalation_after_seconds`` from that anchor. A naive (offsetless) anchor or
    ``now`` disables escalation, failing closed to the un-escalated behavior.
    """
    if capability.escalation_role is None or capability.escalation_after_seconds is None:
        return False
    if request_anchor is None or request_anchor.utcoffset() is None or now.utcoffset() is None:
        return False
    return now - request_anchor > timedelta(seconds=capability.escalation_after_seconds)


def _acceptable_roles(capability: ToolCapability, *, request_anchor: datetime | None, now: datetime) -> set[str] | None:
    """The roles a normal-path approver may hold, or None when no role is required.

    The base ``approver_role`` is always acceptable; the widened ``escalation_role`` is added only
    when escalation is active for this now and anchor. Computed once per call.
    """
    if capability.approver_role is None:
        return None
    roles = {capability.approver_role}
    if _escalation_active(capability, request_anchor=request_anchor, now=now):
        assert capability.escalation_role is not None  # guaranteed by _escalation_active
        roles.add(capability.escalation_role)
    return roles


def approval_satisfied(
    capability: ToolCapability,
    *,
    fingerprint: str,
    approvals: Iterable[ApprovalRecord],
    initiator_id: str,
    now: datetime,
    request_anchor: datetime | None = None,
) -> ApprovalDecision:
    """Whether a required human approval is satisfied for this fingerprint, fail-closed.

    Enforces separation of duties (an initiator can never approve its own action), a required
    approver role, an exact fingerprint match, approval expiry, and distinct approvers counted
    up to ``min_approvals`` so a four-eyes control needs that many different people. Identity
    comparison is case-insensitive so a case variant of the same id cannot self-approve or pad a
    four-eyes count. ``now`` and each ``granted_at`` must be timezone-aware UTC.

    When the capability declares escalation and ``request_anchor`` is past the escalation delay,
    the set of acceptable approver roles widens from the base role to also include the escalation
    role. Escalation only widens who may approve: it never lowers ``min_approvals`` or the ttl and
    never lets the initiator count.

    A run is satisfied by the normal path or, only when the capability opts in, by the explicit
    break-glass path. The two paths are partitioned by record kind: a normal record never counts
    toward break-glass and a break-glass record never counts toward the normal path.
    """
    if not capability.requires_approval:
        return ApprovalDecision(True, "no_approval_required")
    if not fingerprint:
        return ApprovalDecision(False, "missing_fingerprint")
    initiator = _normalize_id(initiator_id)
    if not initiator:
        return ApprovalDecision(False, "missing_initiator")
    if capability.approval_ttl_seconds is not None and now.utcoffset() is None:
        return ApprovalDecision(False, "invalid_now")
    acceptable_roles = _acceptable_roles(capability, request_anchor=request_anchor, now=now)
    records = tuple(approvals)
    distinct: set[str] = set()
    consumed_ids: list[str] = []
    for record in records:
        if record.kind != "normal":
            continue
        if record.fingerprint != fingerprint:
            continue
        approver = _normalize_id(record.approver_id)
        if not approver or approver == initiator or approver in distinct:
            continue
        if acceptable_roles is not None and acceptable_roles.isdisjoint(record.approver_roles):
            continue
        if capability.approval_ttl_seconds is not None:
            if record.granted_at is None or record.granted_at.utcoffset() is None:
                continue
            if record.granted_at > now:
                continue
            if now > record.granted_at + timedelta(seconds=capability.approval_ttl_seconds):
                continue
        distinct.add(approver)
        # Bind the run to the exact durable approvals that satisfied it, so consume marks only those
        # and a concurrent cancel denies rather than letting a sibling row absorb the consume.
        if record.record_id is not None:
            consumed_ids.append(record.record_id)
        if len(distinct) >= capability.min_approvals:
            break
    if len(distinct) >= capability.min_approvals:
        return ApprovalDecision(True, "approved", tuple(consumed_ids))
    if capability.break_glass_enabled:
        decided = _break_glass_satisfied(
            capability, fingerprint=fingerprint, records=records, initiator=initiator, now=now
        )
        if decided is not None:
            return decided
    return ApprovalDecision(False, f"insufficient_approvals:{len(distinct)}/{capability.min_approvals}")


def _break_glass_satisfied(
    capability: ToolCapability,
    *,
    fingerprint: str,
    records: tuple[ApprovalRecord, ...],
    initiator: str,
    now: datetime,
) -> ApprovalDecision | None:
    """The explicit, attributable alternate approval path, stricter than the normal one.

    It counts only break-glass records, enforces the same separation of duties and expiry as the
    normal path, requires the break-glass approver role, and needs at least the dual-approval floor
    of distinct approvers. Returns the satisfied decision, or None when the path does not satisfy so
    the caller falls through to fail closed. A break-glass ttl with a naive ``now`` fails closed.
    """
    ttl = capability.break_glass_ttl_seconds
    if ttl is not None and now.utcoffset() is None:
        return None
    role = capability.break_glass_approver_role
    distinct: set[str] = set()
    consumed_ids: list[str] = []
    for record in records:
        if record.kind != "break_glass":
            continue
        if record.fingerprint != fingerprint:
            continue
        approver = _normalize_id(record.approver_id)
        if not approver or approver == initiator or approver in distinct:
            continue
        if role is not None and role not in record.approver_roles:
            continue
        if ttl is not None:
            if record.granted_at is None or record.granted_at.utcoffset() is None:
                continue
            if record.granted_at > now:
                continue
            if now > record.granted_at + timedelta(seconds=ttl):
                continue
        distinct.add(approver)
        if record.record_id is not None:
            consumed_ids.append(record.record_id)
        if len(distinct) >= capability.break_glass_min_approvals:
            break
    if len(distinct) >= capability.break_glass_min_approvals:
        return ApprovalDecision(True, "break_glass_approved", tuple(consumed_ids), break_glass=True)
    return None


__all__ = [
    "ApprovalDecision",
    "ApprovalRecord",
    "CapabilityDecision",
    "DirectSkillPolicy",
    "ToolCapability",
    "approval_fingerprint",
    "approval_satisfied",
    "capability_permits",
]
