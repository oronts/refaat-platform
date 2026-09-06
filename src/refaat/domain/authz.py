"""Canonical authorization vocabulary and the fail-closed Policy Decision Point.

Defines the value objects and rules that decide whether a principal may see an
evidence unit. The enforcement points that apply the decision live in the retrieval,
RLM, and query services. The decision is fail-closed: any ambiguity denies, and a
model can never influence any of these values. INTERNAL with no allow list is a
policy decision, so the shipped canonical policy denies it rather than inventing an
organizational meaning; a customer policy may grant it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any, Protocol

from refaat.errors import AuthorizationError


class Classification(IntEnum):
    """Ordered sensitivity levels. Higher is more sensitive.

    Distinct from source trust. A document can be PUBLIC and AUTHORITATIVE or
    CONFIDENTIAL and UNVERIFIED, so classification is never overloaded with trust.
    """

    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    RESTRICTED = 3


class SourceTrust(IntEnum):
    """How much a piece of evidence is trusted, independent of confidentiality."""

    UNVERIFIED = 0
    EXTERNAL = 1
    USER_UPLOAD = 2
    INTERNAL = 3
    AUTHORITATIVE = 4


class PrincipalKind(StrEnum):
    """An identity is not always a person."""

    HUMAN = "human"
    SERVICE = "service"
    SYSTEM = "system"


@dataclass(frozen=True, slots=True)
class Principal:
    """An already-authenticated identity, resolved server-side from trusted context.

    Never constructed from a client-supplied HTTP payload. The in-process Python API
    may pass one directly because the caller is trusted application code; the HTTP
    surface resolves it through a ``PrincipalResolver`` and ignores any client-sent
    groups, roles, tenant, or kind.

    ``groups``, ``roles``, and ``attributes`` are the policy-relevant entitlements. An
    authorization scope hash and the audit trail use only these normalized
    entitlements, never arbitrary identity claims such as an email, so the cache key
    stays stable and minimal.
    """

    id: str
    kind: PrincipalKind = PrincipalKind.HUMAN
    tenant: str | None = None
    groups: frozenset[str] = frozenset()
    roles: frozenset[str] = frozenset()
    attributes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AccessTag:
    """The canonical access control metadata on an evidence unit.

    One shared meaning across every connector even though the decision is pluggable.
    A connector that carries source permissions maps them into this shape at
    ingestion, so the permission travels source to document to chunk to graph fact to
    retrieval to citation.
    """

    classification: Classification = Classification.PUBLIC
    tenant: str | None = None
    allow_groups: frozenset[str] = frozenset()
    allow_roles: frozenset[str] = frozenset()
    deny_groups: frozenset[str] = frozenset()
    deny_users: frozenset[str] = frozenset()
    residency_zone: str | None = None


@dataclass(frozen=True, slots=True)
class AccessDecision:
    """The Policy Decision Point result.

    Carries a machine-readable ``reason`` plus the deciding ``policy_id`` and
    ``policy_version`` for auditing, support, cache invalidation, and compliance
    evidence. Sensitive ACL internals are never surfaced to the caller; this metadata
    is recorded internally.
    """

    allowed: bool
    reason: str
    policy_id: str = "canonical"
    policy_version: str = "1"


def _evaluate(principal: Principal, tag: AccessTag) -> tuple[bool, str]:
    """Apply the canonical fail-closed rules, returning the verdict and its reason.

    Rules in order:

    1. Tenant mismatch denies. Defense in depth; the tenant and workspace boundary is
       also a hard invariant enforced before any pluggable policy.
    2. An explicit deny (``deny_users`` or ``deny_groups``) beats any allow.
    3. If an allow list is present, allow only when the principal satisfies
       ``allow_groups`` or ``allow_roles``.
    4. With no allow list, PUBLIC is visible within the workspace, INTERNAL fails
       closed here because its organizational meaning belongs to the customer policy,
       and CONFIDENTIAL and RESTRICTED fail closed as sensitive-without-a-grant.
    """
    if tag.tenant is not None and tag.tenant != principal.tenant:
        return False, "tenant_mismatch"
    if principal.id in tag.deny_users:
        return False, "explicit_deny_user"
    if principal.groups & tag.deny_groups:
        return False, "explicit_deny_group"
    if tag.allow_groups or tag.allow_roles:
        if principal.groups & tag.allow_groups:
            return True, "allow_group"
        if principal.roles & tag.allow_roles:
            return True, "allow_role"
        return False, "no_matching_allow"
    if tag.classification == Classification.PUBLIC:
        return True, "public_open"
    if tag.classification == Classification.INTERNAL:
        return False, "internal_no_allow_policy_decides"
    return False, "sensitive_no_allow_denied"


def authorize(principal: Principal, tag: AccessTag) -> bool:
    """Boolean convenience over :func:`_evaluate` for the common allow check."""
    return _evaluate(principal, tag)[0]


UNTAGGED_ZONE = "\x00zone:untagged"
MALFORMED_ZONE = "\x00zone:malformed"


def residency_zones(metadatas: Iterable[Mapping[str, Any] | None]) -> frozenset[str]:
    """The declared residency zones across evidence metadata, fail-closed on a malformed tag.

    An untagged unit contributes ``UNTAGGED_ZONE`` and a malformed ACL or zone contributes
    ``MALFORMED_ZONE``; neither is ever a member of an operator's allowed zones, so a rule that gates
    zones denies untagged and malformed evidence. Mirrors :func:`max_classification` so the query and
    RLM egress derivations cannot drift. The sentinels are control-char prefixed so an operator zone
    string can never collide with one.
    """
    zones: set[str] = set()
    for metadata in metadatas:
        tag = access_tag_from_metadata(metadata)
        if tag is None:
            zones.add(MALFORMED_ZONE if isinstance(metadata, Mapping) and "acl" in metadata else UNTAGGED_ZONE)
        elif tag.residency_zone is None:
            zones.add(UNTAGGED_ZONE)
        else:
            zones.add(tag.residency_zone)
    return frozenset(zones)


def max_classification(metadatas: Iterable[Mapping[str, Any] | None]) -> Classification:
    """The maximum classification across evidence metadata, fail-closed on a malformed tag.

    Derived deterministically, never by a model. An absent ACL is PUBLIC; a present but
    unparseable ACL fails closed to RESTRICTED, the maximum sensitivity, matching the Policy
    Enforcement Point which denies a present-but-malformed ACL, so a broken tag can never lower
    the derived ceiling. Shared by the query and RLM model-routing egress gates so they derive
    the same ceiling and cannot drift.
    """
    highest = Classification.PUBLIC
    for metadata in metadatas:
        tag = access_tag_from_metadata(metadata)
        if tag is not None:
            level = tag.classification
        elif isinstance(metadata, Mapping) and "acl" in metadata:
            level = Classification.RESTRICTED
        else:
            level = Classification.PUBLIC
        if level > highest:
            highest = level
    return highest


def summary_evidence_from_documents(
    cited_ids: frozenset[str] | None, documents: Iterable[Mapping[str, Any]]
) -> tuple[Classification, frozenset[str]]:
    """Classification and residency of a summary's cited documents, fail-closed on unresolved provenance.

    A compacted summary still carries a cited document's content after that document is erased or
    replaced away, so an id that no longer resolves, and a whole summary whose provenance was never
    recorded (``cited_ids`` is None: compacted before a router existed), are unresolved provenance:
    they floor the classification to RESTRICTED and contribute a fail-closed zone, never allowed to
    lower the ceiling for evidence the model still receives. An empty recorded set is a summary that
    folded no cited documents, genuinely PUBLIC. Recomputed at read time from current document ACLs,
    so a re-tag after compaction is reflected, matching the graph and hit derivations.
    """
    if cited_ids is None:
        return Classification.RESTRICTED, residency_zones([None])
    by_id = {str(document["id"]): document.get("metadata") for document in documents if "id" in document}
    metadatas: list[Mapping[str, Any] | None] = []
    missing = False
    for cited in cited_ids:
        if cited in by_id:
            metadatas.append(by_id[cited])
        else:
            missing = True
            metadatas.append(None)
    classification = max_classification(metadatas)
    if missing:
        classification = max(classification, Classification.RESTRICTED)
    return classification, residency_zones(metadatas)


def summary_documents_authorized(
    policy: AccessPolicy | None,
    principal: Principal | None,
    cited_ids: frozenset[str] | None,
    documents: Iterable[Mapping[str, Any]],
) -> bool:
    """Whether every document a session summary cited is still authorized for the reading principal.

    Re-authorizes the summary's cited documents against the current policy at read time, so a
    document-side ACL tightening after compaction suppresses the summary, mirroring the answer-cache
    re-validation. A no-op that allows when no policy or principal is active, and when provenance was
    never recorded (cited_ids is None). A cited document that no longer resolves is left to the egress
    classification ceiling; only a resolved-but-now-forbidden cited document suppresses the summary.
    """
    if policy is None or principal is None or cited_ids is None:
        return True
    by_id = {str(document["id"]): document.get("metadata") for document in documents if "id" in document}
    return all(cited not in by_id or chunk_allowed(policy, principal, by_id[cited]) for cited in cited_ids)


def require_principal_present(policy: AccessPolicy | None, principal: Principal | None, *, strict: bool) -> None:
    """Fail closed when strict authorization requires a principal and none was supplied.

    On by default: an in-process call under an active policy that omits the principal is denied, so
    an omitted principal cannot silently widen access or reach a model, a quota, or stored evidence
    metadata before the check. Set strict to off to restore the trusted-in-process behavior where
    such a call runs unfiltered. With no policy, or with a principal supplied, this is a no-op, so
    those paths are byte-identical. The HTTP boundary already fails closed.
    """
    if strict and policy is not None and principal is None:
        raise AuthorizationError("an access policy is active and strict authorization requires a principal")


def access_tag_from_metadata(metadata: Mapping[str, Any] | None) -> AccessTag | None:
    """Parse an ``acl`` block out of free-form evidence metadata.

    Returns ``None`` when there is no ACL block, and also when the block is malformed
    or names an unknown classification. In authorization mode the caller treats a
    ``None`` from a present-but-malformed block as a denial, so a broken ACL never
    silently becomes public. An absent block is a separate concept handled by the
    caller under the active mode.
    """
    if not isinstance(metadata, Mapping):
        return None
    raw = metadata.get("acl")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        return None
    try:
        if "classification" in raw:
            classification = _parse_classification(raw["classification"])
            if classification is None:
                return None
        else:
            classification = Classification.PUBLIC
        return AccessTag(
            classification=classification,
            tenant=_opt_str(raw.get("tenant")),
            allow_groups=_str_set(raw.get("allow_groups")),
            allow_roles=_str_set(raw.get("allow_roles")),
            deny_groups=_str_set(raw.get("deny_groups")),
            deny_users=_str_set(raw.get("deny_users")),
            residency_zone=_parse_zone(raw),
        )
    except (TypeError, ValueError):
        return None


def _parse_zone(raw: Mapping[str, Any]) -> str | None:
    """None when absent (untagged), the value when a clean string, else the malformed sentinel.

    A present-but-malformed zone yields ``MALFORMED_ZONE`` rather than nulling the whole tag, so a
    broken zone never lowers the derived classification. A non-printable value (any control character)
    is malformed, so an operator string can never impersonate a sentinel and a broken zone fails closed.
    """
    if "residency_zone" not in raw:
        return None
    value = raw["residency_zone"]
    # Base str methods are called unbound so a str subclass cannot override isprintable or strip to
    # smuggle a sentinel; printability is checked before trimming so a leading or trailing control
    # character also fails closed, and str.strip yields a genuine base str, not a subclass instance.
    if isinstance(value, str) and str.isprintable(value):
        stripped = str.strip(value)
        if stripped:
            return stripped
    return MALFORMED_ZONE


def _parse_classification(value: Any) -> Classification | None:
    if isinstance(value, Classification):
        return value
    if isinstance(value, str):
        try:
            return Classification[value.strip().upper()]
        except KeyError:
            return None
    return None


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError("tenant must be a non-empty string or absent")


def _str_set(value: Any) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        raise ValueError("expected a list of strings, not a bare string")
    if isinstance(value, (list, tuple, set, frozenset)):
        items = {str(item).strip() for item in value if str(item).strip()}
        return frozenset(items)
    raise ValueError("expected a list of strings")


def source_trust_from_metadata(metadata: Mapping[str, Any] | None) -> SourceTrust:
    """Parse a top-level ``source_trust`` tag out of free-form evidence metadata.

    Trust is independent of confidentiality, so it is read from a separate top-level key,
    never from the ``acl`` block. The value is a case-insensitive :class:`SourceTrust` enum
    name. An absent or unparseable tag defaults to ``UNVERIFIED`` (the minimum): trust must be
    asserted, never assumed, so untagged content cannot evade a trust down-rank and a malformed
    tag can never grant maximum trust. This only affects ranking, and only when a positive
    source-trust weight is configured; at the default weight of zero the value is never read, so
    the default path is byte-identical. Pure and deterministic; a model never sets or influences it.
    """
    if not isinstance(metadata, Mapping):
        return SourceTrust.UNVERIFIED
    raw = metadata.get("source_trust")
    if isinstance(raw, SourceTrust):
        return raw
    if isinstance(raw, str):
        try:
            return SourceTrust[raw.strip().upper()]
        except KeyError:
            return SourceTrust.UNVERIFIED
    return SourceTrust.UNVERIFIED


class CanonicalAccessPolicy:
    """The shipped default Policy Decision Point applying the canonical rules.

    A use case can use this directly or replace it with a stricter policy, or route
    the decision to an external engine such as OPA or Cedar behind the same
    ``decide`` contract, without any engine entering refaat core. It is pure and
    stateless, so a decision is reproducible.
    """

    name = "canonical"
    version = "1"

    def decide(self, principal: Principal, tag: AccessTag) -> AccessDecision:
        allowed, reason = _evaluate(principal, tag)
        return AccessDecision(allowed=allowed, reason=reason, policy_id=self.name, policy_version=self.version)


class AccessPolicy(Protocol):
    """A Policy Decision Point: decide visibility for a principal and an access tag."""

    def decide(self, principal: Principal, tag: AccessTag) -> AccessDecision: ...


class PrincipalResolver(Protocol):
    """Turn already-authenticated, trusted server-side context into a canonical principal.

    Refaat does not authenticate. A deployment adapts its identity provider to this port,
    and the HTTP surface resolves the principal from trusted context only, never from a
    client-supplied request body. Returns None when no identity is present.
    """

    def resolve(self, context: Mapping[str, Any]) -> Principal | None: ...


class PrincipalDirectory(Protocol):
    """Re-resolve a principal by stable id at execution time, the counterpart to PrincipalResolver.

    A governed approval consume or a durable job has no HTTP request, so it looks a principal up by
    id rather than from request context. Returns the current principal, None when the principal is
    unknown or revoked, and raises when the directory itself is unavailable, so a caller fails closed
    on any of the three. A deployment adapts its identity provider to this port; unset, execution
    uses the grant-time or enqueue-time snapshot exactly as before.
    """

    def refresh(self, principal_id: str) -> Principal | None: ...


class EntitlementScopeResolver(Protocol):
    """Map a resolved principal's entitlements to the concrete scopes it effectively holds.

    Pure and deterministic; a model never influences it. A deployment whose vocabulary differs from
    roles-and-groups supplies its own implementation. Unset, tool governance is byte-identical to
    before: the caller's scopes are not intersected with a program's self-declared grants.
    """

    def scopes_for(self, principal: Principal) -> frozenset[str]: ...


class RoleGroupScopeResolver:
    """Reference resolver: a principal effectively holds the scopes named by its roles and groups."""

    def scopes_for(self, principal: Principal) -> frozenset[str]:
        return principal.roles | principal.groups


def _claims_set(value: Any) -> frozenset[str]:
    if isinstance(value, (list, tuple, set, frozenset)):
        return frozenset(str(item).strip() for item in value if str(item).strip())
    return frozenset()


def principal_from_claims(claims: Mapping[str, Any]) -> Principal | None:
    """Build a principal from a trusted claims mapping, or None when it names no identity."""
    if not isinstance(claims, Mapping):
        return None
    identifier = claims.get("id")
    if not isinstance(identifier, str) or not identifier.strip():
        return None
    kind_raw = claims.get("kind")
    try:
        kind = PrincipalKind(kind_raw) if isinstance(kind_raw, str) else PrincipalKind.HUMAN
    except ValueError:
        kind = PrincipalKind.HUMAN
    tenant = claims.get("tenant")
    return Principal(
        id=identifier.strip(),
        kind=kind,
        tenant=tenant.strip() if isinstance(tenant, str) and tenant.strip() else None,
        groups=_claims_set(claims.get("groups")),
        roles=_claims_set(claims.get("roles")),
    )


class HeaderPrincipalResolver:
    """Reference resolver that reads a JSON claims object from a configured trusted header.

    A hardened deployment sets this header at a trusted upstream auth proxy and strips any
    client-supplied copy at the edge, so the identity is server-side. Returns None when the
    header is absent or malformed; the fail-closed decision for an active policy is the
    caller's, at the HTTP boundary.
    """

    def __init__(self, header: str = "x-refaat-principal") -> None:
        self._header = header.lower()

    def resolve(self, context: Mapping[str, Any]) -> Principal | None:
        headers = context.get("headers")
        if not isinstance(headers, Mapping):
            return None
        raw = headers.get(self._header)
        if not isinstance(raw, str) or not raw.strip():
            return None
        try:
            claims = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
        return principal_from_claims(claims) if isinstance(claims, Mapping) else None


# An evidence unit with no ACL block is treated as INTERNAL under authorization mode,
# which the canonical policy denies, so enabling authorization on an untagged corpus is
# fail-closed until the evidence is tagged. A deployment may override the default.
DEFAULT_UNTAGGED_TAG = AccessTag(classification=Classification.INTERNAL)


def chunk_allowed(
    policy: AccessPolicy,
    principal: Principal,
    metadata: Mapping[str, Any] | None,
    *,
    default_tag: AccessTag = DEFAULT_UNTAGGED_TAG,
) -> bool:
    """Policy Enforcement Point helper over an evidence unit's free-form metadata.

    An absent ACL block uses ``default_tag`` (a deployment choice); a present but
    malformed block fails closed. The tenant boundary is a framework-level pre-policy
    invariant enforced here before any pluggable policy runs, so a tenant-blind custom
    policy can never grant cross-tenant access; the rest of the decision is the
    pluggable policy's.
    """
    if isinstance(metadata, Mapping) and "acl" in metadata:
        parsed = access_tag_from_metadata(metadata)
        if parsed is None:
            return False
        tag = parsed
    else:
        tag = default_tag
    if tag.tenant is not None and tag.tenant != principal.tenant:
        return False
    return policy.decide(principal, tag).allowed


def authorized_document_ids(
    policy: AccessPolicy,
    principal: Principal,
    documents: Iterable[Mapping[str, Any]],
    *,
    default_tag: AccessTag = DEFAULT_UNTAGGED_TAG,
) -> frozenset[str]:
    """The set of document ids the principal may see, for provenance-aware graph authorization.

    A graph fact or boost is visible only if its supporting document is authorized, so the
    caller filters entities and relations to this set before deriving any graph signal, and
    an unauthorized document can never contribute a fact, a boost, or a side channel.
    """
    return frozenset(
        str(document["id"])
        for document in documents
        if "id" in document and chunk_allowed(policy, principal, document.get("metadata"), default_tag=default_tag)
    )


def canonical_acl(metadata: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """A stable, bounded view of an evidence unit's ACL for a permission-history record.

    None when there is no acl block; a malformed sentinel when the block is present but does not parse,
    so a broken ACL is recorded as such rather than stored raw; otherwise the parsed AccessTag as sorted
    scalars, carrying no free-form blob.
    """
    if not isinstance(metadata, Mapping) or "acl" not in metadata:
        return None
    tag = access_tag_from_metadata(metadata)
    if tag is None:
        return {"malformed": True}
    acl: dict[str, Any] = {"classification": tag.classification.name}
    if tag.tenant is not None:
        acl["tenant"] = tag.tenant
    if tag.allow_groups:
        acl["allow_groups"] = sorted(tag.allow_groups)
    if tag.allow_roles:
        acl["allow_roles"] = sorted(tag.allow_roles)
    if tag.deny_groups:
        acl["deny_groups"] = sorted(tag.deny_groups)
    if tag.deny_users:
        acl["deny_users"] = sorted(tag.deny_users)
    if tag.residency_zone is not None:
        acl["residency_zone"] = tag.residency_zone
    return acl


RESERVED_GOVERNANCE_KEYS: frozenset[str] = frozenset(
    {"acl", "tenant", "classification", "residency_zone", "source_trust"}
)


def is_reserved_governance_key(key: object) -> bool:
    """A metadata key only the ingesting caller may set; a plugin extractor never overrides governance."""
    if not isinstance(key, str):
        return False
    folded = key.casefold()
    return folded in RESERVED_GOVERNANCE_KEYS or folded.startswith(("allow_", "deny_"))


def strip_reserved_governance_keys(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Drop caller-owned governance keys from adapter-supplied metadata, so an extractor cannot set an ACL."""
    return {key: value for key, value in metadata.items() if not is_reserved_governance_key(key)}


def permission_fingerprint(metadata: Mapping[str, Any] | None) -> str:
    """A content-independent digest of an evidence unit's ACL, for permission versioning.

    An ACL-only change (identical document bytes and checksum) produces a different
    fingerprint, so ingestion can detect it and re-tag and re-version the document even
    though the content is unchanged. An absent block and a present but malformed block
    digest to distinct sentinels, because an absent block uses the deployment default
    while a malformed block fails closed, so an omitted-to-malformed transition is
    detected as a permission change. The payload is a structured mapping hashed as
    canonical JSON, never a delimiter join, so a group, role, or user name containing a
    comma can never collide two distinct ACLs into one fingerprint and silently drop a
    real permission change.
    """
    tag = access_tag_from_metadata(metadata)
    if tag is None:
        if isinstance(metadata, Mapping) and "acl" in metadata:
            return "malformed"
        return "none"
    payload: dict[str, Any] = {
        "c": int(tag.classification),
        "t": tag.tenant or "",
        "ag": sorted(tag.allow_groups),
        "ar": sorted(tag.allow_roles),
        "dg": sorted(tag.deny_groups),
        "du": sorted(tag.deny_users),
    }
    # The zone key is present only when set, so a zone-free ACL omits it.
    if tag.residency_zone is not None:
        payload["rz"] = tag.residency_zone
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def auth_scope_hash(principal: Principal, policy: AccessPolicy) -> str:
    """Injective digest of the policy-relevant authorization scope for the answer-cache and session
    identity.

    Folds the principal id, kind, tenant, groups, roles, and attributes as a canonical mapping, never
    a delimiter join, so no permission-relevant difference (a group or attribute value containing the
    delimiter, or a kind or attribute a custom policy reads for access) can collide two distinct
    principals into one cache or session scope; a permission change misses old entries. It is the same
    digest as ``durable_entitlement_digest``, which the durable-job re-resolution uses.
    """
    return durable_entitlement_digest(principal, policy)


def durable_entitlement_digest(principal: Principal, policy: AccessPolicy) -> str:
    """Injective digest of a principal's full policy-relevant entitlements for durable re-resolution.

    Unlike auth_scope_hash, this folds the principal kind and attributes as well, so any change an
    execution-time re-resolution must detect flips the digest, closing a custom-policy gap at the
    framework level. The payload is a structured mapping hashed as canonical JSON, never a delimiter
    join, so an attribute value, group, or role containing the delimiter can never collide two
    distinct principals into one digest. ``auth_scope_hash`` delegates to this, so the answer-cache
    and session scope use the same injective digest.
    """
    payload = {
        "id": principal.id,
        "kind": principal.kind.value,
        "tenant": principal.tenant or "",
        "groups": sorted(principal.groups),
        "roles": sorted(principal.roles),
        "attributes": {key: principal.attributes[key] for key in sorted(principal.attributes)},
        "policy": str(getattr(policy, "name", type(policy).__name__)),
        "version": str(getattr(policy, "version", "")),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def session_summary_visible(
    stored_scope: str | None,
    principal: Principal | None,
    policy: AccessPolicy | None,
) -> bool:
    """Fail-closed gate for surfacing a compacted session summary across the principal boundary.

    A compacted summary is prior evidence bound to the principal who wrote it. With no active
    policy the summary path is unchanged and the summary is always visible; a principal-less
    request under a policy is denied earlier by strict authorization. Under an active policy the
    summary is revealed only when its stored writer scope equals this reader's
    scope; an absent or different writer scope is never revealed, so a cross-principal or
    unknown-writer summary stays hidden. All three read sites (query, retrieval, RLM) share this
    one gate so they cannot drift.
    """
    if policy is None or principal is None:
        return True
    return stored_scope == auth_scope_hash(principal, policy)


def preference_scope(policy: AccessPolicy | None, principal: Principal | None, user_id: str | None) -> str:
    """Identity scope for learned preferences: entitlement-bound under an active policy, else user_id.

    Under a policy the scope folds the full authorization digest, not the bare principal id, so two
    principals that share an id across tenants (or differ in entitlements) never share a profile.
    """
    if policy is not None and principal is not None:
        return f"user:{auth_scope_hash(principal, policy)}"
    if user_id:
        return f"user:{user_id}"
    return "anonymous"


__all__ = [
    "DEFAULT_UNTAGGED_TAG",
    "MALFORMED_ZONE",
    "RESERVED_GOVERNANCE_KEYS",
    "UNTAGGED_ZONE",
    "AccessDecision",
    "AccessPolicy",
    "AccessTag",
    "CanonicalAccessPolicy",
    "Classification",
    "EntitlementScopeResolver",
    "HeaderPrincipalResolver",
    "Principal",
    "PrincipalDirectory",
    "PrincipalKind",
    "PrincipalResolver",
    "RoleGroupScopeResolver",
    "SourceTrust",
    "access_tag_from_metadata",
    "auth_scope_hash",
    "authorize",
    "authorized_document_ids",
    "chunk_allowed",
    "durable_entitlement_digest",
    "is_reserved_governance_key",
    "max_classification",
    "permission_fingerprint",
    "preference_scope",
    "principal_from_claims",
    "residency_zones",
    "session_summary_visible",
    "source_trust_from_metadata",
    "strip_reserved_governance_keys",
]
