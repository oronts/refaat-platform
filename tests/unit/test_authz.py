"""Canonical authorization decision matrix.

These prove the fail-closed rules of the pure decision function. The enforcement-path
isolation suite lives in tests/integration/test_authz_enforcement_e2e.py.
"""

from __future__ import annotations

from refaat.domain.authz import (
    AccessDecision,
    AccessTag,
    CanonicalAccessPolicy,
    Classification,
    Principal,
    PrincipalKind,
    SourceTrust,
    access_tag_from_metadata,
    auth_scope_hash,
    authorize,
    chunk_allowed,
    durable_entitlement_digest,
    is_reserved_governance_key,
    max_classification,
    permission_fingerprint,
    strip_reserved_governance_keys,
    summary_documents_authorized,
)

ALICE = Principal(id="alice", tenant="merck", groups=frozenset({"research-a"}), roles=frozenset({"scientist"}))
POLICY = CanonicalAccessPolicy()


def test_principal_defaults_to_a_human_with_no_entitlements() -> None:
    bare = Principal(id="p")
    assert bare.kind is PrincipalKind.HUMAN
    assert bare.groups == frozenset()
    assert bare.attributes == {}


def test_service_and_system_principals_are_first_class() -> None:
    svc = Principal(id="ingest-worker", kind=PrincipalKind.SERVICE, tenant="merck")
    assert svc.kind is PrincipalKind.SERVICE
    assert Principal(id="scheduler", kind=PrincipalKind.SYSTEM).kind is PrincipalKind.SYSTEM


def test_source_trust_is_separate_from_classification() -> None:
    assert SourceTrust.AUTHORITATIVE > SourceTrust.UNVERIFIED
    assert Classification.RESTRICTED > Classification.PUBLIC


def test_tenant_mismatch_denies() -> None:
    assert authorize(ALICE, AccessTag(classification=Classification.PUBLIC, tenant="other")) is False


def test_tenant_none_on_tag_is_not_tenant_gated() -> None:
    assert authorize(ALICE, AccessTag(classification=Classification.PUBLIC)) is True


def test_explicit_deny_user_beats_any_allow() -> None:
    tag = AccessTag(allow_groups=frozenset({"research-a"}), deny_users=frozenset({"alice"}))
    assert authorize(ALICE, tag) is False


def test_explicit_deny_group_beats_any_allow() -> None:
    tag = AccessTag(allow_roles=frozenset({"scientist"}), deny_groups=frozenset({"research-a"}))
    assert authorize(ALICE, tag) is False


def test_allow_group_match_allows() -> None:
    assert authorize(ALICE, AccessTag(allow_groups=frozenset({"research-a"}))) is True


def test_allow_role_match_allows() -> None:
    assert authorize(ALICE, AccessTag(allow_roles=frozenset({"scientist"}))) is True


def test_allow_list_present_without_match_denies() -> None:
    assert authorize(ALICE, AccessTag(allow_groups=frozenset({"research-b"}))) is False


def test_unknown_group_does_not_broaden_access() -> None:
    tag = AccessTag(classification=Classification.CONFIDENTIAL, allow_groups=frozenset({"research-b"}))
    assert authorize(ALICE, tag) is False


def test_no_allow_list_public_is_visible() -> None:
    assert authorize(ALICE, AccessTag(classification=Classification.PUBLIC)) is True


def test_no_allow_list_internal_fails_closed_at_the_canonical_policy() -> None:
    # INTERNAL organizational meaning belongs to the customer policy, so the shipped
    # canonical policy denies rather than inventing a tenant-wide grant.
    assert authorize(ALICE, AccessTag(classification=Classification.INTERNAL)) is False


def test_no_allow_list_confidential_and_restricted_fail_closed() -> None:
    assert authorize(ALICE, AccessTag(classification=Classification.CONFIDENTIAL)) is False
    assert authorize(ALICE, AccessTag(classification=Classification.RESTRICTED)) is False


def test_canonical_policy_returns_a_reasoned_decision() -> None:
    policy = CanonicalAccessPolicy()
    decision = policy.decide(ALICE, AccessTag(allow_groups=frozenset({"research-a"})))
    assert isinstance(decision, AccessDecision)
    assert decision.allowed is True
    assert decision.reason == "allow_group"
    assert decision.policy_id == "canonical"
    assert decision.policy_version == "1"


def test_decision_reasons_are_machine_readable() -> None:
    policy = CanonicalAccessPolicy()
    assert policy.decide(ALICE, AccessTag(tenant="other")).reason == "tenant_mismatch"
    assert policy.decide(ALICE, AccessTag(deny_users=frozenset({"alice"}))).reason == "explicit_deny_user"
    assert policy.decide(ALICE, AccessTag(classification=Classification.PUBLIC)).reason == "public_open"
    assert (
        policy.decide(ALICE, AccessTag(classification=Classification.INTERNAL)).reason
        == "internal_no_allow_policy_decides"
    )
    assert policy.decide(ALICE, AccessTag(allow_groups=frozenset({"research-b"}))).reason == "no_matching_allow"


def test_metadata_without_acl_yields_none() -> None:
    assert access_tag_from_metadata(None) is None
    assert access_tag_from_metadata({"language": "en"}) is None


def test_malformed_acl_block_fails_to_none() -> None:
    assert access_tag_from_metadata({"acl": "public"}) is None
    assert access_tag_from_metadata({"acl": {"classification": "SECRET"}}) is None
    assert access_tag_from_metadata({"acl": {"allow_groups": "research-a"}}) is None


def test_null_acl_forms_fail_closed() -> None:
    public_default = AccessTag(classification=Classification.PUBLIC)
    for metadata in ({"acl": None}, {"acl": {"classification": None}}):
        assert access_tag_from_metadata(metadata) is None
        assert max_classification([metadata]) is Classification.RESTRICTED
        # A present but malformed ACL is denied even under a permissive deployment default.
        assert chunk_allowed(POLICY, ALICE, metadata, default_tag=public_default) is False
    # An absent ACL still uses the deployment default, so the two are distinguished.
    assert chunk_allowed(POLICY, ALICE, {"language": "en"}, default_tag=public_default) is True
    assert permission_fingerprint({"acl": None}) == "malformed"
    assert permission_fingerprint({"acl": None}) != permission_fingerprint({})


def test_permission_fingerprint_is_injective_over_multi_valued_members() -> None:
    # A member name containing a comma must not collide with two split members, so a real
    # permission change is never silently dropped by the ingest fast path or the hold ledger.
    one = {"acl": {"classification": "INTERNAL", "allow_groups": ["eng,ops"]}}
    two = {"acl": {"classification": "INTERNAL", "allow_groups": ["eng", "ops"]}}
    assert permission_fingerprint(one) != permission_fingerprint(two)
    # Reordering the same members is not a permission change.
    reordered = {"acl": {"classification": "INTERNAL", "allow_groups": ["b", "a"]}}
    ordered = {"acl": {"classification": "INTERNAL", "allow_groups": ["a", "b"]}}
    assert permission_fingerprint(reordered) == permission_fingerprint(ordered)


def test_reserved_governance_keys_cover_every_acl_axis_and_nothing_else() -> None:
    for key in ("acl", "ACL", "tenant", "classification", "residency_zone", "source_trust"):
        assert is_reserved_governance_key(key)
    for prefix in ("allow_groups", "allow_roles", "deny_groups", "deny_users", "Deny_Users"):
        assert is_reserved_governance_key(prefix)
    for benign in ("title", "language", "source_id", "heading", 3, None):
        assert not is_reserved_governance_key(benign)


def test_strip_reserved_governance_keys_removes_only_governance_keys() -> None:
    hostile = {
        "acl": {"classification": "public", "allow_groups": ["everyone"]},
        "tenant": "attacker",
        "classification": "public",
        "allow_groups": ["everyone"],
        "deny_users": ["victim"],
        "residency_zone": "us",
        "source_trust": "authoritative",
        "note": "kept",
        "page": 4,
    }
    assert strip_reserved_governance_keys(hostile) == {"note": "kept", "page": 4}


def test_summary_documents_authorized_pure() -> None:
    allowed = {"id": "doc_a", "metadata": {"acl": {"classification": "confidential", "allow_groups": ["research-a"]}}}
    denied = {"id": "doc_b", "metadata": {"acl": {"classification": "confidential", "allow_groups": ["research-b"]}}}
    docs = [allowed, denied]
    # No policy, no principal, or unrecorded provenance is a no-op that allows (deferred to egress).
    assert summary_documents_authorized(None, ALICE, frozenset({"doc_a"}), docs) is True
    assert summary_documents_authorized(POLICY, None, frozenset({"doc_a"}), docs) is True
    assert summary_documents_authorized(POLICY, ALICE, None, docs) is True
    # A missing cited doc is left to the egress ceiling; only a resolved-but-forbidden doc suppresses.
    assert summary_documents_authorized(POLICY, ALICE, frozenset({"doc_missing"}), docs) is True
    assert summary_documents_authorized(POLICY, ALICE, frozenset({"doc_b"}), docs) is False
    # Every resolved cited doc allowed, or nothing cited, passes.
    assert summary_documents_authorized(POLICY, ALICE, frozenset({"doc_a"}), docs) is True
    assert summary_documents_authorized(POLICY, ALICE, frozenset(), docs) is True


def test_max_classification_over_well_formed_tags() -> None:
    assert max_classification([{"acl": {"allow_groups": ["research-a"]}}]) is Classification.PUBLIC
    assert max_classification([{"language": "en"}]) is Classification.PUBLIC
    assert max_classification([{"acl": {"classification": "restricted"}}]) is Classification.RESTRICTED


def test_valid_acl_block_parses() -> None:
    tag = access_tag_from_metadata(
        {
            "acl": {
                "classification": "confidential",
                "tenant": "merck",
                "allow_groups": ["research-a", "research-b"],
                "deny_users": ["mallory"],
            }
        }
    )
    assert tag is not None
    assert tag.classification is Classification.CONFIDENTIAL
    assert tag.tenant == "merck"
    assert tag.allow_groups == frozenset({"research-a", "research-b"})
    assert tag.deny_users == frozenset({"mallory"})


def test_absent_classification_defaults_public() -> None:
    tag = access_tag_from_metadata({"acl": {"allow_groups": ["research-a"]}})
    assert tag is not None
    assert tag.classification is Classification.PUBLIC


def test_parsed_restricted_denies_without_matching_principal() -> None:
    tag = access_tag_from_metadata({"acl": {"classification": "restricted", "allow_groups": ["research-b"]}})
    assert tag is not None
    assert authorize(ALICE, tag) is False


class _VersionedPolicy:
    def __init__(self, version: str) -> None:
        self.name = "canonical"
        self.version = version

    def decide(self, principal: Principal, tag: AccessTag) -> AccessDecision:  # pragma: no cover - unused here
        raise NotImplementedError


def test_durable_entitlement_digest_is_stable_for_an_unchanged_principal() -> None:
    assert durable_entitlement_digest(ALICE, POLICY) == durable_entitlement_digest(ALICE, POLICY)


def test_durable_entitlement_digest_changes_when_the_kind_changes() -> None:
    service = Principal(
        id="alice",
        kind=PrincipalKind.SERVICE,
        tenant="merck",
        groups=frozenset({"research-a"}),
        roles=frozenset({"scientist"}),
    )
    assert durable_entitlement_digest(service, POLICY) != durable_entitlement_digest(ALICE, POLICY)


def test_durable_entitlement_digest_changes_when_an_attribute_changes() -> None:
    with_attr = Principal(id="alice", tenant="merck", attributes={"clearance": "high"})
    other_attr = Principal(id="alice", tenant="merck", attributes={"clearance": "low"})
    assert durable_entitlement_digest(with_attr, POLICY) != durable_entitlement_digest(other_attr, POLICY)


def test_durable_entitlement_digest_changes_on_group_role_and_tenant_changes() -> None:
    base = durable_entitlement_digest(ALICE, POLICY)
    grouped = Principal(id="alice", tenant="merck", groups=frozenset({"research-b"}), roles=frozenset({"scientist"}))
    roled = Principal(id="alice", tenant="merck", groups=frozenset({"research-a"}), roles=frozenset({"lead"}))
    moved = Principal(id="alice", tenant="pfizer", groups=frozenset({"research-a"}), roles=frozenset({"scientist"}))
    assert durable_entitlement_digest(grouped, POLICY) != base
    assert durable_entitlement_digest(roled, POLICY) != base
    assert durable_entitlement_digest(moved, POLICY) != base


def test_durable_entitlement_digest_changes_when_the_policy_version_changes() -> None:
    assert durable_entitlement_digest(ALICE, _VersionedPolicy("1")) != durable_entitlement_digest(
        ALICE, _VersionedPolicy("2")
    )


def test_auth_scope_hash_is_injective_and_matches_the_durable_digest() -> None:
    # The answer-cache and session scope use the same injective digest as durable re-resolution, so a
    # kind or attribute distinction a custom policy can read for access flips the scope and can never
    # collide two distinct principals into one cache or session scope.
    service = Principal(id="alice", kind=PrincipalKind.SERVICE, tenant="merck", groups=frozenset({"research-a"}))
    human = Principal(id="alice", kind=PrincipalKind.HUMAN, tenant="merck", groups=frozenset({"research-a"}))
    assert auth_scope_hash(service, POLICY) != auth_scope_hash(human, POLICY)
    assert auth_scope_hash(service, POLICY) == durable_entitlement_digest(service, POLICY)
    attr_a = Principal(id="alice", tenant="merck", attributes={"clearance": "high"})
    attr_b = Principal(id="alice", tenant="merck", attributes={"clearance": "low"})
    assert auth_scope_hash(attr_a, POLICY) != auth_scope_hash(attr_b, POLICY)


def test_durable_entitlement_digest_is_injective_across_a_delimiter_collision() -> None:
    # A delimiter join would collide these two distinct attribute maps; canonical-JSON hashing keeps
    # them distinct, so a scope_changed re-resolution can never pass as fresh through a collision.
    collide_a = Principal(id="alice", tenant="merck", attributes={"a": "b,c=d"})
    collide_b = Principal(id="alice", tenant="merck", attributes={"a": "b", "c": "d"})
    assert durable_entitlement_digest(collide_a, POLICY) != durable_entitlement_digest(collide_b, POLICY)
