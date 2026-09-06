"""Tool-governance and approval decision matrix.

These prove the fail-closed rules of the pure decision core: capability admission
(capability_permits) and the human-approval checks (separation of duties, four-eyes, approver
role, expiry, and fingerprint binding). The approval checks are applied by the governed ReAct
loop and the approval service; capability_permits is not yet wired into the loop.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from refaat.domain.authz import Classification, Principal
from refaat.domain.enums import AuditLevel, SideEffectClass
from refaat.domain.governance import (
    ApprovalRecord,
    ToolCapability,
    approval_fingerprint,
    approval_satisfied,
    capability_permits,
)

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
ALICE = Principal(id="alice", tenant="merck", roles=frozenset({"engineer"}))


def _cap(**kwargs: object) -> ToolCapability:
    return ToolCapability(name="deploy", **kwargs)  # type: ignore[arg-type]


def _fingerprint(cap_name: str = "deploy", **overrides: object) -> str:
    base: dict[str, object] = {
        "capability_name": cap_name,
        "arguments": {"target": "prod"},
        "principal": ALICE,
        "policy_id": "canonical",
        "policy_version": "1",
        "action_version": "1",
    }
    base.update(overrides)
    return approval_fingerprint(**base)  # type: ignore[arg-type]


def test_capability_validation() -> None:
    with pytest.raises(ValueError, match="min_approvals"):
        _cap(min_approvals=0)
    with pytest.raises(ValueError, match="approval_ttl_seconds"):
        _cap(approval_ttl_seconds=0)
    with pytest.raises(ValueError, match="approval_ttl_seconds"):
        _cap(approval_ttl_seconds=10**20)


def test_audit_level_defaults_to_standard_and_enhanced_constructs() -> None:
    assert _cap().audit_level == AuditLevel.STANDARD
    assert _cap(audit_level=AuditLevel.ENHANCED).audit_level == AuditLevel.ENHANCED


def test_raw_audit_level_string_raises() -> None:
    # A StrEnum member equals its string value, so a raw "enhanced" would pass a membership test;
    # the isinstance validation rejects it so only a real AuditLevel reaches the deterministic fold.
    with pytest.raises(ValueError, match="audit_level"):
        _cap(audit_level="enhanced")


def test_capability_permits_missing_scope_denies() -> None:
    cap = _cap(required_scopes=frozenset({"deploy:prod"}))
    decision = capability_permits(
        cap,
        granted_scopes=(),
        side_effect_ceiling=SideEffectClass.EXTERNAL,
        evidence_classification=Classification.PUBLIC,
    )
    assert not decision.allowed
    assert decision.reason.startswith("missing_scopes")


def test_capability_permits_side_effect_ceiling() -> None:
    cap = _cap(side_effect=SideEffectClass.EXTERNAL)
    decision = capability_permits(
        cap,
        granted_scopes=(),
        side_effect_ceiling=SideEffectClass.WRITE,
        evidence_classification=Classification.PUBLIC,
    )
    assert not decision.allowed
    assert decision.reason == "side_effect_exceeds_ceiling"


def test_finer_side_effect_taxonomy_is_ordered_and_enforced() -> None:
    order = [
        SideEffectClass.READ_ONLY,
        SideEffectClass.WRITE,
        SideEffectClass.SENSITIVE_WRITE,
        SideEffectClass.IRREVERSIBLE,
        SideEffectClass.EXTERNAL,
    ]
    assert [member.rank for member in order] == [0, 1, 2, 3, 4]
    assert (str(SideEffectClass.READ_ONLY), str(SideEffectClass.WRITE), str(SideEffectClass.EXTERNAL)) == (
        "read_only",
        "write",
        "external",
    )
    for elevated in (SideEffectClass.SENSITIVE_WRITE, SideEffectClass.IRREVERSIBLE):
        denied = capability_permits(
            _cap(side_effect=elevated),
            granted_scopes=(),
            side_effect_ceiling=SideEffectClass.WRITE,
            evidence_classification=Classification.PUBLIC,
        )
        assert not denied.allowed
        assert denied.reason == "side_effect_exceeds_ceiling"
    admitted = capability_permits(
        _cap(side_effect=SideEffectClass.IRREVERSIBLE),
        granted_scopes=(),
        side_effect_ceiling=SideEffectClass.EXTERNAL,
        evidence_classification=Classification.PUBLIC,
    )
    assert admitted.allowed


def test_capability_permits_classification_ceiling() -> None:
    cap = _cap(allowed_classification=Classification.INTERNAL)
    denied = capability_permits(
        cap,
        granted_scopes=(),
        side_effect_ceiling=SideEffectClass.EXTERNAL,
        evidence_classification=Classification.CONFIDENTIAL,
    )
    assert not denied.allowed
    assert denied.reason == "classification_exceeds_capability"
    allowed = capability_permits(
        cap,
        granted_scopes=(),
        side_effect_ceiling=SideEffectClass.EXTERNAL,
        evidence_classification=Classification.INTERNAL,
    )
    assert allowed.allowed


def test_capability_permits_payload_cap() -> None:
    cap = _cap(max_payload_bytes=100)
    assert not capability_permits(
        cap,
        granted_scopes=(),
        side_effect_ceiling=SideEffectClass.EXTERNAL,
        evidence_classification=Classification.PUBLIC,
        payload_bytes=101,
    ).allowed
    assert capability_permits(
        cap,
        granted_scopes=(),
        side_effect_ceiling=SideEffectClass.EXTERNAL,
        evidence_classification=Classification.PUBLIC,
        payload_bytes=100,
    ).allowed
    unknown = capability_permits(
        cap,
        granted_scopes=(),
        side_effect_ceiling=SideEffectClass.EXTERNAL,
        evidence_classification=Classification.PUBLIC,
    )
    assert not unknown.allowed
    assert unknown.reason == "payload_size_unknown"


def test_no_approval_required_is_satisfied() -> None:
    cap = _cap()
    decision = approval_satisfied(cap, fingerprint="x", approvals=(), initiator_id="alice", now=NOW)
    assert decision.satisfied
    assert decision.reason == "no_approval_required"


def test_approval_required_with_no_approvals_fails_closed() -> None:
    cap = _cap(requires_approval=True)
    assert not approval_satisfied(cap, fingerprint="x", approvals=(), initiator_id="alice", now=NOW).satisfied


def test_one_valid_approval_by_another_principal_satisfies() -> None:
    cap = _cap(requires_approval=True)
    fp = _fingerprint()
    approvals = [ApprovalRecord(fingerprint=fp, approver_id="bob")]
    assert approval_satisfied(cap, fingerprint=fp, approvals=approvals, initiator_id="alice", now=NOW).satisfied


def test_separation_of_duties_initiator_cannot_approve_own_action() -> None:
    cap = _cap(requires_approval=True)
    fp = _fingerprint()
    approvals = [ApprovalRecord(fingerprint=fp, approver_id="alice")]
    decision = approval_satisfied(cap, fingerprint=fp, approvals=approvals, initiator_id="alice", now=NOW)
    assert not decision.satisfied
    # A case variant of the initiator's id cannot self-approve either.
    variant = [ApprovalRecord(fingerprint=fp, approver_id="Alice")]
    assert not approval_satisfied(cap, fingerprint=fp, approvals=variant, initiator_id="alice", now=NOW).satisfied


def test_missing_initiator_or_fingerprint_fails_closed() -> None:
    cap = _cap(requires_approval=True)
    fp = _fingerprint()
    approvals = [ApprovalRecord(fingerprint=fp, approver_id="bob")]
    assert not approval_satisfied(cap, fingerprint=fp, approvals=approvals, initiator_id="", now=NOW).satisfied
    assert not approval_satisfied(cap, fingerprint=fp, approvals=approvals, initiator_id="   ", now=NOW).satisfied
    assert not approval_satisfied(cap, fingerprint="", approvals=approvals, initiator_id="alice", now=NOW).satisfied


def test_empty_or_whitespace_approver_never_counts() -> None:
    cap = _cap(requires_approval=True)
    fp = _fingerprint()
    for empty in ("", "   "):
        approvals = [ApprovalRecord(fingerprint=fp, approver_id=empty)]
        assert not approval_satisfied(cap, fingerprint=fp, approvals=approvals, initiator_id="alice", now=NOW).satisfied


def test_identity_normalization_strips_and_folds() -> None:
    cap = _cap(requires_approval=True, min_approvals=2)
    fp = _fingerprint()
    # 'bob' and 'bob ' are one person, so a four-eyes control is not satisfied by a padded variant.
    padded = [ApprovalRecord(fingerprint=fp, approver_id="bob"), ApprovalRecord(fingerprint=fp, approver_id="bob ")]
    assert not approval_satisfied(cap, fingerprint=fp, approvals=padded, initiator_id="alice", now=NOW).satisfied
    # A whitespace variant of the initiator's id cannot self-approve.
    self_approve = [
        ApprovalRecord(fingerprint=fp, approver_id=" Alice "),
        ApprovalRecord(fingerprint=fp, approver_id="bob"),
    ]
    assert not approval_satisfied(cap, fingerprint=fp, approvals=self_approve, initiator_id="alice", now=NOW).satisfied


def test_identity_normalization_folds_compatibility_and_invisible_variants() -> None:
    cap = _cap(requires_approval=True, min_approvals=3)
    fp = _fingerprint()
    fullwidth_bob = "\uff42\uff4f\uff42"
    mono_bob = "\U0001d68b\U0001d698\U0001d68b"
    # Fullwidth and math-monospace spellings of one person do not pad a four-eyes count.
    compat = [
        ApprovalRecord(fingerprint=fp, approver_id="bob"),
        ApprovalRecord(fingerprint=fp, approver_id=fullwidth_bob),
        ApprovalRecord(fingerprint=fp, approver_id=mono_bob),
    ]
    assert not approval_satisfied(cap, fingerprint=fp, approvals=compat, initiator_id="alice", now=NOW).satisfied
    zwsp, zwnj, bom = "\u200b", "\u200c", "\ufeff"
    # Zero-width and format characters create no distinct approver, and an invisible-only id
    # collapses to empty and never counts.
    invisible = [
        ApprovalRecord(fingerprint=fp, approver_id=zwsp),
        ApprovalRecord(fingerprint=fp, approver_id=zwnj),
        ApprovalRecord(fingerprint=fp, approver_id=bom),
    ]
    assert not approval_satisfied(cap, fingerprint=fp, approvals=invisible, initiator_id="alice", now=NOW).satisfied
    single = _cap(requires_approval=True)
    zwsp_self = [ApprovalRecord(fingerprint=fp, approver_id="alice" + zwsp)]
    assert not approval_satisfied(single, fingerprint=fp, approvals=zwsp_self, initiator_id="alice", now=NOW).satisfied


def test_identity_normalization_folds_default_ignorable_variants() -> None:
    cap = _cap(requires_approval=True, min_approvals=2)
    fp = _fingerprint()
    vs, cgj = "\ufe0f", "\u034f"  # variation selector and combining grapheme joiner (category Mn)
    padded = [
        ApprovalRecord(fingerprint=fp, approver_id="bob"),
        ApprovalRecord(fingerprint=fp, approver_id="bob" + vs),
    ]
    assert not approval_satisfied(cap, fingerprint=fp, approvals=padded, initiator_id="alice", now=NOW).satisfied
    single = _cap(requires_approval=True)
    self_approve = [ApprovalRecord(fingerprint=fp, approver_id="bob" + cgj)]
    assert not approval_satisfied(single, fingerprint=fp, approvals=self_approve, initiator_id="bob", now=NOW).satisfied
    # A reserved/unassigned default-ignorable code point (U+2065, category Cn) is covered too.
    reserved = "\u2065"
    reserved_self = [ApprovalRecord(fingerprint=fp, approver_id="bob" + reserved)]
    assert not approval_satisfied(
        single, fingerprint=fp, approvals=reserved_self, initiator_id="bob", now=NOW
    ).satisfied


def test_naive_now_fails_closed_for_an_expiring_capability() -> None:
    cap = _cap(requires_approval=True, approval_ttl_seconds=3600)
    fp = _fingerprint()
    approvals = [ApprovalRecord(fingerprint=fp, approver_id="bob", granted_at=NOW)]
    naive_now = datetime(2026, 1, 1, 12, 0)
    decision = approval_satisfied(cap, fingerprint=fp, approvals=approvals, initiator_id="alice", now=naive_now)
    assert not decision.satisfied
    assert decision.reason == "invalid_now"


def test_naive_timestamp_never_counts() -> None:
    cap = _cap(requires_approval=True, approval_ttl_seconds=3600)
    fp = _fingerprint()
    naive = [ApprovalRecord(fingerprint=fp, approver_id="bob", granted_at=datetime(2026, 1, 1, 11, 30))]
    assert not approval_satisfied(cap, fingerprint=fp, approvals=naive, initiator_id="alice", now=NOW).satisfied


def test_future_dated_approval_never_counts() -> None:
    cap = _cap(requires_approval=True, approval_ttl_seconds=3600)
    fp = _fingerprint()
    future = [ApprovalRecord(fingerprint=fp, approver_id="bob", granted_at=NOW + timedelta(days=3650))]
    assert not approval_satisfied(cap, fingerprint=fp, approvals=future, initiator_id="alice", now=NOW).satisfied


def test_four_eyes_needs_distinct_approvers() -> None:
    cap = _cap(requires_approval=True, min_approvals=2)
    fp = _fingerprint()
    one = [ApprovalRecord(fingerprint=fp, approver_id="bob")]
    assert not approval_satisfied(cap, fingerprint=fp, approvals=one, initiator_id="alice", now=NOW).satisfied
    duplicate = [
        ApprovalRecord(fingerprint=fp, approver_id="bob"),
        ApprovalRecord(fingerprint=fp, approver_id="bob"),
    ]
    assert not approval_satisfied(cap, fingerprint=fp, approvals=duplicate, initiator_id="alice", now=NOW).satisfied
    # Case variants of one approver still count as one person, not two.
    case_variant = [
        ApprovalRecord(fingerprint=fp, approver_id="bob"),
        ApprovalRecord(fingerprint=fp, approver_id="Bob"),
    ]
    assert not approval_satisfied(cap, fingerprint=fp, approvals=case_variant, initiator_id="alice", now=NOW).satisfied
    two = [
        ApprovalRecord(fingerprint=fp, approver_id="bob"),
        ApprovalRecord(fingerprint=fp, approver_id="carol"),
    ]
    assert approval_satisfied(cap, fingerprint=fp, approvals=two, initiator_id="alice", now=NOW).satisfied


def test_required_approver_role() -> None:
    cap = _cap(requires_approval=True, approver_role="release-manager")
    fp = _fingerprint()
    without_role = [ApprovalRecord(fingerprint=fp, approver_id="bob")]
    assert not approval_satisfied(cap, fingerprint=fp, approvals=without_role, initiator_id="alice", now=NOW).satisfied
    with_role = [
        ApprovalRecord(fingerprint=fp, approver_id="bob", approver_roles=frozenset({"release-manager"})),
    ]
    assert approval_satisfied(cap, fingerprint=fp, approvals=with_role, initiator_id="alice", now=NOW).satisfied


def test_approval_expiry() -> None:
    cap = _cap(requires_approval=True, approval_ttl_seconds=3600)
    fp = _fingerprint()
    fresh = [ApprovalRecord(fingerprint=fp, approver_id="bob", granted_at=NOW - timedelta(seconds=60))]
    assert approval_satisfied(cap, fingerprint=fp, approvals=fresh, initiator_id="alice", now=NOW).satisfied
    expired = [ApprovalRecord(fingerprint=fp, approver_id="bob", granted_at=NOW - timedelta(seconds=7200))]
    assert not approval_satisfied(cap, fingerprint=fp, approvals=expired, initiator_id="alice", now=NOW).satisfied
    no_timestamp = [ApprovalRecord(fingerprint=fp, approver_id="bob")]
    assert not approval_satisfied(cap, fingerprint=fp, approvals=no_timestamp, initiator_id="alice", now=NOW).satisfied


def test_fingerprint_mismatch_never_counts() -> None:
    cap = _cap(requires_approval=True)
    fp = _fingerprint()
    stale = [ApprovalRecord(fingerprint=_fingerprint(arguments={"target": "staging"}), approver_id="bob")]
    assert not approval_satisfied(cap, fingerprint=fp, approvals=stale, initiator_id="alice", now=NOW).satisfied


def test_fingerprint_changes_with_binding_inputs() -> None:
    from refaat.domain.authz import PrincipalKind

    base = _fingerprint()
    assert base != _fingerprint(arguments={"target": "staging"})
    assert base != _fingerprint(policy_version="2")
    assert base != _fingerprint(action_version="2")
    assert base != _fingerprint(principal=Principal(id="mallory"))
    # A service principal with the same id and tenant does not share a human's approval.
    assert base != _fingerprint(principal=Principal(id="alice", tenant="merck", kind=PrincipalKind.SERVICE))
    assert base == _fingerprint()


BREAK_GLASS_ROLE = "incident-commander"


def _bg_cap(**kwargs: object) -> ToolCapability:
    base: dict[str, object] = {
        "requires_approval": True,
        "break_glass_enabled": True,
        "break_glass_approver_role": BREAK_GLASS_ROLE,
    }
    base.update(kwargs)
    return _cap(**base)


def _bg_record(
    approver_id: str,
    *,
    fp: str,
    granted_at: datetime | None = NOW,
    roles: frozenset[str] = frozenset({BREAK_GLASS_ROLE}),
) -> ApprovalRecord:
    return ApprovalRecord(
        fingerprint=fp,
        approver_id=approver_id,
        approver_roles=roles,
        granted_at=granted_at,
        kind="break_glass",
    )


def test_break_glass_capability_validation() -> None:
    with pytest.raises(ValueError, match="break_glass_approver_role"):
        _cap(break_glass_enabled=True)
    with pytest.raises(ValueError, match="break_glass_approver_role"):
        _cap(break_glass_enabled=True, break_glass_approver_role="")
    with pytest.raises(ValueError, match="break_glass_min_approvals"):
        _cap(break_glass_enabled=True, break_glass_approver_role=BREAK_GLASS_ROLE, break_glass_min_approvals=1)
    with pytest.raises(ValueError, match="break_glass_ttl_seconds"):
        _cap(break_glass_enabled=True, break_glass_approver_role=BREAK_GLASS_ROLE, break_glass_ttl_seconds=0)
    # A disabled break-glass group is inert and never validated.
    _cap(break_glass_min_approvals=1)


def test_break_glass_single_approval_fails_dual_floor() -> None:
    cap = _bg_cap()
    fp = _fingerprint()
    decision = approval_satisfied(
        cap, fingerprint=fp, approvals=[_bg_record("bob", fp=fp)], initiator_id="alice", now=NOW
    )
    assert not decision.satisfied
    assert not decision.break_glass


def test_two_break_glass_approvers_within_ttl_satisfy() -> None:
    cap = _bg_cap(break_glass_ttl_seconds=3600)
    fp = _fingerprint()
    approvals = [
        _bg_record("bob", fp=fp, granted_at=NOW - timedelta(seconds=60)),
        _bg_record("carol", fp=fp, granted_at=NOW - timedelta(seconds=60)),
    ]
    decision = approval_satisfied(cap, fingerprint=fp, approvals=approvals, initiator_id="alice", now=NOW)
    assert decision.satisfied
    assert decision.reason == "break_glass_approved"
    assert decision.break_glass
    # An approver lacking the break-glass role does not count, so the dual floor is not met.
    without_role = [
        _bg_record("bob", fp=fp, granted_at=NOW - timedelta(seconds=60)),
        _bg_record("carol", fp=fp, granted_at=NOW - timedelta(seconds=60), roles=frozenset({"engineer"})),
    ]
    assert not approval_satisfied(cap, fingerprint=fp, approvals=without_role, initiator_id="alice", now=NOW).satisfied


def test_break_glass_initiator_never_counts() -> None:
    cap = _bg_cap()
    fp = _fingerprint()
    approvals = [_bg_record("alice", fp=fp), _bg_record("bob", fp=fp)]
    assert not approval_satisfied(cap, fingerprint=fp, approvals=approvals, initiator_id="alice", now=NOW).satisfied


def test_expired_break_glass_approval_never_counts() -> None:
    cap = _bg_cap(break_glass_ttl_seconds=3600)
    fp = _fingerprint()
    approvals = [
        _bg_record("bob", fp=fp, granted_at=NOW - timedelta(seconds=60)),
        _bg_record("carol", fp=fp, granted_at=NOW - timedelta(seconds=7200)),
    ]
    assert not approval_satisfied(cap, fingerprint=fp, approvals=approvals, initiator_id="alice", now=NOW).satisfied


def test_record_kind_partitions_the_two_paths() -> None:
    fp = _fingerprint()
    # A normal-kind record never satisfies the break-glass path: the normal path here fails because
    # its required role is absent, and the break-glass path must not borrow the normal-kind records.
    bg_cap = _bg_cap(approver_role="release-manager")
    normal_records = [
        ApprovalRecord(fingerprint=fp, approver_id="bob", approver_roles=frozenset({BREAK_GLASS_ROLE})),
        ApprovalRecord(fingerprint=fp, approver_id="carol", approver_roles=frozenset({BREAK_GLASS_ROLE})),
    ]
    assert not approval_satisfied(
        bg_cap, fingerprint=fp, approvals=normal_records, initiator_id="alice", now=NOW
    ).satisfied
    # A break-glass-kind record never satisfies a normal (break-glass-disabled) capability's path.
    normal_cap = _cap(requires_approval=True)
    bg_records = [ApprovalRecord(fingerprint=fp, approver_id="bob", kind="break_glass")]
    assert not approval_satisfied(
        normal_cap, fingerprint=fp, approvals=bg_records, initiator_id="alice", now=NOW
    ).satisfied


def test_break_glass_disabled_leaves_the_branch_inert() -> None:
    cap = _cap(requires_approval=True)
    fp = _fingerprint()
    ok = approval_satisfied(
        cap,
        fingerprint=fp,
        approvals=[ApprovalRecord(fingerprint=fp, approver_id="bob")],
        initiator_id="alice",
        now=NOW,
    )
    assert ok.satisfied
    assert ok.reason == "approved"
    assert not ok.break_glass
    inert = [
        ApprovalRecord(fingerprint=fp, approver_id="carol", kind="break_glass"),
        ApprovalRecord(fingerprint=fp, approver_id="dave", kind="break_glass"),
    ]
    assert not approval_satisfied(cap, fingerprint=fp, approvals=inert, initiator_id="alice", now=NOW).satisfied


BASE_ROLE = "release-manager"
ESCALATION_ROLE = "senior-release-manager"
ANCHOR = NOW
BEFORE_DELAY = NOW + timedelta(seconds=60)
AFTER_DELAY = NOW + timedelta(seconds=7200)


def _esc_cap(**kwargs: object) -> ToolCapability:
    base: dict[str, object] = {
        "requires_approval": True,
        "approver_role": BASE_ROLE,
        "escalation_role": ESCALATION_ROLE,
        "escalation_after_seconds": 3600,
    }
    base.update(kwargs)
    return _cap(**base)


def _role_record(approver_id: str, *, fp: str, role: str) -> ApprovalRecord:
    return ApprovalRecord(fingerprint=fp, approver_id=approver_id, approver_roles=frozenset({role}))


def test_escalation_field_validation() -> None:
    with pytest.raises(ValueError, match="set together"):
        _cap(approver_role=BASE_ROLE, escalation_role=ESCALATION_ROLE)
    with pytest.raises(ValueError, match="set together"):
        _cap(approver_role=BASE_ROLE, escalation_after_seconds=3600)
    with pytest.raises(ValueError, match="escalation_role must be a non-empty string"):
        _cap(approver_role=BASE_ROLE, escalation_role="", escalation_after_seconds=3600)
    with pytest.raises(ValueError, match="escalation_after_seconds"):
        _cap(approver_role=BASE_ROLE, escalation_role=ESCALATION_ROLE, escalation_after_seconds=0)
    with pytest.raises(ValueError, match="escalation_after_seconds"):
        _cap(approver_role=BASE_ROLE, escalation_role=ESCALATION_ROLE, escalation_after_seconds=10**20)
    with pytest.raises(ValueError, match="approver_role to widen"):
        _cap(escalation_role=ESCALATION_ROLE, escalation_after_seconds=3600)
    with pytest.raises(ValueError, match="must differ from approver_role"):
        _cap(approver_role=BASE_ROLE, escalation_role=BASE_ROLE, escalation_after_seconds=3600)


def test_before_delay_only_base_role_counts() -> None:
    cap = _esc_cap()
    fp = _fingerprint()
    base = [_role_record("bob", fp=fp, role=BASE_ROLE)]
    assert approval_satisfied(
        cap, fingerprint=fp, approvals=base, initiator_id="alice", now=BEFORE_DELAY, request_anchor=ANCHOR
    ).satisfied
    escalation_only = [_role_record("bob", fp=fp, role=ESCALATION_ROLE)]
    assert not approval_satisfied(
        cap, fingerprint=fp, approvals=escalation_only, initiator_id="alice", now=BEFORE_DELAY, request_anchor=ANCHOR
    ).satisfied


def test_after_delay_escalation_role_counts() -> None:
    cap = _esc_cap()
    fp = _fingerprint()
    escalation_only = [_role_record("bob", fp=fp, role=ESCALATION_ROLE)]
    assert approval_satisfied(
        cap, fingerprint=fp, approvals=escalation_only, initiator_id="alice", now=AFTER_DELAY, request_anchor=ANCHOR
    ).satisfied
    # The base role still counts after the delay: escalation only widens, never narrows.
    base = [_role_record("bob", fp=fp, role=BASE_ROLE)]
    assert approval_satisfied(
        cap, fingerprint=fp, approvals=base, initiator_id="alice", now=AFTER_DELAY, request_anchor=ANCHOR
    ).satisfied


def test_escalation_never_lowers_min_approvals() -> None:
    cap = _esc_cap(min_approvals=2)
    fp = _fingerprint()
    one = [_role_record("bob", fp=fp, role=ESCALATION_ROLE)]
    assert not approval_satisfied(
        cap, fingerprint=fp, approvals=one, initiator_id="alice", now=AFTER_DELAY, request_anchor=ANCHOR
    ).satisfied
    two = [
        _role_record("bob", fp=fp, role=BASE_ROLE),
        _role_record("carol", fp=fp, role=ESCALATION_ROLE),
    ]
    assert approval_satisfied(
        cap, fingerprint=fp, approvals=two, initiator_id="alice", now=AFTER_DELAY, request_anchor=ANCHOR
    ).satisfied


def test_separation_of_duties_holds_under_escalation() -> None:
    cap = _esc_cap()
    fp = _fingerprint()
    # After the delay the initiator holds the escalation role, but still never counts.
    self_escalation = [_role_record("alice", fp=fp, role=ESCALATION_ROLE)]
    assert not approval_satisfied(
        cap, fingerprint=fp, approvals=self_escalation, initiator_id="alice", now=AFTER_DELAY, request_anchor=ANCHOR
    ).satisfied


def test_naive_or_missing_anchor_disables_escalation() -> None:
    cap = _esc_cap()
    fp = _fingerprint()
    escalation_only = [_role_record("bob", fp=fp, role=ESCALATION_ROLE)]
    base = [_role_record("bob", fp=fp, role=BASE_ROLE)]
    naive_anchor = datetime(2026, 1, 1, 12, 0)
    naive_now = datetime(2026, 1, 1, 14, 0)
    # A naive anchor disables escalation, so an escalation-only approver never counts.
    assert not approval_satisfied(
        cap,
        fingerprint=fp,
        approvals=escalation_only,
        initiator_id="alice",
        now=AFTER_DELAY,
        request_anchor=naive_anchor,
    ).satisfied
    # A missing anchor does the same.
    assert not approval_satisfied(
        cap, fingerprint=fp, approvals=escalation_only, initiator_id="alice", now=AFTER_DELAY, request_anchor=None
    ).satisfied
    # A naive now disables escalation (the capability declares no ttl, so this is reachable).
    assert not approval_satisfied(
        cap, fingerprint=fp, approvals=escalation_only, initiator_id="alice", now=naive_now, request_anchor=ANCHOR
    ).satisfied
    # The un-escalated normal path is unchanged: the base role still counts with no anchor.
    assert approval_satisfied(
        cap, fingerprint=fp, approvals=base, initiator_id="alice", now=AFTER_DELAY, request_anchor=None
    ).satisfied


def test_capability_extra_is_byte_identical_without_escalation() -> None:
    from refaat.programs.react import GovernedReActProgram

    plain = _cap(required_scopes=frozenset({"deploy:prod"}), requires_approval=True)
    extra = GovernedReActProgram._capability_extra(plain)
    assert "escalation_role" not in extra
    assert "escalation_after_seconds" not in extra
    escalated = _cap(
        required_scopes=frozenset({"deploy:prod"}),
        requires_approval=True,
        approver_role=BASE_ROLE,
        escalation_role=ESCALATION_ROLE,
        escalation_after_seconds=3600,
    )
    esc_extra = GovernedReActProgram._capability_extra(escalated)
    assert esc_extra["escalation_role"] == ESCALATION_ROLE
    assert esc_extra["escalation_after_seconds"] == 3600


def test_capability_extra_is_byte_identical_without_break_glass() -> None:
    from refaat.programs.react import GovernedReActProgram

    plain = _cap(required_scopes=frozenset({"deploy:prod"}), requires_approval=True)
    expected = {
        "name": "deploy",
        "scopes": ["deploy:prod"],
        "side_effect": "read_only",
        "classification": int(Classification.PUBLIC),
        "max_payload_bytes": None,
        "requires_approval": True,
        "min_approvals": 1,
        "approver_role": None,
        "approval_ttl_seconds": None,
    }
    assert GovernedReActProgram._capability_extra(plain) == expected
    # A disabled break-glass group with non-default values still yields the exact same dict, so a
    # capability that never enables break-glass keeps every existing program artifact id unchanged.
    inert = _cap(required_scopes=frozenset({"deploy:prod"}), requires_approval=True, break_glass_min_approvals=5)
    assert GovernedReActProgram._capability_extra(inert) == expected
    # Enabling break-glass folds the diverging fields, so the artifact contribution changes.
    enabled = _cap(
        required_scopes=frozenset({"deploy:prod"}),
        requires_approval=True,
        break_glass_enabled=True,
        break_glass_approver_role=BREAK_GLASS_ROLE,
        break_glass_ttl_seconds=3600,
    )
    extra = GovernedReActProgram._capability_extra(enabled)
    assert extra["break_glass_enabled"] is True
    assert extra["break_glass_approver_role"] == BREAK_GLASS_ROLE
    assert extra["break_glass_ttl_seconds"] == 3600
    assert "break_glass_min_approvals" not in extra
    assert extra != expected


NETWORK_DEST = "api.acme.com:443"
NAMESPACE = "corpus:acme"
SECRET = "acme/deploy-key"


def _permits(cap: ToolCapability, **kwargs: object) -> object:
    return capability_permits(
        cap,
        granted_scopes=(),
        side_effect_ceiling=SideEffectClass.EXTERNAL,
        evidence_classification=Classification.PUBLIC,
        **kwargs,  # type: ignore[arg-type]
    )


def test_declared_governance_field_validation() -> None:
    with pytest.raises(ValueError, match="allowed_network"):
        _cap(allowed_network=frozenset({""}))
    with pytest.raises(ValueError, match="allowed_network"):
        _cap(allowed_network=frozenset({"has space"}))
    with pytest.raises(ValueError, match="allowed_network"):
        _cap(allowed_network=frozenset(str(i) for i in range(65)))
    with pytest.raises(ValueError, match="allowed_network"):
        _cap(allowed_network=frozenset({"x" * 257}))
    with pytest.raises(ValueError, match="resource_namespace"):
        _cap(resource_namespace="")
    with pytest.raises(ValueError, match="resource_namespace"):
        _cap(resource_namespace="has space")
    with pytest.raises(ValueError, match="resource_namespace"):
        _cap(resource_namespace="n" * 513)
    with pytest.raises(ValueError, match="required_secrets"):
        _cap(required_secrets=frozenset({""}))
    with pytest.raises(ValueError, match="required_secrets"):
        _cap(required_secrets=frozenset({"has space"}))
    with pytest.raises(ValueError, match="required_secrets"):
        _cap(required_secrets=frozenset(str(i) for i in range(65)))
    cap = _cap(
        allowed_network=frozenset({NETWORK_DEST}),
        resource_namespace=NAMESPACE,
        required_secrets=frozenset({SECRET}),
    )
    assert cap.allowed_network == frozenset({NETWORK_DEST})
    assert cap.resource_namespace == NAMESPACE
    assert cap.required_secrets == frozenset({SECRET})


def test_declared_governance_fields_default_to_unset() -> None:
    cap = _cap()
    assert cap.allowed_network == frozenset()
    assert cap.resource_namespace is None
    assert cap.required_secrets == frozenset()


def test_permits_network_admission() -> None:
    cap = _cap(allowed_network=frozenset({NETWORK_DEST}))
    denied = _permits(cap, requested_destinations={"evil.example:443"})
    assert not denied.allowed  # type: ignore[attr-defined]
    assert denied.reason.startswith("network_destination_denied")  # type: ignore[attr-defined]
    assert _permits(cap, requested_destinations={NETWORK_DEST}).allowed  # type: ignore[attr-defined]
    # A declared allowlist with no requested destinations is declared-only, so it is not checked.
    assert _permits(cap, requested_destinations=None).allowed  # type: ignore[attr-defined]


def test_permits_namespace_admission() -> None:
    cap = _cap(resource_namespace=NAMESPACE)
    denied = _permits(cap, requested_namespace="corpus:other")
    assert not denied.allowed  # type: ignore[attr-defined]
    assert denied.reason == "resource_namespace_denied"  # type: ignore[attr-defined]
    assert _permits(cap, requested_namespace=NAMESPACE).allowed  # type: ignore[attr-defined]
    assert _permits(cap, requested_namespace=None).allowed  # type: ignore[attr-defined]


def test_permits_secret_availability() -> None:
    cap = _cap(required_secrets=frozenset({SECRET}))
    denied = _permits(cap, available_secrets=frozenset())
    assert not denied.allowed  # type: ignore[attr-defined]
    assert denied.reason == "required_secret_unavailable:" + SECRET  # type: ignore[attr-defined]
    assert _permits(cap, available_secrets=frozenset({SECRET})).allowed  # type: ignore[attr-defined]
    assert _permits(cap, available_secrets=None).allowed  # type: ignore[attr-defined]


def test_permits_without_new_kwargs_matches_prior_decision() -> None:
    # A capability declaring all three governance fields but passing none of the new kwargs still
    # permits, so an existing capability_permits call is byte-identical to before.
    cap = _cap(
        allowed_network=frozenset({NETWORK_DEST}),
        resource_namespace=NAMESPACE,
        required_secrets=frozenset({SECRET}),
    )
    decision = _permits(cap)
    assert decision.allowed  # type: ignore[attr-defined]
    assert decision.reason == "permitted"  # type: ignore[attr-defined]
