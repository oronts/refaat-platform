"""Governance of the direct skill-run path through the single SkillService.run chokepoint.

The four direct surfaces (HTTP run_skill, sdk.run_skill, project.run_skill, and the MCP bridge)
funnel through SkillService.run. A DirectSkillPolicy governs that chokepoint: capability admission
for the four measurable dimensions and an outright refusal of a capability that requires approval.
Every denial here is paired with a passing control so the test cannot pass vacuously; a spy skill
records whether its run body executed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from refaat.application.runtime import RefaatProject, RuntimeBuilder
from refaat.config import Settings
from refaat.domain.authz import CanonicalAccessPolicy, Classification, Principal
from refaat.domain.enums import SideEffectClass
from refaat.domain.governance import DirectSkillPolicy, ToolCapability
from refaat.domain.models import DocumentInput, JsonObject, ReActPolicy, ToolGrant
from refaat.errors import AuthorizationError
from refaat.mcp.bridge import MCPToolBridge

pytestmark = pytest.mark.integration

ALICE = Principal(id="alice", groups=frozenset({"research-a"}))
QUERY = "reindex the production catalog nightly job"


@dataclass(slots=True)
class SpySkill:
    name: str = "reindex"
    description: str = "Reindex the catalog."
    calls: int = 0
    seen_evidence: list[int] = field(default_factory=list)

    def run(self, context: object, arguments: JsonObject) -> JsonObject:
        self.calls += 1
        hits = getattr(context, "hits", ())
        self.seen_evidence.append(len(hits))
        return {"ok": True, "evidence": len(hits), "arguments": arguments}


def _doc(source_id: str, classification: str | None, allow_groups: list[str] | None = None) -> DocumentInput:
    metadata: JsonObject = {"doc": source_id}
    if classification is not None:
        acl: JsonObject = {"classification": classification}
        if allow_groups is not None:
            acl["allow_groups"] = allow_groups
        metadata["acl"] = acl
    content = f"The {source_id} reindex production catalog nightly job requires an indexed rollback plan."
    return DocumentInput(
        source_id=source_id,
        title=source_id,
        media_type="text/markdown",
        content=content.encode(),
        metadata=metadata,
    )


def _build(
    settings: Settings,
    *,
    corpus: str,
    direct_policy: DirectSkillPolicy | None = None,
    access_policy: CanonicalAccessPolicy | None = None,
    docs: list[DocumentInput] | None = None,
) -> tuple[object, RefaatProject, SpySkill]:
    spy = SpySkill()
    builder = RuntimeBuilder(settings).with_skill(spy)
    if access_policy is not None:
        builder = builder.with_access_policy(access_policy)
    if direct_policy is not None:
        builder = builder.with_direct_skill_policy(direct_policy)
    runtime = builder.build()
    project = runtime.project(workspace="acme", corpus=corpus)
    for document in docs or []:
        runtime.ingest(project.workspace_id, project.corpus_id, document)
    return runtime, project, spy


def _capability(**overrides: object) -> ToolCapability:
    base: dict[str, object] = {
        "name": "reindex",
        "required_scopes": frozenset(),
        "side_effect": SideEffectClass.READ_ONLY,
        "allowed_classification": Classification.RESTRICTED,
        "max_payload_bytes": None,
    }
    base.update(overrides)
    return ToolCapability(**base)  # type: ignore[arg-type]


def _policy(capability: ToolCapability, *, granted_scopes: frozenset[str] = frozenset()) -> DirectSkillPolicy:
    return DirectSkillPolicy(
        granted_scopes=granted_scopes,
        side_effect_ceiling=SideEffectClass.WRITE,
        capabilities=(capability,),
    )


def test_undeclared_skill_is_byte_identical(settings: Settings) -> None:
    # No DirectSkillPolicy: the normal one-liner runs the skill unchanged.
    runtime, project, spy = _build(settings, corpus="none", docs=[_doc("public.md", "public")])
    try:
        result = project.run_skill("reindex", QUERY, arguments={"label": "x"})
        assert spy.calls == 1
        assert result["ok"] is True
        assert result["arguments"] == {"label": "x"}
    finally:
        runtime.close()

    # A DirectSkillPolicy installed for a different skill leaves this one on the same one-liner.
    other = _policy(ToolCapability(name="other_skill"))
    runtime2, project2, spy2 = _build(settings, corpus="other", direct_policy=other, docs=[_doc("public.md", "public")])
    try:
        result2 = project2.run_skill("reindex", QUERY, arguments={"label": "y"})
        assert spy2.calls == 1
        assert result2["arguments"] == {"label": "y"}
    finally:
        runtime2.close()


def test_over_classified_evidence_denies(settings: Settings) -> None:
    doc = [_doc("secret.md", "confidential")]
    deny = _policy(_capability(allowed_classification=Classification.PUBLIC))
    runtime, project, spy = _build(settings, corpus="deny", direct_policy=deny, docs=doc)
    try:
        with pytest.raises(AuthorizationError):
            project.run_skill("reindex", QUERY, arguments={})
        assert spy.calls == 0
    finally:
        runtime.close()

    allow = _policy(_capability(allowed_classification=Classification.RESTRICTED))
    runtime2, project2, spy2 = _build(settings, corpus="allow", direct_policy=allow, docs=doc)
    try:
        result = project2.run_skill("reindex", QUERY, arguments={})
        assert spy2.calls == 1
        assert result["ok"] is True
    finally:
        runtime2.close()


def test_over_payload_denies(settings: Settings) -> None:
    doc = [_doc("public.md", "public")]
    capped = _policy(_capability(max_payload_bytes=8))
    runtime, project, spy = _build(settings, corpus="cap", direct_policy=capped, docs=doc)
    try:
        with pytest.raises(AuthorizationError):
            project.run_skill("reindex", QUERY, arguments={"blob": "x" * 200})
        assert spy.calls == 0
        result = project.run_skill("reindex", QUERY, arguments={})
        assert spy.calls == 1
        assert result["ok"] is True
    finally:
        runtime.close()


def test_missing_scope_denies(settings: Settings) -> None:
    doc = [_doc("public.md", "public")]
    capability = _capability(required_scopes=frozenset({"kb.write"}))
    deny = _policy(capability, granted_scopes=frozenset())
    runtime, project, spy = _build(settings, corpus="noscope", direct_policy=deny, docs=doc)
    try:
        with pytest.raises(AuthorizationError):
            project.run_skill("reindex", QUERY, arguments={})
        assert spy.calls == 0
    finally:
        runtime.close()

    allow = _policy(capability, granted_scopes=frozenset({"kb.write"}))
    runtime2, project2, spy2 = _build(settings, corpus="scoped", direct_policy=allow, docs=doc)
    try:
        result = project2.run_skill("reindex", QUERY, arguments={})
        assert spy2.calls == 1
        assert result["ok"] is True
    finally:
        runtime2.close()


def test_requires_approval_refuses_on_direct_path(settings: Settings) -> None:
    doc = [_doc("public.md", "public")]
    approval_capability = _policy(_capability(requires_approval=True))
    runtime, project, spy = _build(settings, corpus="approval", direct_policy=approval_capability, docs=doc)
    try:
        with pytest.raises(AuthorizationError) as excinfo:
            project.run_skill("reindex", QUERY, arguments={})
        assert "governed" in str(excinfo.value) or "program" in str(excinfo.value)
        assert spy.calls == 0

        # Even a durable approval for the capability cannot admit the direct path: it derives no
        # fingerprint, so no stored approval is ever consulted.
        project.approvals.grant(
            fingerprint="a" * 64,
            capability_name="reindex",
            approver_id="carol",
            approver_roles=("approver",),
        )
        with pytest.raises(AuthorizationError):
            project.run_skill("reindex", QUERY, arguments={})
        assert spy.calls == 0
    finally:
        runtime.close()


def _bridge_policy() -> ReActPolicy:
    return ReActPolicy(
        max_iterations=1,
        granted_scopes=("kb.write",),
        side_effect_ceiling=SideEffectClass.WRITE,
        grants=(ToolGrant(skill_name="reindex", required_scopes=("kb.write",), side_effect=SideEffectClass.WRITE),),
    )


def test_mcp_bridge_inherits_enforcement(settings: Settings) -> None:
    doc = [_doc("public.md", "public")]
    capability = _capability(required_scopes=frozenset({"kb.write"}), side_effect=SideEffectClass.WRITE)

    deny = _policy(capability, granted_scopes=frozenset())
    runtime, project, spy = _build(settings, corpus="mcpdeny", direct_policy=deny, docs=doc)
    try:
        bridge = MCPToolBridge(
            registry=runtime.skill_registry,
            skill_service=runtime.skills,
            policy=_bridge_policy(),
            workspace_id=project.workspace_id,
            corpus_id=project.corpus_id,
        )
        with pytest.raises(AuthorizationError):
            bridge.call_tool("reindex", {"query": QUERY})
        assert spy.calls == 0
    finally:
        runtime.close()

    allow = _policy(capability, granted_scopes=frozenset({"kb.write"}))
    runtime2, project2, spy2 = _build(settings, corpus="mcpallow", direct_policy=allow, docs=doc)
    try:
        bridge2 = MCPToolBridge(
            registry=runtime2.skill_registry,
            skill_service=runtime2.skills,
            policy=_bridge_policy(),
            workspace_id=project2.workspace_id,
            corpus_id=project2.corpus_id,
        )
        result = bridge2.call_tool("reindex", {"query": QUERY})
        assert result.authorized is True
        assert result.observation["ok"] is True
        assert spy2.calls == 1
    finally:
        runtime2.close()


def test_ceiling_skill_requires_a_principal_under_an_active_policy(settings: Settings) -> None:
    docs = [
        _doc("alpha.md", "internal", allow_groups=["research-a"]),
        _doc("locked.md", "restricted", allow_groups=["research-x"]),
    ]
    policy = _policy(_capability(allowed_classification=Classification.INTERNAL))
    runtime, project, spy = _build(
        settings,
        corpus="refine",
        direct_policy=policy,
        access_policy=CanonicalAccessPolicy(),
        docs=docs,
    )
    try:
        before = spy.calls
        result = project.run_skill("reindex", QUERY, principal=ALICE, arguments={})
        assert spy.calls == before + 1
        assert result["ok"] is True

        # Without a principal the PEP cannot filter, and it filters before the top-k truncation, so the
        # derived ceiling is not a sound bound. A real ceiling is refused before any evidence is retrieved.
        with pytest.raises(AuthorizationError):
            project.run_skill("reindex", QUERY, principal=None, arguments={})
        assert spy.calls == before + 1
    finally:
        runtime.close()


def test_no_ceiling_skill_still_runs_without_a_principal(settings: Settings) -> None:
    policy = _policy(_capability(allowed_classification=Classification.RESTRICTED))
    # A no-ceiling skill runs without a principal by design, so this test pins strict_authorization off.
    runtime, project, spy = _build(
        settings.model_copy(update={"strict_authorization": False}),
        corpus="noceiling",
        direct_policy=policy,
        access_policy=CanonicalAccessPolicy(),
        docs=[_doc("public.md", "public")],
    )
    try:
        before = spy.calls
        result = project.run_skill("reindex", QUERY, principal=None, arguments={})
        assert spy.calls == before + 1
        assert result["ok"] is True
    finally:
        runtime.close()


def test_strict_denial_precedes_the_quota_charge_on_the_direct_skill_path(settings: Settings) -> None:
    # A principal-less skill call under an active policy is denied by strict authorization BEFORE the quota
    # charge, so it raises AuthorizationError (not QuotaExceededError) and burns no quota unit. Regression
    # for the ordering: the auth gate must run before charge_served_request.
    quota_settings = settings.model_copy(
        update={
            "quota_backend": "database",
            "quota_rules": [{"scope": "workspace", "dimension": "skill_requests", "limit": 1, "window_seconds": 3600}],
        }
    )
    runtime, project, spy = _build(
        quota_settings,
        corpus="strict-quota",
        direct_policy=_policy(_capability(allowed_classification=Classification.RESTRICTED)),
        access_policy=CanonicalAccessPolicy(),
        docs=[_doc("public.md", "public")],
    )
    try:
        # Under the flipped strict default, no principal is denied by authorization and never billed.
        with pytest.raises(AuthorizationError):
            project.run_skill("reindex", QUERY, principal=None, arguments={})
        assert spy.calls == 0
        # The unit was not consumed, so an authorized call still admits under the limit of 1.
        result = project.run_skill("reindex", QUERY, principal=ALICE, arguments={})
        assert result["ok"] is True
    finally:
        runtime.close()


def test_direct_policy_denial_precedes_the_quota_charge(settings: Settings) -> None:
    # A direct-skill-policy denial (requires_approval) is raised BEFORE the skill_requests charge, so a
    # denied call burns no quota unit and never raises QuotaExceededError; the skill never runs.
    quota_settings = settings.model_copy(
        update={
            "quota_backend": "database",
            "quota_rules": [{"scope": "workspace", "dimension": "skill_requests", "limit": 1, "window_seconds": 3600}],
        }
    )
    runtime, project, spy = _build(
        quota_settings,
        corpus="approval-quota",
        direct_policy=_policy(_capability(requires_approval=True)),
        docs=[_doc("public.md", "public")],
    )
    try:
        for _ in range(3):
            with pytest.raises(AuthorizationError):
                project.run_skill("reindex", QUERY, principal=ALICE, arguments={})
        assert spy.calls == 0
    finally:
        runtime.close()
