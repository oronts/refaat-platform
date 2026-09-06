from __future__ import annotations

from typing import ClassVar

import pytest

from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.config import Settings
from refaat.domain.authz import CanonicalAccessPolicy, Principal, RoleGroupScopeResolver
from refaat.domain.enums import SideEffectClass
from refaat.domain.models import ContextPacket, DocumentInput, JsonObject, ReActPolicy, ToolGrant
from refaat.mcp.bridge import PERMISSIVE_INPUT_SCHEMA, MCPToolBridge

pytestmark = pytest.mark.integration


class _NotifySkill:
    name = "notify"
    description = "External notification, denied under a read-only ceiling."

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        del context, arguments
        return {"sent": True}


class _SchemaSkill:
    name = "schema_tool"
    description = "A skill that publishes its own argument schema."
    input_schema: ClassVar[JsonObject] = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        del arguments
        return {"query": context.query}


def _ingest(runtime: Refaat, workspace_id: str, corpus_id: str) -> None:
    runtime.ingest(
        workspace_id,
        corpus_id,
        DocumentInput(
            source_id="ops.txt",
            title="Ops",
            media_type="text/plain",
            content=b"Rollback restores the previously approved harness snapshot before deployment.",
        ),
    )


def _bridge(runtime: Refaat, workspace_id: str, corpus_id: str, policy: ReActPolicy) -> MCPToolBridge:
    return MCPToolBridge(
        registry=runtime.skills.registry,
        skill_service=runtime.skills,
        policy=policy,
        workspace_id=workspace_id,
        corpus_id=corpus_id,
    )


def _read_only_policy() -> ReActPolicy:
    return ReActPolicy(
        granted_scopes=("read:corpus", "write:crm"),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        grants=(
            ToolGrant("corpus_health", required_scopes=("read:corpus",), side_effect=SideEffectClass.READ_ONLY),
            ToolGrant("notify", required_scopes=("write:crm",), side_effect=SideEffectClass.EXTERNAL),
        ),
    )


def test_list_tools_advertises_only_authorized_grants(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="mcp")
    runtime.skills.registry.register(_NotifySkill())
    _ingest(runtime, project.workspace_id, project.corpus_id)
    bridge = _bridge(runtime, project.workspace_id, project.corpus_id, _read_only_policy())

    names = [descriptor.name for descriptor in bridge.list_tools()]

    assert names == ["corpus_health"]
    descriptor = bridge.list_tools()[0]
    assert descriptor.input_schema == PERMISSIVE_INPUT_SCHEMA
    assert descriptor.input_schema["additionalProperties"] is True


def test_call_tool_runs_the_authorized_skill(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="mcp")
    _ingest(runtime, project.workspace_id, project.corpus_id)
    bridge = _bridge(runtime, project.workspace_id, project.corpus_id, _read_only_policy())

    result = bridge.call_tool("corpus_health", {"query": "how does rollback work"})

    assert result.authorized is True
    assert result.reason == "authorized"
    assert result.observation["query"] == "how does rollback work"
    assert result.observation["retrieved_chunks"] >= 1


def test_call_tool_denies_over_ceiling_without_raising(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="mcp")
    runtime.skills.registry.register(_NotifySkill())
    _ingest(runtime, project.workspace_id, project.corpus_id)
    bridge = _bridge(runtime, project.workspace_id, project.corpus_id, _read_only_policy())

    result = bridge.call_tool("notify", {"query": "rollback"})

    assert result.authorized is False
    assert "exceeds ceiling" in result.reason
    assert result.observation == {}


def test_call_tool_denies_ungranted_tool_without_raising(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="mcp")
    _ingest(runtime, project.workspace_id, project.corpus_id)
    bridge = _bridge(runtime, project.workspace_id, project.corpus_id, _read_only_policy())

    result = bridge.call_tool("unregistered", {"query": "rollback"})

    assert result.authorized is False
    assert "no grant registered" in result.reason


def test_manifest_hash_is_stable_and_reflects_the_input_schema(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="mcp")
    runtime.skills.registry.register(_SchemaSkill())
    _ingest(runtime, project.workspace_id, project.corpus_id)
    schema_policy = ReActPolicy(
        granted_scopes=("read:corpus",),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        grants=(ToolGrant("schema_tool", required_scopes=("read:corpus",), side_effect=SideEffectClass.READ_ONLY),),
    )
    bridge = _bridge(runtime, project.workspace_id, project.corpus_id, schema_policy)

    descriptor = bridge.list_tools()[0]
    assert descriptor.input_schema == _SchemaSkill.input_schema
    assert bridge.manifest_hash() == bridge.manifest_hash()

    permissive_bridge = _bridge(
        runtime,
        project.workspace_id,
        project.corpus_id,
        ReActPolicy(
            granted_scopes=("read:corpus",),
            side_effect_ceiling=SideEffectClass.READ_ONLY,
            grants=(
                ToolGrant("corpus_health", required_scopes=("read:corpus",), side_effect=SideEffectClass.READ_ONLY),
            ),
        ),
    )
    assert bridge.manifest_hash() != permissive_bridge.manifest_hash()


def test_grant_without_a_registered_skill_is_skipped_and_denied(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="mcp")
    _ingest(runtime, project.workspace_id, project.corpus_id)
    ghost_policy = ReActPolicy(
        granted_scopes=("read:corpus",),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        grants=(ToolGrant("ghost", required_scopes=("read:corpus",), side_effect=SideEffectClass.READ_ONLY),),
    )
    bridge = _bridge(runtime, project.workspace_id, project.corpus_id, ghost_policy)

    assert bridge.list_tools() == ()

    result = bridge.call_tool("ghost", {"query": "rollback"})
    assert result.authorized is False
    assert "no skill registered" in result.reason
    assert result.to_json() == {
        "name": "ghost",
        "authorized": False,
        "reason": result.reason,
        "observation": {},
    }


def test_manifest_hash_changes_when_policy_changes(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="mcp")
    runtime.skills.registry.register(_NotifySkill())
    _ingest(runtime, project.workspace_id, project.corpus_id)
    read_only = _bridge(runtime, project.workspace_id, project.corpus_id, _read_only_policy())
    external = _bridge(
        runtime,
        project.workspace_id,
        project.corpus_id,
        ReActPolicy(
            granted_scopes=("read:corpus", "write:crm"),
            side_effect_ceiling=SideEffectClass.EXTERNAL,
            grants=(
                ToolGrant("corpus_health", required_scopes=("read:corpus",), side_effect=SideEffectClass.READ_ONLY),
                ToolGrant("notify", required_scopes=("write:crm",), side_effect=SideEffectClass.EXTERNAL),
            ),
        ),
    )

    assert [descriptor.name for descriptor in external.list_tools()] == ["corpus_health", "notify"]
    assert read_only.manifest_hash() != external.manifest_hash()


# Per-caller identity seam on the inbound MCP serve path.

ALICE = Principal(id="alice", tenant="acme", roles=frozenset({"read:corpus"}))
BOB = Principal(id="bob", tenant="acme", roles=frozenset({"other"}))


class _MapResolver:
    """A trusted-context resolver: maps a context token to a principal, or None (identity from context only)."""

    def __init__(self, principals: dict[str, Principal]) -> None:
        self._principals = principals

    def resolve(self, context: JsonObject) -> Principal | None:
        token = context.get("token")
        return self._principals.get(token) if isinstance(token, str) else None


def _ctx(token: str) -> JsonObject:
    return {"token": token}


def _health_policy() -> ReActPolicy:
    return ReActPolicy(
        granted_scopes=("read:corpus", "write:crm"),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        grants=(ToolGrant("corpus_health", required_scopes=("read:corpus",), side_effect=SideEffectClass.READ_ONLY),),
    )


def _identity_runtime(settings: Settings) -> Refaat:
    return (
        RuntimeBuilder(settings)
        .with_access_policy(CanonicalAccessPolicy())
        .with_principal_resolver(_MapResolver({"alice": ALICE, "bob": BOB}))
        .with_entitlement_scope_resolver(RoleGroupScopeResolver())
        .build()
    )


def test_per_caller_identity_scopes_advertised_and_callable_tools(settings: Settings) -> None:
    runtime = _identity_runtime(settings)
    try:
        project = runtime.project(workspace="acme", corpus="mcp")
        _ingest(runtime, project.workspace_id, project.corpus_id)
        bridge = MCPToolBridge.from_runtime(
            runtime, policy=_health_policy(), workspace_id=project.workspace_id, corpus_id=project.corpus_id
        )
        # Alice holds read:corpus: the tool is advertised and callable, and her principal threads through so
        # the SkillService's strict principal gate passes (a no-principal call would be denied there).
        assert [d.name for d in bridge.list_tools(context=_ctx("alice"))] == ["corpus_health"]
        assert bridge.call_tool("corpus_health", {"query": "how does rollback work"}, context=_ctx("alice")).authorized
        # Bob lacks read:corpus: the tool is neither advertised nor callable, even though the policy grants it.
        assert bridge.list_tools(context=_ctx("bob")) == ()
        assert bridge.call_tool("corpus_health", {"query": "rollback"}, context=_ctx("bob")).authorized is False
    finally:
        runtime.close()


def test_bridge_fails_closed_under_strict_policy_without_identity(settings: Settings) -> None:
    runtime = _identity_runtime(settings)
    try:
        project = runtime.project(workspace="acme", corpus="mcp")
        _ingest(runtime, project.workspace_id, project.corpus_id)
        bridge = MCPToolBridge.from_runtime(
            runtime, policy=_health_policy(), workspace_id=project.workspace_id, corpus_id=project.corpus_id
        )
        # No context and an unresolvable token both deny rather than serving the policy-blanket surface.
        assert bridge.list_tools() == ()
        assert bridge.list_tools(context=_ctx("unknown")) == ()
        no_context = bridge.call_tool("corpus_health", {"query": "x"})
        assert no_context.authorized is False
        assert no_context.reason == "identity_required"
        unknown = bridge.call_tool("corpus_health", {"query": "x"}, context=_ctx("unknown"))
        assert unknown.authorized is False
        assert unknown.reason == "identity_required"
    finally:
        runtime.close()


def test_manifest_hash_is_identity_independent(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="mcp")
    _ingest(runtime, project.workspace_id, project.corpus_id)
    policy = _health_policy()
    plain = _bridge(runtime, project.workspace_id, project.corpus_id, policy)
    identity = MCPToolBridge(
        registry=runtime.skills.registry,
        skill_service=runtime.skills,
        policy=policy,
        workspace_id=project.workspace_id,
        corpus_id=project.corpus_id,
        principal_resolver=_MapResolver({"alice": ALICE}),
        entitlement_scope_resolver=RoleGroupScopeResolver(),
        access_policy=CanonicalAccessPolicy(),
        strict=True,
    )
    # The content address folds only the policy-level surface, so it is identical whether or not the
    # per-caller identity seam is configured.
    assert identity.manifest_hash() == plain.manifest_hash()


def test_missing_entitlement_resolver_fails_closed_on_scope(settings: Settings) -> None:
    # A policy plus a principal resolver but NO entitlement resolver under strict must not grant the full
    # policy surface to a resolved caller: with no way to determine scopes, a scope-requiring tool is denied.
    runtime = (
        RuntimeBuilder(settings)
        .with_access_policy(CanonicalAccessPolicy())
        .with_principal_resolver(_MapResolver({"bob": BOB}))
        .build()
    )
    try:
        project = runtime.project(workspace="acme", corpus="mcp")
        _ingest(runtime, project.workspace_id, project.corpus_id)
        bridge = MCPToolBridge.from_runtime(
            runtime, policy=_health_policy(), workspace_id=project.workspace_id, corpus_id=project.corpus_id
        )
        # Bob resolves, but corpus_health requires read:corpus and scopes are undeterminable, so it is denied.
        assert bridge.list_tools(context=_ctx("bob")) == ()
        assert bridge.call_tool("corpus_health", {"query": "x"}, context=_ctx("bob")).authorized is False
    finally:
        runtime.close()


def test_no_seams_ignores_context_and_is_byte_identical(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="mcp")
    _ingest(runtime, project.workspace_id, project.corpus_id)
    bridge = _bridge(runtime, project.workspace_id, project.corpus_id, _health_policy())
    # With no seams configured a context is ignored and the surface is the policy-level default.
    assert bridge.list_tools() == bridge.list_tools(context=_ctx("alice"))
    assert [d.name for d in bridge.list_tools()] == ["corpus_health"]
    assert bridge.call_tool("corpus_health", {"query": "rollback"}, context=_ctx("alice")).authorized is True
