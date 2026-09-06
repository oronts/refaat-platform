from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from refaat.canonical import canonical_digest
from refaat.domain.authz import AccessPolicy, EntitlementScopeResolver, Principal, PrincipalResolver
from refaat.domain.models import JsonObject, QueryRequest, ReActPolicy
from refaat.errors import NotFoundError
from refaat.ports import Skill
from refaat.programs.react import ToolGovernor
from refaat.skills.registry import SkillRegistry, SkillService

if TYPE_CHECKING:
    from refaat.application.runtime import Refaat

PERMISSIVE_INPUT_SCHEMA: JsonObject = {"type": "object", "properties": {}, "additionalProperties": True}


def _permissive_schema() -> JsonObject:
    return dict(PERMISSIVE_INPUT_SCHEMA)


@dataclass(frozen=True, slots=True)
class MCPToolDescriptor:
    """One advertised tool on the served MCP surface.

    ``input_schema`` is a JSON Schema for the tool's arguments. Skills accept an
    arbitrary :class:`JsonObject`, so the default is a permissive object schema
    with no required fields; a skill that exposes its own ``input_schema`` object
    surfaces that instead. The schema is folded into ``manifest_hash`` so the
    served-surface content address is stable and testable offline.
    """

    name: str
    description: str
    input_schema: JsonObject = field(default_factory=_permissive_schema)

    def to_json(self) -> JsonObject:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


@dataclass(frozen=True, slots=True)
class MCPToolResult:
    """The structured outcome of an MCP ``call_tool``.

    A governance denial returns ``authorized=False`` with the governor's reason
    rather than raising, so a client sees a well-formed result on every path.
    """

    name: str
    authorized: bool
    reason: str
    observation: JsonObject = field(default_factory=dict)

    def to_json(self) -> JsonObject:
        return {
            "name": self.name,
            "authorized": self.authorized,
            "reason": self.reason,
            "observation": self.observation,
        }


class MCPToolBridge:
    """Serves Refaat skills over MCP under one governance policy.

    The bridge holds a single :class:`ToolGovernor` built from the policy and
    delegates every authorization decision to it. ``list_tools`` advertises only
    skills the governor admits; ``call_tool`` re-authorizes at the boundary
    (defense in depth) and, on success, runs the skill through
    :class:`SkillService` so retrieval and citations are untouched. The corpus
    binding is fixed at construction and the query is taken from the call
    arguments.
    """

    def __init__(
        self,
        *,
        registry: SkillRegistry,
        skill_service: SkillService,
        policy: ReActPolicy,
        workspace_id: str,
        corpus_id: str,
        principal_resolver: PrincipalResolver | None = None,
        entitlement_scope_resolver: EntitlementScopeResolver | None = None,
        access_policy: AccessPolicy | None = None,
        strict: bool = False,
    ) -> None:
        self._registry = registry
        self._skills = skill_service
        self._policy = policy
        self._governor = ToolGovernor(policy)
        self._workspace_id = workspace_id
        self._corpus_id = corpus_id
        self._principal_resolver = principal_resolver
        self._entitlement_scope_resolver = entitlement_scope_resolver
        self._access_policy = access_policy
        self._strict = strict

    @classmethod
    def from_runtime(cls, runtime: Refaat, *, policy: ReActPolicy, workspace_id: str, corpus_id: str) -> MCPToolBridge:
        """Build a bridge wired to the runtime's identity, entitlement, and policy seams.

        The per-caller identity seam is active only when the runtime installs a principal resolver and an
        access policy under strict authorization; otherwise the bridge behaves exactly as a plain one.
        """
        return cls(
            registry=runtime.skill_registry,
            skill_service=runtime.skills,
            policy=policy,
            workspace_id=workspace_id,
            corpus_id=corpus_id,
            principal_resolver=runtime.principal_resolver,
            entitlement_scope_resolver=runtime.entitlement_scope_resolver,
            access_policy=runtime.access_policy,
            strict=runtime.settings.strict_authorization,
        )

    def list_tools(self, *, context: Mapping[str, Any] | None = None) -> tuple[MCPToolDescriptor, ...]:
        principal = self._resolve_identity(context)
        if self._identity_required() and principal is None:
            return ()
        return self._descriptors(self._caller_scopes(principal))

    def call_tool(
        self, name: str, arguments: JsonObject | None = None, *, context: Mapping[str, Any] | None = None
    ) -> MCPToolResult:
        args = arguments or {}
        principal = self._resolve_identity(context)
        if self._identity_required() and principal is None:
            return MCPToolResult(name=name, authorized=False, reason="identity_required")
        authorized, reason = self._governor.authorize(name, caller_scopes=self._caller_scopes(principal))
        if not authorized:
            return MCPToolResult(name=name, authorized=False, reason=reason)
        try:
            skill = self._registry.get(name)
        except NotFoundError:
            return MCPToolResult(name=name, authorized=False, reason=f"no skill registered for tool {name!r}")
        del skill
        observation = self._skills.run(name, self._request(args, principal), args)
        return MCPToolResult(name=name, authorized=True, reason=reason, observation=observation)

    def manifest_hash(self) -> str:
        # Folds the policy-level surface (no caller identity), so the content address is identity-independent
        # and byte-identical to the pre-identity manifest for the same policy.
        payload: JsonObject = {
            "tools": [descriptor.to_json() for descriptor in self._policy_descriptors()],
            "side_effect_ceiling": str(self._policy.side_effect_ceiling),
            "granted_scopes": sorted(self._policy.granted_scopes),
        }
        return f"sha256:{canonical_digest(payload)}"

    def _identity_required(self) -> bool:
        # Fail closed only when strict authorization is on and a policy is installed: then an unresolved
        # caller is denied rather than served the policy-blanket surface.
        return self._strict and self._access_policy is not None

    def _resolve_identity(self, context: Mapping[str, Any] | None) -> Principal | None:
        # Identity comes only from the trusted server-side context, never from the tool arguments.
        if context is None or self._principal_resolver is None:
            return None
        return self._principal_resolver.resolve(context)

    def _caller_scopes(self, principal: Principal | None) -> frozenset[str] | None:
        resolver = self._entitlement_scope_resolver
        if resolver is not None and principal is not None:
            # A resolver that breaks its frozenset contract and returns None must not become a full grant.
            return resolver.scopes_for(principal) or frozenset()
        # No entitlement resolver: fail closed to no scopes when identity is enforced, so a missing resolver
        # cannot silently grant the full policy surface; otherwise None keeps the unconfigured path byte-identical.
        if principal is not None and self._identity_required():
            return frozenset()
        return None

    def _descriptors(self, caller_scopes: frozenset[str] | None) -> tuple[MCPToolDescriptor, ...]:
        descriptors: list[MCPToolDescriptor] = []
        for grant in self._policy.grants:
            authorized, _ = self._governor.authorize(grant.skill_name, caller_scopes=caller_scopes)
            if not authorized:
                continue
            try:
                skill = self._registry.get(grant.skill_name)
            except NotFoundError:
                continue
            descriptors.append(self._descriptor(skill))
        return tuple(sorted(descriptors, key=lambda descriptor: descriptor.name))

    def _policy_descriptors(self) -> tuple[MCPToolDescriptor, ...]:
        return self._descriptors(None)

    def _descriptor(self, skill: Skill) -> MCPToolDescriptor:
        schema = getattr(skill, "input_schema", None)
        input_schema = schema if isinstance(schema, dict) else _permissive_schema()
        return MCPToolDescriptor(name=skill.name, description=skill.description, input_schema=input_schema)

    def _request(self, arguments: JsonObject, principal: Principal | None = None) -> QueryRequest:
        query = arguments.get("query")
        return QueryRequest(
            workspace_id=self._workspace_id,
            corpus_id=self._corpus_id,
            query=query if isinstance(query, str) else "",
            principal=principal,
        )
