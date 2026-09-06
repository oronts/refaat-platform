from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from refaat.application.approvals import StoreApprovalBackend, active_approval_context
from refaat.application.budget import (
    CostMeter,
    active_meter,
    budget_scope,
    prices_from,
    resolve_stream_meter,
    use_meter,
)
from refaat.application.outbound import CapabilityOutbound, OutboundGateway
from refaat.application.secrets import SecretResolver, redact_secret_values, sanitize_secret_exception
from refaat.application.tool_timeout import run_tool_within_deadline
from refaat.canonical import canonical_json
from refaat.domain.authz import Classification, Principal, max_classification
from refaat.domain.enums import AuditLevel, SideEffectClass
from refaat.domain.governance import (
    ToolCapability,
    approval_fingerprint,
    approval_satisfied,
    capability_permits,
)
from refaat.domain.kill_switch import KillSwitchRegistry, KillSwitchScope
from refaat.domain.models import (
    JsonObject,
    QueryRequest,
    ReActPolicy,
    ReActStep,
    ToolGrant,
    ToolProposal,
    ToolSpec,
)
from refaat.domain.secrets import SecretBundle, SecretScope
from refaat.errors import (
    TOOL_OUTCOME_UNKNOWN,
    TOOL_TIMEOUT_REASON_CONFIRMED,
    TOOL_TIMEOUT_REASON_UNKNOWN,
    ToolTimeoutError,
    ValidationError,
)
from refaat.observability import get_logger, span
from refaat.ports import Skill, ToolPlanner
from refaat.programs.program import (
    GroundedRagInput,
    GroundedRagOutput,
    Program,
    ProgramResult,
    Tunables,
)
from refaat.programs.toolcall import _ToolCallAccumulator, validate_tool_arguments

if TYPE_CHECKING:
    from refaat.application.runtime import RefaatProject
    from refaat.skills.registry import SkillRegistry


logger = get_logger(__name__)


# The bounded, developer-authored vocabulary a governed-step reason maps to before it reaches the
# observability egress. Every current reason literal (from the governed loop, the ToolGovernor, the
# approval gate, and capability_permits) reduces to one of these or to the "other" fallback, so no
# model-chosen substring (an argument field name, a denied host list) is ever emitted.
_REASON_CATEGORY_CODES = frozenset(
    {
        "authorized",
        "unauthorized",
        "approval_required",
        "argument_validation_failed",
        "missing_scopes",
        "side_effect_exceeds_ceiling",
        "classification_exceeds_capability",
        "payload_size_unknown",
        "payload_exceeds_max",
        "network_destination_denied",
        "resource_namespace_denied",
        "required_secret_unavailable",
        "tool_timeout",
    }
)


def governance_reason_code(reason: str) -> str:
    """Map a raw governed-step reason to a fixed-vocabulary opaque code.

    The model-influenced detail of a reason always follows the first ``:`` (an unexpected argument
    field name, a denied host list), so only the developer-authored prefix before the first ``:`` is
    considered, normalized to a snake_case identifier, and returned only when it is a known category.
    Anything else, including a reason that embeds a model-chosen tool name with no separator, maps to
    the fixed ``other`` fallback, so only a bounded developer code ever reaches the egress.
    """
    prefix = reason.split(":", 1)[0].strip().lower().replace(" ", "_")
    return prefix if prefix in _REASON_CATEGORY_CODES else "other"


def _effective_scopes(granted: tuple[str, ...], caller_scopes: frozenset[str] | None) -> tuple[str, ...]:
    # The scopes actually in force: the program's declared grants intersected with the caller's held
    # scopes when known, so a program cannot self-grant authority the caller lacks. Unset is unchanged.
    if caller_scopes is None:
        return granted
    return tuple(scope for scope in granted if scope in caller_scopes)


class ToolGovernor:
    """Pure authorization for a governed tool call.

    A tool is admitted only when it has a registered grant, the caller holds
    every scope the grant requires, and the grant's side-effect class does not
    exceed the policy ceiling. No I/O, so the decision is reproducible.
    """

    def __init__(self, policy: ReActPolicy) -> None:
        self._policy = policy
        self._grants = {grant.skill_name: grant for grant in policy.grants}

    def authorize(self, tool: str, *, caller_scopes: frozenset[str] | None = None) -> tuple[bool, str]:
        grant = self._grants.get(tool)
        if grant is None:
            return False, f"no grant registered for tool {tool!r}"
        # Least-privilege: a grant is admitted only when the CALLING principal holds its required scopes,
        # not merely because the program declares them. Unset caller_scopes is byte-identical to before.
        granted = _effective_scopes(self._policy.granted_scopes, caller_scopes)
        missing = tuple(scope for scope in grant.required_scopes if scope not in granted)
        if missing:
            return False, f"missing scopes: {', '.join(missing)}"
        if grant.side_effect.rank > self._policy.side_effect_ceiling.rank:
            return False, f"side effect {grant.side_effect} exceeds ceiling {self._policy.side_effect_ceiling}"
        return True, "authorized"


class KeywordToolPlanner:
    """Deterministic, model-free planner.

    Routes a query to the first not-yet-used tool whose keywords appear in it,
    in registration order, and returns None to stop. Model-free by design, so a
    governed ReAct run is reproducible offline; a model-backed planner can drop
    in behind the same interface later.
    """

    kind = "keyword"

    def __init__(self, routes: Mapping[str, Sequence[str]]) -> None:
        self._routes = {tool: tuple(keywords) for tool, keywords in routes.items()}

    def tools(self) -> tuple[str, ...]:
        return tuple(self._routes)

    def plan(self, query: str, used: frozenset[str]) -> ToolProposal | None:
        lowered = query.lower()
        for tool, keywords in self._routes.items():
            if tool in used:
                continue
            if any(keyword.lower() in lowered for keyword in keywords):
                return ToolProposal(tool=tool, arguments={})
        return None


def _program_skill_artifact_id(skill: object) -> str | None:
    """The inner program's content-addressed artifact id for a granted ProgramSkill, or None otherwise.

    Folds the inner id so changing the wrapped program forces re-approval, and two ProgramSkills that
    wrap different programs stay distinct. A subclass that overrides the id accessor to forge it is a
    registration-trust residual, the same class as a plain skill spoofing its class identity, not an
    ordinary-caller bypass. A plain or plugin skill contributes its class identity instead (see
    ``_grant_extra``), never a skill-supplied artifact_id attribute.
    """
    from refaat.skills.program_skill import ProgramSkill

    return skill.artifact_id if isinstance(skill, ProgramSkill) else None


def _revealed_values(bundle: SecretBundle | None) -> frozenset[str]:
    """The resolved values of a bundle, used to scrub a secret-aware skill's observation and errors."""
    if bundle is None:
        return frozenset()
    return frozenset(bundle.reveal(name) for name in bundle.names)


class GovernedReActProgram(Program):
    """A bounded, governed think-act-observe program.

    Each iteration plans a tool with a deterministic planner, admits it through
    a pure :class:`ToolGovernor`, and, if authorized, executes only the tool's
    ``skill.run(context, arguments)`` (no host code, no code generation). The
    loop is hard-capped by ``policy.max_iterations``. The final answer is the
    runtime's grounded query, so retrieval, generation, caching, and citations
    are unchanged; the governed steps are recorded in the result outputs. The
    full governance policy is pinned into the content-addressed artifact.
    """

    name = "governed-react"
    input_signature = GroundedRagInput
    output_signature = GroundedRagOutput

    def __init__(
        self,
        *,
        policy: ReActPolicy,
        planner: ToolPlanner,
        skills: Mapping[str, Skill],
        manifest: Sequence[ToolSpec] | None = None,
        tunables: Tunables | None = None,
    ) -> None:
        super().__init__(tunables=tunables)
        self._policy = policy
        self._planner = planner
        self._governor = ToolGovernor(policy)
        self._capabilities = {capability.name: capability for capability in policy.capabilities}
        self._skills = dict(skills)
        self._planner_kind = str(getattr(planner, "kind", "keyword"))
        self._is_model_planner = self._planner_kind == "model"
        self._planner_model_id = str(getattr(planner, "model_id", ""))
        resolved = manifest if manifest is not None else self._resolve_manifest(planner)
        self._manifest = tuple(resolved)
        self._manifest_by_name = {spec.name: spec for spec in self._manifest}

    @classmethod
    def from_registry(
        cls,
        *,
        policy: ReActPolicy,
        planner: ToolPlanner,
        registry: SkillRegistry,
        manifest: Sequence[ToolSpec] | None = None,
        tunables: Tunables | None = None,
    ) -> GovernedReActProgram:
        """Build the skills map by resolving each policy grant from a registry.

        This is the resolved-from-registry seam: a promoted program wrapped as a
        :class:`ProgramSkill` is looked up by grant name exactly like a plain
        skill, with no change to the governed loop body.
        """
        skills = {grant.skill_name: registry.get(grant.skill_name) for grant in policy.grants}
        return cls(policy=policy, planner=planner, skills=skills, manifest=manifest, tunables=tunables)

    @staticmethod
    def _resolve_manifest(planner: ToolPlanner) -> tuple[ToolSpec, ...]:
        provider = getattr(planner, "manifest", None)
        return tuple(provider()) if callable(provider) else ()

    def steps(self) -> tuple[str, ...]:
        return ("plan", "govern", "act", "observe", "answer")

    @property
    def shadow_safe(self) -> bool:
        """False when the policy admits a non-read-only tool or a capability that requires approval.

        A shadow run still executes authorized tools through ``skill.run``, and a write-or-external
        side effect is real and cannot be undone by a write-suppression scope. Consuming a durable
        single-use approval is likewise a real side effect the scope does not neutralize, so a
        capability that requires approval is never shadow-safe (an unserved run would burn the
        approval and deny the served run). A grant is only admittable within the ceiling, so a write
        grant under a read-only ceiling (always denied by the governor) stays safe to shadow.
        """
        if any(capability.requires_approval for capability in self._policy.capabilities):
            return False
        # A write or external side effect (from a grant OR a capability, within the ceiling) is real and
        # cannot be undone by a write-suppression scope, so only a fully read-only program is shadow-safe.
        return self.declared_side_effect() is SideEffectClass.READ_ONLY

    def declared_side_effect(self) -> SideEffectClass:
        # The strongest side effect a governed step could execute: the max over grants AND capabilities
        # within the ceiling, since capability admission can grant a stronger side effect than its grant.
        ceiling = self._policy.side_effect_ceiling.rank
        admissible = [grant.side_effect for grant in self._policy.grants if grant.side_effect.rank <= ceiling]
        admissible += [
            capability.side_effect for capability in self._policy.capabilities if capability.side_effect.rank <= ceiling
        ]
        return max(admissible, key=lambda side_effect: side_effect.rank, default=SideEffectClass.READ_ONLY)

    def _caller_scopes(self, project: RefaatProject, principal: Principal | None) -> frozenset[str] | None:
        resolver = project.runtime.entitlement_scope_resolver
        if resolver is not None and principal is not None:
            # A resolver that breaks its frozenset contract and returns None must not become a full grant.
            return resolver.scopes_for(principal) or frozenset()
        # No entitlement resolver: fail closed to no scopes when identity is enforced (strict plus a policy)
        # with a resolved principal, so a missing resolver cannot silently grant the full tool surface; else None.
        enforced = project.runtime.access_policy is not None and project.runtime.settings.strict_authorization
        if principal is not None and enforced:
            return frozenset()
        return None

    def _definition_extra(self) -> JsonObject:
        extra: JsonObject = {
            "react": {
                "max_iterations": self._policy.max_iterations,
                "granted_scopes": list(self._policy.granted_scopes),
                "side_effect_ceiling": str(self._policy.side_effect_ceiling),
                "grants": [self._grant_extra(grant) for grant in self._policy.grants],
                "tools": list(self._planner.tools()),
            }
        }
        if self._policy.capabilities:
            extra["react"]["capabilities"] = [self._capability_extra(c) for c in self._policy.capabilities]
        if self._is_model_planner:
            extra["planner"] = {"kind": "model", "model_id": self._planner_model_id}
            extra["manifest"] = [
                {"name": spec.name, "description": spec.description, "parameters": spec.parameters}
                for spec in self._manifest
            ]
        return extra

    def _grant_extra(self, grant: ToolGrant) -> JsonObject:
        entry: JsonObject = {
            "skill": grant.skill_name,
            "scopes": list(grant.required_scopes),
            "side_effect": str(grant.side_effect),
        }
        # Bind the grant to which skill runs so a swap forces re-approval: a ProgramSkill by its
        # content-addressed inner id, a plain skill by its class module and qualname (separate fields
        # so they cannot alias), and only under capabilities so a no-capability program stays
        # byte-identical. A skill-supplied artifact_id is never trusted here.
        skill = self._skills.get(grant.skill_name)
        artifact_id = _program_skill_artifact_id(skill)
        if artifact_id:
            entry["artifact_id"] = artifact_id
        elif skill is not None and self._policy.capabilities:
            entry["skill_identity"] = {
                "module": type(skill).__module__,
                "qualname": type(skill).__qualname__,
            }
        return entry

    @staticmethod
    def _capability_extra(capability: ToolCapability) -> JsonObject:
        entry: JsonObject = {
            "name": capability.name,
            "scopes": sorted(capability.required_scopes),
            "side_effect": str(capability.side_effect),
            "classification": int(capability.allowed_classification),
            "max_payload_bytes": capability.max_payload_bytes,
            "requires_approval": capability.requires_approval,
            "min_approvals": capability.min_approvals,
            "approver_role": capability.approver_role,
            "approval_ttl_seconds": capability.approval_ttl_seconds,
        }
        # Fold break-glass only when the capability opts in and each field diverges from its default,
        # so a capability that never enables break-glass yields the exact same dict as before and
        # every existing program artifact id, including the pinned base grounded-rag id, is unchanged.
        if capability.break_glass_enabled:
            entry["break_glass_enabled"] = True
            if capability.break_glass_approver_role is not None:
                entry["break_glass_approver_role"] = capability.break_glass_approver_role
            if capability.break_glass_min_approvals != 2:
                entry["break_glass_min_approvals"] = capability.break_glass_min_approvals
            if capability.break_glass_ttl_seconds is not None:
                entry["break_glass_ttl_seconds"] = capability.break_glass_ttl_seconds
        # Fold escalation only when declared, so a capability that never escalates yields the exact
        # same dict as before and every existing program artifact id stays unchanged.
        if capability.escalation_role is not None:
            entry["escalation_role"] = capability.escalation_role
            entry["escalation_after_seconds"] = capability.escalation_after_seconds
        # Fold each declared governance field only when set, so a capability that declares none yields the
        # exact same dict as before; sorted() makes the fold order-independent for equal frozensets.
        if capability.allowed_network:
            entry["allowed_network"] = sorted(capability.allowed_network)
        if capability.resource_namespace is not None:
            entry["resource_namespace"] = capability.resource_namespace
        if capability.required_secrets:
            entry["required_secrets"] = sorted(capability.required_secrets)
        # Fold audit_level only when it diverges from STANDARD, so a default capability yields the
        # exact same dict as before and every existing program artifact id stays unchanged.
        if capability.audit_level != AuditLevel.STANDARD:
            entry["audit_level"] = str(capability.audit_level)
        return entry

    def _authorize_approval(
        self,
        capability: ToolCapability,
        arguments: JsonObject,
        principal: Principal | None,
        now: datetime | None,
        backend: StoreApprovalBackend | None,
        action_version: str | None,
    ) -> str | None:
        """Authorize a required approval and consume it once, or return a denial reason.

        A required approval cannot be evaluated without a resolved initiator, a clock, and the
        program's action version, so any absent fails closed. Approvals come from the run-scoped
        context and, when a store is present, the durable granted approvals for the exact action
        fingerprint. A durable approval is consumed atomically before execution so it authorizes
        exactly one run; if a concurrent run consumed it first the action is denied.
        """
        if principal is None or now is None or action_version is None:
            return "approval_required:missing_initiator"
        approval = active_approval_context()
        # Bind the approval to the exact program version: action_version is the content-addressed
        # artifact id, so a change to the program, its policy, or its capabilities flips the
        # fingerprint and invalidates every prior approval.
        fingerprint = approval_fingerprint(
            capability_name=capability.name,
            arguments=arguments,
            principal=principal,
            policy_id=approval.policy_id,
            policy_version=approval.policy_version,
            action_version=action_version,
        )
        # Strip any record_id from run-scoped context approvals: only the durable backend, whose
        # record ids and fields come from the same stored row, may direct a consume. This keeps a
        # context-supplied approval from pointing the consume at an unrelated durable row.
        context_approvals = tuple(replace(record, record_id=None) for record in approval.approvals)
        durable = backend.active(fingerprint) if backend is not None else ()
        # Only a capability that declares escalation reads the request anchor, so a non-escalation
        # capability takes no extra query and its decision stays byte-identical.
        request_anchor = None
        if backend is not None and capability.escalation_role is not None:
            request_anchor = backend.request_anchor(fingerprint)
        decision = approval_satisfied(
            capability,
            fingerprint=fingerprint,
            approvals=context_approvals + durable,
            initiator_id=principal.id,
            now=now,
            request_anchor=request_anchor,
        )
        if not decision.satisfied:
            return f"approval_required:{decision.reason}"
        # Consume and audit as one atomic unit under a durable sink, so a consumed single-use approval can
        # never exist without its audit record. A false return means a concurrent run consumed it first.
        if (
            backend is not None
            and decision.consumed_ids
            and not backend.consume_with_audit(
                fingerprint,
                decision.consumed_ids,
                principal_id=principal.id,
                principal_kind=principal.kind.value,
                tenant=principal.tenant,
                capability_name=capability.name,
                policy_id=approval.policy_id,
                policy_version=approval.policy_version,
                occurred_at=now,
            )
        ):
            return "approval_required:already_consumed"
        return None

    def execute(
        self,
        project: RefaatProject,
        *,
        inputs: Mapping[str, Any],
        user_id: str | None,
        session_id: str | None,
        principal: Principal | None = None,
    ) -> ProgramResult:
        question = inputs.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValidationError("Program input 'question' is required and must be a non-empty string")

        settings = project.runtime.settings
        with budget_scope(
            settings.run_token_budget,
            settings.run_cost_budget_usd,
            prices=prices_from(settings.model_prices),
        ):
            return self._run(project, question, user_id=user_id, session_id=session_id, principal=principal)

    def _query_request(
        self,
        project: RefaatProject,
        question: str,
        *,
        user_id: str | None,
        session_id: str | None,
        principal: Principal | None = None,
    ) -> QueryRequest:
        # The governed trajectory answers with the runtime's grounded query on
        # top_k/token_budget only (fusion, instructions, and demonstrations are
        # not part of a governed answer), so this narrows the base helper to the
        # exact fields the run path has always sent, keeping the cache identity
        # byte-identical for the streamed and executed answer alike.
        return QueryRequest(
            workspace_id=project.workspace_id,
            corpus_id=project.corpus_id,
            query=question,
            user_id=user_id,
            session_id=session_id,
            top_k=self.tunables.top_k,
            token_budget=self.tunables.token_budget,
            principal=principal,
        )

    def _admissible(self, tool: str, used: set[str]) -> bool:
        """Whether a model-proposed tool may proceed to govern and execute.

        A provider can return a tool the loop already used or one that was never
        advertised, which would let a tool run twice or an unadvertised tool
        reach ``skill.run``. A model proposal is admitted only when the tool is in
        the advertised manifest and not yet used. The deterministic keyword
        planner already excludes used tools and returns only routed tools, so its
        offline path is untouched.
        """
        if tool in used:
            return False
        return not (self._is_model_planner and tool not in self._manifest_by_name)

    def _invoke_skill(
        self,
        skill: Skill,
        context: Any,
        arguments: JsonObject,
        *,
        principal: Principal | None,
        secrets: SecretBundle | None,
        outbound: CapabilityOutbound | None = None,
    ) -> JsonObject:
        """Dispatch one authorized skill call, honoring the opt-in markers.

        A skill opts into secret injection with a ``wants_secrets`` marker and into governed outbound
        HTTP with a ``wants_outbound`` marker (its EXTERNAL capability is enforced by the caller); each
        arrives as a separate keyword, never merged into ``arguments``, so neither enters ``react_steps``
        or the model prompt. A plain skill keeps the exact ``skill.run(context, arguments)`` call, so
        every existing skill is byte-identical. A secret-aware skill can copy a revealed value into its
        return or raise it in an error, so its observation is scrubbed and any exception sanitized
        whenever secrets are live, regardless of the other markers.
        """
        if getattr(skill, "wants_secrets", False):
            kwargs: dict[str, Any] = {"secrets": secrets}
            if outbound is not None:
                kwargs["outbound"] = outbound
            values = _revealed_values(secrets)
            if not values:
                unscrubbed: JsonObject = cast(Any, skill).run(context, arguments, **kwargs)
                return unscrubbed
            failure: BaseException | None = None
            observation: JsonObject | None = None
            try:
                observation = cast(Any, skill).run(context, arguments, **kwargs)
            except BaseException as exc:
                # Contain every failure form (even a bare KeyboardInterrupt carrying the secret); the
                # wrapper is re-raised below, outside the handler, so __context__ drops the original.
                failure = sanitize_secret_exception(exc, values)
            if failure is not None:
                raise failure from None
            assert observation is not None
            return cast(JsonObject, redact_secret_values(observation, values))
        if outbound is not None:
            observation = cast(Any, skill).run(context, arguments, outbound=outbound)
            return observation
        if getattr(skill, "is_program_skill", False):
            observation = cast(Any, skill).run(context, arguments, principal=principal)
            return observation
        return skill.run(context, arguments)

    def _tool_side_effect(self, tool: str, capability: ToolCapability | None) -> SideEffectClass:
        # The tool's authoritative declared side effect: its grant's, widened by any capability, so an
        # abandoned grant-only write is treated as a write (raise), never silently as a read (continue).
        grant = next((grant for grant in self._policy.grants if grant.skill_name == tool), None)
        declared = grant.side_effect if grant is not None else SideEffectClass.READ_ONLY
        if capability is not None and capability.side_effect.rank > declared.rank:
            return capability.side_effect
        return declared

    def _execute_tool(
        self,
        context: Any,
        proposal: ToolProposal,
        principal: Principal | None = None,
        now: datetime | None = None,
        approvals_backend: StoreApprovalBackend | None = None,
        action_version: str | None = None,
        evidence_classification: Classification = Classification.PUBLIC,
        secret_backend: SecretResolver | None = None,
        outbound_gateway: OutboundGateway | None = None,
        *,
        workspace_id: str,
        kill_switches: KillSwitchRegistry | None,
        caller_scopes: frozenset[str] | None = None,
        tool_timeout_seconds: float | None = None,
        tool_timeout_max_workers: int = 8,
    ) -> ReActStep:
        """Validate, authorize, and run one proposed tool call.

        The single governed body shared by the non-streaming ``_run`` loop and
        the streamed loop: model-proposed arguments are validated only once they
        are complete, the pure governor authorizes the tool, and only an
        authorized tool's ``skill.run`` executes, exactly once with the complete
        argument dict.
        """
        tool = proposal.tool
        arguments = proposal.arguments
        if kill_switches is not None and kill_switches.engaged().denies(KillSwitchScope.TOOL_EXECUTION, workspace_id):
            # A tool-execution switch denies the tool but not the run: record a denied step and let the
            # agent answer from evidence without it, since generation is a separate scope.
            return ReActStep(
                thought=f"consider {tool}",
                tool=tool,
                arguments=arguments,
                authorized=False,
                reason="tool execution blocked by an incident kill switch",
            )
        if self._is_model_planner:
            spec = self._manifest_by_name.get(tool)
            schema = spec.parameters if spec is not None else {}
            valid, detail = validate_tool_arguments(arguments, schema)
            if not valid:
                return ReActStep(
                    thought=f"consider {tool}",
                    tool=tool,
                    arguments=arguments,
                    authorized=False,
                    reason=f"argument validation failed: {detail}",
                )
        authorized, reason = self._governor.authorize(tool, caller_scopes=caller_scopes)
        skill = self._skills.get(tool) if authorized else None
        if not authorized or skill is None:
            detail = f"tool {tool!r} not available" if authorized else reason
            return ReActStep(
                thought=f"consider {tool}", tool=tool, arguments=arguments, authorized=False, reason=detail
            )
        capability = self._capabilities.get(tool)
        scope: SecretScope | None = None
        if capability is not None:
            # Admission runs after grant authorization and before the approval gate so a
            # capability_permits denial never consumes a single-use durable approval. Network and
            # namespace stay unchecked (None): they are skill-internal, not measurable here. A
            # declared required secret with no provider installed denies (available is the empty set),
            # so a declared "required" secret is never silently unenforced.
            available: frozenset[str] | None = None
            if capability.required_secrets:
                if secret_backend is None:
                    available = frozenset()
                else:
                    scope = SecretScope(
                        workspace_id=workspace_id,
                        capability_name=capability.name,
                        side_effect=capability.side_effect,
                        principal_id=None if principal is None else principal.id,
                        principal_kind=None if principal is None else principal.kind.value,
                        tenant=None if principal is None else principal.tenant,
                    )
                    available = secret_backend.available(capability, scope)
            permit = capability_permits(
                capability,
                granted_scopes=_effective_scopes(self._policy.granted_scopes, caller_scopes),
                side_effect_ceiling=self._policy.side_effect_ceiling,
                evidence_classification=evidence_classification,
                payload_bytes=(
                    len(canonical_json(arguments).encode()) if capability.max_payload_bytes is not None else None
                ),
                available_secrets=available,
            )
            if not permit.allowed:
                return ReActStep(
                    thought=f"consider {tool}",
                    tool=tool,
                    arguments=arguments,
                    authorized=False,
                    reason=permit.reason,
                )
        bundle: SecretBundle | None = None
        scrub_deferred = False
        try:
            if capability is not None and capability.required_secrets and secret_backend is not None:
                assert scope is not None
                # Materialize before the approval consume so an unresolved required secret (rotated or
                # deleted after the availability check) denies the step without burning a single-use
                # approval; the bundle lives only until skill.run, then scrub in the finally.
                bundle = secret_backend.materialize(capability, scope)
                if bundle is None:
                    return ReActStep(
                        thought=f"consider {tool}",
                        tool=tool,
                        arguments=arguments,
                        authorized=False,
                        reason="required_secret_unavailable:unresolved",
                    )
            if capability is not None and capability.requires_approval:
                denial = self._authorize_approval(
                    capability, arguments, principal, now, approvals_backend, action_version
                )
                if denial is not None:
                    return ReActStep(
                        thought=f"consider {tool}", tool=tool, arguments=arguments, authorized=False, reason=denial
                    )
            outbound = None
            if getattr(skill, "wants_outbound", False):
                # Governed outbound reach is an EXTERNAL side effect: require a capability declaring it, so
                # a READ_ONLY capability cannot send bytes and a no-capability skill cannot egress at all.
                if capability is None or capability.side_effect is not SideEffectClass.EXTERNAL:
                    return ReActStep(
                        thought=f"consider {tool}",
                        tool=tool,
                        arguments=arguments,
                        authorized=False,
                        reason="network_destination_denied:requires_external_capability",
                    )
                if outbound_gateway is None:
                    return ReActStep(
                        thought=f"consider {tool}",
                        tool=tool,
                        arguments=arguments,
                        authorized=False,
                        reason="network_destination_denied:governed_egress_unavailable",
                    )
                outbound = outbound_gateway.for_capability(capability, principal)

            if tool_timeout_seconds is None:
                observation = self._invoke_skill(
                    skill, context, arguments, principal=principal, secrets=bundle, outbound=outbound
                )
                return ReActStep(
                    thought=f"consider {tool}",
                    tool=tool,
                    arguments=arguments,
                    authorized=True,
                    reason=reason,
                    observation=observation,
                )

            def _invoke() -> JsonObject:
                return self._invoke_skill(
                    skill, context, arguments, principal=principal, secrets=bundle, outbound=outbound
                )

            # run_tool_within_deadline owns the bundle scrub in every path (via on_finish), so the outer
            # finally must not scrub even if the call is interrupted while a worker is still using it.
            scrub_deferred = True
            outcome = run_tool_within_deadline(
                _invoke,
                timeout=tool_timeout_seconds,
                max_workers=tool_timeout_max_workers,
                on_finish=(bundle.scrub if bundle is not None else None),
            )
            if not outcome.started:
                return ReActStep(
                    thought=f"consider {tool}",
                    tool=tool,
                    arguments=arguments,
                    authorized=False,
                    reason=TOOL_TIMEOUT_REASON_CONFIRMED,
                )
            if outcome.completed:
                if outcome.error is not None:
                    raise outcome.error
                return ReActStep(
                    thought=f"consider {tool}",
                    tool=tool,
                    arguments=arguments,
                    authorized=True,
                    reason=reason,
                    observation=cast(JsonObject, outcome.observation),
                )
            # Abandoned and not terminated, so the outcome is unknown. Severity is the tool's authoritative
            # declared side effect (the grant, widened by any capability), never capability-only, so a
            # grant-only write is not silently continued: a read tool answers from evidence, a write raises.
            if self._tool_side_effect(tool, capability) is SideEffectClass.READ_ONLY:
                return ReActStep(
                    thought=f"consider {tool}",
                    tool=tool,
                    arguments=arguments,
                    authorized=False,
                    reason=TOOL_TIMEOUT_REASON_UNKNOWN,
                )
            raise ToolTimeoutError(
                outcome=TOOL_OUTCOME_UNKNOWN,
                side_effect=self._tool_side_effect(tool, capability).value,
                tool=tool,
                execution_terminated=False,
            )
        finally:
            if bundle is not None and not scrub_deferred:
                bundle.scrub()

    def _secret_backend(self, project: RefaatProject) -> SecretResolver | None:
        # Only a governed program with capabilities and an installed provider resolves secrets; with
        # no provider this is None and the loop is byte-identical (a declared required secret then
        # fails closed inside _execute_tool rather than running unenforced).
        provider = project.runtime.secret_provider
        if not self._capabilities or provider is None:
            return None
        from refaat.application.services import in_unserved_run

        # An unserved shadow run serves nothing, so it must not emit a secret-resolution audit event,
        # matching how the query and RLM audit sinks are suppressed under in_unserved_run().
        audit_sink = None if in_unserved_run() else project.runtime.audit_sink
        return SecretResolver(
            provider,
            project.workspace_id,
            clock=project.runtime.store.clock,
            audit_sink=audit_sink,
            audit_fail_closed=project.runtime.settings.audit_fail_closed,
        )

    def _outbound_gateway(self, project: RefaatProject) -> OutboundGateway | None:
        # Only a governed program with capabilities builds a gateway; with none this is None and the loop
        # is byte-identical. No client is built until a wants_outbound tool actually egresses.
        if not self._capabilities:
            return None
        from refaat.application.services import in_unserved_run

        audit_sink = None if in_unserved_run() else project.runtime.audit_sink
        return OutboundGateway(
            project.workspace_id,
            egress_policy=project.runtime.egress_policy,
            clock=project.runtime.store.clock,
            audit_sink=audit_sink,
            audit_fail_closed=project.runtime.settings.audit_fail_closed,
            timeout=project.runtime.settings.request_timeout_seconds,
            max_bytes=project.runtime.settings.connector_max_bytes,
        )

    def _emit_step_observability(self, step: ReActStep, iteration: int) -> None:
        """Emit one redacted governed-step record for an ENHANCED capability, nothing otherwise.

        STANDARD (the default) is byte-identical on the observability path: no span, no log. Under
        ENHANCED the record carries only ids, counts, enums, and the opaque reason code; it never
        reads an argument value, an observation's content, or the step thought.
        """
        capability = self._capabilities.get(step.tool)
        if capability is None or capability.audit_level.rank <= AuditLevel.STANDARD.rank:
            return
        # dict[str, Any] so the shared field map unpacks into both the span (attributes) and the
        # structured logger, whose keyword parameters the logging stubs type individually.
        fields: dict[str, Any] = {
            "tool": step.tool,
            "capability": capability.name,
            "authorized": step.authorized,
            "reason_code": governance_reason_code(step.reason),
            "side_effect": str(capability.side_effect),
            "classification": int(capability.allowed_classification),
            "requires_approval": capability.requires_approval,
            "arguments_count": len(step.arguments),
            "observation_present": step.observation is not None,
            "iteration": iteration,
        }
        with span("refaat.tool.step", **fields):
            pass
        logger.info("governed_tool_step", **fields)

    def _run(
        self,
        project: RefaatProject,
        question: str,
        *,
        user_id: str | None,
        session_id: str | None,
        principal: Principal | None = None,
    ) -> ProgramResult:
        meter = active_meter()
        # No approval backend, clock reads, or artifact id when the policy declares no capabilities,
        # so a governed run without approvals stays strictly inert on the approval path. The action
        # version is the content-addressed artifact id, computed once per run since it is constant
        # for an immutable program.
        backend = (
            StoreApprovalBackend(
                project.runtime.store,
                project.workspace_id,
                audit_sink=project.runtime.audit_sink,
                audit_fail_closed=project.runtime.settings.audit_fail_closed,
                directory=project.runtime.principal_directory,
            )
            if self._capabilities
            else None
        )
        action_version = self.artifact().artifact_id if self._capabilities else None
        secret_backend = self._secret_backend(project)
        outbound_gateway = self._outbound_gateway(project)
        caller_scopes = self._caller_scopes(project, principal)
        context = project.retrieve(question, user_id=user_id, session_id=session_id, principal=principal)
        # Deterministic evidence-classification ceiling for capability admission, derived from the
        # retrieved chunks' ACL tags the same way the query egress gate does, never from the model.
        evidence_classification = (
            max_classification(hit.chunk.metadata for hit in context.hits)
            if self._capabilities
            else Classification.PUBLIC
        )
        steps: list[ReActStep] = []
        used: set[str] = set()
        try:
            for iteration in range(self._policy.max_iterations):
                # Stop starting new tool iterations once the meter is exhausted; the
                # final grounded answer below still runs, so an exact-exhaustion run
                # returns a grounded result rather than crashing.
                if meter is not None and meter.exhausted():
                    break
                proposal = self._planner.plan(question, frozenset(used))
                if proposal is None or not self._admissible(proposal.tool, used):
                    break
                used.add(proposal.tool)
                # Fresh clock per iteration so an approval whose TTL lapses during retrieval or planning
                # is not honored at the decision; read it only when approvals can apply.
                now = project.runtime.store.clock.now() if self._capabilities else None
                step = self._execute_tool(
                    context,
                    proposal,
                    principal=principal,
                    now=now,
                    approvals_backend=backend,
                    action_version=action_version,
                    evidence_classification=evidence_classification,
                    secret_backend=secret_backend,
                    outbound_gateway=outbound_gateway,
                    workspace_id=project.workspace_id,
                    kill_switches=project.runtime.kill_switches,
                    caller_scopes=caller_scopes,
                    tool_timeout_seconds=project.runtime.settings.tool_timeout_seconds,
                    tool_timeout_max_workers=project.runtime.settings.tool_timeout_max_workers,
                )
                steps.append(step)
                self._emit_step_observability(step, iteration)

            response = project.runtime.query(
                self._query_request(project, question, user_id=user_id, session_id=session_id, principal=principal)
            )
            result = self._result(response)
            return replace(result, outputs={**result.outputs, "react_steps": [self._step_json(step) for step in steps]})
        finally:
            if outbound_gateway is not None:
                outbound_gateway.close()

    def stream(
        self,
        project: RefaatProject,
        *,
        inputs: Mapping[str, Any],
        user_id: str | None,
        session_id: str | None,
        principal: Principal | None = None,
    ) -> Iterator[dict[str, Any]]:
        question = inputs.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValidationError("Program input 'question' is required and must be a non-empty string")

        settings = project.runtime.settings
        # The streamed governed loop is driven across per-iteration context copies,
        # so the run meter is resolved once and re-bound transiently around each
        # synchronous unit rather than held open across the yields by a budget_scope
        # whose ContextVar reset would break at the copy boundary.
        meter = resolve_stream_meter(
            settings.run_token_budget,
            settings.run_cost_budget_usd,
            prices=prices_from(settings.model_prices),
        )
        yield from self._stream(project, question, meter, user_id=user_id, session_id=session_id, principal=principal)

    def _stream(
        self,
        project: RefaatProject,
        question: str,
        meter: CostMeter | None,
        *,
        user_id: str | None,
        session_id: str | None,
        principal: Principal | None = None,
    ) -> Iterator[dict[str, Any]]:
        backend = (
            StoreApprovalBackend(
                project.runtime.store,
                project.workspace_id,
                audit_sink=project.runtime.audit_sink,
                audit_fail_closed=project.runtime.settings.audit_fail_closed,
                directory=project.runtime.principal_directory,
            )
            if self._capabilities
            else None
        )
        action_version = self.artifact().artifact_id if self._capabilities else None
        secret_backend = self._secret_backend(project)
        outbound_gateway = self._outbound_gateway(project)
        caller_scopes = self._caller_scopes(project, principal)
        with use_meter(meter):
            context = project.retrieve(question, user_id=user_id, session_id=session_id, principal=principal)
        evidence_classification = (
            max_classification(hit.chunk.metadata for hit in context.hits)
            if self._capabilities
            else Classification.PUBLIC
        )
        steps: list[ReActStep] = []
        used: set[str] = set()
        stream_plan = getattr(self._planner, "stream_plan", None)
        try:
            for iteration in range(self._policy.max_iterations):
                if meter is not None and meter.exhausted():
                    break
                frozen_used = frozenset(used)
                if self._is_model_planner and callable(stream_plan):
                    accumulator = _ToolCallAccumulator()
                    plan_stream = stream_plan(question, frozen_used)
                    while True:
                        with use_meter(meter):
                            try:
                                delta = next(plan_stream)
                            except StopIteration:
                                break
                        accumulator.feed(delta)
                        yield {
                            "type": "tool_call_delta",
                            "tool": delta.tool,
                            "arguments_fragment": delta.arguments_fragment,
                        }
                    proposal = accumulator.result()
                else:
                    # Offline default (keyword planner) and any non-streaming model
                    # planner emit completed steps only: no tool_call_delta is ever
                    # fabricated for a path that cannot honestly stream one.
                    with use_meter(meter):
                        proposal = self._planner.plan(question, frozen_used)
                if proposal is None or not self._admissible(proposal.tool, used):
                    break
                used.add(proposal.tool)
                now = project.runtime.store.clock.now() if self._capabilities else None
                with use_meter(meter):
                    step = self._execute_tool(
                        context,
                        proposal,
                        principal=principal,
                        now=now,
                        approvals_backend=backend,
                        action_version=action_version,
                        evidence_classification=evidence_classification,
                        secret_backend=secret_backend,
                        outbound_gateway=outbound_gateway,
                        workspace_id=project.workspace_id,
                        kill_switches=project.runtime.kill_switches,
                        caller_scopes=caller_scopes,
                        tool_timeout_seconds=project.runtime.settings.tool_timeout_seconds,
                        tool_timeout_max_workers=project.runtime.settings.tool_timeout_max_workers,
                    )
                steps.append(step)
                self._emit_step_observability(step, iteration)
                yield {"type": "tool_step", **self._public_step_json(step)}
        finally:
            if outbound_gateway is not None:
                outbound_gateway.close()

        react_steps = [self._public_step_json(step) for step in steps]
        answer_request = self._query_request(
            project, question, user_id=user_id, session_id=session_id, principal=principal
        )
        answer_stream = project.runtime.stream_query(answer_request)
        while True:
            with use_meter(meter):
                try:
                    event = next(answer_stream)
                except StopIteration:
                    break
            if event.get("type") == "final":
                yield {**event, "react_steps": react_steps}
            else:
                yield event

    @staticmethod
    def _step_json(step: ReActStep) -> JsonObject:
        return {
            "tool": step.tool,
            "arguments": step.arguments,
            "authorized": step.authorized,
            "reason": step.reason,
            "observation": step.observation,
        }

    @staticmethod
    def _public_step_json(step: ReActStep) -> JsonObject:
        # Streamed to the caller, so it omits the raw ungoverned observation the output governor never saw.
        return {
            "tool": step.tool,
            "arguments": step.arguments,
            "authorized": step.authorized,
            "reason": step.reason,
        }
