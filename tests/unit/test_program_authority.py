"""A program's self-declared grants are authority only when the calling principal holds them.

The tool governor intersects the caller's effectively-held scopes with the program's granted scopes, and
the HTTP program surface refuses a side-effectful program to a read-scoped caller.
"""

from __future__ import annotations

import pytest

from refaat.api.dependencies import Principal as ApiPrincipal
from refaat.api.routes import _require_side_effect_scope
from refaat.domain.authz import Principal, RoleGroupScopeResolver
from refaat.domain.enums import SideEffectClass
from refaat.domain.models import ReActPolicy, ToolGrant
from refaat.errors import AuthorizationError
from refaat.programs.program import GroundedRagProgram
from refaat.programs.react import GovernedReActProgram, ToolGovernor, _effective_scopes


def _policy(required: tuple[str, ...], granted: tuple[str, ...], side_effect: SideEffectClass) -> ReActPolicy:
    grant = ToolGrant(skill_name="lookup", required_scopes=required, side_effect=side_effect)
    return ReActPolicy(grants=(grant,), granted_scopes=granted, side_effect_ceiling=SideEffectClass.WRITE)


def test_effective_scopes_intersects_or_passes_through() -> None:
    granted = ("read", "write")
    assert _effective_scopes(granted, None) == granted  # no caller scopes: byte-identical
    assert _effective_scopes(granted, frozenset({"read"})) == ("read",)  # intersection drops write
    assert _effective_scopes(granted, frozenset()) == ()


def test_governor_admits_only_when_the_caller_holds_the_required_scope() -> None:
    governor = ToolGovernor(_policy(("write",), ("write",), SideEffectClass.READ_ONLY))
    # No caller scopes threaded: byte-identical to before, admitted because the policy grants write.
    admitted, _ = governor.authorize("lookup")
    assert admitted is True
    # Caller lacks write: denied (least-privilege), even though the program declares it.
    denied, reason = governor.authorize("lookup", caller_scopes=frozenset({"read"}))
    assert denied is False
    assert "missing scopes" in reason
    # Caller holds write: admitted.
    ok, _ = governor.authorize("lookup", caller_scopes=frozenset({"write"}))
    assert ok is True


def test_role_group_resolver_maps_entitlements_to_scopes() -> None:
    principal = Principal(id="p", roles=frozenset({"write"}), groups=frozenset({"research-a"}))
    assert RoleGroupScopeResolver().scopes_for(principal) == frozenset({"write", "research-a"})


def test_declared_side_effect_reflects_the_admissible_grants() -> None:
    assert GroundedRagProgram().declared_side_effect() is SideEffectClass.READ_ONLY
    governed = GovernedReActProgram(
        policy=_policy(("write",), ("write",), SideEffectClass.WRITE),
        planner=_planner(),
        skills={},
    )
    assert governed.declared_side_effect() is SideEffectClass.WRITE


def _planner() -> object:
    from refaat.programs.react import KeywordToolPlanner

    return KeywordToolPlanner({})


class _WriteProgram:
    def declared_side_effect(self) -> SideEffectClass:
        return SideEffectClass.WRITE


class _ReadProgram:
    def declared_side_effect(self) -> SideEffectClass:
        return SideEffectClass.READ_ONLY


def _api_principal(*scopes: str) -> ApiPrincipal:
    return ApiPrincipal(workspace_id="w", scopes=frozenset(scopes), api_key_id="k")


def test_undeclared_custom_program_is_side_effectful_by_default() -> None:
    # Safe-by-default: a custom program that supplies its own execute() but does not assert a side-effect
    # profile is treated as maximally side-effectful, so the HTTP gate refuses it to a read-scoped caller.
    from refaat.programs.program import Program

    class _Undeclared(Program):
        name = "undeclared"
        input_signature = GroundedRagProgram.input_signature
        output_signature = GroundedRagProgram.output_signature

        def execute(self, project, *, inputs, user_id, session_id, principal=None):  # type: ignore[override]
            raise NotImplementedError

    assert _Undeclared().declared_side_effect() is SideEffectClass.EXTERNAL
    with pytest.raises(AuthorizationError):
        _require_side_effect_scope(_Undeclared(), _api_principal("read"))  # type: ignore[arg-type]
    # The pure built-ins assert read-only, so the stock read-only surface stays byte-identical.
    assert GroundedRagProgram().declared_side_effect() is SideEffectClass.READ_ONLY

    # A subclass of a pure program that overrides execute() no longer inherits the pure assertion.
    class _SneakyRag(GroundedRagProgram):
        def execute(self, project, *, inputs, user_id, session_id, principal=None):  # type: ignore[override]
            raise NotImplementedError

    assert _SneakyRag().declared_side_effect() is SideEffectClass.EXTERNAL
    assert _SneakyRag().shadow_safe is False
    with pytest.raises(AuthorizationError):
        _require_side_effect_scope(_SneakyRag(), _api_principal("read"))  # type: ignore[arg-type]


def test_declared_side_effect_folds_in_capabilities() -> None:
    # A WRITE capability under a READ_ONLY grant must report WRITE, since capability admission can execute
    # the stronger side effect; otherwise the HTTP gate would admit a read-only caller to a write program.
    from refaat.domain.governance import ToolCapability

    grant = ToolGrant(skill_name="w", required_scopes=("read",), side_effect=SideEffectClass.READ_ONLY)
    policy = ReActPolicy(
        grants=(grant,),
        capabilities=(ToolCapability(name="w", side_effect=SideEffectClass.WRITE),),
        granted_scopes=("read",),
        side_effect_ceiling=SideEffectClass.WRITE,
    )
    governed = GovernedReActProgram(policy=policy, planner=_planner(), skills={})
    assert governed.declared_side_effect() is SideEffectClass.WRITE


def test_write_capability_makes_a_program_not_shadow_safe() -> None:
    # shadow_safe must account for capability side effects, not only grants: a WRITE capability executes a
    # real write a suppression scope cannot undo, so the program is not shadow-safe and cannot be canaried.
    from refaat.domain.governance import ToolCapability

    read_only = GovernedReActProgram(
        policy=_policy(("read",), ("read",), SideEffectClass.READ_ONLY), planner=_planner(), skills={}
    )
    assert read_only.shadow_safe is True
    with_write_cap = GovernedReActProgram(
        policy=ReActPolicy(
            grants=(ToolGrant(skill_name="c", required_scopes=("read",), side_effect=SideEffectClass.READ_ONLY),),
            capabilities=(ToolCapability(name="c", side_effect=SideEffectClass.WRITE),),
            granted_scopes=("read",),
            side_effect_ceiling=SideEffectClass.WRITE,
        ),
        planner=_planner(),
        skills={},
    )
    assert with_write_cap.shadow_safe is False


def test_handoff_orchestrator_declares_a_write_and_is_not_shadow_safe() -> None:
    # A handoff creates durable agents and sends messages, so a handoff orchestrator is a write even over
    # purely read-only members and cannot be admitted to a read-scoped caller.
    from refaat.programs.orchestrator import MultiAgentOrchestrator, TaskRouter

    members = {"a": GroundedRagProgram(), "b": GroundedRagProgram()}
    router = TaskRouter({"a": ("x",), "b": ("y",)})
    handoff = MultiAgentOrchestrator(members, router, handoff=True)
    assert handoff.declared_side_effect() is SideEffectClass.WRITE
    assert handoff.shadow_safe is False
    plain = MultiAgentOrchestrator(members, router, handoff=False)
    assert plain.declared_side_effect() is SideEffectClass.READ_ONLY


def test_wrappers_delegate_declared_side_effect_to_the_inner_program() -> None:
    # A wrapper must not hide an inner governed program's side effects; the HTTP gate examines the outer
    # program, so every wrapper delegates declared_side_effect to what it wraps.
    from refaat.programs.policy import BoundedInferencePolicy

    governed = GovernedReActProgram(
        policy=_policy(("write",), ("write",), SideEffectClass.WRITE),
        planner=_planner(),
        skills={},
    )
    assert BoundedInferencePolicy(governed).declared_side_effect() is SideEffectClass.WRITE
    assert BoundedInferencePolicy(GroundedRagProgram()).declared_side_effect() is SideEffectClass.READ_ONLY


def test_every_program_executing_route_applies_the_side_effect_gate() -> None:
    # Every HTTP route that runs a program (executes its pipeline) must gate side effects, so a future
    # program-executing route cannot silently skip the check.
    import inspect

    from refaat.api import routes

    for name in ("run_program", "run_program_stream", "run_deployment", "evaluate_program", "enqueue_optimization"):
        source = inspect.getsource(getattr(routes, name))
        assert "_require_side_effect_scope" in source, name


def test_require_side_effect_scope_gates_the_http_surface() -> None:
    # A read-only program is served to any caller (byte-identical for the stock grounded-rag surface).
    _require_side_effect_scope(_ReadProgram(), _api_principal("read"))  # type: ignore[arg-type]
    # A side-effectful program is refused a read-scoped caller.
    with pytest.raises(AuthorizationError):
        _require_side_effect_scope(_WriteProgram(), _api_principal("read"))  # type: ignore[arg-type]
    # A write- or admin-scoped caller is admitted.
    _require_side_effect_scope(_WriteProgram(), _api_principal("read", "write"))  # type: ignore[arg-type]
    _require_side_effect_scope(_WriteProgram(), _api_principal("admin"))  # type: ignore[arg-type]


def test_governed_react_caller_scopes_fail_closed_without_an_entitlement_resolver(settings: object) -> None:
    # A governed program's caller-scope resolution mirrors the MCP bridge: with an active policy under strict
    # authorization and a resolved principal but no entitlement resolver, the scopes are undeterminable, so it
    # fails closed to no scopes rather than silently granting the full tool surface.
    from refaat.application.runtime import RuntimeBuilder
    from refaat.domain.authz import CanonicalAccessPolicy, RoleGroupScopeResolver

    alice = Principal(id="alice", roles=frozenset({"read"}))
    program = GovernedReActProgram(
        policy=_policy(("read",), ("read",), SideEffectClass.READ_ONLY), planner=_planner(), skills={}
    )

    enforced = RuntimeBuilder(settings).with_access_policy(CanonicalAccessPolicy()).build()  # type: ignore[arg-type]
    try:
        project = enforced.project(workspace="w", corpus="c")
        assert program._caller_scopes(project, alice) == frozenset()
        assert program._caller_scopes(project, None) is None
    finally:
        enforced.close()

    with_resolver = (
        RuntimeBuilder(settings)  # type: ignore[arg-type]
        .with_access_policy(CanonicalAccessPolicy())
        .with_entitlement_scope_resolver(RoleGroupScopeResolver())
        .build()
    )
    try:
        project = with_resolver.project(workspace="w", corpus="c")
        assert program._caller_scopes(project, alice) == frozenset({"read"})
    finally:
        with_resolver.close()

    plain = RuntimeBuilder(settings).build()  # type: ignore[arg-type]
    try:
        project = plain.project(workspace="w", corpus="c")
        assert program._caller_scopes(project, alice) is None
    finally:
        plain.close()
