from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request

from refaat.application.approvals import ApprovalContext, active_approval_context
from refaat.application.runtime import Refaat
from refaat.domain.authz import Principal as AuthzPrincipal
from refaat.errors import AuthenticationError, AuthorizationError, KillSwitchEngagedError


@dataclass(frozen=True, slots=True)
class Principal:
    workspace_id: str
    scopes: frozenset[str]
    api_key_id: str


def get_runtime(request: Request) -> Refaat:
    runtime: Refaat = request.app.state.runtime
    return runtime


def get_principal(request: Request, runtime: Annotated[Refaat, Depends(get_runtime)]) -> Principal:
    authorization = request.headers.get("authorization", "")
    raw = request.headers.get("x-api-key")
    if authorization.lower().startswith("bearer "):
        raw = authorization[7:].strip()
    if not raw:
        raise AuthenticationError("API key required")
    workspace_id, scopes, key_id = runtime.authentication.authenticate(raw)
    # The single HTTP-edge freeze chokepoint: every /v1 route funnels through a *Principal dependency,
    # so a frozen workspace's whole HTTP surface is dark, including its own kill-switch routes. A
    # freeze is lifted only by the operator (CLI or direct database), never by a workspace credential.
    registry = runtime.kill_switches
    if registry is not None and registry.engaged().freezes(workspace_id):
        raise KillSwitchEngagedError("This workspace is frozen by an incident kill switch")
    return Principal(workspace_id=workspace_id, scopes=scopes, api_key_id=key_id)


def require_scope(scope: str) -> Callable[..., Principal]:
    def dependency(principal: Annotated[Principal, Depends(get_principal)]) -> Principal:
        if scope not in principal.scopes and "admin" not in principal.scopes:
            raise AuthorizationError(f"Scope required: {scope}")
        return principal

    return dependency


def _resolve_principal(request: Request, runtime: Refaat, principal: Principal) -> AuthzPrincipal | None:
    resolver = runtime.principal_resolver
    if resolver is None:
        return None
    context = {
        "workspace_id": principal.workspace_id,
        "scopes": sorted(principal.scopes),
        "api_key_id": principal.api_key_id,
        "headers": {key.lower(): value for key, value in request.headers.items()},
    }
    return resolver.resolve(context)


def resolve_identity(
    request: Request,
    runtime: Annotated[Refaat, Depends(get_runtime)],
    principal: Annotated[Principal, Depends(get_principal)],
) -> AuthzPrincipal | None:
    """Resolve the request's authorization principal from trusted server-side context.

    Identity comes only from the authenticated context and the resolver, never from the
    request body. With no access policy installed the HTTP path carries no principal and is
    byte-identical to the unauthorized default. When an access policy is installed the request
    is in authorization mode, so a request that resolves to no identity is denied rather than
    served unfiltered (a resolver is how identity is supplied).
    """
    identity = _resolve_principal(request, runtime, principal)
    if identity is None and runtime.access_policy is not None:
        raise AuthorizationError("A resolved identity is required")
    return identity


def resolve_identity_optional(
    request: Request,
    runtime: Annotated[Refaat, Depends(get_runtime)],
    principal: Annotated[Principal, Depends(get_principal)],
) -> AuthzPrincipal | None:
    """Resolve the identity for audit attribution only, never fail closed and never raise.

    Admin-scoped management routes are authorized by the admin api key, so they must still act with
    no resolver installed or when the resolver errors; they attribute an actor when one resolves and
    otherwise proceed unattributed.
    """
    try:
        return _resolve_principal(request, runtime, principal)
    except Exception:
        return None


def deny_program_surface_under_policy(
    runtime: Annotated[Refaat, Depends(get_runtime)],
    principal: Annotated[Principal, Depends(get_principal)],
) -> None:
    """Fail-closed interim for the skill-run and program-evaluate surfaces under an active policy.

    These endpoints run retrieval but do not yet thread the request principal, so they fail
    closed rather than serve unfiltered evidence. The program run, run-stream, and deployment-run
    surfaces now thread the resolved principal end to end and enforce per-principal ACLs, so they
    resolve identity instead of using this guard. Inert with no policy installed.
    """
    if runtime.access_policy is not None:
        raise AuthorizationError("This endpoint is unavailable while an access policy is active")


def deny_ambient_approvals() -> None:
    """Reject any non-default ambient approval context on an untrusted surface.

    The in-memory ApprovalContext is a trusted in-process channel that no HTTP request populates, so
    any deviation from the default at the HTTP boundary would mean request data was piped into it. The
    check is on the whole context, not just its approvals, so a custom policy or action version cannot
    slip through and change the fingerprint. Fail closed rather than downgrade the approval trust
    boundary.
    """
    if active_approval_context() != ApprovalContext():
        raise AuthorizationError("Ambient approvals are not accepted on this surface")


ProgramSurfaceGuard = Annotated[None, Depends(deny_program_surface_under_policy)]
AmbientApprovalGuard = Annotated[None, Depends(deny_ambient_approvals)]
ReadPrincipal = Annotated[Principal, Depends(require_scope("read"))]
WritePrincipal = Annotated[Principal, Depends(require_scope("write"))]
AdminPrincipal = Annotated[Principal, Depends(require_scope("admin"))]
Identity = Annotated[AuthzPrincipal | None, Depends(resolve_identity)]
OptionalIdentity = Annotated[AuthzPrincipal | None, Depends(resolve_identity_optional)]
RuntimeDependency = Annotated[Refaat, Depends(get_runtime)]
