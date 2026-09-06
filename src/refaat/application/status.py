"""The unified admin status plane aggregator.

Reads the posture and health of every governance and operational subsystem from the already
composed runtime and projects each into a redacted SubsystemStatus. It is read-only: it opens no
write, mutates no state, and folds into no artifact id or cache identity, so it is byte-identical
to the rest of the runtime.

Two guarantees hold. Degrade-to-unknown-never-raise: every accessor is wrapped so one unreadable
subsystem reports UNKNOWN while the rest report normally and the plane still returns. Redaction by
projection: each accessor emits only counts, booleans, enum values, cache identifiers, and
timestamps, sourced from COUNT and boolean readers, never a secret, evidence text, a raw principal,
a reason, or a raw quota scope key.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from typing import TYPE_CHECKING

from refaat.domain.audit import AUDIT_SCHEMA_VERSION
from refaat.domain.kill_switch import KillSwitchScope
from refaat.domain.status import (
    AdminStatus,
    BuildInfo,
    JsonScalar,
    SchemaInfo,
    SubsystemState,
    SubsystemStatus,
)
from refaat.infrastructure.quota import DatabaseQuotaPolicy
from refaat.version import __version__

if TYPE_CHECKING:
    from refaat.application.runtime import Refaat

AUDIT_BACKLOG_STALE_SECONDS = 300

_Accessor = Callable[[str | None, bool], SubsystemStatus]
_Detail = dict[str, JsonScalar | tuple[JsonScalar, ...]]


class StatusService:
    def __init__(self, runtime: Refaat) -> None:
        self._rt = runtime

    def snapshot(self, *, workspace_id: str | None, operator: bool = False) -> AdminStatus:
        subs = tuple(self._safe(name, partial(fn, workspace_id, operator)) for name, fn in self._accessors())
        return AdminStatus(
            generated_at=datetime.now(UTC),
            scope="operator" if operator else "workspace",
            workspace_id=workspace_id,
            build=self._build_info(),
            schema=self._schema_info(),
            subsystems=subs,
        )

    @staticmethod
    def _safe(name: str, fn: Callable[[], SubsystemStatus]) -> SubsystemStatus:
        try:
            return fn()
        except Exception:
            return SubsystemStatus.unknown(name)

    def _accessors(self) -> tuple[tuple[str, _Accessor], ...]:
        return (
            ("kill_switches", self._kill_switches),
            ("quotas", self._quotas),
            ("audit", self._audit),
            ("legal_holds", self._legal_holds),
            ("approvals", self._approvals),
            ("jobs", self._jobs),
            ("connectors", self._connectors),
            ("degradation", self._degradation),
            ("model_routing", self._model_routing),
            ("output_governance", self._output_governance),
            ("egress", self._egress),
            ("egress_policy", self._egress_policy),
            ("source_trust", self._source_trust),
            ("identity", self._identity),
            ("plugin_pins", self._plugin_pins),
            ("secrets", self._secrets),
        )

    def _kill_switches(self, workspace_id: str | None, operator: bool) -> SubsystemStatus:
        registry = self._rt.kill_switches
        if registry is None:
            return SubsystemStatus.not_configured("kill_switches")
        snapshot = registry.engaged()
        ws = workspace_id or ""
        frozen = snapshot.freezes(ws)
        if not snapshot.available:
            state = SubsystemState.DEGRADED
        elif frozen:
            state = SubsystemState.ENGAGED
        else:
            state = SubsystemState.ACTIVE
        detail: _Detail = {"workspace_frozen": frozen}
        for scope in KillSwitchScope:
            detail[f"denies_{scope.value}"] = snapshot.denies(scope, ws)
        if operator:
            detail["backend"] = self._rt.settings.kill_switch_backend
            detail["management_enabled"] = self._rt.store.kill_switch_management_enabled
            try:
                detail["global_engaged_count"] = self._rt.store.count_active_kill_switches(None)
            except Exception:
                state = SubsystemState.DEGRADED
        return SubsystemStatus(name="kill_switches", state=state, configured=True, detail=detail)

    def _quotas(self, workspace_id: str | None, operator: bool) -> SubsystemStatus:
        policy = self._rt.quota_policy
        if policy is None:
            return SubsystemStatus.not_configured("quotas")
        detail: _Detail = {}
        rules: tuple[object, ...] = ()
        if isinstance(policy, DatabaseQuotaPolicy):
            rules = policy.rules
        detail["rule_count"] = len(rules)
        state = SubsystemState.ACTIVE
        if workspace_id:
            try:
                for row in self._rt.store.quota_usage_for_workspace(workspace_id):
                    dimension = str(row["dimension"])
                    detail[f"spent_{dimension}"] = int(row["spent"])
                    detail[f"reserved_{dimension}"] = int(row["reserved"])
            except Exception:
                state = SubsystemState.DEGRADED
        if operator:
            detail["backend"] = policy.cache_id
            detail["rules"] = tuple(
                f"{getattr(rule, 'scope', '')}:{getattr(rule, 'dimension', '')}:"
                f"{getattr(rule, 'limit', '')}:{getattr(rule, 'window_seconds', '')}"
                for rule in rules
            )
        return SubsystemStatus(name="quotas", state=state, configured=True, detail=detail)

    def _audit(self, workspace_id: str | None, operator: bool) -> SubsystemStatus:
        sink = self._rt.audit_sink
        if sink is None:
            return SubsystemStatus.not_configured("audit")
        settings = self._rt.settings
        durable = settings.audit_durable
        detail: _Detail = {"durable": durable, "fail_closed": settings.audit_fail_closed}
        state = SubsystemState.ACTIVE
        if durable:
            try:
                depth, oldest_age, _ = self._rt.store.audit_outbox_backlog(workspace_id if not operator else None)
                stuck = self._rt.store.count_stuck_audit(workspace_id if not operator else None)
                detail["backlog_depth"] = depth
                detail["oldest_undelivered_age_s"] = oldest_age
                detail["stuck_count"] = stuck
                if stuck > 0 or (oldest_age is not None and oldest_age > AUDIT_BACKLOG_STALE_SECONDS):
                    state = SubsystemState.DEGRADED
            except Exception:
                state = SubsystemState.DEGRADED
        if operator:
            detail["schema_version"] = AUDIT_SCHEMA_VERSION
            detail["delivery_sink_count"] = len(self._rt.audit_delivery_sinks)
        return SubsystemStatus(name="audit", state=state, configured=True, detail=detail)

    def _legal_holds(self, workspace_id: str | None, operator: bool) -> SubsystemStatus:
        state = SubsystemState.ACTIVE
        detail: _Detail = {"management_enabled": self._rt.store.legal_hold_management_enabled}
        try:
            detail["active_hold_count"] = self._rt.store.count_active_legal_holds(workspace_id)
        except Exception:
            state = SubsystemState.DEGRADED
        return SubsystemStatus(name="legal_holds", state=state, configured=True, detail=detail)

    def _approvals(self, workspace_id: str | None, operator: bool) -> SubsystemStatus:
        state = SubsystemState.ACTIVE
        detail: _Detail = {}
        try:
            detail["active_grant_count"] = self._rt.store.count_active_tool_approvals(workspace_id)
            detail["open_request_count"] = self._rt.store.count_open_approval_requests(workspace_id)
        except Exception:
            state = SubsystemState.DEGRADED
        return SubsystemStatus(name="approvals", state=state, configured=True, detail=detail)

    def _jobs(self, workspace_id: str | None, operator: bool) -> SubsystemStatus:
        try:
            pending, running, stuck, failed = self._rt.store.job_posture(workspace_id)
        except Exception:
            return SubsystemStatus(name="jobs", state=SubsystemState.DEGRADED, configured=True, detail={})
        state = SubsystemState.DEGRADED if stuck > 0 else SubsystemState.ACTIVE
        detail: _Detail = {
            "pending": pending,
            "running": running,
            "stuck_lease": stuck,
            "failed": failed,
        }
        return SubsystemStatus(name="jobs", state=state, configured=True, detail=detail)

    def _connectors(self, workspace_id: str | None, operator: bool) -> SubsystemStatus:
        schemes = tuple(sorted(self._rt.connectors.schemes))
        detail: _Detail = {}
        if operator:
            detail["scheme_count"] = len(schemes)
            detail["schemes"] = schemes
        return SubsystemStatus(
            name="connectors",
            state=SubsystemState.ACTIVE if schemes else SubsystemState.NOT_CONFIGURED,
            configured=bool(schemes),
            detail=detail,
        )

    def _degradation(self, workspace_id: str | None, operator: bool) -> SubsystemStatus:
        policy = self._rt.degradation_policy
        if policy is None:
            return SubsystemStatus.not_configured("degradation")
        detail: _Detail = {}
        if operator:
            detail["policy_id"] = policy.cache_id
        return SubsystemStatus(name="degradation", state=SubsystemState.ACTIVE, configured=True, detail=detail)

    def _model_routing(self, workspace_id: str | None, operator: bool) -> SubsystemStatus:
        router = self._rt.queries.model_router
        if router is None:
            return SubsystemStatus.not_configured("model_routing")
        detail: _Detail = {}
        if operator:
            detail["cache_id"] = router.cache_id
        return SubsystemStatus(name="model_routing", state=SubsystemState.ACTIVE, configured=True, detail=detail)

    def _output_governance(self, workspace_id: str | None, operator: bool) -> SubsystemStatus:
        governor = self._rt.queries.output_governor
        if governor is None:
            return SubsystemStatus.not_configured("output_governance")
        detail: _Detail = {}
        if operator:
            detail["cache_id"] = governor.cache_id
        return SubsystemStatus(name="output_governance", state=SubsystemState.ACTIVE, configured=True, detail=detail)

    def _egress(self, workspace_id: str | None, operator: bool) -> SubsystemStatus:
        transform = self._rt.queries.context_transform
        if transform is None:
            return SubsystemStatus.not_configured("egress")
        detail: _Detail = {}
        if operator:
            detail["cache_id"] = transform.cache_id
        return SubsystemStatus(name="egress", state=SubsystemState.ACTIVE, configured=True, detail=detail)

    def _egress_policy(self, workspace_id: str | None, operator: bool) -> SubsystemStatus:
        policy = self._rt.egress_policy
        if policy is None:
            return SubsystemStatus.not_configured("egress_policy")
        detail: _Detail = {}
        if operator and policy.cache_id is not None:
            detail["cache_id"] = policy.cache_id
        return SubsystemStatus(name="egress_policy", state=SubsystemState.ACTIVE, configured=True, detail=detail)

    def _source_trust(self, workspace_id: str | None, operator: bool) -> SubsystemStatus:
        weight = self._rt.settings.source_trust_weight
        if weight <= 0.0:
            return SubsystemStatus.not_configured("source_trust")
        detail: _Detail = {}
        if operator:
            detail["weight"] = weight
        return SubsystemStatus(name="source_trust", state=SubsystemState.ACTIVE, configured=True, detail=detail)

    def _identity(self, workspace_id: str | None, operator: bool) -> SubsystemStatus:
        resolver = self._rt.principal_resolver
        directory = self._rt.principal_directory
        policy = self._rt.access_policy
        if resolver is None and directory is None and policy is None:
            return SubsystemStatus.not_configured("identity")
        detail: _Detail = {"authorization_active": policy is not None}
        if operator:
            detail["resolver_installed"] = resolver is not None
            detail["directory_installed"] = directory is not None
            detail["access_policy_installed"] = policy is not None
        return SubsystemStatus(name="identity", state=SubsystemState.ACTIVE, configured=True, detail=detail)

    def _plugin_pins(self, workspace_id: str | None, operator: bool) -> SubsystemStatus:
        settings = self._rt.settings
        pins = settings.plugin_pins
        allowlist = settings.plugin_allowlist
        if not pins and not allowlist:
            return SubsystemStatus.not_configured("plugin_pins")
        detail: _Detail = {"pinning_enforced": bool(pins)}
        if operator:
            detail["pin_count"] = len(pins)
            detail["allowlist_size"] = len(allowlist)
        return SubsystemStatus(name="plugin_pins", state=SubsystemState.ACTIVE, configured=True, detail=detail)

    def _secrets(self, workspace_id: str | None, operator: bool) -> SubsystemStatus:
        provider = self._rt.secret_provider
        if provider is None:
            return SubsystemStatus.not_configured("secrets")
        detail: _Detail = {}
        if operator:
            detail["provider_kind"] = self._rt.settings.tool_secret_provider
        return SubsystemStatus(name="secrets", state=SubsystemState.ACTIVE, configured=True, detail=detail)

    def _build_info(self) -> BuildInfo:
        try:
            rt = self._rt
            settings = rt.settings
            features = {
                "kill_switches": rt.kill_switches is not None,
                "quotas": rt.quota_policy is not None,
                "audit": rt.audit_sink is not None,
                "degradation": rt.degradation_policy is not None,
                "model_routing": rt.queries.model_router is not None,
                "output_governance": rt.queries.output_governor is not None,
                "egress": rt.queries.context_transform is not None,
                "egress_policy": rt.egress_policy is not None,
                "source_trust": settings.source_trust_weight > 0.0,
                "authorization": rt.access_policy is not None,
                "identity_resolver": rt.principal_resolver is not None,
                "plugin_pinning": bool(settings.plugin_pins),
                "secrets": rt.secret_provider is not None,
                "legal_hold_management": rt.store.legal_hold_management_enabled,
                "connectors": bool(rt.connectors.schemes),
            }
            return BuildInfo(
                version=__version__,
                environment=settings.environment,
                features=features,
                state=SubsystemState.ACTIVE,
            )
        except Exception:
            return BuildInfo(
                version=__version__,
                environment="",
                features={},
                state=SubsystemState.UNKNOWN,
            )

    def _schema_info(self) -> SchemaInfo:
        try:
            db_revision = self._rt.store.schema_revision()
        except Exception:
            return SchemaInfo(
                code_head=None, db_revision=None, at_head=False, managed=False, state=SubsystemState.UNKNOWN
            )
        code_head = _derive_code_head()
        if db_revision is None:
            return SchemaInfo(
                code_head=code_head, db_revision=None, at_head=False, managed=False, state=SubsystemState.UNMANAGED
            )
        at_head = code_head is not None and db_revision == code_head
        return SchemaInfo(
            code_head=code_head,
            db_revision=db_revision,
            at_head=at_head,
            managed=True,
            state=SubsystemState.ACTIVE,
        )


def _derive_code_head() -> str | None:
    """The migration head from the alembic script tree, or None when it is unreadable.

    Best-effort: an installed deployment that ships no migration tree falls back to None cleanly
    rather than raising, so the plane still returns.
    """
    try:
        from alembic.config import Config
        from alembic.script import ScriptDirectory

        return ScriptDirectory.from_config(Config("alembic.ini")).get_current_head()
    except Exception:
        return None


__all__ = ["StatusService"]
