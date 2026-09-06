from __future__ import annotations

import re
from contextlib import AbstractContextManager
from pathlib import Path

from refaat.application.budget import CostMeter, budget_scope, prices_from
from refaat.application.quota_charge import charge_served_request
from refaat.canonical import canonical_json
from refaat.config import Settings
from refaat.core.harness import HarnessService
from refaat.core.retrieval import AdaptiveRetriever
from refaat.domain.audit import AuditSink
from refaat.domain.authz import Classification, max_classification, preference_scope, require_principal_present
from refaat.domain.governance import DirectSkillPolicy, capability_permits
from refaat.domain.kill_switch import KillSwitchRegistry, KillSwitchScope, ensure_scope_active, ensure_workspace_active
from refaat.domain.models import ContextPacket, JsonObject, QueryRequest
from refaat.domain.quota import QuotaDimension, QuotaPolicy
from refaat.errors import AuthorizationError, ConflictError, NotFoundError, ValidationError
from refaat.infrastructure.database.store import SQLStore
from refaat.infrastructure.plugins import PluginRegistry
from refaat.ports import Embedder, Skill
from refaat.security import safe_join
from refaat.skills.builtin import CorpusHealthSkill
from refaat.version import __version__

SKILL_NAME = re.compile(r"^[a-z][a-z0-9_]{2,63}$")


class SkillRegistry:
    def __init__(self, plugin_registry: PluginRegistry | None = None) -> None:
        self._skills: dict[str, Skill] = {}
        self.register(CorpusHealthSkill())
        for name, plugin in (plugin_registry or PluginRegistry(set())).discover("refaat.skills").items():
            skill = plugin() if isinstance(plugin, type) else plugin
            if getattr(skill, "name", name) in self._skills:
                continue
            self.register(skill)

    def register(self, skill: Skill) -> None:
        if skill.name in self._skills:
            raise ConflictError(f"Skill already registered: {skill.name}")
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill:
        try:
            return self._skills[name]
        except KeyError as exc:
            raise NotFoundError(f"Skill not found: {name}") from exc

    def list(self) -> list[JsonObject]:
        return [
            {"name": skill.name, "description": skill.description}
            for skill in sorted(self._skills.values(), key=lambda value: value.name)
        ]


class SkillService:
    def __init__(
        self,
        *,
        registry: SkillRegistry,
        store: SQLStore,
        embedder: Embedder,
        retriever: AdaptiveRetriever,
        harness: HarnessService,
        settings: Settings,
        direct_skill_policy: DirectSkillPolicy | None = None,
        kill_switches: KillSwitchRegistry | None = None,
        quota_policy: QuotaPolicy | None = None,
        audit_sink: AuditSink | None = None,
    ) -> None:
        self.registry = registry
        self.store = store
        self.embedder = embedder
        self.retriever = retriever
        self.harness = harness
        self.settings = settings
        self.kill_switches = kill_switches
        self.quota_policy = quota_policy
        self.audit_sink = audit_sink
        self._direct_skill_policy = direct_skill_policy
        self._capabilities = (
            {capability.name: capability for capability in direct_skill_policy.capabilities}
            if direct_skill_policy is not None
            else {}
        )

    def _budget_scope(self) -> AbstractContextManager[CostMeter | None]:
        return budget_scope(
            self.settings.run_token_budget,
            self.settings.run_cost_budget_usd,
            prices=prices_from(self.settings.model_prices),
        )

    def prepare(self, request: QueryRequest) -> ContextPacket:
        ensure_workspace_active(self.kill_switches, request.workspace_id)
        # Fail closed before the embedder is called, so a principal-less strict call never reaches a
        # model. The retriever repeats the check as defense in depth.
        require_principal_present(
            self.retriever.access_policy, request.principal, strict=self.retriever.strict_authorization
        )
        with self._budget_scope():
            self.store.get_corpus(request.workspace_id, request.corpus_id)
            query_embedding = self.embedder.embed([request.query])[0]
            profile = self.store.get_preferences(
                request.workspace_id,
                request.corpus_id,
                preference_scope(self.retriever.access_policy, request.principal, request.user_id),
            )
            supplemental = self.harness.active(request.workspace_id, request.corpus_id)
            return self.retriever.retrieve(
                request,
                query_embedding,
                profile=profile,
                supplemental_harness=supplemental,
            )

    def run(self, name: str, request: QueryRequest, arguments: JsonObject | None = None) -> JsonObject:
        ensure_workspace_active(self.kill_switches, request.workspace_id)
        # Running a skill directly is a tool execution, so a tool-execution switch denies it; the direct
        # path has nothing to continue to, so it raises rather than recording a step like the ReAct loop.
        ensure_scope_active(self.kill_switches, request.workspace_id, KillSwitchScope.TOOL_EXECUTION)
        # Resolve the skill (raises for an unknown name) before the charge, so a bad name is not billed.
        skill = self.registry.get(name)
        if getattr(skill, "wants_outbound", False):
            # The direct path (HTTP run_skill, the MCP bridge, in-process run_skill) has no governed
            # outbound client, so a skill that performs governed outbound HTTP must run as a tool inside a
            # program, where the run-scoped gateway audits and bounds every request.
            raise AuthorizationError(
                f"skill {name!r} performs governed outbound HTTP and cannot run on the direct path; "
                "invoke it as a governed tool inside a program"
            )
        # A principal-less call under an active policy in strict mode is denied before the quota charge, so
        # an unauthorized request is never billed and the denial is an AuthorizationError, not a spurious
        # QuotaExceededError. prepare() repeats the check as defense in depth.
        require_principal_present(
            self.retriever.access_policy, request.principal, strict=self.retriever.strict_authorization
        )
        policy = self._direct_skill_policy
        capability = self._capabilities.get(name)
        if capability is None or policy is None:
            # No direct-skill policy governs this skill, so there is no later denial to precede the charge.
            self._charge_skill_request(request)
            return skill.run(self.prepare(request), arguments or {})
        # Every direct-skill-policy denial runs before the charge, so a policy-denied call is never billed.
        if capability.requires_approval:
            raise AuthorizationError(
                f"skill {name!r} requires approval and cannot run on the direct path; "
                "invoke it as a governed tool inside a program"
            )
        if (
            capability.allowed_classification < Classification.RESTRICTED
            and self.retriever.access_policy is not None
            and request.principal is None
        ):
            raise AuthorizationError(
                f"skill {name!r} declares a classification ceiling and cannot run on the direct path "
                "without a resolved principal while an access policy is active"
            )
        context = self.prepare(request)
        evidence_classification = max_classification(hit.chunk.metadata for hit in context.hits)
        permit = capability_permits(
            capability,
            granted_scopes=policy.granted_scopes,
            side_effect_ceiling=policy.side_effect_ceiling,
            evidence_classification=evidence_classification,
            payload_bytes=(
                len(canonical_json(arguments or {}).encode()) if capability.max_payload_bytes is not None else None
            ),
        )
        if not permit.allowed:
            raise AuthorizationError(f"skill {name!r} denied by direct skill policy")
        self._charge_skill_request(request)
        return skill.run(context, arguments or {})

    def _charge_skill_request(self, request: QueryRequest) -> None:
        # The single served-skill funnel (HTTP run_skill, MCP bridge, in-process run_skill all reach here;
        # a governed-ReAct tool calls the skill's own run instead), so charging skill_requests here covers
        # every served skill surface with no double-count.
        charge_served_request(
            quota_policy=self.quota_policy,
            kill_switches=self.kill_switches,
            store=self.store,
            audit_sink=self.audit_sink,
            workspace_id=request.workspace_id,
            corpus_id=request.corpus_id,
            principal=request.principal,
            dimension=QuotaDimension.SKILL_REQUESTS,
        )


class SkillCreator:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, name: str, description: str) -> Path:
        if not SKILL_NAME.fullmatch(name):
            raise ValidationError("Skill names must be lowercase Python identifiers with 3-64 characters")
        target = safe_join(self.root, name)
        if target.exists():
            raise ConflictError(f"Skill project already exists: {name}")
        package = target / "src" / name
        tests = target / "tests"
        package.mkdir(parents=True)
        tests.mkdir(parents=True)
        (target / "pyproject.toml").write_text(self._pyproject(name, description), encoding="utf-8")
        (package / "__init__.py").write_text(self._implementation(name, description), encoding="utf-8")
        (tests / "test_skill.py").write_text(self._test(name), encoding="utf-8")
        (target / "README.md").write_text(f"# {name}\n\n{description}\n", encoding="utf-8")
        return target

    @staticmethod
    def _pyproject(name: str, description: str) -> str:
        return (
            "[build-system]\n"
            'requires = ["setuptools>=75"]\n'
            'build-backend = "setuptools.build_meta"\n\n'
            "[project]\n"
            f'name = "refaat-skill-{name.replace("_", "-")}"\n'
            'version = "0.1.0"\n'
            f"description = {description!r}\n"
            'requires-python = ">=3.12"\n'
            f'dependencies = ["refaat>={__version__},<1"]\n\n'
            '[project.entry-points."refaat.skills"]\n'
            f'{name} = "{name}:{name.title().replace("_", "")}Skill"\n\n'
            "[tool.setuptools.packages.find]\n"
            'where = ["src"]\n'
            f'include = ["{name}*"]\n'
        )

    @staticmethod
    def _implementation(name: str, description: str) -> str:
        class_name = name.title().replace("_", "") + "Skill"
        return (
            "from __future__ import annotations\n\n"
            "from refaat.domain.models import ContextPacket, JsonObject\n\n\n"
            f"class {class_name}:\n"
            f"    name = {name!r}\n"
            f"    description = {description!r}\n\n"
            "    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:\n"
            '        return {"query": context.query, "arguments": arguments, "chunks": len(context.hits)}\n'
        )

    @staticmethod
    def _test(name: str) -> str:
        class_name = name.title().replace("_", "") + "Skill"
        return (
            f"from {name} import {class_name}\n\n\n"
            "def test_skill_metadata() -> None:\n"
            f"    skill = {class_name}()\n"
            f"    assert skill.name == {name!r}\n"
            "    assert skill.description\n"
        )
