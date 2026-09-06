from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from refaat.application.budget import isolated_budget_scope, prices_from
from refaat.application.services import unserved_run
from refaat.canonical import canonical_digest
from refaat.core.text import sha256_text, stable_digest
from refaat.domain.authz import Principal
from refaat.domain.models import JsonObject
from refaat.errors import ValidationError
from refaat.observability import get_logger
from refaat.optimization.telemetry import ShadowComparison, compare_results
from refaat.programs.program import ProgramResult, Tunables
from refaat.programs.registry import BoundProgram
from refaat.version import __version__

logger = get_logger(__name__)

if TYPE_CHECKING:
    from refaat.application.registry import RegistryService
    from refaat.application.runtime import RefaatProject
    from refaat.config import Settings
    from refaat.infrastructure.database.store import SQLStore
    from refaat.programs.program import Program


def _canonicalize(value: Any) -> Any:
    """Normalize a programmatic input into a JSON-canonical, hash-seed-independent form.

    Sets and frozensets become order-independent sorted lists and mappings are
    rebuilt with string keys, so the canonical digest of the inputs is identical
    across processes and PYTHONHASHSEED values.
    """
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(value[key]) for key in sorted(value, key=repr)}
    if isinstance(value, (set, frozenset)):
        return sorted((_canonicalize(item) for item in value), key=repr)
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class DeploymentRunResult:
    served: str
    result: ProgramResult
    active_artifact_id: str
    candidate_artifact_id: str | None
    bucket: int | None
    comparison: ShadowComparison | None
    shadow_run_id: str | None

    def to_json(self) -> JsonObject:
        return {
            "served": self.served,
            "result": self.result.to_json(),
            "active_artifact_id": self.active_artifact_id,
            "candidate_artifact_id": self.candidate_artifact_id,
            "bucket": self.bucket,
            "comparison": self.comparison.to_json() if self.comparison is not None else None,
            "shadow_run_id": self.shadow_run_id,
        }


class DeploymentRouter:
    """Serve a program through its environment's active artifact and optional canary.

    With no canary attached the router collapses to resolve-active-and-run through
    the unchanged query pipeline, so every artifact id, cache key, and route is
    byte-identical to running the resolved active artifact directly. When a canary
    is attached both the active and the candidate run through the same pipeline
    (both cite real stored chunks), a deterministic bucket picks the one served,
    and the deterministic comparison of the two is appended to append-only
    telemetry. The unserved result is used only to compute the comparison.
    """

    def __init__(self, store: SQLStore, registry: RegistryService, settings: Settings) -> None:
        self.store = store
        self.registry = registry
        self.settings = settings

    def run(
        self,
        project: RefaatProject,
        program: str,
        environment: str,
        *,
        inputs: Mapping[str, Any],
        user_id: str | None = None,
        session_id: str | None = None,
        idempotency_key: str | None = None,
        principal: Principal | None = None,
        allow_defer: bool = True,
    ) -> DeploymentRunResult:
        workspace_id = project.workspace_id
        active_id, active_document = self.registry.resolve_deployment(workspace_id, program, environment)
        base = project.programs.get(program).program
        active = self._rehydrate(base, active_document)

        canary = self.store.current_canary(workspace_id, program, environment)
        if canary is None:
            result = BoundProgram(project, active).run(
                user_id=user_id, session_id=session_id, principal=principal, **inputs
            )
            return DeploymentRunResult(
                served="active",
                result=result,
                active_artifact_id=active_id,
                candidate_artifact_id=None,
                bucket=None,
                comparison=None,
                shadow_run_id=None,
            )

        candidate_id = str(canary["candidate_artifact_id"])
        candidate_document = self.registry.get(workspace_id, candidate_id)
        candidate = self._rehydrate(base, candidate_document)

        # Backstop for the attach-time guard: a program whose execution can leak
        # durable or external side effects a suppression scope cannot neutralize is
        # never shadowed, even if a canary row was somehow attached out of band.
        if not active.shadow_safe or not candidate.shadow_safe:
            raise ValidationError(
                "Program is not shadow-safe; refusing to run a canary that would execute an unserved side-effectful run"
            )

        percent = int(canary["percent"])
        query_text = self._query_text(inputs)
        identity = session_id or user_id or sha256_text(query_text)
        bucket = int(stable_digest([workspace_id, program, environment, identity], length=8), 16) % 100
        served = "candidate" if bucket < percent else "active"

        question = inputs.get("question")
        if (
            allow_defer
            and self.settings.defer_shadow
            and project.runtime.access_policy is None
            and isinstance(question, str)
            and question.strip()
        ):
            # Serve-first: return the authoritative served answer now and defer the unserved twin and the
            # comparison to a durable job, so the shadow re-derivation never delays this response. The job
            # carries the query like any durable evidence job and only runs when no access policy is active.
            served_program = candidate if served == "candidate" else active
            served_result = BoundProgram(project, served_program).run(
                user_id=user_id, session_id=session_id, principal=principal, **inputs
            )
            deadline = self.store.clock.now() + timedelta(seconds=self.settings.shadow_defer_timeout_seconds)
            self.store.enqueue_job(
                workspace_id,
                "program.shadow_compare",
                {
                    "corpus_id": project.corpus_id,
                    "program": program,
                    "environment": environment,
                    "questions": [query_text],
                    "served": served,
                    "bucket": bucket,
                    "compare": bool(canary["compare"]),
                    "deferred": True,
                    "deadline": deadline.isoformat(),
                },
            )
            return DeploymentRunResult(
                served=served,
                result=served_result,
                active_artifact_id=active_id,
                candidate_artifact_id=candidate_id,
                bucket=bucket,
                comparison=None,
                shadow_run_id=None,
            )

        # Run the UNSERVED shadow first, fully isolated: inside the shadow-execution
        # scope (no session, answer-cache, or model-call-cache write) and wrapped so any
        # exception is swallowed. A shadow failure can never corrupt or crash the served
        # response, and both runs read the same pre-served-write session state.
        unserved_program = active if served == "candidate" else candidate
        unserved_result: ProgramResult | None
        try:
            # The unserved run meters against its own isolated budget, never the served run's.
            with (
                unserved_run(),
                isolated_budget_scope(
                    self.settings.run_token_budget,
                    self.settings.run_cost_budget_usd,
                    prices=prices_from(self.settings.model_prices),
                ),
            ):
                unserved_result = BoundProgram(project, unserved_program).run(
                    user_id=user_id, session_id=session_id, principal=principal, **inputs
                )
        except Exception:
            logger.warning("shadow_run_failed", program=program, environment=environment, served=served)
            unserved_result = None

        # The served run is authoritative: it runs OUTSIDE the scope and writes the
        # session turn and cache exactly once. It is always returned.
        served_program = candidate if served == "candidate" else active
        served_result = BoundProgram(project, served_program).run(
            user_id=user_id, session_id=session_id, principal=principal, **inputs
        )

        comparison: ShadowComparison | None = None
        shadow_run_id: str | None = None
        if unserved_result is not None:
            if served == "candidate":
                active_result, candidate_result = unserved_result, served_result
            else:
                active_result, candidate_result = served_result, unserved_result
            comparison = compare_results(active_result, candidate_result) if bool(canary["compare"]) else None
            shadow_run_id = self.store.record_shadow_run(
                workspace_id,
                program,
                environment,
                active_id,
                candidate_id,
                served,
                bucket,
                sha256_text(query_text),
                comparison.to_json() if comparison is not None else None,
                idempotency_key=idempotency_key,
            )
        return DeploymentRunResult(
            served=served,
            result=served_result,
            active_artifact_id=active_id,
            candidate_artifact_id=candidate_id,
            bucket=bucket,
            comparison=comparison,
            shadow_run_id=shadow_run_id,
        )

    def run_deferred_compare(
        self,
        project: RefaatProject,
        program: str,
        environment: str,
        *,
        query_text: str,
        served: str,
        bucket: int,
        compare: bool,
        idempotency_key: str | None = None,
    ) -> str | None:
        """Re-derive a deferred canary comparison in a worker job, persisting no answer text.

        Runs the active and candidate programs over the query under the same suppression and isolated-budget
        scopes as the inline shadow, so nothing is served, cached, or charged, then records only the hash-only
        telemetry row record_shadow_run already writes. The caller returns telemetry only, never a result.
        """
        workspace_id = project.workspace_id
        active_id, active_document = self.registry.resolve_deployment(workspace_id, program, environment)
        base = project.programs.get(program).program
        active = self._rehydrate(base, active_document)
        canary = self.store.current_canary(workspace_id, program, environment)
        if canary is None:
            return None
        candidate_id = str(canary["candidate_artifact_id"])
        candidate = self._rehydrate(base, self.registry.get(workspace_id, candidate_id))
        if not active.shadow_safe or not candidate.shadow_safe:
            raise ValidationError(
                "Program is not shadow-safe; refusing a deferred canary comparison over a side-effectful run"
            )
        with (
            unserved_run(),
            isolated_budget_scope(
                self.settings.run_token_budget,
                self.settings.run_cost_budget_usd,
                prices=prices_from(self.settings.model_prices),
            ),
        ):
            active_result = BoundProgram(project, active).run(question=query_text)
            candidate_result = BoundProgram(project, candidate).run(question=query_text)
        comparison = compare_results(active_result, candidate_result) if compare else None
        return self.store.record_shadow_run(
            workspace_id,
            program,
            environment,
            active_id,
            candidate_id,
            served,
            bucket,
            sha256_text(query_text),
            comparison.to_json() if comparison is not None else None,
            idempotency_key=idempotency_key,
        )

    @staticmethod
    def _rehydrate(base: Program, document: JsonObject) -> Program:
        """Rebuild the registered variant and refuse if its identity drifted from the approved one.

        Both the mutable definition (``definition_hash``, which folds tunables,
        signatures, and any subclass policy) and the immutable answer-affecting
        state (``immutable_policy_hash``, the base system prompt) must match the
        stored deployment document. The full ``artifact_id`` is deliberately NOT
        compared: it also folds ``dependency_versions`` and ``__version__``, so a
        benign version bump would otherwise over-refuse a program whose behavior is
        unchanged. A base-prompt or policy change that shifts the immutable hash
        without shifting the definition hash is a real approval bypass and is refused.
        """
        variant = base.with_tunables(Tunables.from_json(document.get("tunables", {})))
        artifact = variant.artifact()
        if artifact.definition_hash != document.get("definition_hash"):
            raise ValidationError(
                "Registered artifact definition does not match its rehydrated program; refusing to serve"
            )
        if artifact.immutable_policy_hash != document.get("immutable_policy_hash"):
            raise ValidationError(
                "Registered artifact immutable policy does not match its rehydrated program; refusing to serve"
            )
        stored_versions = document.get("dependency_versions")
        if stored_versions != {"refaat": __version__}:
            # Behavior identity matches (both hashes above); only the build drifted, so record, never refuse.
            logger.info(
                "deployment_dependency_drift",
                program=base.name,
                behavior_hash=artifact.behavior_hash,
                stored=stored_versions,
                current={"refaat": __version__},
            )
        return variant

    @staticmethod
    def _query_text(inputs: Mapping[str, Any]) -> str:
        """A deterministic routing identity for the inputs.

        A string ``question`` routes on its own text. Any other programmatic input
        falls back to a canonical, key-sorted, type-stable digest so the bucket is
        reproducible regardless of key insertion order, set iteration order, or a
        custom object's ``__str__``; ``json.dumps(default=str)`` would not be pure.
        """
        question = inputs.get("question")
        if isinstance(question, str) and question.strip():
            return question
        return canonical_digest(_canonicalize(dict(inputs)))


class BoundDeployment:
    """A :class:`DeploymentRouter` and its canary controls bound to one project."""

    def __init__(self, project: RefaatProject) -> None:
        self._project = project

    def run(
        self,
        program: str,
        environment: str,
        *,
        inputs: Mapping[str, Any],
        user_id: str | None = None,
        session_id: str | None = None,
        idempotency_key: str | None = None,
        principal: Principal | None = None,
        allow_defer: bool = True,
    ) -> DeploymentRunResult:
        return self._project.runtime.deployment.run(
            self._project,
            program,
            environment,
            inputs=inputs,
            user_id=user_id,
            session_id=session_id,
            idempotency_key=idempotency_key,
            principal=principal,
            allow_defer=allow_defer,
        )

    def set_canary(
        self, program: str, environment: str, candidate_artifact_id: str, *, percent: int = 0, compare: bool = True
    ) -> JsonObject:
        # The active deployment and the candidate rehydrate from the same registered
        # base program, so its shadow_safe property governs both. Refuse the attach
        # for a side-effectful program whose unserved run a suppression scope cannot
        # neutralize; DeploymentRouter.run backstops the same guard at run time.
        if not self._project.programs.get(program).program.shadow_safe:
            raise ValidationError(
                "Program is not shadow-safe; refusing to attach a canary whose unserved run would leak side effects"
            )
        return self._project.registry.set_canary(program, environment, candidate_artifact_id, percent, compare)

    def clear_canary(self, program: str, environment: str) -> JsonObject:
        return self._project.registry.clear_canary(program, environment)

    def current_canary(self, program: str, environment: str) -> JsonObject | None:
        return self._project.registry.current_canary(program, environment)

    def canary_history(self, program: str, environment: str) -> list[JsonObject]:
        return self._project.registry.canary_history(program, environment)

    def shadow_runs(self, program: str, environment: str, limit: int = 50) -> list[JsonObject]:
        return self._project.runtime.store.shadow_runs(self._project.workspace_id, program, environment, limit)
