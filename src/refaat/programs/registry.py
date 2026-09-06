from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

from refaat.domain.authz import Principal
from refaat.domain.kill_switch import ensure_workspace_active
from refaat.domain.models import JsonObject
from refaat.errors import ConflictError, NotFoundError
from refaat.observability import span
from refaat.programs.program import GroundedRagProgram, Program, ProgramResult

if TYPE_CHECKING:
    from refaat.application.runtime import RefaatProject
    from refaat.evaluation.bundle import MetricBundle
    from refaat.evaluation.dataset import Dataset
    from refaat.evaluation.runner import EvaluationReport
    from refaat.optimization.candidates import OptimizationResult
    from refaat.programs.program import Tunables


class BoundProgram:
    """A program bound to a project, ready to run against its corpus."""

    def __init__(self, project: RefaatProject, program: Program) -> None:
        self._project = project
        self.program = program

    @property
    def name(self) -> str:
        return self.program.name

    @property
    def project(self) -> RefaatProject:
        return self._project

    @property
    def artifact(self) -> JsonObject:
        return self.program.artifact().to_json()

    def run(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        principal: Principal | None = None,
        **inputs: Any,
    ) -> ProgramResult:
        from refaat.application.services import in_shadow_execution

        ensure_workspace_active(self._project.runtime.kill_switches, self._project.workspace_id)
        # Fail closed before a program's own preprocessing (a KNN embedding, a Self-RAG necessity gate)
        # reaches a model, so a principal-less strict run is denied first.
        self._project.runtime.require_principal(principal)
        with span("refaat.program.run", program=self.program.name, shadow=in_shadow_execution()):
            return self.program.execute(
                self._project, inputs=inputs, user_id=user_id, session_id=session_id, principal=principal
            )

    def stream(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        principal: Principal | None = None,
        **inputs: Any,
    ) -> Iterator[dict[str, Any]]:
        ensure_workspace_active(self._project.runtime.kill_switches, self._project.workspace_id)
        self._project.runtime.require_principal(principal)
        with span("refaat.program.stream", program=self.program.name):
            yield from self.program.stream(
                self._project, inputs=inputs, user_id=user_id, session_id=session_id, principal=principal
            )

    def evaluate(self, dataset: Dataset, bundle: MetricBundle | None = None) -> EvaluationReport:
        from refaat.evaluation.runner import ProgramEvaluator

        ensure_workspace_active(self._project.runtime.kill_switches, self._project.workspace_id)
        with span("refaat.program.evaluate", program=self.program.name, cases=len(dataset.cases)):
            return ProgramEvaluator(self).evaluate(dataset, bundle)

    def optimize(
        self,
        dataset: Dataset,
        *,
        bundle: MetricBundle | None = None,
        search_space: Sequence[Tunables] | None = None,
        budget: int | None = None,
        token_budget: int | None = None,
        cost_budget_usd: float | None = None,
    ) -> OptimizationResult:
        from refaat.optimization.optimizer import RetrievalOptimizer

        ensure_workspace_active(self._project.runtime.kill_switches, self._project.workspace_id)
        with span("refaat.program.optimize", program=self.program.name):
            return RetrievalOptimizer(self._project).optimize(
                self.program,
                dataset,
                bundle=bundle,
                search_space=search_space,
                budget=budget,
                token_budget=token_budget,
                cost_budget_usd=cost_budget_usd,
            )

    def enqueue_evaluation(self, dataset: Dataset) -> str:
        return self._project.runtime.store.enqueue_job(
            self._project.workspace_id,
            "program.evaluate",
            {"program": self.name, "corpus_id": self._project.corpus_id, "cases": dataset.to_records()},
        )

    def enqueue_optimization(
        self,
        dataset: Dataset,
        *,
        search_space: Sequence[Tunables] | None = None,
        budget: int | None = None,
    ) -> str:
        """Persist a durable, resumable optimization run and enqueue it on the worker.

        The materialized search space and dataset are stored so a restart replays
        the exact candidate set; the run executes on the shared ``DurableWorker``
        via the ``optimization.run`` job kind and never mutates the active program.

        Refused up front when the program cannot be resolved after a restart: a
        custom program registered only in-process is gone once the worker rebuilds
        a fresh project, so its durable run could never resume. An identical
        already-enqueued plan reuses the in-flight run instead of duplicating it.
        """
        from refaat.optimization.durable import plan_digest
        from refaat.optimization.search import default_search_space

        if not ProgramRegistry(self._project).resolvable(self.name):
            raise ValueError(
                "Durable optimization requires a program resolvable after a restart (a built-in program); "
                f"{self.name!r} is only registered in-process and could never resume on the worker"
            )

        space = list(search_space) if search_space is not None else default_search_space()
        if budget is not None:
            space = space[:budget]
        if not space:
            raise ValueError("Optimization search space is empty")
        search_space_json = [tunables.as_json() for tunables in space]
        bundle_id = "grounded-rag"
        store = self._project.runtime.store
        from refaat.infrastructure.model_cache import resolved_cache_identity

        corpus_version = store.get_corpus(self._project.workspace_id, self._project.corpus_id).version
        embedder = self._project.runtime.queries.embedder
        generator = self._project.runtime.queries.generator
        plan_hash = plan_digest(
            program=self.name,
            corpus_id=self._project.corpus_id,
            corpus_version=corpus_version,
            dataset_version=dataset.version,
            search_space=search_space_json,
            bundle_id=bundle_id,
            embedding_model=embedder.model_id,
            generation_model=generator.model_id,
            embedding_cache_identity=resolved_cache_identity(embedder),
            generation_cache_identity=resolved_cache_identity(generator),
        )
        existing = store.get_optimization_run_by_plan(self._project.workspace_id, self.name, plan_hash)
        if existing is not None:
            return str(existing["id"])
        artifact = self.program.artifact()
        try:
            run_id = store.create_optimization_run(
                self._project.workspace_id,
                program=self.name,
                corpus_id=self._project.corpus_id,
                dataset_name=dataset.name,
                dataset_version=dataset.version,
                dataset_cases=dataset.to_records(),
                search_space=search_space_json,
                bundle_id=bundle_id,
                budget=budget,
                plan_hash=plan_hash,
                program_definition_hash=artifact.definition_hash,
                program_immutable_policy_hash=artifact.immutable_policy_hash,
            )
        except ConflictError:
            # A concurrent enqueue of the identical plan won the partial-unique guard;
            # converge on its in-flight run and its already-enqueued job rather than
            # creating a duplicate.
            winner = store.get_optimization_run_by_plan(self._project.workspace_id, self.name, plan_hash)
            if winner is not None:
                return str(winner["id"])
            raise
        store.enqueue_job(self._project.workspace_id, "optimization.run", {"run_id": run_id}, dedup_key=run_id)
        return run_id


class ProgramRegistry:
    """In-process registry of typed programs available to a project.

    The built-in ``grounded-rag`` program is always registered so the runtime,
    CLI, and examples can invoke a program without any user code.
    """

    def __init__(self, project: RefaatProject) -> None:
        self._project = project
        self._programs: dict[str, Program] = {}
        self.register(GroundedRagProgram())

    def register(self, program: Program) -> BoundProgram:
        if program.name in self._programs:
            raise ConflictError(f"Program already registered: {program.name}")
        self._programs[program.name] = program
        return BoundProgram(self._project, program)

    def get(self, name: str) -> BoundProgram:
        try:
            program = self._programs[name]
        except KeyError as exc:
            raise NotFoundError(f"Program not found: {name}") from exc
        return BoundProgram(self._project, program)

    def resolvable(self, name: str) -> bool:
        return name in self._programs

    def list(self) -> list[JsonObject]:
        return [
            program.artifact().to_json() for program in sorted(self._programs.values(), key=lambda value: value.name)
        ]
