from __future__ import annotations

from typing import TYPE_CHECKING

from refaat.canonical import canonical_digest
from refaat.domain.enums import EvaluationStatus
from refaat.domain.models import JsonObject
from refaat.errors import AuthorizationError
from refaat.evaluation.bundle import default_bundle
from refaat.evaluation.dataset import Dataset
from refaat.optimization.candidates import (
    Candidate,
    OptimizationResult,
    assemble_result,
    evaluate_candidate,
)
from refaat.programs.program import Program, Tunables

if TYPE_CHECKING:
    from refaat.application.runtime import RefaatProject


def plan_digest(
    *,
    program: str,
    corpus_id: str,
    corpus_version: int,
    dataset_version: str,
    search_space: list[JsonObject],
    bundle_id: str,
    embedding_model: str,
    generation_model: str,
    embedding_cache_identity: str | None = None,
    generation_cache_identity: str | None = None,
) -> str:
    """Content-address the materialized optimization plan for idempotent enqueue.

    The plan is the full untruncated search space and deliberately excludes the
    live run token and cost budgets, because the durable path applies no
    spend-gated early stop, so the persisted plan replays exactly. It binds the
    corpus version and the embedding and generation model identity so two runs
    that differ in evidence version or model never share a non-terminal dedup
    slot, mirroring the answer-cache context identity.
    """
    payload: JsonObject = {
        "program": program,
        "corpus_id": corpus_id,
        "corpus_version": corpus_version,
        "dataset_version": dataset_version,
        "search_space": search_space,
        "bundle_id": bundle_id,
        "embedding_model": embedding_model,
        "generation_model": generation_model,
    }
    if embedding_cache_identity is not None:
        payload["embedding_cache_identity"] = embedding_cache_identity
    if generation_cache_identity is not None:
        payload["generation_cache_identity"] = generation_cache_identity
    return canonical_digest(payload)


def resume_optimization_run(project: RefaatProject, run_id: str) -> JsonObject:
    """Evaluate the persisted plan, skipping already-checkpointed ordinals.

    Idempotent: an already-succeeded run returns its stored result untouched.
    Otherwise every not-yet-checkpointed ordinal is evaluated and checkpointed,
    the candidate set is reassembled in ordinal order, and the ranked result is
    assembled exactly as the synchronous path would. The durable path deliberately
    evaluates the full materialized search space with no live run-budget early
    stop: a token or cost run budget accrues provider spend that is known only
    after each model call and resets per process, so it cannot be replayed
    deterministically across a resume. Because each candidate evaluation is
    otherwise deterministic and independent, a resumed run produces a
    byte-identical :class:`OptimizationResult` to a synchronous ``optimize()`` run
    whenever that synchronous run is not truncated by a live run budget, which is
    always the case offline and whenever no ``run_token_budget`` or
    ``run_cost_budget_usd`` ceiling is configured. The run registers the winning
    and Pareto-front artifacts as immutable registry entries but never promotes
    them and never mutates the active program.
    """
    if project.runtime.access_policy is not None:
        raise AuthorizationError("Async optimization is unavailable while an access policy is active")
    store = project.runtime.store
    workspace_id = project.workspace_id
    run = store.get_optimization_run(workspace_id, run_id)
    if run["status"] == EvaluationStatus.SUCCEEDED:
        return dict(run["result"])

    program = project.programs.get(str(run["program"])).program
    _verify_program_identity(program, run)
    store.mark_optimization_run_running(workspace_id, run_id)

    search_space = [Tunables.from_json(item) for item in run["search_space"]]
    dataset = Dataset.from_records(str(run["dataset_name"]), run["dataset_cases"])
    # The persisted version is authoritative: it was content-addressed at enqueue,
    # so replaying against it keeps the result byte-identical even if the case
    # reconstruction ever drifted from the original object.
    dataset_version = str(run["dataset_version"])
    bundle = default_bundle()
    checkpoints = {int(row["ordinal"]): Candidate.from_json(row) for row in store.list_optimization_candidates(run_id)}

    candidates: list[Candidate] = []
    for ordinal, tunables in enumerate(search_space):
        checkpoint = checkpoints.get(ordinal)
        if checkpoint is not None:
            candidates.append(checkpoint)
            continue
        candidate = evaluate_candidate(project, program, tunables, dataset, bundle)
        store.record_optimization_candidate(run_id, ordinal, candidate.to_json())
        # Read back the committed checkpoint rather than trusting the locally
        # evaluated candidate: the record is insert-if-absent, so if a concurrent
        # resumer won the ordinal every resumer still converges on the one
        # committed row, and the assembled result always equals the checkpoint table.
        committed = store.get_optimization_candidate(run_id, ordinal)
        candidates.append(Candidate.from_json(committed) if committed is not None else candidate)

    result = assemble_result(program.name, dataset_version, candidates)
    _register_artifacts(project, program, result)
    store.finish_optimization_run(
        workspace_id,
        run_id,
        result=result.to_json(),
        best_artifact_id=result.best.artifact_id,
    )
    return result.to_json()


def _verify_program_identity(program: Program, run: JsonObject) -> None:
    """Refuse a resume when the program definition drifted since enqueue.

    The run pins the program's ``definition_hash`` and ``immutable_policy_hash``.
    If either changed between enqueue and resume (a base-prompt or response-schema
    change), the persisted checkpoints and the new definition disagree, so the
    result could reference an artifact id the run never registered. Refuse instead,
    and the worker terminalizes the run as FAILED with this reason. Legacy runs
    without a stored hash skip the check.
    """
    stored_definition = run.get("program_definition_hash")
    stored_policy = run.get("program_immutable_policy_hash")
    # Fail closed on a legacy run with no pinned identity: without both hashes the
    # drift check cannot run, so the program could have changed under it. A run
    # created after B1 always pins both, so this only rejects un-verifiable rows.
    if stored_definition is None or stored_policy is None:
        raise ValueError(
            "Optimization run has no pinned program identity; refusing to resume a run whose program cannot be "
            f"verified against the enqueued definition for {program.name!r}"
        )
    artifact = program.artifact()
    if artifact.definition_hash != stored_definition or artifact.immutable_policy_hash != stored_policy:
        raise ValueError(
            "Program definition drifted since the optimization run was enqueued; refusing to resume against a "
            f"changed program {program.name!r}"
        )


def _register_artifacts(project: RefaatProject, program: Program, result: OptimizationResult) -> None:
    registry = project.registry
    # The winner backs best_artifact_id, which finalize persists as the stored id for
    # registry.get to resolve. It must be registered under that exact id, so it is
    # deliberately compared on the full artifact_id rather than the leniency the
    # Pareto members below allow: registering the winner under any rebuilt id (even
    # one a benign dependency bump produced) would leave best_artifact_id dangling.
    # This asymmetry is intentional. A Pareto member is only re-registered as an
    # immutable entry and never resolved by a stored pointer, so it tolerates a
    # version-only id shift; the winner cannot. The offline single-process path never
    # straddles a version, so this rebuilds the identical id.
    winner = program.with_tunables(Tunables.from_json(result.best.tunables)).artifact()
    if winner.artifact_id != result.best.artifact_id:
        raise ValueError(
            "Winning candidate identity drifted since it was checkpointed; refusing to finalize a run whose "
            f"best_artifact_id would not resolve for {program.name!r}"
        )
    registry.save(winner)
    seen: set[str] = {winner.artifact_id}
    for candidate in result.pareto_front:
        if candidate.artifact_id in seen:
            continue
        seen.add(candidate.artifact_id)
        variant = program.with_tunables(Tunables.from_json(candidate.tunables))
        artifact = variant.artifact()
        # Compare the version-independent identity, not the full artifact_id: the
        # id folds dependency_versions and __version__, so a benign library bump
        # would spuriously refuse a checkpoint from an older version. A real
        # definition, base-prompt, or response-schema change still shifts the
        # definition or policy hash and is refused. A legacy checkpoint that never
        # stored these hashes falls back to the full-id equality it was written under.
        if candidate.definition_hash is not None and candidate.immutable_policy_hash is not None:
            drifted = (
                artifact.definition_hash != candidate.definition_hash
                or artifact.immutable_policy_hash != candidate.immutable_policy_hash
            )
        else:
            drifted = artifact.artifact_id != candidate.artifact_id
        if drifted:
            raise ValueError(
                "Re-registered variant identity does not match the checkpoint; refusing to register a drifted artifact"
            )
        registry.save(artifact)
