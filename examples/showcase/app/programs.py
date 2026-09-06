"""The program and optimization layer over the Oronts knowledge base.

Every function here is a small, honest demonstration of one refaat program-layer
capability driven through the public API over an already-ingested project. Each is a
standalone callable a later API layer can wrap, and each returns a plain dict of the
facts the entrypoint asserts. The whole file is deterministic and offline: no paid
model, no network, no mocks.

The load-bearing invariants every demonstration preserves:

- The citation contract. Every answer cites only real stored chunks; the resolution
  helper checks each cited id against the stored-chunk universe.
- Register, do not promote. Optimizer output is registered as immutable candidates and
  the active program is never mutated. Promotion is explicit and approval gated;
  rollback is atomic.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from refaat import RefaatProject
from refaat.application.budget import budget_scope
from refaat.application.fine_tuning import fine_tune_artifact, fine_tune_program_name
from refaat.application.worker import DurableWorker
from refaat.domain.enums import SideEffectClass
from refaat.domain.models import Citation, ContextPacket, JsonObject, ReActPolicy, ToolGrant
from refaat.errors import NotFoundError, ValidationError
from refaat.evaluation import Dataset, ProgramEvaluator, default_bundle, trajectory_bundle
from refaat.optimization import RetrievalOptimizer, retrieval_search_space
from refaat.programs import (
    BEST_OF,
    REFINE,
    BoundedInferencePolicy,
    GovernedReActProgram,
    GroundedRagProgram,
    InputField,
    KeywordToolPlanner,
    MultiAgentOrchestrator,
    Signature,
    StructuredOutputProgram,
    TaskRouter,
    Tunables,
)
from refaat.skills import ProgramSkill, SkillRegistry

from .config import ShowcaseConfig

DIGEST_TOOL = "evidence_digest"
PAGER_TOOL = "pager_escalation"
PLAYBOOK_TOOL = "oronts_playbook"

TYPED_QUESTION = "What is OGuardAI and what does it protect for the enterprise?"
POLICY_QUESTION = "How does the 90-Day Production Pilot take a use case from validation to production?"
REACT_QUESTION = "During a security incident, how does OGuardAI stop sensitive data drift to an LLM provider?"
STRUCTURED_QUESTION = "Where is Oronts GmbH based and who founded it?"
COMPANY_QUESTION = "Who founded Oronts GmbH and where is the company registered?"
PRODUCT_QUESTION = "Which Oronts products exist and what does each one do?"


def _resolves(citations: tuple[Citation, ...], chunk_ids: set[str]) -> bool:
    return bool(citations) and all(citation.chunk_id in chunk_ids for citation in citations)


class OrontsQuestion(Signature):
    question: str = InputField(description="A question answered from the Oronts knowledge base.")


class OrontsAssistant(GroundedRagProgram):
    name = "oronts-assistant"
    input_signature = OrontsQuestion


class ProductSpecialist(GroundedRagProgram):
    name = "oronts-product-specialist"


class CompanySpecialist(GroundedRagProgram):
    name = "oronts-company-specialist"


class RegistryDemoProgram(GroundedRagProgram):
    name = "oronts-registry-demo"


class CanaryDemoProgram(GroundedRagProgram):
    name = "oronts-canary-demo"


class EvidenceDigestSkill:
    name = DIGEST_TOOL
    description = "Read-only digest of the top retrieved Oronts evidence for a governed step."

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        del arguments
        return {
            "sources": sorted({str(hit.chunk.metadata.get("source")) for hit in context.hits}),
            "evidence": [hit.chunk.text.split(".", maxsplit=1)[0] for hit in context.hits[:3]],
        }


class PagerEscalationSkill:
    name = PAGER_TOOL
    description = "External paging side effect, correctly denied under a read-only ceiling."

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        del context, arguments
        return {"paged": True}


def typed_program(project: RefaatProject, chunk_ids: set[str]) -> dict[str, Any]:
    assistant = project.programs.register(OrontsAssistant(tunables=Tunables(top_k=6, minimum_citation_coverage=0.5)))
    result = assistant.run(question=TYPED_QUESTION)
    return {
        "program": result.program,
        "artifact_id": result.artifact_id,
        "grounded": result.grounded and _resolves(result.citations, chunk_ids),
        "citations": len(result.citations),
    }


def inference_policies(project: RefaatProject, chunk_ids: set[str]) -> dict[str, Any]:
    base = GroundedRagProgram()
    best_of = project.programs.register(BoundedInferencePolicy(base, mode=BEST_OF, attempts=3))
    refine = project.programs.register(BoundedInferencePolicy(base, mode=REFINE, attempts=3))
    best_result = best_of.run(question=POLICY_QUESTION)
    refine_result = refine.run(question=POLICY_QUESTION)
    return {
        "best_of_grounded": best_result.grounded and _resolves(best_result.citations, chunk_ids),
        "refine_grounded": refine_result.grounded and _resolves(refine_result.citations, chunk_ids),
        "distinct_artifacts": best_of.program.artifact().artifact_id != refine.program.artifact().artifact_id,
    }


def _react_pieces() -> tuple[ReActPolicy, KeywordToolPlanner, dict[str, Any]]:
    policy = ReActPolicy(
        max_iterations=4,
        granted_scopes=("read",),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        grants=(
            ToolGrant(DIGEST_TOOL, required_scopes=("read",), side_effect=SideEffectClass.READ_ONLY),
            ToolGrant(PAGER_TOOL, required_scopes=("read",), side_effect=SideEffectClass.EXTERNAL),
        ),
    )
    planner = KeywordToolPlanner({DIGEST_TOOL: ("incident", "drift"), PAGER_TOOL: ("incident", "drift")})
    skills: dict[str, Any] = {DIGEST_TOOL: EvidenceDigestSkill(), PAGER_TOOL: PagerEscalationSkill()}
    return policy, planner, skills


def build_governed_react(project: RefaatProject) -> Any:
    """A bound governed-ReAct program the agent layer can compile as a subgraph."""
    policy, planner, skills = _react_pieces()
    return project.programs.register(GovernedReActProgram(policy=policy, planner=planner, skills=skills))


def governed_react(bound_react: Any, chunk_ids: set[str]) -> dict[str, Any]:
    result = bound_react.run(question=REACT_QUESTION)
    steps = result.outputs["react_steps"]
    return {
        "authorized": sorted(step["tool"] for step in steps if step["authorized"]),
        "denied": sorted(step["tool"] for step in steps if not step["authorized"]),
        "grounded": result.grounded and _resolves(result.citations, chunk_ids),
    }


def structured_output(project: RefaatProject, chunk_ids: set[str]) -> dict[str, Any]:
    structured = StructuredOutputProgram(GroundedRagProgram())
    result = project.programs.register(structured).run(question=STRUCTURED_QUESTION)
    return {
        "grounded": result.grounded and _resolves(result.citations, chunk_ids),
        "citations": len(result.citations),
    }


def multi_agent(project: RefaatProject, chunk_ids: set[str]) -> dict[str, Any]:
    members: dict[str, Any] = {"products": ProductSpecialist(), "company": CompanySpecialist()}
    router = TaskRouter({"products": ("product", "products"), "company": ("founded", "based", "company")})
    orchestrator = project.programs.register(MultiAgentOrchestrator(members, router))
    product_result = orchestrator.run(question=PRODUCT_QUESTION)
    company_result = orchestrator.run(question=COMPANY_QUESTION)
    product_selected = list(product_result.outputs["orchestrator"]["selected"])
    company_selected = list(company_result.outputs["orchestrator"]["selected"])
    return {
        "product_selected": product_selected,
        "company_selected": company_selected,
        "routes_differ": product_selected != company_selected,
        "grounded": (
            product_result.grounded
            and company_result.grounded
            and _resolves(product_result.citations, chunk_ids)
            and _resolves(company_result.citations, chunk_ids)
        ),
    }


def _dataset(name: str) -> Dataset:
    return Dataset.from_records(
        name,
        [
            {"id": "d1", "query": COMPANY_QUESTION, "expected_terms": ["oronts", "munich"]},
            {"id": "d2", "query": PRODUCT_QUESTION, "expected_terms": ["oronts", "gmbh"]},
        ],
    )


def program_evaluation(project: RefaatProject) -> dict[str, Any]:
    evaluator = ProgramEvaluator(project.programs.get("grounded-rag"))
    report = evaluator.evaluate(_dataset("oronts-eval"), default_bundle())
    return {
        "cases": len(report.cases),
        "grounded": report.aggregate.get("grounded"),
        "hard_passed": report.hard_constraints_passed,
    }


def retrieval_optimization(project: RefaatProject) -> dict[str, Any]:
    before = project.programs.get("grounded-rag").program.artifact().artifact_id
    result = RetrievalOptimizer(project).optimize(
        GroundedRagProgram(),
        _dataset("oronts-optimize"),
        search_space=retrieval_search_space(top_k=(4, 8)),
    )
    after = project.programs.get("grounded-rag").program.artifact().artifact_id
    return {
        "candidate_count": result.candidate_count,
        "pareto_size": len(result.pareto_front),
        "no_mutation": before == after,
    }


def registry_lifecycle(project: RefaatProject) -> dict[str, Any]:
    registry = project.registry
    program = RegistryDemoProgram.name
    base = RegistryDemoProgram().artifact()
    tuned = RegistryDemoProgram(tunables=Tunables(top_k=4)).artifact()
    registry.save(base)
    registry.save(tuned)

    registry.promote(program, "development", base.artifact_id)
    registry.promote(program, "staging", base.artifact_id)

    registry.approve(program, "production", base.artifact_id)
    registry.promote(program, "production", base.artifact_id)
    resolved_base = registry.resolve(program, "production")["definition_hash"] == base.definition_hash

    registry.approve(program, "production", tuned.artifact_id)
    registry.promote(program, "production", tuned.artifact_id)
    resolved_tuned = registry.resolve(program, "production")["definition_hash"] == tuned.definition_hash

    rolled = registry.rollback(program, "production")
    rolled_back = (
        registry.resolve(program, "production")["definition_hash"] == base.definition_hash
        and rolled["artifact_id"] == base.artifact_id
    )
    history = registry.history(program, "production")
    return {
        "environments": ["development", "staging", "production"],
        "promoted": resolved_base and resolved_tuned,
        "rolled_back": rolled_back,
        "history_len": len(history),
    }


def program_as_skill(project: RefaatProject, chunk_ids: set[str]) -> dict[str, Any]:
    registry = project.registry
    artifact = GroundedRagProgram().artifact()
    promoted_id = registry.save(artifact)
    registry.approve("grounded-rag", "production", promoted_id)
    registry.promote("grounded-rag", "production", promoted_id)

    playbook = ProgramSkill(
        project.programs.get("grounded-rag"),
        name=PLAYBOOK_TOOL,
        description="Run the promoted grounded Oronts program as one governed tool.",
    )
    skills = SkillRegistry()
    skills.register(playbook)
    skills.register(EvidenceDigestSkill())
    skills.register(PagerEscalationSkill())
    policy = ReActPolicy(
        max_iterations=4,
        granted_scopes=("read",),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        grants=(
            ToolGrant(PLAYBOOK_TOOL, required_scopes=("read",), side_effect=SideEffectClass.READ_ONLY),
            ToolGrant(DIGEST_TOOL, required_scopes=("read",), side_effect=SideEffectClass.READ_ONLY),
            ToolGrant(PAGER_TOOL, required_scopes=("read",), side_effect=SideEffectClass.EXTERNAL),
        ),
    )
    planner = KeywordToolPlanner(
        {PLAYBOOK_TOOL: ("incident", "drift"), DIGEST_TOOL: ("incident", "drift"), PAGER_TOOL: ("incident", "drift")}
    )
    react = GovernedReActProgram.from_registry(policy=policy, planner=planner, registry=skills)
    react.name = "oronts-incident-agent"
    bound = project.programs.register(react)

    evaluation = ProgramEvaluator(bound).evaluate(
        Dataset.from_records(
            "oronts-trajectory",
            [{"id": "t1", "query": REACT_QUESTION}, {"id": "t2", "query": COMPANY_QUESTION}],
        ),
        trajectory_bundle(),
    )
    run = react.execute(project, inputs={"question": REACT_QUESTION}, user_id=None, session_id=None)
    return {
        "skill_matches_promotion": playbook.artifact_id == promoted_id,
        "trajectory_grounded": run.grounded and _resolves(run.citations, chunk_ids),
        "trajectory_tool_safety": evaluation.aggregate.get("tool_safety"),
        "trajectory_hard_passed": evaluation.hard_constraints_passed,
    }


def model_cache_and_budget(project: RefaatProject, config: ShowcaseConfig, chunk_ids: set[str]) -> dict[str, Any]:
    """Model-call cache plus a per-run token budget, both honest offline.

    The model-call cache is enabled in the runtime settings. Offline the deterministic
    embedder and extractive generator are never wrapped, so the cache is a transparent
    no-op and two runs are byte-identical; online it serves repeated real model calls.
    The budget opens a run-scoped meter through the public budget scope; a bounded
    refine loop consults it at each attempt boundary. Offline the deterministic stack is
    unmetered, so accrual stays zero and the run completes within budget.

    The offline no-op alone would pass even if the meter never enforced, so a low-ceiling
    scope charges the exact CostMeter the refine loop consults at its attempt boundary and
    proves the ceiling is reachable: after one charge that exceeds it, ``exhausted`` flips
    to True, the condition under which the loop denies the next attempt. An unreachable or
    no-op meter would leave it False.
    """
    refine = BoundedInferencePolicy(GroundedRagProgram(), mode=REFINE, attempts=2)
    with budget_scope(token_ceiling=200_000) as meter:
        first = refine.execute(project, inputs={"question": POLICY_QUESTION}, user_id=None, session_id=None)
        second = refine.execute(project, inputs={"question": POLICY_QUESTION}, user_id=None, session_id=None)
    with budget_scope(token_ceiling=8) as low_meter:
        assert low_meter is not None
        before_charge = low_meter.exhausted()
        low_meter.charge({"prompt_tokens": 10, "completion_tokens": 6})
        budget_enforces = low_meter.exhausted() and not before_charge
    return {
        "cache_enabled": config.settings.model_call_cache_enabled,
        "byte_identical": first.answer == second.answer,
        "budget_active": meter is not None,
        "within_budget": meter is not None and not meter.exhausted(),
        "budget_enforces": budget_enforces,
        "grounded": first.grounded and _resolves(first.citations, chunk_ids),
    }


def durable_optimization(project: RefaatProject) -> dict[str, Any]:
    runtime = project.runtime
    dataset = _dataset("oronts-durable-optimize")
    search_space = retrieval_search_space(top_k=(4, 8))
    synchronous = project.programs.get("grounded-rag").optimize(dataset, search_space=search_space)
    run_id = project.programs.get("grounded-rag").enqueue_optimization(dataset, search_space=search_space)
    worked = DurableWorker(runtime).run_once()
    run = runtime.store.get_optimization_run(project.workspace_id, run_id)
    best_artifact_id = str(run["best_artifact_id"])
    registered_best = runtime.registry.get(project.workspace_id, best_artifact_id)
    best_promoted_to_any_env = False
    for environment in ("development", "staging", "production"):
        try:
            promoted_id, _ = runtime.registry.resolve_deployment(project.workspace_id, "grounded-rag", environment)
        except NotFoundError:
            continue
        if promoted_id == best_artifact_id:
            best_promoted_to_any_env = True
    return {
        "worked": worked,
        "status": run["status"],
        "candidate_count": len(runtime.store.list_optimization_candidates(run_id)),
        "matches_synchronous": run["result"] == synchronous.to_json(),
        "registered_best_program": registered_best["program"],
        "best_promoted_to_any_env": best_promoted_to_any_env,
    }


def shadow_canary(project: RefaatProject, chunk_ids: set[str]) -> dict[str, Any]:
    program = CanaryDemoProgram.name
    project.programs.register(CanaryDemoProgram())
    registry = project.registry
    base = CanaryDemoProgram().artifact()
    candidate = CanaryDemoProgram(tunables=Tunables(top_k=4)).artifact()
    registry.save(base)
    registry.save(candidate)
    registry.promote(program, "staging", base.artifact_id)

    project.deployment.set_canary(program, "staging", candidate.artifact_id, percent=0)
    result = project.deployment.run(program, "staging", inputs={"question": POLICY_QUESTION})
    runs = project.deployment.shadow_runs(program, "staging")
    return {
        "served": result.served,
        "comparison_present": result.comparison is not None,
        "shadow_recorded": bool(runs) and runs[0]["served"] == "active",
        "grounded": result.result.grounded and _resolves(result.result.citations, chunk_ids),
    }


FINE_TUNE_BASE_MODEL = "refaat-base-small"


def _fine_tune_dataset() -> Dataset:
    return Dataset.from_records(
        "oronts-fine-tune",
        [
            {"query": COMPANY_QUESTION, "reference_answer": "Oronts GmbH was founded by Refaat Al Ktifan in Munich."},
            {"query": PRODUCT_QUESTION, "reference_answer": "OGuardAI, PimTx, and the Oronts AI Assistant."},
        ],
    )


def governed_fine_tuning(project: RefaatProject) -> dict[str, Any]:
    """Governed fine-tuning: request, approve, durable run, immutable registration, never promoted.

    Offline the deterministic provider trains nothing and records honest provenance
    (``trained`` is False); the run still exercises the full approval gate, durable
    execution, and immutable registration, and the produced model reference is never
    auto-promoted to any environment. Runs over the primary project, building no new
    runtime.
    """
    dataset = _fine_tune_dataset()
    run_id = project.fine_tuning.request(
        dataset,
        base_model=FINE_TUNE_BASE_MODEL,
        hyperparameters={"epochs": 3},
        requester="founder",
    )

    refused_without_approval = False
    try:
        project.fine_tuning.run(run_id)
    except ValidationError:
        refused_without_approval = True

    project.fine_tuning.approve(run_id, approver="lead-reviewer")
    project.fine_tuning.enqueue(run_id)
    worker = DurableWorker(project.runtime)
    for _ in range(20):
        if project.fine_tuning.get(run_id)["status"] == "succeeded":
            break
        if not worker.run_once():
            break

    run = project.fine_tuning.get(run_id)
    base_model = str(run["base_model"])
    program_name = fine_tune_program_name(base_model)
    artifact = fine_tune_artifact(
        base_model=base_model,
        dataset_version=str(run["dataset_version"]),
        hyperparameters=dict(run["hyperparameters"]),
        model_ref=str(run["result_model_ref"]),
    )
    registered = project.runtime.registry.get(project.workspace_id, artifact.artifact_id)

    not_auto_promoted = False
    try:
        project.runtime.registry.resolve(project.workspace_id, program_name, "production")
    except NotFoundError:
        not_auto_promoted = True

    return {
        "refused_without_approval": refused_without_approval,
        "run_status": run["status"],
        "model_ref": str(run["result_model_ref"]),
        "provenance_trained": run["provenance"].get("trained"),
        "provenance_dataset_version": run["provenance"].get("dataset_version"),
        "artifact_registered": registered["program"] == program_name,
        "not_auto_promoted": not_auto_promoted,
    }


def run_program_layer(
    project: RefaatProject, config: ShowcaseConfig, chunk_ids: set[str], bound_react: Any
) -> dict[str, Any]:
    return {
        "typed_program": typed_program(project, chunk_ids),
        "inference_policies": inference_policies(project, chunk_ids),
        "governed_react": governed_react(bound_react, chunk_ids),
        "structured_output": structured_output(project, chunk_ids),
        "multi_agent": multi_agent(project, chunk_ids),
        "program_evaluation": program_evaluation(project),
        "retrieval_optimization": retrieval_optimization(project),
        "registry_lifecycle": registry_lifecycle(project),
        "program_as_skill": program_as_skill(project, chunk_ids),
        "model_cache_and_budget": model_cache_and_budget(project, config, chunk_ids),
        "durable_optimization": durable_optimization(project),
        "shadow_canary": shadow_canary(project, chunk_ids),
    }


def program_layer_grounded(report: Mapping[str, Any]) -> bool:
    return all(
        [
            report["typed_program"]["grounded"],
            report["inference_policies"]["best_of_grounded"],
            report["inference_policies"]["refine_grounded"],
            report["governed_react"]["grounded"],
            report["structured_output"]["grounded"],
            report["multi_agent"]["grounded"],
            report["program_as_skill"]["trajectory_grounded"],
            report["model_cache_and_budget"]["grounded"],
            report["shadow_canary"]["grounded"],
        ]
    )


__all__ = [
    "DIGEST_TOOL",
    "PAGER_TOOL",
    "PLAYBOOK_TOOL",
    "build_governed_react",
    "durable_optimization",
    "governed_fine_tuning",
    "governed_react",
    "inference_policies",
    "model_cache_and_budget",
    "multi_agent",
    "program_as_skill",
    "program_evaluation",
    "program_layer_grounded",
    "registry_lifecycle",
    "retrieval_optimization",
    "run_program_layer",
    "shadow_canary",
    "structured_output",
    "typed_program",
]
