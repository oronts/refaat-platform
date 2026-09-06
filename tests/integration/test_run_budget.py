from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from refaat.application.budget import active_meter, budget_scope
from refaat.application.runtime import Refaat, RefaatProject, RuntimeBuilder
from refaat.config import Settings
from refaat.domain.enums import SideEffectClass
from refaat.domain.models import (
    Citation,
    ContextPacket,
    DocumentInput,
    GenerationRequest,
    GenerationResult,
    JsonObject,
    ReActPolicy,
    ToolGrant,
)
from refaat.errors import ConfigurationError
from refaat.evaluation import Dataset
from refaat.evaluation.dataset import EvaluationCase
from refaat.optimization import RANDOM_SEARCH, BootstrapFewShotOptimizer, retrieval_search_space
from refaat.optimization.bootstrap import _bootstrap_traces
from refaat.optimization.instructions import InstructionOptimizer
from refaat.programs import (
    BEST_OF,
    REFINE,
    BoundedInferencePolicy,
    GovernedReActProgram,
    InputField,
    KeywordToolPlanner,
    OutputField,
    ProgramResult,
    Signature,
)
from refaat.programs.program import GroundedRagProgram, Program, ValidationOutcome

pytestmark = pytest.mark.integration


class _Sig(Signature):
    question: str = InputField(description="q")
    answer: str = OutputField(description="a", grounded=True)


def _scripted(*, grounded: bool, tag: str) -> ProgramResult:
    citations = (Citation(chunk_id="c0", document_id="doc", label="[0]", quote="q"),) if grounded else ()
    return ProgramResult(
        program="charging",
        artifact_id="sha256:x",
        answer=tag,
        citations=citations,
        routes=(),
        cached=False,
        corpus_version=1,
        trace_id="t",
        outputs={},
        validations=(ValidationOutcome(validator="grounding", passed=grounded, detail=""),),
    )


class _ChargingProgram(Program):
    name = "charging"
    input_signature = _Sig
    output_signature = _Sig

    def __init__(self, *, cost: int, grounded: bool = False) -> None:
        super().__init__()
        self._cost = cost
        self._grounded = grounded
        self.calls = 0

    def execute(
        self, project: RefaatProject, *, inputs: Any, user_id: Any, session_id: Any, principal: Any = None
    ) -> ProgramResult:
        self.calls += 1
        meter = active_meter()
        if meter is not None:
            meter.charge({"prompt_tokens": self._cost, "completion_tokens": 0})
        return _scripted(grounded=self._grounded, tag=f"a{self.calls}")


class _ChargingSkill:
    description = "Charges the run meter to simulate a model-backed tool."

    def __init__(self, name: str, cost: int) -> None:
        self.name = name
        self._cost = cost
        self.calls = 0

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        del context, arguments
        self.calls += 1
        meter = active_meter()
        if meter is not None:
            meter.charge({"prompt_tokens": self._cost, "completion_tokens": 0})
        return {"ok": True}


class _CountingGenerator:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "fake-gen-v1"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls += 1
        cited = (request.context.hits[0].chunk.id,) if request.context.hits else ()
        return GenerationResult(
            answer=f"answer {self.calls}",
            cited_chunk_ids=cited,
            usage={"prompt_tokens": 10, "completion_tokens": 0},
        )

    def stream(self, request: GenerationRequest):  # type: ignore[no-untyped-def]
        yield "answer"


def _project(settings: Settings, update: dict[str, Any]) -> tuple[Refaat, RefaatProject]:
    runtime = RuntimeBuilder(settings.model_copy(update=update)).build()
    return runtime, runtime.project(workspace="acme", corpus="budget")


def test_best_of_stops_early_and_selects_best_under_a_low_budget(settings: Settings) -> None:
    runtime, project = _project(settings, {"run_token_budget": 25})
    try:
        inner = _ChargingProgram(cost=10)
        policy = BoundedInferencePolicy(inner, mode=BEST_OF, attempts=5)
        result = policy.execute(project, inputs={"question": "q"}, user_id=None, session_id=None)
        # Boundary enforcement: runs while not yet exhausted (10, 20, 30), then stops
        # before the 4th attempt once the ceiling is met. Never returns nothing.
        assert inner.calls == 3
        assert result.answer in {"a1", "a2", "a3"}
    finally:
        runtime.close()


def test_best_of_runs_every_attempt_when_no_budget_is_set(settings: Settings) -> None:
    runtime, project = _project(settings, {})
    try:
        inner = _ChargingProgram(cost=10)
        policy = BoundedInferencePolicy(inner, mode=BEST_OF, attempts=5)
        policy.execute(project, inputs={"question": "q"}, user_id=None, session_id=None)
        assert inner.calls == 5
    finally:
        runtime.close()


def test_refine_stops_on_budget_before_exhausting_attempts(settings: Settings) -> None:
    runtime, project = _project(settings, {"run_token_budget": 25})
    try:
        inner = _ChargingProgram(cost=10, grounded=False)
        policy = BoundedInferencePolicy(inner, mode=REFINE, attempts=6)
        result = policy.execute(project, inputs={"question": "q"}, user_id=None, session_id=None)
        assert inner.calls == 3
        assert result.answer in {"a1", "a2", "a3"}
    finally:
        runtime.close()


def test_governed_react_tool_loop_stops_early_under_budget(settings: Settings) -> None:
    runtime, project = _project(settings, {"run_token_budget": 25})
    try:
        runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id="ops.txt",
                title="Ops",
                media_type="text/plain",
                content=b"Rollback restores the previously approved snapshot before deployment.",
            ),
        )
        skills = {f"t{index}": _ChargingSkill(f"t{index}", cost=10) for index in range(5)}
        grants = tuple(
            ToolGrant(name, required_scopes=("read:corpus",), side_effect=SideEffectClass.READ_ONLY) for name in skills
        )
        policy = ReActPolicy(max_iterations=5, granted_scopes=("read:corpus",), grants=grants)
        planner = KeywordToolPlanner(dict.fromkeys(skills, ("rollback",)))
        program = GovernedReActProgram(policy=policy, planner=planner, skills=dict(skills))

        result = program.execute(project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None)

        assert sum(skill.calls for skill in skills.values()) == 3
        # The final grounded generation always runs, so an exhausted run still answers.
        assert result.grounded
    finally:
        runtime.close()


def test_program_artifact_identity_is_unchanged_by_a_budget(settings: Settings) -> None:
    plain, plain_project = _project(settings, {})
    budgeted, budgeted_project = _project(settings, {"run_token_budget": 5, "model_call_cache_enabled": True})
    try:
        assert (
            plain_project.programs.get("grounded-rag").artifact
            == budgeted_project.programs.get("grounded-rag").artifact
        )
    finally:
        plain.close()
        budgeted.close()


def test_optimization_stops_evaluating_candidates_under_a_low_budget(settings: Settings) -> None:
    generator = _CountingGenerator()
    runtime = RuntimeBuilder(settings.model_copy(update={"run_token_budget": 15})).with_generator(generator).build()
    try:
        project = runtime.project(workspace="acme", corpus="optbudget")
        runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id="run.txt",
                title="Runbook",
                media_type="text/plain",
                content=b"Escalate a Sev1 incident to on-call within fifteen minutes.",
            ),
        )
        dataset = Dataset.from_records("opt", [{"query": "incident escalation", "expected_terms": ["incident"]}])
        result = project.programs.get("grounded-rag").optimize(
            dataset, search_space=retrieval_search_space(top_k=(2, 6, 12))
        )
        assert result.candidate_count < 3
    finally:
        runtime.close()


class _MeterProbeGenerator:
    """A metered generator that records whether a run meter was active per call."""

    def __init__(self) -> None:
        self.meter_active: list[bool] = []

    @property
    def model_id(self) -> str:
        return "probe-gen-v1"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.meter_active.append(active_meter() is not None)
        cited = (request.context.hits[0].chunk.id,) if request.context.hits else ()
        return GenerationResult(answer="probe answer", cited_chunk_ids=cited, usage={"prompt_tokens": 3})

    def stream(self, request: GenerationRequest):  # type: ignore[no-untyped-def]
        yield "probe answer"


def _probe_project(settings: Settings, update: dict[str, Any]) -> tuple[Refaat, RefaatProject, _MeterProbeGenerator]:
    probe = _MeterProbeGenerator()
    runtime = RuntimeBuilder(settings.model_copy(update=update)).with_generator(probe).build()
    project = runtime.project(workspace="acme", corpus="probe")
    runtime.ingest(
        project.workspace_id,
        project.corpus_id,
        DocumentInput(
            source_id="run.txt",
            title="Runbook",
            media_type="text/plain",
            content=b"Escalate a Sev1 incident to on-call within fifteen minutes.",
        ),
    )
    return runtime, project, probe


def test_plain_ask_opens_a_budget_scope_when_configured(settings: Settings) -> None:
    runtime, project, probe = _probe_project(settings, {"run_token_budget": 1000})
    try:
        project.ask("incident escalation")
        assert probe.meter_active == [True]
    finally:
        runtime.close()


def test_plain_ask_installs_no_meter_without_a_budget(settings: Settings) -> None:
    runtime, project, probe = _probe_project(settings, {})
    try:
        project.ask("incident escalation")
        assert probe.meter_active == [False]
    finally:
        runtime.close()


class _EchoChargingGenerator:
    """A metered generator that echoes the top hit so teacher cases pass the gate."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "echo-gen-v1"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls += 1
        hits = request.context.hits
        answer = hits[0].chunk.text if hits else "no grounded context"
        cited = (hits[0].chunk.id,) if hits else ()
        return GenerationResult(
            answer=answer, cited_chunk_ids=cited, usage={"prompt_tokens": 10, "completion_tokens": 0}
        )

    def stream(self, request: GenerationRequest):  # type: ignore[no-untyped-def]
        yield "echo"


def _bootstrap_dataset() -> Dataset:
    return Dataset.from_records(
        "bootstrap-budget",
        [
            {"id": "a", "query": "incident escalation path", "expected_terms": ["incident"]},
            {"id": "b", "query": "on-call response window", "expected_terms": ["minutes"]},
            {"id": "c", "query": "sev1 handling", "expected_terms": ["sev1"]},
            {"id": "d", "query": "unicorn taxonomy", "expected_terms": ["unicorn"]},
            {"id": "e", "query": "runbook location", "expected_terms": ["runbook"]},
        ],
    )


def _run_bootstrap(settings: Settings, update: dict[str, Any], corpus: str) -> int:
    runtime = RuntimeBuilder(settings.model_copy(update=update)).with_generator(_EchoChargingGenerator()).build()
    try:
        project = runtime.project(workspace="acme", corpus=corpus)
        for index in range(4):
            runtime.ingest(
                project.workspace_id,
                project.corpus_id,
                DocumentInput(
                    source_id=f"doc{index}.txt",
                    title=f"Doc {index}",
                    media_type="text/plain",
                    content=f"Runbook {index}: escalate a Sev1 incident to on-call within fifteen minutes.".encode(),
                ),
            )
        result = BootstrapFewShotOptimizer(project).optimize(
            GroundedRagProgram(), _bootstrap_dataset(), mode=RANDOM_SEARCH, trials=8, seed=0
        )
        return result.candidate_count
    finally:
        runtime.close()


def test_bootstrap_optimizer_stops_evaluating_candidates_under_a_low_budget(settings: Settings) -> None:
    # Distinct corpora so the second run does not read the first run's answer cache
    # (a shared cache would skip the generator and never charge the meter).
    full = _run_bootstrap(settings, {}, corpus="bootfull")
    bounded = _run_bootstrap(settings, {"run_token_budget": 15}, corpus="bootbounded")
    assert full > 1
    assert 1 <= bounded < full


def test_cost_only_budget_without_prices_fails_closed() -> None:
    update = {
        "run_cost_budget_usd": 1.0,
        "generation_provider": "openai_compatible",
        "generation_model": "gpt-4o-mini",
        "openai_api_key": "sk-test",
    }
    with pytest.raises(ValueError, match="REFAAT_MODEL_PRICES"):
        Settings(environment="test", **update).validate_for_startup()

    priced = Settings(
        environment="test",
        model_prices={"gpt-4o-mini": {"input_per_1k": 0.15, "output_per_1k": 0.6}},
        **update,
    )
    priced.validate_for_startup()


def test_cost_only_budget_without_embedding_prices_fails_closed() -> None:
    update = {
        "run_cost_budget_usd": 1.0,
        "embedding_provider": "openai_compatible",
        "embedding_model": "text-embedding-3-small",
        "openai_api_key": "sk-test",
    }
    with pytest.raises(ValueError, match="embedding model"):
        Settings(environment="test", **update).validate_for_startup()

    priced = Settings(
        environment="test",
        model_prices={"text-embedding-3-small": {"input_per_1k": 0.02, "output_per_1k": 0.0}},
        **update,
    )
    priced.validate_for_startup()


class _MeterProbeEmbedder:
    """A metered embedder that records whether a run meter was active per call."""

    def __init__(self) -> None:
        self.meter_active: list[bool] = []

    @property
    def model_id(self) -> str:
        return "probe-embed-v1"

    @property
    def dimensions(self) -> int:
        return 8

    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        self.meter_active.append(active_meter() is not None)
        return [tuple(float((sum(map(ord, text)) + index) % 7) for index in range(8)) for text in texts]


def _embed_probe_project(
    settings: Settings, update: dict[str, Any]
) -> tuple[Refaat, RefaatProject, _MeterProbeEmbedder]:
    probe = _MeterProbeEmbedder()
    runtime = RuntimeBuilder(settings.model_copy(update=update)).with_embedder(probe).build()
    project = runtime.project(workspace="acme", corpus="embedprobe")
    return runtime, project, probe


def _ingest_ops(runtime: Refaat, project: RefaatProject) -> None:
    runtime.ingest(
        project.workspace_id,
        project.corpus_id,
        DocumentInput(
            source_id="ops.txt",
            title="Ops",
            media_type="text/plain",
            content=b"Rollback restores the previously approved snapshot before deployment.",
        ),
    )


# R2: an ingest run opens a budget scope, so its embeds are metered.
def test_ingest_opens_a_budget_scope_when_configured(settings: Settings) -> None:
    runtime, project, probe = _embed_probe_project(settings, {"run_token_budget": 1000})
    try:
        _ingest_ops(runtime, project)
        assert probe.meter_active
        assert all(probe.meter_active)
    finally:
        runtime.close()


def test_ingest_installs_no_meter_without_a_budget(settings: Settings) -> None:
    runtime, project, probe = _embed_probe_project(settings, {})
    try:
        _ingest_ops(runtime, project)
        assert probe.meter_active
        assert not any(probe.meter_active)
    finally:
        runtime.close()


# R2: a retrieve run (skill-prepare) opens a budget scope, so its embed is metered.
def test_retrieve_opens_a_budget_scope_when_configured(settings: Settings) -> None:
    runtime, project, probe = _embed_probe_project(settings, {"run_token_budget": 1000})
    try:
        _ingest_ops(runtime, project)
        probe.meter_active.clear()
        project.retrieve("how does rollback work")
        assert probe.meter_active
        assert all(probe.meter_active)
    finally:
        runtime.close()


def test_retrieve_installs_no_meter_without_a_budget(settings: Settings) -> None:
    runtime, project, probe = _embed_probe_project(settings, {})
    try:
        _ingest_ops(runtime, project)
        probe.meter_active.clear()
        project.retrieve("how does rollback work")
        assert probe.meter_active
        assert not any(probe.meter_active)
    finally:
        runtime.close()


# R2: an RLM run opens a budget scope, so its generations are metered.
def test_rlm_opens_a_budget_scope_when_configured(settings: Settings) -> None:
    probe = _MeterProbeGenerator()
    runtime = RuntimeBuilder(settings.model_copy(update={"run_token_budget": 1000})).with_generator(probe).build()
    try:
        project = runtime.project(workspace="acme", corpus="rlmprobe")
        _ingest_ops(runtime, project)
        project.rlm("how does rollback work")
        assert probe.meter_active
        assert all(probe.meter_active)
    finally:
        runtime.close()


def test_rlm_installs_no_meter_without_a_budget(settings: Settings) -> None:
    probe = _MeterProbeGenerator()
    runtime = RuntimeBuilder(settings.model_copy(update={})).with_generator(probe).build()
    try:
        project = runtime.project(workspace="acme", corpus="rlmprobe")
        _ingest_ops(runtime, project)
        project.rlm("how does rollback work")
        assert probe.meter_active
        assert not any(probe.meter_active)
    finally:
        runtime.close()


# R3: GEPA's initial held-out eval runs inside the one shared budget scope, so a low
# budget bounds the whole search (fewer candidates than the unbounded run).
def _run_gepa(settings: Settings, update: dict[str, Any], corpus: str) -> int:
    runtime = RuntimeBuilder(settings.model_copy(update=update)).with_generator(_EchoChargingGenerator()).build()
    try:
        project = runtime.project(workspace="acme", corpus=corpus)
        for index in range(4):
            runtime.ingest(
                project.workspace_id,
                project.corpus_id,
                DocumentInput(
                    source_id=f"doc{index}.txt",
                    title=f"Doc {index}",
                    media_type="text/plain",
                    content=f"Runbook {index}: escalate a Sev1 incident to on-call within fifteen minutes.".encode(),
                ),
            )
        result = InstructionOptimizer(project).optimize_gepa(GroundedRagProgram(), _bootstrap_dataset(), rounds=3)
        return result.candidate_count
    finally:
        runtime.close()


def test_gepa_stops_under_a_low_budget(settings: Settings) -> None:
    full = _run_gepa(settings, {}, "gepafull")
    bounded = _run_gepa(settings, {"run_token_budget": 15}, "gepabounded")
    assert full > 1
    assert 1 <= bounded < full


# R3: the bootstrap teacher loop stops on the shared meter boundary.
class _ChargingRunner:
    def __init__(self, cost: int) -> None:
        self._cost = cost
        self.calls = 0

    def run(self, *, question: str) -> ProgramResult:
        del question
        self.calls += 1
        meter = active_meter()
        if meter is not None:
            meter.charge({"prompt_tokens": self._cost})
        return _scripted(grounded=True, tag="incident")


def _teacher_cases() -> tuple[EvaluationCase, ...]:
    return tuple(
        EvaluationCase(id=f"c{index}", query=f"incident case {index}", expected_terms=("incident",))
        for index in range(6)
    )


def test_bootstrap_teacher_loop_stops_under_a_low_budget() -> None:
    cases = _teacher_cases()

    full = _ChargingRunner(cost=10)
    _bootstrap_traces(full, cases, (), max_demos=10)

    bounded = _ChargingRunner(cost=10)
    with budget_scope(15):
        _bootstrap_traces(bounded, cases, (), max_demos=10)

    assert full.calls == len(cases)
    assert bounded.calls < full.calls


# R6: an empty price record fails closed at startup (membership alone is not enough).
def test_cost_budget_with_empty_generation_price_record_fails_closed() -> None:
    with pytest.raises(ValueError, match="positive input and output"):
        Settings(
            environment="test",
            run_cost_budget_usd=1.0,
            generation_provider="openai_compatible",
            generation_model="gpt-4o-mini",
            openai_api_key="sk-test",
            model_prices={"gpt-4o-mini": {}},
        ).validate_for_startup()


# R6: an injected metered adapter whose model is unpriced fails closed under a cost
# budget, even when the settings providers stay on the extractive/deterministic stack
# and so bypass the startup guard.
def test_injected_metered_generator_unpriced_fails_closed_under_cost_budget(settings: Settings) -> None:
    with pytest.raises(ConfigurationError, match="metered generation"):
        (
            RuntimeBuilder(settings.model_copy(update={"run_cost_budget_usd": 1.0, "run_token_budget": 100}))
            .with_generator(_CountingGenerator())
            .build()
        )


# R6: a call-time optimizer cost budget with a price map that cannot price the active
# generation model fails closed.
def test_call_time_optimizer_cost_budget_without_prices_fails_closed(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings.model_copy(update={})).build()
    try:
        project = runtime.project(workspace="acme", corpus="optcost")
        runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id="run.txt",
                title="Runbook",
                media_type="text/plain",
                content=b"Escalate a Sev1 incident to on-call within fifteen minutes.",
            ),
        )
        dataset = Dataset.from_records("opt", [{"query": "incident escalation", "expected_terms": ["incident"]}])
        with pytest.raises(ValueError, match="cost budget requires REFAAT_MODEL_PRICES"):
            project.programs.get("grounded-rag").optimize(
                dataset, search_space=retrieval_search_space(top_k=(2, 6)), cost_budget_usd=1.0
            )
    finally:
        runtime.close()
