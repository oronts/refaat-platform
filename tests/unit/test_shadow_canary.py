from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError

from refaat.application.deployment import DeploymentRouter, _canonicalize
from refaat.core.text import stable_digest
from refaat.domain.enums import SideEffectClass
from refaat.domain.models import Citation, Demonstration, ReActPolicy, ToolGrant
from refaat.errors import ValidationError
from refaat.infrastructure.database.store import SQLStore
from refaat.optimization.search import default_search_space, retrieval_search_space
from refaat.optimization.telemetry import compare_results
from refaat.programs import (
    GovernedReActProgram,
    KeywordToolPlanner,
    MultiAgentOrchestrator,
    TaskRouter,
)
from refaat.programs.ensemble import EnsembleProgram
from refaat.programs.policy import BoundedInferencePolicy
from refaat.programs.program import (
    GroundedRagProgram,
    Program,
    ProgramResult,
    Tunables,
    ValidationOutcome,
)


def _result(
    *,
    answer: str,
    chunk_ids: tuple[str, ...],
    routes: tuple[str, ...],
    grounded: bool,
    context_tokens: int,
) -> ProgramResult:
    citations = tuple(
        Citation(chunk_id=cid, document_id="doc", label=f"[{index}]", quote="q") for index, cid in enumerate(chunk_ids)
    )
    validations = (ValidationOutcome(validator="grounding", passed=grounded, detail=""),)
    return ProgramResult(
        program="grounded-rag",
        artifact_id="sha256:x",
        answer=answer,
        citations=citations,
        routes=routes,
        cached=False,
        corpus_version=1,
        trace_id="t",
        outputs={},
        validations=validations,
        usage={"context_tokens": context_tokens},
    )


def test_tunables_roundtrip_preserves_definition_hash() -> None:
    program = GroundedRagProgram()
    spaces = [*default_search_space(), *retrieval_search_space()]
    spaces.append(
        Tunables(
            top_k=5,
            token_budget=400,
            minimum_citation_coverage=0.5,
            fusion="rrf",
            rrf_k=60,
            autocut_enabled=True,
            autocut_threshold=0.2,
            instructions="Answer tersely.",
            demonstrations=(Demonstration(question="q1", answer="a1"), Demonstration(question="q2", answer="a2")),
            router={"strategy": "keyword", "routes": []},
        )
    )
    assert any(t.demonstrations for t in spaces)
    for tunables in spaces:
        direct = program.with_tunables(tunables).artifact().definition_hash
        roundtrip = program.with_tunables(Tunables.from_json(tunables.as_json())).artifact().definition_hash
        assert direct == roundtrip


def test_compare_results_exact_signals() -> None:
    active = _result(answer="same", chunk_ids=("a", "b"), routes=("vector",), grounded=True, context_tokens=100)
    candidate = _result(
        answer="same", chunk_ids=("b", "c"), routes=("vector", "bm25"), grounded=False, context_tokens=130
    )
    comparison = compare_results(active, candidate)
    assert comparison.answers_equal is True
    assert comparison.citation_overlap == 1
    assert comparison.citation_jaccard == pytest.approx(1 / 3)
    assert comparison.active_grounded is True
    assert comparison.candidate_grounded is False
    assert comparison.routes_equal is False
    assert comparison.token_delta == 30


def test_compare_results_empty_citations_are_equal() -> None:
    active = _result(answer="x", chunk_ids=(), routes=(), grounded=False, context_tokens=0)
    candidate = _result(answer="y", chunk_ids=(), routes=(), grounded=False, context_tokens=0)
    comparison = compare_results(active, candidate)
    assert comparison.citation_jaccard == 1.0
    assert comparison.citation_overlap == 0
    assert comparison.answers_equal is False


def test_bucket_formula_is_deterministic() -> None:
    parts = ["ws1", "grounded-rag", "production", "user-42"]
    first = int(stable_digest(parts, length=8), 16) % 100
    second = int(stable_digest(parts, length=8), 16) % 100
    assert first == second
    assert 0 <= first < 100


class _DummySkill:
    name = "tool"
    description = "dummy"

    def run(self, context: object, arguments: object) -> dict[str, bool]:
        return {"ok": True}


def _write_react() -> GovernedReActProgram:
    policy = ReActPolicy(
        granted_scopes=("write:crm",),
        side_effect_ceiling=SideEffectClass.WRITE,
        grants=(ToolGrant("update", required_scopes=("write:crm",), side_effect=SideEffectClass.WRITE),),
    )
    return GovernedReActProgram(
        policy=policy, planner=KeywordToolPlanner({"update": ("update",)}), skills={"update": _DummySkill()}
    )


def _read_only_react() -> GovernedReActProgram:
    policy = ReActPolicy(
        granted_scopes=("read:corpus",),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        grants=(ToolGrant("lookup", required_scopes=("read:corpus",)),),
    )
    return GovernedReActProgram(
        policy=policy, planner=KeywordToolPlanner({"lookup": ("policy",)}), skills={"lookup": _DummySkill()}
    )


def test_shadow_safe_reflects_side_effect_and_handoff() -> None:
    assert GroundedRagProgram().shadow_safe is True
    assert _read_only_react().shadow_safe is True
    assert _write_react().shadow_safe is False
    # A write grant above a read-only ceiling is always denied, so it stays safe to shadow.
    denied_policy = ReActPolicy(
        granted_scopes=("write:crm",),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        grants=(ToolGrant("update", required_scopes=("write:crm",), side_effect=SideEffectClass.WRITE),),
    )
    denied = GovernedReActProgram(
        policy=denied_policy, planner=KeywordToolPlanner({"update": ("update",)}), skills={"update": _DummySkill()}
    )
    assert denied.shadow_safe is True

    members = {"a": GroundedRagProgram(), "b": GroundedRagProgram(tunables=Tunables(top_k=2))}
    router = TaskRouter({"a": ("x",), "b": ("y",)})
    assert MultiAgentOrchestrator(members, router, handoff=False).shadow_safe is True
    assert MultiAgentOrchestrator(members, router, handoff=True).shadow_safe is False


def test_shadow_safe_base_default_is_false() -> None:
    # Safe-by-default: a custom program that does not declare its side-effect profile is never
    # silently canaried, so the base default is False and only a program that asserts it is True.
    class _Custom(Program):
        name = "custom"
        input_signature = GroundedRagProgram.input_signature
        output_signature = GroundedRagProgram.output_signature

        def execute(self, project, *, inputs, user_id, session_id, principal=None):  # type: ignore[override]
            raise NotImplementedError

    assert _Custom().shadow_safe is False


def test_shadow_safe_wrappers_delegate_to_what_they_wrap() -> None:
    # A wrapper is shadow-safe only when what it wraps is: a best-of loop, an ensemble, and an
    # orchestrator over a writing ReAct program are each refused a canary, while over pure programs
    # they are allowed.
    assert BoundedInferencePolicy(GroundedRagProgram()).shadow_safe is True
    assert BoundedInferencePolicy(_write_react()).shadow_safe is False

    pure = [GroundedRagProgram(), GroundedRagProgram(tunables=Tunables(top_k=2))]
    assert EnsembleProgram(pure).shadow_safe is True
    assert EnsembleProgram([GroundedRagProgram(), _write_react()]).shadow_safe is False

    router = TaskRouter({"a": ("x",), "b": ("y",)})
    unsafe_members = {"a": GroundedRagProgram(), "b": _write_react()}
    assert MultiAgentOrchestrator(unsafe_members, router, handoff=False).shadow_safe is False


def test_rehydrate_guards_definition_and_immutable_policy_but_not_version() -> None:
    base = GroundedRagProgram()
    document = base.artifact().to_json()

    # An exact match rehydrates.
    assert DeploymentRouter._rehydrate(base, document) is not None

    # A benign version bump changes only artifact_id and dependency_versions, never the
    # definition hash or the immutable policy hash, so it still serves.
    version_bump = {**document, "dependency_versions": {"refaat": "999.0.0"}}
    assert version_bump["dependency_versions"] != document["dependency_versions"]
    assert DeploymentRouter._rehydrate(base, version_bump) is not None

    # An immutable-policy drift (a base-prompt change) is refused even though the
    # definition hash still matches: this is the approval-bypass the guard closes.
    policy_drift = {**document, "immutable_policy_hash": "deadbeef"}
    with pytest.raises(ValidationError, match="immutable policy"):
        DeploymentRouter._rehydrate(base, policy_drift)

    # A definition drift (tampered tunables) is still refused.
    definition_drift = {**document, "definition_hash": "deadbeef"}
    with pytest.raises(ValidationError, match="definition"):
        DeploymentRouter._rehydrate(base, definition_drift)


def test_query_text_fallback_is_canonical_and_order_independent() -> None:
    a = DeploymentRouter._query_text({"tags": {"z", "a", "m"}, "n": 3})
    b = DeploymentRouter._query_text({"n": 3, "tags": {"m", "a", "z"}})
    assert a == b
    # A string question routes on its own text, not the digest.
    assert DeploymentRouter._query_text({"question": "hello"}) == "hello"


def test_canonicalize_normalizes_sets_and_nesting() -> None:
    assert _canonicalize({"s": {3, 1, 2}}) == {"s": [1, 2, 3]}
    assert _canonicalize((1, {2, 3})) == [1, [2, 3]]


def test_query_text_fallback_is_deterministic_across_hash_seeds() -> None:
    src = str(Path(__file__).resolve().parents[2] / "src")
    script = (
        "from refaat.application.deployment import DeploymentRouter;"
        "print(DeploymentRouter._query_text({'tags': {'z', 'a', 'm', 'q'}, 'n': 3}))"
    )
    outputs = []
    for seed in ("0", "1", "12345"):
        completed = subprocess.run(  # noqa: S603
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": src},
            check=True,
        )
        outputs.append(completed.stdout.strip())
    assert len(set(outputs)) == 1
    assert outputs[0]


def test_retry_returning_recomputes_on_integrity_error() -> None:
    attempts = {"count": 0}

    def flaky() -> dict[str, int]:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise IntegrityError("insert", {}, Exception("duplicate sequence"))
        return {"sequence": attempts["count"]}

    result = SQLStore._retry_returning(flaky)
    assert result == {"sequence": 3}


def test_retry_returning_reraises_after_exhausting_attempts() -> None:
    def always_conflicts() -> dict[str, int]:
        raise IntegrityError("insert", {}, Exception("duplicate sequence"))

    with pytest.raises(IntegrityError):
        SQLStore._retry_returning(always_conflicts)
