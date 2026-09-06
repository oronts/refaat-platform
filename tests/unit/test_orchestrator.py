from __future__ import annotations

from dataclasses import replace

import pytest

from refaat.domain.models import Citation
from refaat.errors import ValidationError
from refaat.programs import (
    EnsembleProgram,
    GroundedRagInput,
    GroundedRagOutput,
    InputField,
    MultiAgentOrchestrator,
    OutputField,
    Signature,
    TaskRouter,
)
from refaat.programs.program import Program, ProgramResult, ValidationOutcome


class _AltInput(Signature):
    prompt: str = InputField(description="a different input field")


class _AltOutput(Signature):
    answer: str = OutputField(description="a different output field", grounded=True)


def _result(
    *,
    grounded: bool,
    citations: int,
    program: str = "m",
    answer: str = "answer",
    tokens: float = 10.0,
    chunk_prefix: str = "c",
) -> ProgramResult:
    cites = tuple(
        Citation(chunk_id=f"{chunk_prefix}{index}", document_id="doc", label=f"[{index}]", quote="q")
        for index in range(citations)
    )
    return ProgramResult(
        program=program,
        artifact_id="sha256:x",
        answer=answer,
        citations=cites,
        routes=(),
        cached=False,
        corpus_version=1,
        trace_id="t",
        outputs={},
        validations=(ValidationOutcome(validator="grounding", passed=grounded, detail=""),),
        usage={"context_tokens": tokens},
    )


class _Scripted(Program):
    input_signature = GroundedRagInput
    output_signature = GroundedRagOutput

    def __init__(
        self,
        result: ProgramResult,
        tag: str,
        *,
        input_signature: type[Signature] | None = None,
        output_signature: type[Signature] | None = None,
    ) -> None:
        super().__init__()
        self._result = result
        self.name = tag
        self.calls = 0
        if input_signature is not None:
            self.input_signature = input_signature
        if output_signature is not None:
            self.output_signature = output_signature

    def execute(self, project, *, inputs, user_id, session_id, principal=None) -> ProgramResult:
        self.calls += 1
        return self._result


class _PickFirst:
    name = "pick-first"

    def reduce(self, results):
        return results[0]


class _Fabricate:
    name = "fabricate"

    def reduce(self, results):
        forged = Citation(chunk_id="forged", document_id="doc", label="[x]", quote="q")
        return replace(results[0], citations=(forged,))


class _BorrowChunkId:
    name = "borrow"

    def reduce(self, results):
        borrowed = Citation(chunk_id="a0", document_id="doc", label="[x]", quote="forged-quote")
        return replace(results[0], citations=(borrowed,))


class _ReservedName:
    name = "select_best"

    def reduce(self, results):
        return results[0]


def _orchestrator(**kwargs) -> tuple[MultiAgentOrchestrator, _Scripted, _Scripted]:
    a = _Scripted(_result(grounded=False, citations=5, program="a", chunk_prefix="a"), "a")
    b = _Scripted(_result(grounded=True, citations=1, program="b", chunk_prefix="b"), "b")
    router = TaskRouter({"a": ("alpha",), "b": ("beta",)})
    return MultiAgentOrchestrator({"a": a, "b": b}, router, **kwargs), a, b


def test_route_selects_the_expected_subset_in_sorted_order() -> None:
    router = TaskRouter({"zeta": ("z",), "alpha": ("a",), "mid": ("m",)})
    assert router.route("a and z present") == ("alpha", "zeta")
    assert router.route("a m z") == ("alpha", "mid", "zeta")


def test_no_match_falls_back_to_the_full_sorted_set() -> None:
    router = TaskRouter({"zeta": ("z",), "alpha": ("a",)})
    assert router.route("nothing here") == ("alpha", "zeta")


def test_identical_task_gives_an_identical_tuple() -> None:
    router = TaskRouter({"b": ("beta",), "a": ("alpha",)})
    task = "alpha and beta"
    assert router.route(task) == router.route(task) == ("a", "b")


def test_input_signature_mismatch_is_rejected() -> None:
    a = _Scripted(_result(grounded=True, citations=1), "a")
    b = _Scripted(_result(grounded=True, citations=1), "b", input_signature=_AltInput)
    with pytest.raises(ValidationError, match="input signature"):
        MultiAgentOrchestrator({"a": a, "b": b}, TaskRouter({"a": ("x",)}))


def test_output_signature_mismatch_is_rejected() -> None:
    a = _Scripted(_result(grounded=True, citations=1), "a")
    b = _Scripted(_result(grounded=True, citations=1), "b", output_signature=_AltOutput)
    with pytest.raises(ValidationError, match="output signature"):
        MultiAgentOrchestrator({"a": a, "b": b}, TaskRouter({"a": ("x",)}))


def test_empty_members_are_rejected() -> None:
    with pytest.raises(ValidationError, match="at least one member"):
        MultiAgentOrchestrator({}, TaskRouter({"a": ("x",)}))


def test_unknown_specialist_router_is_rejected() -> None:
    a = _Scripted(_result(grounded=True, citations=1), "a")
    with pytest.raises(ValidationError, match="unknown specialists: ghost"):
        MultiAgentOrchestrator({"a": a}, TaskRouter({"a": ("x",), "ghost": ("y",)}))


def test_default_reducer_selects_the_grounded_best() -> None:
    orch, _a, _b = _orchestrator()
    out = orch.execute(None, inputs={"question": "alpha beta"}, user_id=None, session_id=None)
    assert out.program == "multi-agent"
    assert out.grounded
    assert out.outputs["orchestrator"]["chosen"] == "b"
    assert out.outputs["orchestrator"]["selected"] == ["a", "b"]
    assert out.outputs["orchestrator"]["reducer"] == "select_best"


def test_named_custom_reducer_overrides_selection() -> None:
    # select_best would pick the grounded "b"; the custom reducer takes the first ("a"),
    # so the override is observable rather than coinciding with the default.
    orch, _a, _b = _orchestrator(reducer=_PickFirst())
    out = orch.execute(None, inputs={"question": "alpha beta"}, user_id=None, session_id=None)
    assert out.outputs["orchestrator"]["chosen"] == "a"
    assert out.outputs["orchestrator"]["reducer"] == "pick-first"
    assert not out.grounded


def test_reducer_citations_outside_the_member_union_are_rejected() -> None:
    orch, _a, _b = _orchestrator(reducer=_Fabricate())
    with pytest.raises(ValidationError, match="outside the member results: forged"):
        orch.execute(None, inputs={"question": "alpha beta"}, user_id=None, session_id=None)


def test_reducer_reusing_a_member_chunk_id_with_a_forged_quote_is_rejected() -> None:
    # The chunk id a0 belongs to a member, but the full citation is fabricated, so the
    # whole-identity check rejects it where a chunk-id-only check would have passed it.
    orch, _a, _b = _orchestrator(reducer=_BorrowChunkId())
    with pytest.raises(ValidationError, match="outside the member results: a0"):
        orch.execute(None, inputs={"question": "alpha beta"}, user_id=None, session_id=None)


def test_custom_reducer_cannot_claim_the_reserved_default_name() -> None:
    a = _Scripted(_result(grounded=True, citations=1), "a")
    with pytest.raises(ValidationError, match="reserved"):
        MultiAgentOrchestrator({"a": a}, TaskRouter({"a": ("x",)}), reducer=_ReservedName())


def test_custom_score_changes_the_orchestrator_id() -> None:
    def custom_score(result):
        return (True, 0, 0)

    a = _Scripted(_result(grounded=True, citations=1, chunk_prefix="a"), "a")
    b = _Scripted(_result(grounded=True, citations=1, chunk_prefix="b"), "b")
    members = {"a": a, "b": b}
    router = TaskRouter({"a": ("alpha",), "b": ("beta",)})
    default_id = MultiAgentOrchestrator(members, router).artifact().artifact_id
    custom_id = MultiAgentOrchestrator(members, router, score=custom_score).artifact().artifact_id
    assert default_id != custom_id


def test_merged_citations_are_a_subset_of_the_member_union() -> None:
    orch, _a, _b = _orchestrator()
    out = orch.execute(None, inputs={"question": "alpha beta"}, user_id=None, session_id=None)
    union = {"a0", "a1", "a2", "a3", "a4", "b0"}
    assert {citation.chunk_id for citation in out.citations} <= union


def test_artifact_pins_members_router_and_reducer_identity() -> None:
    orch, a, b = _orchestrator()
    extra = orch._definition_extra()["orchestrator"]
    assert {entry["artifact_id"] for entry in extra["members"]} == {
        a.artifact().artifact_id,
        b.artifact().artifact_id,
    }
    assert extra["router"] == {"a": ["alpha"], "b": ["beta"]}
    assert extra["reducer"] == "select_best"


def test_mutations_change_the_orchestrator_id() -> None:
    a = _Scripted(_result(grounded=True, citations=1, chunk_prefix="a"), "a")
    b = _Scripted(_result(grounded=True, citations=1, chunk_prefix="b"), "b")
    members = {"a": a, "b": b}
    base = MultiAgentOrchestrator(members, TaskRouter({"a": ("alpha",), "b": ("beta",)}))
    other_keyword = MultiAgentOrchestrator(members, TaskRouter({"a": ("gamma",), "b": ("beta",)}))
    named = MultiAgentOrchestrator(members, TaskRouter({"a": ("alpha",), "b": ("beta",)}), reducer=_PickFirst())
    fabricate = MultiAgentOrchestrator(members, TaskRouter({"a": ("alpha",), "b": ("beta",)}), reducer=_Fabricate())

    base_id = base.artifact().artifact_id
    ids = {
        base_id,
        other_keyword.artifact().artifact_id,
        named.artifact().artifact_id,
        fabricate.artifact().artifact_id,
        a.artifact().artifact_id,
        b.artifact().artifact_id,
        EnsembleProgram([a, b]).artifact().artifact_id,
    }
    assert len(ids) == 7


def test_handoff_toggle_changes_the_artifact_id() -> None:
    # Handoff creates durable agents and messages, so it is a real behavioral difference the artifact
    # id must capture; a handoff and a no-handoff orchestrator are distinct, reproducible artifacts.
    a = _Scripted(_result(grounded=True, citations=1), "a")
    b = _Scripted(_result(grounded=True, citations=1), "b")
    members = {"a": a, "b": b}
    router = TaskRouter({"a": ("alpha",), "b": ("beta",)})
    off = MultiAgentOrchestrator(members, router, handoff=False).artifact().artifact_id
    on = MultiAgentOrchestrator(members, router, handoff=True).artifact().artifact_id
    assert off != on


def test_mutable_member_tunable_changes_the_orchestrator_id() -> None:
    from refaat.programs import GroundedRagProgram, Tunables

    lean = {"ops": GroundedRagProgram(tunables=Tunables(top_k=1))}
    wide = {"ops": GroundedRagProgram(tunables=Tunables(top_k=9))}
    router = TaskRouter({"ops": ("rollback",)})
    lean_id = MultiAgentOrchestrator(lean, router).artifact().artifact_id
    wide_id = MultiAgentOrchestrator(wide, router).artifact().artifact_id
    assert lean_id != wide_id


def test_determinism_and_no_member_mutation() -> None:
    orch, a, b = _orchestrator()
    before = orch.artifact().artifact_id
    first = orch.execute(None, inputs={"question": "alpha beta"}, user_id=None, session_id=None)
    second = orch.execute(None, inputs={"question": "alpha beta"}, user_id=None, session_id=None)
    assert first.to_json() == second.to_json()
    assert a.calls == 2
    assert b.calls == 2
    assert orch.artifact().artifact_id == before


def test_usage_sums_member_context_tokens() -> None:
    a = _Scripted(_result(grounded=True, citations=1, program="a", tokens=10.0, chunk_prefix="a"), "a")
    b = _Scripted(_result(grounded=True, citations=1, program="b", tokens=15.0, chunk_prefix="b"), "b")
    orch = MultiAgentOrchestrator({"a": a, "b": b}, TaskRouter({"a": ("alpha",), "b": ("beta",)}))
    out = orch.execute(None, inputs={"question": "alpha beta"}, user_id=None, session_id=None)
    assert out.usage["orchestrator_member_tokens"] == 25.0
    assert out.usage["orchestrator_members"] == 2


def test_missing_task_input_is_rejected() -> None:
    orch, _a, _b = _orchestrator()
    with pytest.raises(ValidationError, match="required"):
        orch.execute(None, inputs={"question": "  "}, user_id=None, session_id=None)
