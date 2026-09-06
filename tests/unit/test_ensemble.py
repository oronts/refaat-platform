from __future__ import annotations

import pytest

from refaat.domain.models import Citation
from refaat.errors import ValidationError
from refaat.programs import EnsembleProgram, GroundedRagInput, GroundedRagOutput
from refaat.programs.program import Program, ProgramResult, ValidationOutcome


def _result(
    *,
    grounded: bool,
    citations: int,
    program: str = "m",
    answer: str = "answer",
    tokens: float = 10.0,
) -> ProgramResult:
    cites = tuple(
        Citation(chunk_id=f"c{index}", document_id="doc", label=f"[{index}]", quote="q") for index in range(citations)
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
    name = "scripted"
    input_signature = GroundedRagInput
    output_signature = GroundedRagOutput

    def __init__(self, result: ProgramResult, tag: str) -> None:
        super().__init__()
        self._result = result
        self.name = tag

    def execute(self, project, *, inputs, user_id, session_id, principal=None) -> ProgramResult:
        return self._result


def _run(members, **kwargs):
    ensemble = EnsembleProgram(members, **kwargs)
    return ensemble, ensemble.execute(None, inputs={"question": "q"}, user_id=None, session_id=None)


def test_best_policy_selects_the_grounded_member() -> None:
    members = [
        _Scripted(_result(grounded=False, citations=5, program="a"), "a"),
        _Scripted(_result(grounded=True, citations=1, program="b"), "b"),
    ]
    _ensemble, out = _run(members, policy="best")
    assert out.program == "ensemble"
    assert out.grounded
    assert out.outputs["ensemble"]["selected"] == "b"


def test_majority_policy_picks_the_most_common_citation_set() -> None:
    members = [
        _Scripted(_result(grounded=True, citations=1, program="a"), "a"),
        _Scripted(_result(grounded=True, citations=1, program="b"), "b"),
        _Scripted(_result(grounded=True, citations=2, program="c"), "c"),
    ]
    _ensemble, out = _run(members, policy="majority")
    assert out.outputs["ensemble"]["selected"] in ("a", "b")


def test_usage_sums_member_tokens() -> None:
    members = [
        _Scripted(_result(grounded=True, citations=1, tokens=10.0), "a"),
        _Scripted(_result(grounded=True, citations=1, tokens=15.0), "b"),
    ]
    _ensemble, out = _run(members)
    assert out.usage["ensemble_member_tokens"] == 25.0
    assert out.usage["ensemble_members"] == 2


def test_artifact_pins_members_and_policy() -> None:
    a = _Scripted(_result(grounded=True, citations=1), "a")
    b = _Scripted(_result(grounded=True, citations=1), "b")
    best = EnsembleProgram([a, b], policy="best").artifact().artifact_id
    majority = EnsembleProgram([a, b], policy="majority").artifact().artifact_id
    single = EnsembleProgram([a], policy="best").artifact().artifact_id
    assert len({best, majority, single}) == 3
    assert best != a.artifact().artifact_id


def test_empty_members_are_rejected() -> None:
    with pytest.raises(ValidationError, match="at least one"):
        EnsembleProgram([])


def test_unknown_policy_is_rejected() -> None:
    a = _Scripted(_result(grounded=True, citations=1), "a")
    with pytest.raises(ValidationError, match="policy"):
        EnsembleProgram([a], policy="bogus")
