from __future__ import annotations

import pytest

from refaat.application.runtime import Refaat
from refaat.domain.models import Citation, DocumentInput
from refaat.programs import (
    BEST_OF,
    REFINE,
    BoundedInferencePolicy,
    GroundedRagProgram,
    InputField,
    OutputField,
    ProgramResult,
    Signature,
)
from refaat.programs.program import Program, ValidationOutcome

pytestmark = pytest.mark.integration


class _Sig(Signature):
    question: str = InputField(description="q")
    answer: str = OutputField(description="a", grounded=True)


def _scripted(*, grounded: bool, tag: str) -> ProgramResult:
    citations = (Citation(chunk_id="c0", document_id="doc", label="[0]", quote="q"),) if grounded else ()
    return ProgramResult(
        program="counting",
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


class _CountingProgram(Program):
    name = "counting"
    input_signature = _Sig
    output_signature = _Sig

    def __init__(self, script: list[ProgramResult]) -> None:
        super().__init__()
        self._script = script
        self.calls = 0

    def execute(self, project, *, inputs, user_id, session_id, principal=None) -> ProgramResult:
        result = self._script[min(self.calls, len(self._script) - 1)]
        self.calls += 1
        return result


def test_best_of_runs_exactly_n_and_selects_best(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="policy")
    inner = _CountingProgram(
        [_scripted(grounded=False, tag="a"), _scripted(grounded=True, tag="b"), _scripted(grounded=False, tag="c")]
    )
    policy = BoundedInferencePolicy(inner, mode=BEST_OF, attempts=3)

    result = policy.execute(project, inputs={"question": "q"}, user_id=None, session_id=None)

    assert inner.calls == 3
    assert result.grounded
    assert result.answer == "b"


def test_refine_short_circuits_on_first_grounded(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="policy")
    inner = _CountingProgram(
        [_scripted(grounded=False, tag="a"), _scripted(grounded=True, tag="b"), _scripted(grounded=True, tag="c")]
    )
    policy = BoundedInferencePolicy(inner, mode=REFINE, attempts=5)

    result = policy.execute(project, inputs={"question": "q"}, user_id=None, session_id=None)

    assert inner.calls == 2
    assert result.answer == "b"


def test_refine_exhausts_the_cap_when_never_grounded(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="policy")
    inner = _CountingProgram([_scripted(grounded=False, tag="a")])
    policy = BoundedInferencePolicy(inner, mode=REFINE, attempts=3)

    result = policy.execute(project, inputs={"question": "q"}, user_id=None, session_id=None)

    assert inner.calls == 3
    assert not result.grounded


def test_policy_pins_mode_and_attempts_into_the_artifact() -> None:
    inner = GroundedRagProgram()
    best3 = BoundedInferencePolicy(inner, mode=BEST_OF, attempts=3).artifact().artifact_id
    best5 = BoundedInferencePolicy(inner, mode=BEST_OF, attempts=5).artifact().artifact_id
    refine3 = BoundedInferencePolicy(inner, mode=REFINE, attempts=3).artifact().artifact_id

    assert len({best3, best5, refine3, inner.artifact().artifact_id}) == 4


def test_best_of_over_the_real_runtime_returns_a_grounded_result(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    runtime.ingest(
        project.workspace_id,
        project.corpus_id,
        DocumentInput(
            source_id="ops.txt",
            title="Ops",
            media_type="text/plain",
            content=b"Rollback restores the previously approved harness snapshot before deployment.",
        ),
    )
    policy = BoundedInferencePolicy(GroundedRagProgram(), mode=BEST_OF, attempts=2)

    result = policy.execute(project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None)

    assert result.grounded
    assert result.citations
