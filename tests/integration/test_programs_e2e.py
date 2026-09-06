from __future__ import annotations

import pytest

from refaat.application.runtime import Refaat
from refaat.domain.models import DocumentInput
from refaat.errors import ConflictError, NotFoundError, ValidationError
from refaat.programs import GroundedRagProgram, InputField, ProgramResult, Signature, Tunables

pytestmark = pytest.mark.integration


def _ingest(runtime: Refaat, workspace_id: str, corpus_id: str, text: bytes) -> None:
    runtime.ingest(
        workspace_id,
        corpus_id,
        DocumentInput(source_id="ops.txt", title="Ops", media_type="text/plain", content=text),
    )


def test_grounded_rag_program_runs_and_is_grounded(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    _ingest(
        runtime,
        project.workspace_id,
        project.corpus_id,
        b"Rollback restores the previously approved harness snapshot before deployment.",
    )

    result = project.programs.get("grounded-rag").run(question="how does rollback work")

    assert isinstance(result, ProgramResult)
    assert result.answer
    assert result.citations
    assert result.grounded
    assert result.artifact_id.startswith("sha256:")
    assert "grounded-rag" in {program["program"] for program in project.programs.list()}
    assert result.to_json()["grounded"] is True


def test_custom_program_registration_and_error_paths(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    _ingest(
        runtime,
        project.workspace_id,
        project.corpus_id,
        b"Escalate incidents to the on-call engineer within fifteen minutes.",
    )

    class IncidentInput(Signature):
        question: str = InputField(description="incident question")

    class IncidentProgram(GroundedRagProgram):
        name = "incident-runbook"
        input_signature = IncidentInput

    bound = project.programs.register(IncidentProgram(tunables=Tunables(top_k=2, minimum_citation_coverage=0.5)))
    result = bound.run(question="who handles incidents")
    assert result.program == "incident-runbook"
    assert result.grounded

    with pytest.raises(ConflictError):
        project.programs.register(IncidentProgram())
    with pytest.raises(NotFoundError):
        project.programs.get("missing")
    with pytest.raises(ValidationError):
        project.programs.get("grounded-rag").run(question="   ")
