from __future__ import annotations

import pytest

from refaat.application.runtime import Refaat
from refaat.domain.models import DocumentInput
from refaat.programs import EnsembleProgram, GroundedRagProgram, Tunables

pytestmark = pytest.mark.integration


def test_ensemble_over_real_programs_is_grounded_and_pins_members(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="ensemble")
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
    member_one = GroundedRagProgram(tunables=Tunables(top_k=1))
    member_two = GroundedRagProgram(tunables=Tunables(top_k=3))
    ensemble = EnsembleProgram([member_one, member_two], policy="best")

    result = ensemble.execute(project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None)

    assert result.program == "ensemble"
    assert result.grounded
    assert result.citations
    assert result.usage["ensemble_members"] == 2
    assert ensemble.artifact().artifact_id != member_one.artifact().artifact_id
