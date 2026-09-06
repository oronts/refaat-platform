from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from _shared import demo_settings, emit

from refaat.application.runtime import RuntimeBuilder
from refaat.domain.authz import Principal
from refaat.domain.enums import SideEffectClass
from refaat.domain.governance import ToolCapability, approval_fingerprint
from refaat.domain.models import ContextPacket, DocumentInput, JsonObject, ReActPolicy, ToolGrant
from refaat.programs import GovernedReActProgram, KeywordToolPlanner

ALICE = Principal(id="alice", tenant="acme")
QUESTION = "how does rollback work"


class LookupSkill:
    name = "lookup"
    description = "Read-only lookup over the retrieved context."

    def __init__(self) -> None:
        self.calls = 0

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        del arguments
        self.calls += 1
        return {"documents": len({hit.chunk.document_id for hit in context.hits})}


def _program(skill: LookupSkill) -> GovernedReActProgram:
    policy = ReActPolicy(
        max_iterations=2,
        granted_scopes=("read:corpus",),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        grants=(ToolGrant("lookup", required_scopes=("read:corpus",), side_effect=SideEffectClass.READ_ONLY),),
        capabilities=(
            ToolCapability(
                name="lookup",
                required_scopes=frozenset({"read:corpus"}),
                side_effect=SideEffectClass.READ_ONLY,
                requires_approval=True,
            ),
        ),
    )
    return GovernedReActProgram(
        policy=policy, planner=KeywordToolPlanner({"lookup": ("rollback",)}), skills={"lookup": skill}
    )


def _fingerprint(program: GovernedReActProgram) -> str:
    # The approval binds to the exact program version: the enforcement path derives action_version
    # from the content-addressed artifact id, so the grant must use the same value to match.
    return approval_fingerprint(
        capability_name="lookup",
        arguments={},
        principal=ALICE,
        policy_id="react",
        policy_version="1",
        action_version=program.artifact().artifact_id,
    )


def _lookup_step(program: GovernedReActProgram, project: object) -> JsonObject:
    result = program.execute(project, inputs={"question": QUESTION}, user_id=None, session_id=None, principal=ALICE)
    return {step["tool"]: step for step in result.outputs["react_steps"]}["lookup"]


def main() -> None:
    with TemporaryDirectory(prefix="refaat-tool-approval-") as directory:
        settings = demo_settings(Path(directory))
        with RuntimeBuilder(settings).build() as runtime:
            project = runtime.project(workspace="acme", corpus="react")
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

            skill = LookupSkill()
            program = _program(skill)

            without_approval = _lookup_step(program, project)
            calls_without_approval = skill.calls

            # A different principal records the approval, so separation of duties holds.
            approval_id = project.approvals.grant(
                fingerprint=_fingerprint(program), capability_name="lookup", approver_id="bob"
            )

            authorized = _lookup_step(program, project)
            calls_after_authorized = skill.calls

            # The single approval was consumed by the run above; the next run is denied.
            second = _lookup_step(program, project)
            calls_after_second = skill.calls

        assert without_approval["authorized"] is False
        assert without_approval["reason"].startswith("approval_required")
        assert calls_without_approval == 0
        assert bool(approval_id)
        assert authorized["authorized"] is True
        assert calls_after_authorized == 1
        assert second["authorized"] is False
        assert second["reason"].startswith("approval_required")
        assert calls_after_second == 1

        emit(
            {
                "approval_id_present": bool(approval_id),
                "denied_without_approval": (
                    without_approval["authorized"] is False
                    and without_approval["reason"].startswith("approval_required")
                    and calls_without_approval == 0
                ),
                "authorized_once": authorized["authorized"] is True and calls_after_authorized == 1,
                "second_run_denied": (
                    second["authorized"] is False
                    and second["reason"].startswith("approval_required")
                    and calls_after_second == 1
                ),
                "single_use_consumed": calls_after_second == calls_after_authorized == 1,
                "skill_call_count": calls_after_second,
            }
        )


if __name__ == "__main__":
    main()
