"""A per-tool deadline continues a read tool from evidence and raises on a write tool, byte-identical off.

With tool_timeout_seconds unset the governed loop runs a tool inline exactly as before. With a deadline
set, a read tool abandoned at the deadline records an unknown step and the grounded answer still serves,
while a write-capable tool abandoned at the deadline raises a 409 ToolTimeoutError so no caller reads it
as a safe retry.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from refaat.application.runtime import RuntimeBuilder
from refaat.config import Settings
from refaat.domain.authz import Classification
from refaat.domain.enums import SideEffectClass
from refaat.domain.governance import ToolCapability
from refaat.domain.models import DocumentInput, JsonObject, ReActPolicy, ToolGrant
from refaat.errors import TOOL_TIMEOUT_REASON_UNKNOWN, ToolTimeoutError
from refaat.programs import GovernedReActProgram, KeywordToolPlanner

pytestmark = pytest.mark.integration

_DOC = DocumentInput(
    source_id="ops.txt",
    title="Ops",
    media_type="text/plain",
    content=b"Rollback restores the previously approved harness snapshot before deployment.",
)


class _HangingSkill:
    def __init__(self, name: str, release: threading.Event) -> None:
        self.name = name
        self.description = "blocks past the deadline"
        self._release = release
        self.calls = 0

    def run(self, context: Any, arguments: JsonObject) -> JsonObject:
        del context, arguments
        self.calls += 1
        self._release.wait(5.0)
        return {"done": True}


class _FastSkill:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = "returns immediately"
        self.calls = 0

    def run(self, context: Any, arguments: JsonObject) -> JsonObject:
        del arguments
        self.calls += 1
        return {"documents": len({hit.chunk.document_id for hit in context.hits})}


def _runtime(settings: Settings, **overrides: object) -> Any:
    return RuntimeBuilder(settings.model_copy(update=overrides)).build()


def test_read_tool_timeout_continues_from_evidence(settings: Settings) -> None:
    runtime = _runtime(settings, tool_timeout_seconds=0.05)
    release = threading.Event()
    try:
        project = runtime.project(workspace="acme", corpus="react")
        runtime.ingest(project.workspace_id, project.corpus_id, _DOC)
        reader = _HangingSkill("reader", release)
        policy = ReActPolicy(
            max_iterations=4,
            granted_scopes=("read:corpus",),
            grants=(ToolGrant("reader", required_scopes=("read:corpus",), side_effect=SideEffectClass.READ_ONLY),),
        )
        planner = KeywordToolPlanner({"reader": ("rollback",)})
        program = GovernedReActProgram(policy=policy, planner=planner, skills={"reader": reader})
        result = program.execute(project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None)
        steps = {step["tool"]: step for step in result.outputs["react_steps"]}
        assert steps["reader"]["authorized"] is False
        assert steps["reader"]["reason"] == TOOL_TIMEOUT_REASON_UNKNOWN
        assert result.grounded
    finally:
        release.set()
        runtime.close()


def test_write_tool_timeout_raises_a_409(settings: Settings) -> None:
    runtime = _runtime(settings, tool_timeout_seconds=0.05)
    release = threading.Event()
    try:
        project = runtime.project(workspace="acme", corpus="react")
        runtime.ingest(project.workspace_id, project.corpus_id, _DOC)
        writer = _HangingSkill("writer", release)
        policy = ReActPolicy(
            max_iterations=4,
            granted_scopes=("write:crm",),
            side_effect_ceiling=SideEffectClass.WRITE,
            grants=(ToolGrant("writer", required_scopes=("write:crm",), side_effect=SideEffectClass.WRITE),),
            capabilities=(
                ToolCapability(
                    name="writer",
                    required_scopes=frozenset({"write:crm"}),
                    side_effect=SideEffectClass.WRITE,
                    allowed_classification=Classification.RESTRICTED,
                ),
            ),
        )
        planner = KeywordToolPlanner({"writer": ("rollback",)})
        program = GovernedReActProgram(policy=policy, planner=planner, skills={"writer": writer})
        with pytest.raises(ToolTimeoutError) as excinfo:
            program.execute(project, inputs={"question": "rollback please"}, user_id=None, session_id=None)
        assert excinfo.value.status_code == 409
        assert excinfo.value.outcome == "unknown"
        assert excinfo.value.execution_terminated is False
        assert excinfo.value.side_effect == SideEffectClass.WRITE.value
    finally:
        release.set()
        runtime.close()


def test_grant_only_write_timeout_still_raises(settings: Settings) -> None:
    # A WRITE grant with NO capability: severity comes from the authoritative grant, so an abandoned
    # write is not silently continued as a read.
    runtime = _runtime(settings, tool_timeout_seconds=0.05)
    release = threading.Event()
    try:
        project = runtime.project(workspace="acme", corpus="react")
        runtime.ingest(project.workspace_id, project.corpus_id, _DOC)
        writer = _HangingSkill("writer", release)
        policy = ReActPolicy(
            max_iterations=4,
            granted_scopes=("write:crm",),
            side_effect_ceiling=SideEffectClass.WRITE,
            grants=(ToolGrant("writer", required_scopes=("write:crm",), side_effect=SideEffectClass.WRITE),),
        )
        planner = KeywordToolPlanner({"writer": ("rollback",)})
        program = GovernedReActProgram(policy=policy, planner=planner, skills={"writer": writer})
        with pytest.raises(ToolTimeoutError) as excinfo:
            program.execute(project, inputs={"question": "rollback please"}, user_id=None, session_id=None)
        assert excinfo.value.side_effect == SideEffectClass.WRITE.value
    finally:
        release.set()
        runtime.close()


def test_fast_tool_completes_under_a_deadline(settings: Settings) -> None:
    runtime = _runtime(settings, tool_timeout_seconds=1.0)
    try:
        project = runtime.project(workspace="acme", corpus="react")
        runtime.ingest(project.workspace_id, project.corpus_id, _DOC)
        reader = _FastSkill("reader")
        policy = ReActPolicy(
            max_iterations=4,
            granted_scopes=("read:corpus",),
            grants=(ToolGrant("reader", required_scopes=("read:corpus",), side_effect=SideEffectClass.READ_ONLY),),
        )
        planner = KeywordToolPlanner({"reader": ("rollback",)})
        program = GovernedReActProgram(policy=policy, planner=planner, skills={"reader": reader})
        result = program.execute(project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None)
        steps = {step["tool"]: step for step in result.outputs["react_steps"]}
        assert reader.calls == 1
        assert steps["reader"]["authorized"] is True
        assert result.grounded
    finally:
        runtime.close()


def test_disabled_runs_the_inline_path(settings: Settings) -> None:
    runtime = _runtime(settings)  # tool_timeout_seconds default None -> inline, byte-identical
    try:
        project = runtime.project(workspace="acme", corpus="react")
        runtime.ingest(project.workspace_id, project.corpus_id, _DOC)
        reader = _FastSkill("reader")
        policy = ReActPolicy(
            max_iterations=4,
            granted_scopes=("read:corpus",),
            grants=(ToolGrant("reader", required_scopes=("read:corpus",), side_effect=SideEffectClass.READ_ONLY),),
        )
        planner = KeywordToolPlanner({"reader": ("rollback",)})
        program = GovernedReActProgram(policy=policy, planner=planner, skills={"reader": reader})
        result = program.execute(project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None)
        steps = {step["tool"]: step for step in result.outputs["react_steps"]}
        assert reader.calls == 1
        assert steps["reader"]["authorized"] is True
        assert steps["reader"]["observation"] == {"documents": 1}
    finally:
        runtime.close()
