"""Incident kill switches, in-process and worker enforcement plus byte-identity.

With the backend off nothing is installed, no table is read, the pinned artifact id is unchanged, and
a query runs normally. With a workspace freeze engaged every in-process serving and mutation path and
every durable job for that workspace is denied fail-closed, while another workspace is untouched. The
management surface is audited and fails closed when not enabled or when a read fails.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from refaat.application.kill_switch import KillSwitchService
from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.application.worker import DurableWorker
from refaat.config import Settings
from refaat.domain.audit import AuditEvent
from refaat.domain.enums import JobStatus
from refaat.domain.kill_switch import KillSwitchScope
from refaat.domain.models import DocumentInput, Feedback, QueryRequest
from refaat.errors import ConfigurationError, KillSwitchEngagedError
from refaat.programs.program import GroundedRagProgram

pytestmark = pytest.mark.integration

PINNED_GROUNDED_RAG_ID = "sha256:b76d73f8d956bcd38ccc9d4c24372b07b387647a189f3a9e300cded186febbb4"
_BODY = b"The checkout production release requires an approved rollback plan before deploy."


class _RecordingSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        self.events.append(event)


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {
        "environment": "test",
        "database_url": f"sqlite:///{tmp_path / 'refaat.db'}",
        "auto_create_schema": True,
        "blob_root": tmp_path / "blobs",
        "connector_file_root": tmp_path / "connectors",
        "skill_root": tmp_path / "skills",
        "api_key_pepper": "test-pepper-with-enough-entropy",
        "bootstrap_token": "bootstrap-test-token",
        "default_chunk_tokens": 48,
        "chunk_overlap_tokens": 8,
        "log_json": False,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _db(tmp_path: Path) -> Settings:
    # ttl 0.0 makes every engaged() re-read, so a switch engaged mid-test is visible at once.
    return _settings(tmp_path, kill_switch_backend="database", kill_switch_cache_ttl_seconds=0.0)


def _service(runtime: Refaat, workspace_id: str) -> KillSwitchService:
    return KillSwitchService(
        runtime.store,
        workspace_id,
        audit_sink=runtime.audit_sink,
        audit_fail_closed=runtime.settings.audit_fail_closed,
    )


def _seed(runtime: Refaat) -> tuple[str, str]:
    ws = runtime.create_workspace("acme", "Acme")
    corpus = runtime.create_corpus(ws, "handbook")
    runtime.ingest(ws, corpus, DocumentInput(source_id="a", title="a", media_type="text/plain", content=_BODY))
    return ws, corpus


def test_pinned_grounded_rag_artifact_id_is_unchanged() -> None:
    assert GroundedRagProgram().artifact().artifact_id == PINNED_GROUNDED_RAG_ID


def test_backend_off_installs_nothing_and_serves_normally(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_settings(tmp_path)).build()
    try:
        assert runtime.kill_switches is None
        assert runtime.store.kill_switch_management_enabled is False
        _seed(runtime)
        answer = runtime.project(workspace="acme", corpus="handbook").ask("rollback plan")
        assert answer.citations
    finally:
        runtime.close()


def test_workspace_freeze_denies_every_in_process_path(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_db(tmp_path)).build()
    try:
        ws, _corpus = _seed(runtime)
        project = runtime.project(workspace="acme", corpus="handbook")
        # Before the freeze the project serves normally.
        assert project.ask("rollback plan").citations
        runtime.store.engage_kill_switch(
            ws,
            scope=KillSwitchScope.WORKSPACE_FREEZE.value,
            engaged_by="operator",
            reason=None,
            engaged_at=runtime.store.clock.now(),
        )
        for call in (
            lambda: project.ask("rollback plan"),
            lambda: project.retrieve("rollback plan"),
            lambda: project.create_session(),
            lambda: project.ingest_source("inline text", source_id="b"),
        ):
            with pytest.raises(KillSwitchEngagedError):
                call()
        # A different workspace is unaffected.
        other = runtime.create_workspace("pfizer", "Pfizer")
        runtime.create_corpus(other, "handbook")
        assert runtime.project(workspace="pfizer", corpus="handbook").create_session()
        # Disengaging restores service.
        runtime.store.disengage_kill_switch(
            ws,
            scope=KillSwitchScope.WORKSPACE_FREEZE.value,
            disengaged_by="operator",
            disengaged_at=runtime.store.clock.now(),
        )
        assert project.ask("rollback plan").citations
    finally:
        runtime.close()


def test_workspace_freeze_denies_direct_service_calls(tmp_path: Path) -> None:
    # These reach the services directly, bypassing the Refaat and RefaatProject wrappers; the freeze
    # guard lives at the service entry, so a direct call is frozen too.
    runtime = RuntimeBuilder(_db(tmp_path)).build()
    try:
        ws, corpus = _seed(runtime)
        request = QueryRequest(workspace_id=ws, corpus_id=corpus, query="rollback plan")
        assert runtime.queries.query(request).answer is not None
        runtime.store.engage_kill_switch(
            ws,
            scope=KillSwitchScope.WORKSPACE_FREEZE.value,
            engaged_by="operator",
            reason=None,
            engaged_at=runtime.store.clock.now(),
        )
        with pytest.raises(KillSwitchEngagedError):
            runtime.queries.query(request)
        with pytest.raises(KillSwitchEngagedError):
            runtime.rlm.run(ws, corpus, "rollback plan")
        with pytest.raises(KillSwitchEngagedError):
            runtime.ingestion.delete(ws, corpus, "does-not-matter")
    finally:
        runtime.close()


def test_workspace_freeze_denies_state_mutations(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_db(tmp_path)).build()
    try:
        ws, corpus = _seed(runtime)
        snapshot = runtime.harness.create_snapshot(
            workspace_id=ws, corpus_id=corpus, supplemental={}, evidence={"note": "seed"}
        )
        runtime.store.engage_kill_switch(
            ws,
            scope=KillSwitchScope.WORKSPACE_FREEZE.value,
            engaged_by="operator",
            reason=None,
            engaged_at=runtime.store.clock.now(),
        )
        feedback = Feedback(user_id="u", query_id=None, rating=1, selected_chunk_ids=(), note=None)
        with pytest.raises(KillSwitchEngagedError):
            runtime.feedback.submit(ws, corpus, feedback)
        with pytest.raises(KillSwitchEngagedError):
            runtime.harness.create_snapshot(workspace_id=ws, corpus_id=corpus, supplemental={}, evidence={"note": "x"})
        with pytest.raises(KillSwitchEngagedError):
            runtime.project(workspace="acme", corpus="handbook").activate_harness_snapshot(snapshot.id)
    finally:
        runtime.close()


def test_workspace_freeze_denies_shadow_optimizer_evidence_read(tmp_path: Path) -> None:
    from refaat.optimization.shadow import active_documents

    runtime = RuntimeBuilder(_db(tmp_path)).build()
    try:
        ws, _corpus = _seed(runtime)
        project = runtime.project(workspace="acme", corpus="handbook")
        assert active_documents(project)
        runtime.store.engage_kill_switch(
            ws,
            scope=KillSwitchScope.WORKSPACE_FREEZE.value,
            engaged_by="operator",
            reason=None,
            engaged_at=runtime.store.clock.now(),
        )
        with pytest.raises(KillSwitchEngagedError):
            active_documents(project)
    finally:
        runtime.close()


def test_workspace_freeze_denies_an_in_process_program_run(tmp_path: Path) -> None:
    # A program run calls runtime.query directly, bypassing RefaatProject; the deep runtime chokepoint
    # closes that surface, so an in-process program run is frozen too.
    runtime = RuntimeBuilder(_db(tmp_path)).build()
    try:
        ws, _corpus = _seed(runtime)
        program = runtime.project(workspace="acme", corpus="handbook").programs.get("grounded-rag")
        assert program.run(question="rollback plan").answer
        runtime.store.engage_kill_switch(
            ws,
            scope=KillSwitchScope.WORKSPACE_FREEZE.value,
            engaged_by="operator",
            reason=None,
            engaged_at=runtime.store.clock.now(),
        )
        with pytest.raises(KillSwitchEngagedError):
            program.run(question="rollback plan")
    finally:
        runtime.close()


def test_workspace_freeze_refuses_a_durable_job_without_exhausting_attempts(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_db(tmp_path)).build()
    try:
        ws, _corpus = _seed(runtime)
        job_id = runtime.store.enqueue_job(ws, "cache.purge", {})
        runtime.store.engage_kill_switch(
            ws,
            scope=KillSwitchScope.WORKSPACE_FREEZE.value,
            engaged_by="operator",
            reason=None,
            engaged_at=runtime.store.clock.now(),
        )
        worker = DurableWorker(runtime)
        assert worker.run_once() is True
        blocked = runtime.store.get_job(ws, job_id)
        assert blocked["status"] == JobStatus.PENDING
        assert blocked["attempts"] == 0
        runtime.store.disengage_kill_switch(
            ws,
            scope=KillSwitchScope.WORKSPACE_FREEZE.value,
            disengaged_by="operator",
            disengaged_at=runtime.store.clock.now(),
        )
        assert worker.run_once() is True
        assert runtime.store.get_job(ws, job_id)["status"] == JobStatus.SUCCEEDED
    finally:
        runtime.close()


def test_engage_and_disengage_emit_redacted_audit(tmp_path: Path) -> None:
    sink = _RecordingSink()
    runtime = RuntimeBuilder(_db(tmp_path)).with_audit_sink(sink).build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        service = _service(runtime, ws)
        service.engage(scope=KillSwitchScope.WORKSPACE_FREEZE, engaged_by="operator", reason="secret-incident-detail")
        assert service.disengage(scope=KillSwitchScope.WORKSPACE_FREEZE, disengaged_by="operator") is True
        types = [event.event_type for event in sink.events]
        assert "killswitch.engaged" in types
        assert "killswitch.disengaged" in types
        engaged = next(event for event in sink.events if event.event_type == "killswitch.engaged")
        assert engaged.capability_name == "killswitch.workspace_freeze"
        # The human reason is row-only and never enters an event.
        assert all("secret-incident-detail" not in str(event.to_json()) for event in sink.events)
    finally:
        runtime.close()


def test_disengage_with_no_active_switch_is_a_noop(tmp_path: Path) -> None:
    sink = _RecordingSink()
    runtime = RuntimeBuilder(_db(tmp_path)).with_audit_sink(sink).build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        assert (
            _service(runtime, ws).disengage(scope=KillSwitchScope.WORKSPACE_FREEZE, disengaged_by="operator") is False
        )
        assert sink.events == []
    finally:
        runtime.close()


def test_management_surface_fails_closed_when_backend_off(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_settings(tmp_path)).build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        with pytest.raises(ConfigurationError):
            _service(runtime, ws).engage(scope=KillSwitchScope.WORKSPACE_FREEZE, engaged_by="operator")
    finally:
        runtime.close()


def test_tool_execution_scope_denies_the_direct_skill_path(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_db(tmp_path)).build()
    try:
        ws, _corpus = _seed(runtime)
        project = runtime.project(workspace="acme", corpus="handbook")
        assert project.run_skill("corpus_health", "health") is not None
        _service(runtime, ws).engage(scope=KillSwitchScope.TOOL_EXECUTION, engaged_by="operator")
        with pytest.raises(KillSwitchEngagedError):
            project.run_skill("corpus_health", "health")
        # A tool-execution switch blocks tools, not reads or generation.
        assert project.ask("rollback plan").citations
        assert project.retrieve("rollback plan").hits
    finally:
        runtime.close()


def test_tool_execution_scope_denies_tools_but_the_agent_still_answers(tmp_path: Path) -> None:
    from refaat.domain.enums import SideEffectClass
    from refaat.domain.models import ContextPacket, JsonObject, ReActPolicy, ToolGrant
    from refaat.programs import GovernedReActProgram, KeywordToolPlanner

    class _Lookup:
        name = "lookup"
        description = "read-only lookup over the retrieved context"

        def __init__(self) -> None:
            self.calls = 0

        def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
            del arguments
            self.calls += 1
            return {"hits": len(context.hits)}

    runtime = RuntimeBuilder(_db(tmp_path)).build()
    try:
        ws, _corpus = _seed(runtime)
        project = runtime.project(workspace="acme", corpus="handbook")
        skill = _Lookup()
        policy = ReActPolicy(
            max_iterations=2,
            granted_scopes=("read:corpus",),
            side_effect_ceiling=SideEffectClass.READ_ONLY,
            grants=(ToolGrant("lookup", required_scopes=("read:corpus",), side_effect=SideEffectClass.READ_ONLY),),
        )
        program = project.programs.register(
            GovernedReActProgram(
                policy=policy, planner=KeywordToolPlanner({"lookup": ("rollback",)}), skills={"lookup": skill}
            )
        )
        result = program.run(question="how does rollback work")
        assert skill.calls == 1
        assert any(step["authorized"] for step in result.outputs["react_steps"])
        _service(runtime, ws).engage(scope=KillSwitchScope.TOOL_EXECUTION, engaged_by="operator")
        denied = program.run(question="how does rollback work")
        assert skill.calls == 1  # the tool never executed under the switch
        steps = denied.outputs["react_steps"]
        assert steps
        assert all(not step["authorized"] for step in steps)
        assert "kill switch" in steps[0]["reason"]
        assert denied.answer is not None  # the agent still answers from evidence without the tool
        stream_events = list(program.stream(question="how does rollback work"))
        assert skill.calls == 1  # the streamed loop denies the tool too
        tool_steps = [event for event in stream_events if event.get("type") == "tool_step"]
        assert tool_steps
        assert all(not event["authorized"] for event in tool_steps)
    finally:
        runtime.close()


def test_engaging_an_unenforced_scope_is_refused(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_db(tmp_path)).build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        # external_egress has no wired seam yet, so the management surface still refuses it.
        with pytest.raises(ConfigurationError):
            _service(runtime, ws).engage(scope=KillSwitchScope.EXTERNAL_EGRESS, engaged_by="operator")
    finally:
        runtime.close()


def test_generation_scope_denies_generation_but_leaves_reads_and_writes(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_db(tmp_path)).build()
    try:
        ws, corpus = _seed(runtime)
        project = runtime.project(workspace="acme", corpus="handbook")
        assert project.ask("rollback plan").citations
        _service(runtime, ws).engage(scope=KillSwitchScope.GENERATION, engaged_by="operator")
        with pytest.raises(KillSwitchEngagedError):
            project.ask("rollback plan")
        with pytest.raises(KillSwitchEngagedError):
            list(runtime.queries.stream_query(QueryRequest(workspace_id=ws, corpus_id=corpus, query="rollback plan")))
        with pytest.raises(KillSwitchEngagedError):
            project.rlm("rollback plan")
        # Retrieval carries no generation, and a writes-only concern is untouched, so both still run.
        assert project.retrieve("rollback plan").hits
        runtime.ingest(ws, corpus, DocumentInput(source_id="b", title="b", media_type="text/plain", content=_BODY))
    finally:
        runtime.close()


def test_writes_scope_denies_mutations_but_leaves_reads(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_db(tmp_path)).build()
    try:
        ws, corpus = _seed(runtime)
        project = runtime.project(workspace="acme", corpus="handbook")
        doc_id = str(project.list_documents()[0]["id"])
        snapshot = project.create_harness_snapshot(supplemental={}, evidence={"reason": "kill-switch test"})
        _service(runtime, ws).engage(scope=KillSwitchScope.WRITES, engaged_by="operator")
        with pytest.raises(KillSwitchEngagedError):
            runtime.ingest(ws, corpus, DocumentInput(source_id="b", title="b", media_type="text/plain", content=_BODY))
        with pytest.raises(KillSwitchEngagedError):
            project.submit_feedback(rating=1, user_id="u", selected_chunk_ids=())
        with pytest.raises(KillSwitchEngagedError):
            project.delete_document(doc_id)
        with pytest.raises(KillSwitchEngagedError):
            project.erase(doc_id)
        # The active harness is generation-config, so writing or activating a snapshot is a mutation too.
        with pytest.raises(KillSwitchEngagedError):
            project.create_harness_snapshot(supplemental={}, evidence={"reason": "kill-switch test"})
        with pytest.raises(KillSwitchEngagedError):
            project.activate_harness_snapshot(snapshot.id)
        # Reads carry no mutation, so query and retrieval still run under a writes-only switch.
        assert project.ask("rollback plan").citations
        assert project.retrieve("rollback plan").hits
    finally:
        runtime.close()
