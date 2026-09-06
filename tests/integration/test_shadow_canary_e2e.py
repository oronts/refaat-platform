from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from datetime import timedelta
from typing import Any

import pytest
from sqlalchemy import func, select, update

from refaat.application.runtime import Refaat, RefaatProject, RuntimeBuilder
from refaat.application.services import shadow_execution
from refaat.application.worker import DurableWorker
from refaat.config import Settings
from refaat.domain.audit import AuditEvent
from refaat.domain.authz import CanonicalAccessPolicy, Principal
from refaat.domain.enums import SideEffectClass
from refaat.domain.models import (
    DocumentInput,
    GenerationRequest,
    GenerationResult,
    ReActPolicy,
    RlmLimits,
    ToolGrant,
)
from refaat.domain.output_governance import OutputContext, OutputDecision
from refaat.errors import QuotaExceededError, ValidationError
from refaat.infrastructure.database.schema import (
    model_call_cache,
    program_artifacts,
    program_shadow_runs,
    query_cache,
)
from refaat.programs import GovernedReActProgram, GroundedRagProgram, KeywordToolPlanner, Tunables
from refaat.programs.registry import BoundProgram

pytestmark = pytest.mark.integration


def _ingest(runtime: Refaat, workspace_id: str, corpus_id: str) -> None:
    runtime.ingest(
        workspace_id,
        corpus_id,
        DocumentInput(
            source_id="ops.txt",
            title="Ops",
            media_type="text/plain",
            content=(
                b"Rollback restores the previously approved harness snapshot before deployment. "
                b"Promotion points an environment alias at an approved artifact. "
                b"A canary serves a fraction of traffic to a candidate artifact."
            ),
        ),
    )


def _promote(project: RefaatProject, environment: str, *, approve: bool) -> tuple[str, str]:
    base = GroundedRagProgram().artifact()
    candidate = GroundedRagProgram(tunables=Tunables(top_k=2)).artifact()
    registry = project.registry
    registry.save(base)
    registry.save(candidate)
    if approve:
        registry.approve("grounded-rag", environment, base.artifact_id)
    registry.promote("grounded-rag", environment, base.artifact_id)
    if approve:
        registry.approve("grounded-rag", environment, candidate.artifact_id)
    return base.artifact_id, candidate.artifact_id


def _resolves(project: RefaatProject, chunk_ids: tuple[str, ...]) -> bool:
    resolved = project.runtime.store.get_chunks(project.workspace_id, project.corpus_id, chunk_ids)
    return len(resolved) == len(chunk_ids) and len(chunk_ids) > 0


def test_shadow_serves_active_and_records_comparison(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    _ingest(runtime, project.workspace_id, project.corpus_id)
    active_id, candidate_id = _promote(project, "staging", approve=False)

    project.deployment.set_canary("grounded-rag", "staging", candidate_id, percent=0)
    result = project.deployment.run("grounded-rag", "staging", inputs={"question": "how does rollback work"})

    assert result.served == "active"
    assert result.result.artifact_id == active_id
    assert result.candidate_artifact_id == candidate_id
    assert result.comparison is not None
    assert _resolves(project, tuple(c.chunk_id for c in result.result.citations))

    runs = project.deployment.shadow_runs("grounded-rag", "staging")
    assert len(runs) == 1
    assert runs[0]["served"] == "active"
    assert runs[0]["comparison"] is not None


def test_canary_deterministic_fraction(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    _ingest(runtime, project.workspace_id, project.corpus_id)
    _, candidate_id = _promote(project, "staging", approve=False)

    project.deployment.set_canary("grounded-rag", "staging", candidate_id, percent=100)
    for identity in ("u1", "u2", "u3"):
        served = project.deployment.run(
            "grounded-rag", "staging", inputs={"question": "how does rollback work"}, user_id=identity
        ).served
        assert served == "candidate"

    project.deployment.set_canary("grounded-rag", "staging", candidate_id, percent=0)
    for identity in ("u1", "u2", "u3"):
        served = project.deployment.run(
            "grounded-rag", "staging", inputs={"question": "how does rollback work"}, user_id=identity
        ).served
        assert served == "active"

    project.deployment.set_canary("grounded-rag", "staging", candidate_id, percent=50)
    first = project.deployment.run(
        "grounded-rag", "staging", inputs={"question": "how does rollback work"}, user_id="sticky"
    ).served
    second = project.deployment.run(
        "grounded-rag", "staging", inputs={"question": "how does rollback work"}, user_id="sticky"
    ).served
    assert first == second


def test_production_canary_requires_approval(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    _ingest(runtime, project.workspace_id, project.corpus_id)
    base = GroundedRagProgram().artifact()
    candidate = GroundedRagProgram(tunables=Tunables(top_k=2)).artifact()
    registry = project.registry
    registry.save(base)
    registry.save(candidate)
    registry.approve("grounded-rag", "production", base.artifact_id)
    registry.promote("grounded-rag", "production", base.artifact_id)

    with pytest.raises(ValidationError):
        project.deployment.set_canary("grounded-rag", "production", candidate.artifact_id, percent=0)

    registry.approve("grounded-rag", "production", candidate.artifact_id)
    attached = project.deployment.set_canary("grounded-rag", "production", candidate.artifact_id, percent=0)
    assert attached["action"] == "set"


def test_identity_guard_refuses_mismatch(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    _ingest(runtime, project.workspace_id, project.corpus_id)
    _, candidate_id = _promote(project, "staging", approve=False)
    project.deployment.set_canary("grounded-rag", "staging", candidate_id, percent=0)

    tampered = project.registry.get(candidate_id)
    tampered["tunables"] = {**tampered["tunables"], "top_k": 99}
    with runtime.store.engine.begin() as conn:
        conn.execute(
            update(program_artifacts).where(program_artifacts.c.artifact_id == candidate_id).values(document=tampered)
        )

    with pytest.raises(ValidationError):
        project.deployment.run("grounded-rag", "staging", inputs={"question": "how does rollback work"})


# A redelivered shadow batch keyed to the same job/question does not duplicate
# shadow rows or outbox events.
def test_shadow_run_idempotency_key_dedupes(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    _ingest(runtime, project.workspace_id, project.corpus_id)
    _, candidate_id = _promote(project, "staging", approve=False)
    project.deployment.set_canary("grounded-rag", "staging", candidate_id, percent=0)

    first = project.deployment.run(
        "grounded-rag", "staging", inputs={"question": "how does rollback work"}, idempotency_key="job-7:0"
    )
    second = project.deployment.run(
        "grounded-rag", "staging", inputs={"question": "how does rollback work"}, idempotency_key="job-7:0"
    )

    assert first.shadow_run_id == second.shadow_run_id
    assert len(project.deployment.shadow_runs("grounded-rag", "staging")) == 1
    events = [row for row in runtime.store.pending_outbox(limit=100) if row["topic"] == "program.shadow_compared"]
    assert len(events) == 1


# The shadow_runs.purge worker job drops expired telemetry and keeps recent rows.
def test_shadow_runs_purge_worker_job(runtime: Refaat) -> None:
    workspace_id = runtime.create_workspace("acme", "Acme")
    old_id = runtime.store.record_shadow_run(
        workspace_id, "grounded-rag", "staging", "sha256:a", "sha256:b", "active", 0, "old", None
    )
    recent_id = runtime.store.record_shadow_run(
        workspace_id, "grounded-rag", "staging", "sha256:a", "sha256:b", "active", 0, "new", None
    )
    with runtime.store.engine.begin() as conn:
        conn.execute(
            update(program_shadow_runs)
            .where(program_shadow_runs.c.id == old_id)
            .values(created_at=runtime.store.clock.now() - timedelta(days=40))
        )

    worker = DurableWorker(runtime)
    assert "shadow_runs.purge" in worker.handlers
    result = worker._purge_shadow_runs(workspace_id, {})
    assert result == {"shadow_runs_purged": 1}

    remaining = {row["id"] for row in runtime.store.shadow_runs(workspace_id, "grounded-rag", "staging")}
    assert recent_id in remaining
    assert old_id not in remaining


def test_byte_identical_when_disabled(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    _ingest(runtime, project.workspace_id, project.corpus_id)
    _promote(project, "staging", approve=False)

    direct = project.programs.get("grounded-rag").run(question="how does rollback work")
    routed = project.deployment.run("grounded-rag", "staging", inputs={"question": "how does rollback work"})

    assert routed.served == "active"
    assert routed.result.answer == direct.answer
    assert tuple(c.chunk_id for c in routed.result.citations) == tuple(c.chunk_id for c in direct.citations)
    assert routed.result.cached is True
    assert routed.comparison is None
    assert project.deployment.shadow_runs("grounded-rag", "staging") == []


def test_canary_session_records_served_answer_only(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    _ingest(runtime, project.workspace_id, project.corpus_id)
    _, candidate_id = _promote(project, "staging", approve=False)
    project.deployment.set_canary("grounded-rag", "staging", candidate_id, percent=0)

    served_active = project.create_session(user_id="u-active")
    active_run = project.deployment.run(
        "grounded-rag", "staging", inputs={"question": "how does rollback work"}, session_id=served_active
    )
    assert active_run.served == "active"
    active_turns = project.runtime.store.list_session_turns(project.workspace_id, served_active)
    assert [turn["role"] for turn in active_turns] == ["user", "assistant"]
    assert active_turns[1]["content"] == active_run.result.answer
    assert active_turns[1]["metadata"]["citations"] == [c.chunk_id for c in active_run.result.citations]

    project.deployment.set_canary("grounded-rag", "staging", candidate_id, percent=100)
    served_candidate = project.create_session(user_id="u-candidate")
    candidate_run = project.deployment.run(
        "grounded-rag", "staging", inputs={"question": "how does rollback work"}, session_id=served_candidate
    )
    assert candidate_run.served == "candidate"
    candidate_turns = project.runtime.store.list_session_turns(project.workspace_id, served_candidate)
    assert [turn["role"] for turn in candidate_turns] == ["user", "assistant"]
    assert candidate_turns[1]["content"] == candidate_run.result.answer
    assert candidate_turns[1]["metadata"]["citations"] == [c.chunk_id for c in candidate_run.result.citations]


def test_clear_canary_reverts_to_active_only(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    _ingest(runtime, project.workspace_id, project.corpus_id)
    _, candidate_id = _promote(project, "staging", approve=False)

    project.deployment.set_canary("grounded-rag", "staging", candidate_id, percent=100)
    project.deployment.run("grounded-rag", "staging", inputs={"question": "how does rollback work"})
    assert len(project.deployment.shadow_runs("grounded-rag", "staging")) == 1

    project.deployment.clear_canary("grounded-rag", "staging")
    assert project.deployment.current_canary("grounded-rag", "staging") is None
    routed = project.deployment.run("grounded-rag", "staging", inputs={"question": "how does rollback work"})
    assert routed.served == "active"
    assert routed.shadow_run_id is None
    assert len(project.deployment.shadow_runs("grounded-rag", "staging")) == 1


class _CountingEmbedder:
    def __init__(self) -> None:
        self.dimensions = 4
        self.model_id = "fake-embed-v1"

    @property
    def cache_identity(self) -> str:
        return "fake-embed:v1:4"

    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [tuple(float((sum(map(ord, t)) + i) % 7) for i in range(self.dimensions)) for t in texts]

    def close(self) -> None:
        return None


class _CountingGenerator:
    def __init__(self) -> None:
        self.model_id = "fake-gen-v1"
        self.calls = 0

    @property
    def cache_identity(self) -> str:
        return "fake-gen:v1"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls += 1
        cited = (request.context.hits[0].chunk.id,) if request.context.hits else ()
        return GenerationResult(answer=f"answer #{self.calls}", cited_chunk_ids=cited, usage={})

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        yield "answer"

    def close(self) -> None:
        return None


def _cached_runtime(settings: Settings) -> Refaat:
    return (
        RuntimeBuilder(settings.model_copy(update={"model_call_cache_enabled": True}))
        .with_generator(_CountingGenerator())
        .with_embedder(_CountingEmbedder())
        .build()
    )


def _mcc(store: Any) -> int:
    with store.engine.connect() as conn:
        return int(conn.execute(select(func.count()).select_from(model_call_cache)).scalar_one())


def _qc(store: Any) -> int:
    with store.engine.connect() as conn:
        return int(conn.execute(select(func.count()).select_from(query_cache)).scalar_one())


def test_unserved_shadow_scope_writes_zero_cache_rows(settings: Settings) -> None:
    runtime = _cached_runtime(settings)
    try:
        project = runtime.project(workspace="acme", corpus="handbook")
        _ingest(runtime, project.workspace_id, project.corpus_id)
        store = runtime.store

        # A normal served run of a distinct query writes model-call and answer-cache rows.
        before_mcc, before_qc = _mcc(store), _qc(store)
        project.programs.get("grounded-rag").run(question="how does rollback work today")
        assert _mcc(store) > before_mcc
        assert _qc(store) > before_qc

        # An unserved shadow run of a distinct query and variant writes NOTHING durable.
        candidate = project.programs.get("grounded-rag").program.with_tunables(Tunables(top_k=1))
        base_mcc, base_qc = _mcc(store), _qc(store)
        with shadow_execution():
            BoundProgram(project, candidate).run(question="a wholly different promotion question here")
        assert _mcc(store) == base_mcc
        assert _qc(store) == base_qc
    finally:
        runtime.close()


def test_canary_served_run_writes_once_shadow_adds_no_rows(settings: Settings) -> None:
    runtime = _cached_runtime(settings)
    try:
        project = runtime.project(workspace="acme", corpus="handbook")
        _ingest(runtime, project.workspace_id, project.corpus_id)
        _, candidate_id = _promote(project, "staging", approve=False)
        project.deployment.set_canary("grounded-rag", "staging", candidate_id, percent=0)
        store = runtime.store

        before_mcc, before_qc = _mcc(store), _qc(store)
        result = project.deployment.run(
            "grounded-rag", "staging", inputs={"question": "how exactly does rollback work"}
        )
        assert result.served == "active"
        # The served run wrote its own answer-cache row exactly once; the unserved shadow
        # (different variant, otherwise a distinct model-call key) contributed nothing.
        assert _qc(store) - before_qc == 1
        served_only = _mcc(store) - before_mcc

        # Re-running the identical served query is a full cache hit: zero new rows. This
        # proves the earlier delta was the served run alone, never a leaked shadow write.
        mcc_after, qc_after = _mcc(store), _qc(store)
        project.deployment.run("grounded-rag", "staging", inputs={"question": "how exactly does rollback work"})
        assert _qc(store) == qc_after
        assert _mcc(store) == mcc_after
        assert served_only >= 1
    finally:
        runtime.close()


class _FlakyProgram(GroundedRagProgram):
    name = "flaky-rag"

    @property
    def shadow_safe(self) -> bool:
        return True

    def execute(
        self,
        project: RefaatProject,
        *,
        inputs: Mapping[str, Any],
        user_id: str | None,
        session_id: str | None,
        principal: Any = None,
    ):  # type: ignore[no-untyped-def]
        if self.tunables.top_k == 999:
            raise RuntimeError("shadow boom")
        return super().execute(project, inputs=inputs, user_id=user_id, session_id=session_id)


def test_raising_shadow_run_still_returns_served_answer(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    _ingest(runtime, project.workspace_id, project.corpus_id)
    project.programs.register(_FlakyProgram())

    base = _FlakyProgram().artifact()
    candidate = _FlakyProgram(tunables=Tunables(top_k=999)).artifact()
    registry = project.registry
    registry.save(base)
    registry.save(candidate)
    registry.promote("flaky-rag", "staging", base.artifact_id)

    # percent=0 => served=active (succeeds), shadow=candidate (raises, swallowed).
    project.deployment.set_canary("flaky-rag", "staging", candidate.artifact_id, percent=0)
    session_id = project.create_session(user_id="u1")
    result = project.deployment.run(
        "flaky-rag", "staging", inputs={"question": "how does rollback work"}, session_id=session_id
    )

    assert result.served == "active"
    assert result.result.artifact_id == base.artifact_id
    assert result.result.answer
    assert result.comparison is None
    assert result.shadow_run_id is None
    assert project.deployment.shadow_runs("flaky-rag", "staging") == []
    # The served run wrote the session exactly once; the raising shadow wrote nothing.
    turns = runtime.store.list_session_turns(project.workspace_id, session_id)
    assert [turn["role"] for turn in turns] == ["user", "assistant"]


def _write_react_program() -> GovernedReActProgram:
    class _WriteSkill:
        name = "update"
        description = "writes"

        def run(self, context: Any, arguments: Any) -> dict[str, Any]:
            return {"ok": True}

    policy = ReActPolicy(
        granted_scopes=("write:crm",),
        side_effect_ceiling=SideEffectClass.WRITE,
        grants=(ToolGrant("update", required_scopes=("write:crm",), side_effect=SideEffectClass.WRITE),),
    )
    planner = KeywordToolPlanner({"update": ("update",)})
    return GovernedReActProgram(policy=policy, planner=planner, skills={"update": _WriteSkill()})


def test_non_shadow_safe_program_refuses_canary_attach(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    _ingest(runtime, project.workspace_id, project.corpus_id)
    react = _write_react_program()
    assert react.shadow_safe is False
    project.programs.register(react)

    with pytest.raises(ValidationError, match="shadow-safe"):
        project.deployment.set_canary("governed-react", "staging", "sha256:whatever", percent=0)

    # A read-only program attaches fine (proven by the other canary tests).
    _, candidate_id = _promote(project, "staging", approve=False)
    attached = project.deployment.set_canary("grounded-rag", "staging", candidate_id, percent=0)
    assert attached["action"] == "set"


def test_sequential_canary_sequences_strictly_increase(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    _ingest(runtime, project.workspace_id, project.corpus_id)
    _, candidate_id = _promote(project, "staging", approve=False)

    first = project.deployment.set_canary("grounded-rag", "staging", candidate_id, percent=10)
    second = project.deployment.set_canary("grounded-rag", "staging", candidate_id, percent=20)
    cleared = project.deployment.clear_canary("grounded-rag", "staging")
    third = project.deployment.set_canary("grounded-rag", "staging", candidate_id, percent=30)

    sequences = [first["sequence"], second["sequence"], cleared["sequence"], third["sequence"]]
    assert sequences == sorted(sequences)
    assert len(set(sequences)) == len(sequences)
    current = project.deployment.current_canary("grounded-rag", "staging")
    assert current is not None
    assert current["sequence"] == third["sequence"]
    assert current["percent"] == 30


class _RecordingSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        self.events.append(event)


def test_shadow_run_does_not_consume_served_request_quota(settings: Settings) -> None:
    quota_settings = settings.model_copy(
        update={
            "quota_backend": "database",
            "quota_rules": [{"scope": "workspace", "dimension": "requests", "limit": 1, "window_seconds": 3600}],
        }
    )
    runtime = RuntimeBuilder(quota_settings).build()
    try:
        project = runtime.project(workspace="acme", corpus="handbook")
        _ingest(runtime, project.workspace_id, project.corpus_id)
        _, candidate_id = _promote(project, "staging", approve=False)
        project.deployment.set_canary("grounded-rag", "staging", candidate_id, percent=0)
        # With a REQUESTS limit of 1, the unserved shadow run must charge nothing: the first deployment
        # run serves (consuming the one unit) and only the second is denied. Before the fix the shadow
        # consumed the unit first and the very first served run was denied.
        first = project.deployment.run("grounded-rag", "staging", inputs={"question": "how does rollback work"})
        assert first.served == "active"
        with pytest.raises(QuotaExceededError):
            project.deployment.run("grounded-rag", "staging", inputs={"question": "how does rollback work"})
    finally:
        runtime.close()


def test_shadow_run_emits_no_served_query_audit(settings: Settings) -> None:
    sink = _RecordingSink()
    runtime = RuntimeBuilder(settings).with_access_policy(CanonicalAccessPolicy()).with_audit_sink(sink).build()
    try:
        project = runtime.project(workspace="merck", corpus="research")
        runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id="ops.txt",
                title="Ops",
                media_type="text/plain",
                content=(
                    b"Rollback restores the previously approved harness snapshot before deployment. "
                    b"A canary serves a fraction of traffic to a candidate artifact."
                ),
                metadata={"acl": {"classification": "confidential", "allow_groups": ["research-a"]}},
            ),
        )
        _, candidate_id = _promote(project, "staging", approve=False)
        project.deployment.set_canary("grounded-rag", "staging", candidate_id, percent=0)
        alice = Principal(id="alice", tenant="merck", groups=frozenset({"research-a"}))
        result = project.deployment.run(
            "grounded-rag", "staging", inputs={"question": "how does rollback work"}, principal=alice
        )
        assert result.served == "active"
        # Only the served run emits a query.authorized event; the unserved shadow run emits none, so no
        # two events collide on the shared request dedup_id.
        query_events = [event for event in sink.events if event.event_type == "query.authorized"]
        assert len(query_events) == 1
    finally:
        runtime.close()


class _DenyingGovernor:
    @property
    def cache_id(self) -> str:
        return "denying-governor:v1"

    def review(self, output: OutputContext) -> OutputDecision:
        return OutputDecision(allowed=False, reason="output_denied")


def test_shadow_run_emits_no_output_denied_audit(settings: Settings) -> None:
    sink = _RecordingSink()
    runtime = (
        RuntimeBuilder(settings)
        .with_access_policy(CanonicalAccessPolicy())
        .with_audit_sink(sink)
        .with_output_governor(_DenyingGovernor())
        .build()
    )
    try:
        project = runtime.project(workspace="merck", corpus="research")
        runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id="ops.txt",
                title="Ops",
                media_type="text/plain",
                content=(
                    b"Rollback restores the previously approved harness snapshot before deployment. "
                    b"A canary serves a fraction of traffic to a candidate artifact."
                ),
                metadata={"acl": {"classification": "confidential", "allow_groups": ["research-a"]}},
            ),
        )
        _, candidate_id = _promote(project, "staging", approve=False)
        project.deployment.set_canary("grounded-rag", "staging", candidate_id, percent=0)
        alice = Principal(id="alice", tenant="merck", groups=frozenset({"research-a"}))
        project.deployment.run(
            "grounded-rag", "staging", inputs={"question": "how does rollback work"}, principal=alice
        )
        # The governor denies both variants, but only the served run may emit an output.denied event;
        # the unserved shadow run emits none, so nothing collides on the shared request dedup_id.
        denied = [event for event in sink.events if event.event_type == "output.denied"]
        assert len(denied) == 1
    finally:
        runtime.close()


def test_unserved_rlm_run_is_isolated_from_served_quota_session_and_audit(settings: Settings) -> None:
    from refaat.application.services import unserved_run

    sink = _RecordingSink()
    quota_settings = settings.model_copy(
        update={
            "quota_backend": "database",
            "quota_rules": [{"scope": "workspace", "dimension": "requests", "limit": 1, "window_seconds": 3600}],
        }
    )
    runtime = RuntimeBuilder(quota_settings).with_access_policy(CanonicalAccessPolicy()).with_audit_sink(sink).build()
    try:
        alice = Principal(id="alice", tenant="merck", groups=frozenset({"research-a"}))
        project = runtime.project(workspace="merck", corpus="research")
        runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id="ops.txt",
                title="Ops",
                media_type="text/plain",
                content=(
                    b"Rollback restores the previously approved harness snapshot before deployment. "
                    b"A canary serves a fraction of traffic to a candidate artifact."
                ),
                metadata={"acl": {"classification": "confidential", "allow_groups": ["research-a"]}},
            ),
        )
        session_id = project.create_session(user_id="u")
        limits = RlmLimits(max_depth=1, max_nodes=4, branch_factor=2, leaf_token_budget=300)
        with unserved_run():
            project.rlm("how does rollback work", session_id=session_id, principal=alice, limits=limits)
        assert runtime.store.list_session_turns(project.workspace_id, session_id) == []
        assert [event for event in sink.events if event.event_type in ("rlm.authorized", "query.authorized")] == []
        # The unserved run consumed no REQUESTS unit, so a served query still admits under the limit of 1.
        assert project.ask("how does rollback work", principal=alice).answer
    finally:
        runtime.close()
