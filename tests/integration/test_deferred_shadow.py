"""Serve-first async shadow. When defer_shadow is on and no access policy is active, the canary
comparison is deferred to a durable job so it never delays the served response. The job persists no answer
text (telemetry only), carries the query like any durable evidence job, and is bounded by a deadline.
"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest
from sqlalchemy import select

from refaat.application.runtime import Refaat, RefaatProject, RuntimeBuilder
from refaat.application.worker import DurableWorker
from refaat.config import Settings
from refaat.domain.authz import CanonicalAccessPolicy, Principal
from refaat.domain.models import DocumentInput
from refaat.infrastructure.database.schema import jobs
from refaat.programs import GroundedRagProgram, Tunables

pytestmark = pytest.mark.integration


def _ingest(runtime: Refaat, workspace_id: str, corpus_id: str, *, acl: dict | None = None) -> None:
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
            metadata={"acl": acl} if acl else {},
        ),
    )


def _setup_canary(project: RefaatProject, environment: str, *, percent: int = 0) -> tuple[str, str]:
    base = GroundedRagProgram().artifact()
    candidate = GroundedRagProgram(tunables=Tunables(top_k=2)).artifact()
    registry = project.registry
    registry.save(base)
    registry.save(candidate)
    registry.promote("grounded-rag", environment, base.artifact_id)
    project.deployment.set_canary("grounded-rag", environment, candidate.artifact_id, percent=percent)
    return base.artifact_id, candidate.artifact_id


def _defer_runtime(settings: Settings, *, policy: bool = False) -> Refaat:
    builder = RuntimeBuilder(settings.model_copy(update={"defer_shadow": True}))
    if policy:
        builder = builder.with_access_policy(CanonicalAccessPolicy())
    return builder.build()


def _shadow_jobs(store: object) -> list[dict]:
    with store.engine.connect() as conn:  # type: ignore[attr-defined]
        rows = conn.execute(select(jobs).where(jobs.c.kind == "program.shadow_compare")).mappings().all()
    return [dict(row) for row in rows]


def test_defer_serves_inline_and_persists_no_answer_at_rest(settings: Settings) -> None:
    runtime = _defer_runtime(settings)
    try:
        project = runtime.project(workspace="acme", corpus="handbook")
        _ingest(runtime, project.workspace_id, project.corpus_id)
        _setup_canary(project, "staging")

        result = project.deployment.run("grounded-rag", "staging", inputs={"question": "how does rollback work"})

        # Serve-first: the authoritative served answer is returned now, the comparison is deferred.
        assert result.served == "active"
        assert result.result.answer
        assert result.comparison is None
        assert result.shadow_run_id is None

        # Exactly one deferred compare job is enqueued, and its payload carries the query but NO answer text.
        shadow_jobs = _shadow_jobs(runtime.store)
        assert len(shadow_jobs) == 1
        payload = shadow_jobs[0]["payload"]
        assert payload["deferred"] is True
        assert payload["questions"] == ["how does rollback work"]
        assert payload["served"] == "active"
        assert "deadline" in payload
        assert "result" not in payload
        assert "answer" not in payload
        assert result.result.answer not in json.dumps(payload)
    finally:
        runtime.close()


def test_worker_deferred_branch_is_telemetry_only_and_hash_only(settings: Settings) -> None:
    runtime = _defer_runtime(settings)
    try:
        project = runtime.project(workspace="acme", corpus="handbook")
        _ingest(runtime, project.workspace_id, project.corpus_id)
        _setup_canary(project, "staging")
        served = project.deployment.run("grounded-rag", "staging", inputs={"question": "how does rollback work"})

        payload = _shadow_jobs(runtime.store)[0]["payload"]
        worker = DurableWorker(runtime)
        outcome = worker._shadow_compare(project.workspace_id, payload)

        # The handler returns telemetry only: shadow run ids, never a program result / answer text.
        assert outcome["deferred"] is True
        assert outcome["shadow_run_ids"]
        assert all(sid.startswith("shd_") for sid in outcome["shadow_run_ids"])
        assert "runs" not in outcome
        assert served.result.answer not in json.dumps(outcome)

        # The re-derive recorded exactly one hash-only telemetry row with a real comparison.
        runs = project.deployment.shadow_runs("grounded-rag", "staging")
        assert len(runs) == 1
        assert runs[0]["served"] == "active"
        assert runs[0]["comparison"] is not None
        # The row stores the query HASH, never the query text (like the inline path).
        assert "how does rollback work" not in json.dumps(runs[0], default=str)
    finally:
        runtime.close()


def test_worker_deferred_branch_skips_a_stale_job(settings: Settings) -> None:
    runtime = _defer_runtime(settings)
    try:
        project = runtime.project(workspace="acme", corpus="handbook")
        _ingest(runtime, project.workspace_id, project.corpus_id)
        _setup_canary(project, "staging")

        past = (runtime.store.clock.now() - timedelta(seconds=1)).isoformat()
        stale_payload = {
            "corpus_id": project.corpus_id,
            "program": "grounded-rag",
            "environment": "staging",
            "questions": ["how does rollback work"],
            "served": "active",
            "bucket": 0,
            "compare": True,
            "deferred": True,
            "deadline": past,
        }
        outcome = DurableWorker(runtime)._shadow_compare(project.workspace_id, stale_payload)

        assert outcome == {"deferred": True, "skipped": "stale", "shadow_run_ids": []}
        assert project.deployment.shadow_runs("grounded-rag", "staging") == []
    finally:
        runtime.close()


def test_defer_does_not_fire_under_an_active_policy(settings: Settings) -> None:
    runtime = _defer_runtime(settings, policy=True)
    try:
        project = runtime.project(workspace="merck", corpus="research")
        _ingest(
            runtime,
            project.workspace_id,
            project.corpus_id,
            acl={"classification": "confidential", "allow_groups": ["research-a"]},
        )
        _setup_canary(project, "staging")
        alice = Principal(id="alice", tenant="merck", groups=frozenset({"research-a"}))

        result = project.deployment.run(
            "grounded-rag", "staging", inputs={"question": "how does rollback work"}, principal=alice
        )

        # Under a policy the defer branch never fires: the comparison is computed inline and no job is enqueued.
        assert result.served == "active"
        assert result.comparison is not None
        assert result.shadow_run_id is not None
        assert _shadow_jobs(runtime.store) == []
    finally:
        runtime.close()


def test_run_once_dispatch_persists_no_answer_in_jobs_result(settings: Settings) -> None:
    runtime = _defer_runtime(settings)
    try:
        project = runtime.project(workspace="acme", corpus="handbook")
        _ingest(runtime, project.workspace_id, project.corpus_id)
        _setup_canary(project, "staging")
        served = project.deployment.run("grounded-rag", "staging", inputs={"question": "how does rollback work"})

        # Drain the durable queue through the real dispatch path (not a direct handler call).
        worker = DurableWorker(runtime)
        for _ in range(25):
            if not worker.run_once():
                break

        with runtime.store.engine.connect() as conn:
            row = conn.execute(select(jobs).where(jobs.c.kind == "program.shadow_compare")).mappings().first()
        assert row is not None
        result = row["result"]
        # The completed job's persisted result is telemetry only: no program result, no served answer text.
        assert result is not None
        assert result.get("deferred") is True
        assert "runs" not in result
        assert served.result.answer not in json.dumps(result)
        assert len(project.deployment.shadow_runs("grounded-rag", "staging")) == 1
    finally:
        runtime.close()


def test_operator_batch_never_defers_even_when_defer_shadow_is_on(settings: Settings) -> None:
    runtime = _defer_runtime(settings)
    try:
        project = runtime.project(workspace="acme", corpus="handbook")
        _ingest(runtime, project.workspace_id, project.corpus_id)
        _setup_canary(project, "staging")

        # The operator batch (non-deferred payload) passes allow_defer=False, so it computes the comparison
        # inline in-job and never re-enters the defer branch (no nested job, no answer-free-fall recursion).
        worker = DurableWorker(runtime)
        outcome = worker._shadow_compare(
            project.workspace_id,
            {
                "corpus_id": project.corpus_id,
                "program": "grounded-rag",
                "environment": "staging",
                "questions": ["how does rollback work"],
            },
        )

        assert "runs" in outcome
        assert _shadow_jobs(runtime.store) == []
        assert len(project.deployment.shadow_runs("grounded-rag", "staging")) == 1
    finally:
        runtime.close()
