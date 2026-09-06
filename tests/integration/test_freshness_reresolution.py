"""Worker-side identity-freshness re-resolution for durable rlm.run jobs.

With a PrincipalDirectory installed under an active access policy, a durable rlm.run job carries the
enqueue-time identity snapshot; the worker re-resolves the principal and admits the run only when the
snapshot comes back fresh, under the refreshed principal. Every staleness or mismatch is a terminal
denial; only a directory outage is retryable. Byte-identical on the no-policy path: no snapshot, no
re-resolution, principal None, unfiltered.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import update

from refaat.application.rlm import capture_durable_snapshot
from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.application.worker import DurableWorker
from refaat.config import Settings
from refaat.domain.audit import AuditEvent
from refaat.domain.authz import CanonicalAccessPolicy, Principal
from refaat.domain.freshness import snapshot_to_payload
from refaat.domain.models import DocumentInput
from refaat.infrastructure.database.schema import jobs

pytestmark = pytest.mark.integration

ALICE = Principal(id="alice", tenant="merck", groups=frozenset({"research-a"}))
QUERY = "What is the production rollback approval control for checkout releases?"
ALICE_TEXT = "Alice team onboarding handbook. " + " ".join(f"onboarding note {index}." for index in range(60))
BOB_TEXT = (
    "BOBSECRET production rollback approval control: checkout releases require a signed production "
    "rollback plan and production approval before any checkout release rollback. "
    + " ".join(f"production rollback approval detail {index}." for index in range(60))
)


class _Directory:
    def __init__(self, principal: Principal | None, *, unavailable: bool = False) -> None:
        self.principal = principal
        self.unavailable = unavailable

    def refresh(self, principal_id: str) -> Principal | None:
        del principal_id
        if self.unavailable:
            raise RuntimeError("directory unavailable")
        return self.principal


class _RecordingSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        self.events.append(event)


class _FailingSink:
    def record(self, event: AuditEvent) -> None:
        del event
        raise RuntimeError("audit sink down")


def _build(
    settings: Settings,
    *,
    directory: _Directory | None = None,
    sink: object | None = None,
    with_policy: bool = True,
    audit_fail_closed: bool = False,
) -> Refaat:
    effective = settings.model_copy(update={"audit_fail_closed": True}) if audit_fail_closed else settings
    builder = RuntimeBuilder(effective)
    if with_policy:
        builder = builder.with_access_policy(CanonicalAccessPolicy())
    if directory is not None:
        builder = builder.with_principal_directory(directory)
    if sink is not None:
        builder = builder.with_audit_sink(sink)  # type: ignore[arg-type]
    return builder.build()


def _seed(runtime: Refaat) -> tuple[str, str, dict[str, str]]:
    workspace_id = runtime.create_workspace("merck", "Merck")
    corpus_id = runtime.create_corpus(workspace_id, "research")
    docs = {
        "alice.md": ({"acl": {"classification": "confidential", "allow_groups": ["research-a"]}}, ALICE_TEXT),
        "bob.md": ({"acl": {"classification": "confidential", "allow_groups": ["research-b"]}}, BOB_TEXT),
    }
    for source_id, (metadata, content) in docs.items():
        runtime.ingest(
            workspace_id,
            corpus_id,
            DocumentInput(
                source_id=source_id,
                title=source_id,
                media_type="text/markdown",
                content=content.encode(),
                metadata=dict(metadata),
            ),
        )
    ids = {str(doc["source_id"]): str(doc["id"]) for doc in runtime.store.list_documents(workspace_id, corpus_id)}
    return workspace_id, corpus_id, ids


def _make_claimable(runtime: Refaat, job_id: str) -> None:
    # Reset the backoff run_after so the next claim can pick the job up immediately, standing in for
    # the retry window elapsing (the SystemClock cannot be advanced in the test).
    with runtime.store.engine.begin() as conn:
        conn.execute(update(jobs).where(jobs.c.id == job_id).values(run_after=runtime.store.clock.now()))


def _event_types(sink: _RecordingSink) -> list[str]:
    return [event.event_type for event in sink.events]


def _direct_rlm_job(runtime: Refaat, workspace_id: str, corpus_id: str, snapshot_payload: dict[str, str]) -> str:
    return runtime.store.enqueue_job(
        workspace_id,
        "rlm.run",
        {
            "corpus_id": corpus_id,
            "query": QUERY,
            "user_id": None,
            "session_id": None,
            "limits": {"max_depth": 2, "max_nodes": 16, "branch_factor": 4, "leaf_token_budget": 1_500},
            "_identity_snapshot": snapshot_payload,
        },
    )


@pytest.fixture
def fresh_runtime(settings: Settings) -> Iterator[tuple[Refaat, _RecordingSink, _Directory]]:
    sink = _RecordingSink()
    directory = _Directory(Principal(id="alice", tenant="merck", groups=frozenset({"research-a"})))
    runtime = _build(settings, directory=directory, sink=sink)
    try:
        yield runtime, sink, directory
    finally:
        runtime.close()


def test_fresh_snapshot_runs_under_the_refreshed_principal(
    fresh_runtime: tuple[Refaat, _RecordingSink, _Directory],
) -> None:
    runtime, sink, _directory = fresh_runtime
    workspace_id, corpus_id, ids = _seed(runtime)
    job_id = runtime.enqueue_rlm(workspace_id, corpus_id, QUERY, principal=ALICE)
    assert DurableWorker(runtime).run_once() is True

    job = runtime.store.get_job(workspace_id, job_id)
    assert str(job["status"]) == "succeeded"
    citations = job["result"]["citations"]
    assert citations
    cited_documents = {str(citation["document_id"]) for citation in citations}
    # The BOB-ONLY document is filtered out under the refreshed research-a principal, so it never
    # appears in the citations. The run genuinely failed if the worker had passed principal None.
    assert ids["bob.md"] not in cited_documents
    assert all("BOBSECRET" not in str(citation["quote"]) for citation in citations)
    assert "identity.reresolved" in _event_types(sink)
    reresolved = next(event for event in sink.events if event.event_type == "identity.reresolved")
    assert reresolved.capability_name == "fresh"
    assert reresolved.principal_id == "alice"


def test_no_snapshot_is_a_terminal_denial(settings: Settings) -> None:
    sink = _RecordingSink()
    runtime = _build(settings, directory=_Directory(ALICE), sink=sink)
    try:
        workspace_id, corpus_id, _ids = _seed(runtime)
        job_id = runtime.enqueue_rlm(workspace_id, corpus_id, QUERY)
        assert DurableWorker(runtime).run_once() is True
        job = runtime.store.get_job(workspace_id, job_id)
        assert str(job["status"]) == "failed"
        refused = [event for event in sink.events if event.event_type == "identity.refused"]
        assert [event.capability_name for event in refused] == ["no_snapshot"]
        assert "rlm.authorized" not in _event_types(sink)
    finally:
        runtime.close()


def test_identity_revoked_is_a_terminal_denial(settings: Settings) -> None:
    sink = _RecordingSink()
    runtime = _build(settings, directory=_Directory(None), sink=sink)
    try:
        workspace_id, corpus_id, _ids = _seed(runtime)
        job_id = runtime.enqueue_rlm(workspace_id, corpus_id, QUERY, principal=ALICE)
        assert DurableWorker(runtime).run_once() is True
        assert str(runtime.store.get_job(workspace_id, job_id)["status"]) == "failed"
        assert "identity_revoked" in [
            event.capability_name for event in sink.events if event.event_type == "identity.refused"
        ]
        assert "rlm.authorized" not in _event_types(sink)
    finally:
        runtime.close()


def test_scope_changed_is_rejected_and_the_rlm_does_not_run(settings: Settings) -> None:
    sink = _RecordingSink()
    changed = _Directory(Principal(id="alice", tenant="merck", groups=frozenset({"research-b"})))
    runtime = _build(settings, directory=changed, sink=sink)
    try:
        workspace_id, corpus_id, _ids = _seed(runtime)
        job_id = runtime.enqueue_rlm(workspace_id, corpus_id, QUERY, principal=ALICE)
        assert DurableWorker(runtime).run_once() is True
        assert str(runtime.store.get_job(workspace_id, job_id)["status"]) == "failed"
        assert "scope_changed" in [
            event.capability_name for event in sink.events if event.event_type == "identity.refused"
        ]
        # A rejected re-resolution never runs the RLM, so there is no evidence-serving audit event.
        assert "rlm.authorized" not in _event_types(sink)
    finally:
        runtime.close()


def test_principal_mismatch_is_a_terminal_denial(settings: Settings) -> None:
    sink = _RecordingSink()
    wrong = _Directory(Principal(id="mallory", tenant="merck", groups=frozenset({"research-a"})))
    runtime = _build(settings, directory=wrong, sink=sink)
    try:
        workspace_id, corpus_id, _ids = _seed(runtime)
        job_id = runtime.enqueue_rlm(workspace_id, corpus_id, QUERY, principal=ALICE)
        assert DurableWorker(runtime).run_once() is True
        assert str(runtime.store.get_job(workspace_id, job_id)["status"]) == "failed"
        assert "principal_mismatch" in [
            event.capability_name for event in sink.events if event.event_type == "identity.refused"
        ]
    finally:
        runtime.close()


def test_tenant_changed_is_a_terminal_denial(settings: Settings) -> None:
    sink = _RecordingSink()
    moved = _Directory(Principal(id="alice", tenant="pfizer", groups=frozenset({"research-a"})))
    runtime = _build(settings, directory=moved, sink=sink)
    try:
        workspace_id, corpus_id, _ids = _seed(runtime)
        job_id = runtime.enqueue_rlm(workspace_id, corpus_id, QUERY, principal=ALICE)
        assert DurableWorker(runtime).run_once() is True
        assert str(runtime.store.get_job(workspace_id, job_id)["status"]) == "failed"
        assert "tenant_changed" in [
            event.capability_name for event in sink.events if event.event_type == "identity.refused"
        ]
    finally:
        runtime.close()


def test_snapshot_too_old_is_a_terminal_denial(settings: Settings) -> None:
    sink = _RecordingSink()
    runtime = _build(settings, directory=_Directory(ALICE), sink=sink)
    try:
        workspace_id, corpus_id, _ids = _seed(runtime)
        stale = snapshot_to_payload(
            capture_durable_snapshot(ALICE, CanonicalAccessPolicy(), datetime.now(UTC) - timedelta(seconds=86_401))
        )
        job_id = _direct_rlm_job(runtime, workspace_id, corpus_id, stale)
        assert DurableWorker(runtime).run_once() is True
        assert str(runtime.store.get_job(workspace_id, job_id)["status"]) == "failed"
        assert "snapshot_too_old" in [
            event.capability_name for event in sink.events if event.event_type == "identity.refused"
        ]
        assert "rlm.authorized" not in _event_types(sink)
    finally:
        runtime.close()


def test_directory_outage_is_retryable_then_a_recovered_directory_succeeds(settings: Settings) -> None:
    sink = _RecordingSink()
    directory = _Directory(ALICE, unavailable=True)
    runtime = _build(settings, directory=directory, sink=sink)
    try:
        workspace_id, corpus_id, _ids = _seed(runtime)
        job_id = runtime.enqueue_rlm(workspace_id, corpus_id, QUERY, principal=ALICE)
        assert DurableWorker(runtime).run_once() is True
        # A transient directory outage is retryable, not terminal: the job returns to PENDING.
        assert str(runtime.store.get_job(workspace_id, job_id)["status"]) == "pending"
        assert "directory_unavailable" in [
            event.capability_name for event in sink.events if event.event_type == "identity.refused"
        ]
        directory.unavailable = False
        _make_claimable(runtime, job_id)
        assert DurableWorker(runtime).run_once() is True
        assert str(runtime.store.get_job(workspace_id, job_id)["status"]) == "succeeded"
    finally:
        runtime.close()


def test_audit_fail_closed_retries_the_allow_path_rather_than_running_unaudited(settings: Settings) -> None:
    runtime = _build(settings, directory=_Directory(ALICE), sink=_FailingSink(), audit_fail_closed=True)
    try:
        workspace_id, corpus_id, _ids = _seed(runtime)
        job_id = runtime.enqueue_rlm(workspace_id, corpus_id, QUERY, principal=ALICE)
        assert DurableWorker(runtime).run_once() is True
        # The allow-path audit is fail-closed, so a sink outage raises before the run and the job
        # retries rather than serving the RLM unaudited: it is back to PENDING with no result.
        job = runtime.store.get_job(workspace_id, job_id)
        assert str(job["status"]) == "pending"
        assert job["result"] is None
    finally:
        runtime.close()


def test_no_policy_path_is_unfiltered_and_byte_identical(settings: Settings) -> None:
    runtime = _build(settings, with_policy=False)
    try:
        workspace_id, corpus_id, ids = _seed(runtime)
        job_id = runtime.enqueue_rlm(workspace_id, corpus_id, QUERY)
        assert DurableWorker(runtime).run_once() is True
        job = runtime.store.get_job(workspace_id, job_id)
        assert str(job["status"]) == "succeeded"
        citations = job["result"]["citations"]
        assert citations
        # No snapshot was written and the gate is a no-op with no policy, so the run is unfiltered:
        # the BOB-ONLY document is a candidate here, which is exactly what the policy path excludes.
        assert ids["bob.md"] in {str(citation["document_id"]) for citation in citations}
    finally:
        runtime.close()
