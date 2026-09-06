"""Fail-closed audit for the synchronous pre-return paths, byte-identical when off."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from refaat.application.approvals import ApprovalService, StoreApprovalBackend
from refaat.application.runtime import RefaatProject, RuntimeBuilder
from refaat.config import Settings
from refaat.core.text import stable_digest
from refaat.domain.audit import AuditEvent
from refaat.domain.authz import CanonicalAccessPolicy, Principal
from refaat.domain.models import DocumentInput, QueryRequest, RlmLimits
from refaat.errors import AuditUnavailableError, ConfigurationError
from refaat.infrastructure.audit import DurableAuditSink

pytestmark = pytest.mark.integration

ALICE = Principal(id="alice", tenant="merck", groups=frozenset({"research-a"}))
QUERY = "checkout production release rollback plan"
FINGERPRINT = "f" * 64


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        self.events.append(event)


class FailingSink:
    def record(self, event: AuditEvent) -> None:
        raise RuntimeError("sink down")


def _ingest(project: RefaatProject) -> None:
    project.runtime.ingest(
        project.workspace_id,
        project.corpus_id,
        DocumentInput(
            source_id="alpha.md",
            title="alpha.md",
            media_type="text/markdown",
            content=b"The alpha checkout production release requires an approved rollback plan.",
            metadata={"acl": {"classification": "confidential", "allow_groups": ["research-a"]}},
        ),
    )


def _runtime(settings: Settings, sink: object) -> object:
    return RuntimeBuilder(settings).with_access_policy(CanonicalAccessPolicy()).with_audit_sink(sink).build()


def _fail_closed(settings: Settings) -> Settings:
    return settings.model_copy(update={"audit_fail_closed": True})


def test_query_fails_closed_when_the_sink_is_down(settings: Settings) -> None:
    runtime = _runtime(_fail_closed(settings), FailingSink())
    try:
        project = runtime.project(workspace="merck", corpus="research")
        _ingest(project)
        with pytest.raises(AuditUnavailableError):
            project.ask(QUERY, principal=ALICE)
    finally:
        runtime.close()


def test_query_is_best_effort_when_fail_closed_is_off(settings: Settings) -> None:
    runtime = _runtime(settings, FailingSink())
    try:
        project = runtime.project(workspace="merck", corpus="research")
        _ingest(project)
        response = project.ask(QUERY, principal=ALICE)
        assert response.citations
    finally:
        runtime.close()


def test_rlm_fails_closed_when_the_sink_is_down(settings: Settings) -> None:
    runtime = _runtime(_fail_closed(settings), FailingSink())
    try:
        project = runtime.project(workspace="merck", corpus="research")
        _ingest(project)
        with pytest.raises(AuditUnavailableError):
            project.rlm(
                QUERY,
                principal=ALICE,
                limits=RlmLimits(max_depth=1, max_nodes=4, branch_factor=2, leaf_token_budget=300),
            )
    finally:
        runtime.close()


def test_streaming_fails_closed_before_any_byte_when_the_sink_is_down(settings: Settings) -> None:
    # Under fail-closed audit a streamed governed query buffers so the terminal event fires before any
    # delta, and a sink outage denies rather than serving unaudited: consuming the stream raises and no
    # answer delta was yielded.
    runtime = _runtime(_fail_closed(settings), FailingSink())
    try:
        project = runtime.project(workspace="merck", corpus="research")
        _ingest(project)
        request = QueryRequest(
            workspace_id=project.workspace_id,
            corpus_id=project.corpus_id,
            query=QUERY,
            principal=ALICE,
        )
        emitted: list[dict[str, object]] = []

        def drain() -> None:
            for event in runtime.stream_query(request):
                emitted.append(event)

        with pytest.raises(AuditUnavailableError):
            drain()
        assert not any(event.get("type") == "delta" for event in emitted)
    finally:
        runtime.close()


def test_streaming_is_best_effort_when_fail_closed_is_off(settings: Settings) -> None:
    # With the flag off (the default) a failing sink is swallowed and the stream serves incrementally,
    # byte-identical to before.
    runtime = _runtime(settings, FailingSink())
    try:
        project = runtime.project(workspace="merck", corpus="research")
        _ingest(project)
        request = QueryRequest(
            workspace_id=project.workspace_id,
            corpus_id=project.corpus_id,
            query=QUERY,
            principal=ALICE,
        )
        events = list(runtime.stream_query(request))
        assert any(event.get("type") == "final" for event in events)
    finally:
        runtime.close()


def test_fail_closed_without_a_sink_refuses_to_build(settings: Settings) -> None:
    with pytest.raises(ConfigurationError):
        RuntimeBuilder(_fail_closed(settings)).with_access_policy(CanonicalAccessPolicy()).build()


def test_approval_grant_fails_closed_and_leaves_no_active_approval(settings: Settings) -> None:
    runtime = _runtime(_fail_closed(settings), RecordingSink())
    try:
        project = runtime.project(workspace="merck", corpus="research")
        service = ApprovalService(runtime.store, project.workspace_id, audit_sink=FailingSink(), audit_fail_closed=True)
        with pytest.raises(AuditUnavailableError):
            service.grant(fingerprint=FINGERPRINT, capability_name="deploy", approver_id="ops")
        assert ApprovalService(runtime.store, project.workspace_id).list_active(FINGERPRINT) == []
    finally:
        runtime.close()


def test_grant_audits_before_the_approval_is_consumable(settings: Settings) -> None:
    # The approval must not be consumable at the moment its grant is audited (emit-then-insert).
    runtime = _runtime(settings, RecordingSink())
    seen: dict[str, object] = {}

    class OrderingSink:
        def __init__(self, store: object, workspace_id: str) -> None:
            self._store = store
            self._workspace_id = workspace_id

        def record(self, event: AuditEvent) -> None:
            del event
            seen["active_at_audit"] = ApprovalService(self._store, self._workspace_id).list_active(FINGERPRINT)

    try:
        project = runtime.project(workspace="merck", corpus="research")
        sink = OrderingSink(runtime.store, project.workspace_id)
        ApprovalService(runtime.store, project.workspace_id, audit_sink=sink).grant(
            fingerprint=FINGERPRINT, capability_name="deploy", approver_id="ops"
        )
        assert seen["active_at_audit"] == []
        assert ApprovalService(runtime.store, project.workspace_id).list_active(FINGERPRINT)
    finally:
        runtime.close()


def test_approval_cancel_fails_closed_even_without_a_canceller_identity(settings: Settings) -> None:
    runtime = _runtime(_fail_closed(settings), RecordingSink())
    try:
        project = runtime.project(workspace="merck", corpus="research")
        good = ApprovalService(runtime.store, project.workspace_id, audit_sink=RecordingSink())
        approval_id = good.grant(fingerprint=FINGERPRINT, capability_name="deploy", approver_id="ops")
        failing = ApprovalService(runtime.store, project.workspace_id, audit_sink=FailingSink(), audit_fail_closed=True)
        with pytest.raises(AuditUnavailableError):
            failing.cancel(approval_id)
    finally:
        runtime.close()


def test_approval_consume_fails_closed_when_the_sink_is_down(settings: Settings) -> None:
    runtime = _runtime(_fail_closed(settings), RecordingSink())
    try:
        project = runtime.project(workspace="merck", corpus="research")
        backend = StoreApprovalBackend(
            runtime.store, project.workspace_id, audit_sink=FailingSink(), audit_fail_closed=True
        )
        with pytest.raises(AuditUnavailableError):
            backend.record_consumed(
                FINGERPRINT,
                principal_id="alice",
                principal_kind="human",
                tenant="merck",
                capability_name="deploy",
                policy_id="react",
                policy_version="1",
                consumed_ids=["a1"],
                occurred_at=datetime.now(UTC),
            )
    finally:
        runtime.close()


def test_consume_with_audit_enqueues_the_audit_atomically_under_a_durable_sink(settings: Settings) -> None:
    runtime = _runtime(settings, RecordingSink())
    try:
        project = runtime.project(workspace="merck", corpus="research")
        approval_id = ApprovalService(runtime.store, project.workspace_id, audit_sink=RecordingSink()).grant(
            fingerprint=FINGERPRINT, capability_name="deploy", approver_id="ops"
        )
        backend = StoreApprovalBackend(runtime.store, project.workspace_id, audit_sink=DurableAuditSink(runtime.store))
        consumed = backend.consume_with_audit(
            FINGERPRINT,
            [approval_id],
            principal_id="alice",
            principal_kind="human",
            tenant="merck",
            capability_name="deploy",
            policy_id="react",
            policy_version="1",
            occurred_at=datetime.now(UTC),
        )
        assert consumed is True
        assert ApprovalService(runtime.store, project.workspace_id).list_active(FINGERPRINT) == []
        depth, _oldest, _attempts = runtime.store.audit_outbox_backlog(project.workspace_id)
        assert depth == 1
        claimed = runtime.store.claim_audit_outbox("tw", 60, 10)
        assert claimed[0]["payload"]["event_type"] == "approval.consumed"
        assert claimed[0]["payload"]["dedup_id"] == stable_digest(
            ("approval.consumed", project.workspace_id, FINGERPRINT, "alice", approval_id)
        )
    finally:
        runtime.close()


def test_consume_with_audit_rolls_back_the_consume_when_the_audit_enqueue_fails(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime(settings, RecordingSink())
    try:
        project = runtime.project(workspace="merck", corpus="research")
        approval_id = ApprovalService(runtime.store, project.workspace_id, audit_sink=RecordingSink()).grant(
            fingerprint=FINGERPRINT, capability_name="deploy", approver_id="ops"
        )
        backend = StoreApprovalBackend(runtime.store, project.workspace_id, audit_sink=DurableAuditSink(runtime.store))

        def boom(*args: object, **kwargs: object) -> str:
            raise RuntimeError("audit insert failed")

        monkeypatch.setattr(runtime.store, "_insert_audit_outbox", boom)
        with pytest.raises(RuntimeError, match="audit insert failed"):
            backend.consume_with_audit(
                FINGERPRINT,
                [approval_id],
                principal_id="alice",
                principal_kind="human",
                tenant="merck",
                capability_name="deploy",
                policy_id="react",
                policy_version="1",
                occurred_at=datetime.now(UTC),
            )
        # The consume rolled back with the failed audit: the approval is still active and consumable.
        assert ApprovalService(runtime.store, project.workspace_id).list_active(FINGERPRINT) != []
        depth, _oldest, _attempts = runtime.store.audit_outbox_backlog(project.workspace_id)
        assert depth == 0
    finally:
        runtime.close()


def test_cancel_with_audit_rolls_back_the_cancel_when_the_audit_enqueue_fails(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime(settings, RecordingSink())
    try:
        project = runtime.project(workspace="merck", corpus="research")
        approval_id = ApprovalService(runtime.store, project.workspace_id, audit_sink=RecordingSink()).grant(
            fingerprint=FINGERPRINT, capability_name="deploy", approver_id="ops"
        )
        canceller = ApprovalService(runtime.store, project.workspace_id, audit_sink=DurableAuditSink(runtime.store))

        def boom(*args: object, **kwargs: object) -> str:
            raise RuntimeError("audit insert failed")

        monkeypatch.setattr(runtime.store, "_insert_audit_outbox", boom)
        with pytest.raises(RuntimeError, match="audit insert failed"):
            canceller.cancel(approval_id, canceller_id="ops")
        # The cancel rolled back with the failed audit: the approval is still active.
        assert ApprovalService(runtime.store, project.workspace_id).list_active(FINGERPRINT) != []
    finally:
        runtime.close()


def test_cancel_with_audit_enqueues_atomically_under_a_durable_sink(settings: Settings) -> None:
    runtime = _runtime(settings, RecordingSink())
    try:
        project = runtime.project(workspace="merck", corpus="research")
        approval_id = ApprovalService(runtime.store, project.workspace_id, audit_sink=RecordingSink()).grant(
            fingerprint=FINGERPRINT, capability_name="deploy", approver_id="ops"
        )
        canceller = ApprovalService(runtime.store, project.workspace_id, audit_sink=DurableAuditSink(runtime.store))
        assert canceller.cancel(approval_id, canceller_id="ops") is True
        assert ApprovalService(runtime.store, project.workspace_id).list_active(FINGERPRINT) == []
        depth, _oldest, _attempts = runtime.store.audit_outbox_backlog(project.workspace_id)
        assert depth == 1
        claimed = runtime.store.claim_audit_outbox("tw", 60, 10)
        assert claimed[0]["payload"]["event_type"] == "approval.cancelled"
        assert claimed[0]["payload"]["dedup_id"] == stable_digest(
            ("approval.cancelled", project.workspace_id, approval_id)
        )
    finally:
        runtime.close()


def test_a_working_sink_records_the_same_events_across_the_flag(settings: Settings) -> None:
    def events_for(active: Settings) -> list[tuple[str, str | None, str, int]]:
        sink = RecordingSink()
        runtime = _runtime(active, sink)
        try:
            project = runtime.project(workspace="merck", corpus="research")
            _ingest(project)
            project.ask(QUERY, principal=ALICE)
        finally:
            runtime.close()
        return [(e.event_type, e.principal_id, e.dedup_id, e.citation_count) for e in sink.events]

    assert events_for(settings) == events_for(_fail_closed(settings))
