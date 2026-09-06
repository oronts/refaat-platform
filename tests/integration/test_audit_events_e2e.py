"""Audit events for authorized queries: emitted, redacted, and pluggable."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import fields

import pytest

from refaat.application.runtime import RefaatProject, RuntimeBuilder
from refaat.config import Settings
from refaat.domain.audit import AuditEvent
from refaat.domain.authz import CanonicalAccessPolicy, Principal
from refaat.domain.models import DocumentInput

pytestmark = pytest.mark.integration

ALICE = Principal(id="alice", tenant="merck", groups=frozenset({"research-a"}))
QUERY = "checkout production release rollback plan"


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        self.events.append(event)


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


@pytest.fixture
def audited(settings: Settings) -> Iterator[tuple[RefaatProject, RecordingSink]]:
    sink = RecordingSink()
    lenient = settings.model_copy(update={"strict_authorization": False})
    runtime = RuntimeBuilder(lenient).with_access_policy(CanonicalAccessPolicy()).with_audit_sink(sink).build()
    try:
        project = runtime.project(workspace="merck", corpus="research")
        _ingest(project)
        yield project, sink
    finally:
        runtime.close()


def test_an_authorized_query_emits_one_audit_event(audited: tuple[RefaatProject, RecordingSink]) -> None:
    project, sink = audited
    project.ask(QUERY, principal=ALICE)
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.event_type == "query.authorized"
    assert event.principal_id == "alice"
    assert event.principal_kind == "human"
    assert event.tenant == "merck"
    assert event.policy_id == "canonical"
    assert event.policy_version == "1"
    assert event.dedup_id


def test_audit_event_is_redacted_by_construction(audited: tuple[RefaatProject, RecordingSink]) -> None:
    project, sink = audited
    project.ask(QUERY, principal=ALICE)
    event = sink.events[0]
    values = " ".join(str(getattr(event, field.name)) for field in fields(event))
    assert "checkout" not in values
    assert "rollback" not in values


def test_no_principal_emits_no_audit_event(audited: tuple[RefaatProject, RecordingSink]) -> None:
    project, sink = audited
    project.ask(QUERY)
    assert sink.events == []


def test_an_authorized_rlm_run_emits_an_audit_event(audited: tuple[RefaatProject, RecordingSink]) -> None:
    from refaat.domain.models import RlmLimits

    project, sink = audited
    project.rlm(
        QUERY, principal=ALICE, limits=RlmLimits(max_depth=1, max_nodes=4, branch_factor=2, leaf_token_budget=300)
    )
    assert any(event.event_type == "rlm.authorized" for event in sink.events)
    rlm_event = next(event for event in sink.events if event.event_type == "rlm.authorized")
    assert rlm_event.principal_id == "alice"
    assert rlm_event.policy_id == "canonical"


def test_a_sink_failure_does_not_break_the_query(settings: Settings) -> None:
    class FailingSink:
        def record(self, event: AuditEvent) -> None:
            raise RuntimeError("sink down")

    runtime = (
        RuntimeBuilder(settings).with_access_policy(CanonicalAccessPolicy()).with_audit_sink(FailingSink()).build()
    )
    try:
        project = runtime.project(workspace="merck", corpus="research")
        _ingest(project)
        response = project.ask(QUERY, principal=ALICE)
        assert response.citations
    finally:
        runtime.close()
