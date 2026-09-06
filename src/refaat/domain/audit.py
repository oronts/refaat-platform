"""The enterprise audit event contract and its pluggable sink.

An AuditEvent is redacted by construction: it carries ids, counts, the policy, a governance
action capability name and action fingerprint, and timestamps, never a raw query, raw evidence
text, or secrets. A deployment installs an AuditSink to deliver events to an append-only store,
a SIEM, or a signed ledger.

Schema version 2 added the governance fields capability_name and action_fingerprint, so every
serialized event carries them (null on the evidence-serving events); a consumer keyed on the
version knows the shape without inspecting fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from refaat.errors import AuditUnavailableError

AUDIT_SCHEMA_VERSION = "2"


@dataclass(frozen=True, slots=True)
class AuditEvent:
    event_type: str
    occurred_at: datetime
    workspace_id: str
    principal_id: str
    principal_kind: str
    tenant: str | None
    policy_id: str
    policy_version: str
    dedup_id: str
    corpus_id: str = ""
    corpus_version: int = 0
    cited_document_ids: tuple[str, ...] = ()
    citation_count: int = 0
    abstained: bool = False
    cached: bool = False
    trace_id: str | None = None
    capability_name: str | None = None
    action_fingerprint: str | None = None
    schema_version: str = AUDIT_SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        """A JSON-safe mapping for durable persistence: the timestamp as an ISO 8601 string
        and the tuple field as a list. Additive, so the redaction contract is unchanged."""
        return {
            "event_type": self.event_type,
            "occurred_at": self.occurred_at.isoformat(),
            "workspace_id": self.workspace_id,
            "principal_id": self.principal_id,
            "principal_kind": self.principal_kind,
            "tenant": self.tenant,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "dedup_id": self.dedup_id,
            "corpus_id": self.corpus_id,
            "corpus_version": self.corpus_version,
            "cited_document_ids": list(self.cited_document_ids),
            "citation_count": self.citation_count,
            "abstained": self.abstained,
            "cached": self.cached,
            "trace_id": self.trace_id,
            "capability_name": self.capability_name,
            "action_fingerprint": self.action_fingerprint,
            "schema_version": self.schema_version,
        }


def audit_event_from_json(payload: dict[str, Any]) -> AuditEvent:
    """Reconstruct an AuditEvent from a to_json mapping, the faithful inverse: the ISO 8601
    timestamp parses back to a datetime and the document-id list restores to a tuple."""
    return AuditEvent(
        event_type=str(payload["event_type"]),
        occurred_at=datetime.fromisoformat(str(payload["occurred_at"])),
        workspace_id=str(payload["workspace_id"]),
        principal_id=str(payload["principal_id"]),
        principal_kind=str(payload["principal_kind"]),
        tenant=payload["tenant"],
        policy_id=str(payload["policy_id"]),
        policy_version=str(payload["policy_version"]),
        dedup_id=str(payload["dedup_id"]),
        corpus_id=str(payload.get("corpus_id", "")),
        corpus_version=int(payload.get("corpus_version", 0)),
        cited_document_ids=tuple(payload.get("cited_document_ids", ())),
        citation_count=int(payload.get("citation_count", 0)),
        abstained=bool(payload.get("abstained", False)),
        cached=bool(payload.get("cached", False)),
        trace_id=payload.get("trace_id"),
        capability_name=payload.get("capability_name"),
        action_fingerprint=payload.get("action_fingerprint"),
        schema_version=str(payload.get("schema_version", AUDIT_SCHEMA_VERSION)),
    )


class AuditSink(Protocol):
    """Receives audit events. A deployment routes them to its durable audit backend."""

    def record(self, event: AuditEvent) -> None: ...


def emit_audit(
    sink: AuditSink | None,
    event: AuditEvent,
    *,
    fail_closed: bool,
    logger: Any,
    log_key: str = "audit_sink_failed",
) -> None:
    """Deliver one audit event under the configured outage policy.

    No-op when no sink is installed. Best-effort by default: a sink failure is logged and swallowed
    so the operation completes. Fail-closed raises AuditUnavailableError on a sink failure so a
    pre-return operation refuses rather than proceeding unaudited.
    """
    if sink is None:
        return
    try:
        sink.record(event)
    except Exception as exc:
        if fail_closed:
            raise AuditUnavailableError(f"audit sink unavailable for {event.event_type}") from exc
        logger.warning(log_key, event_type=event.event_type)


__all__ = ["AUDIT_SCHEMA_VERSION", "AuditEvent", "AuditSink", "audit_event_from_json", "emit_audit"]
