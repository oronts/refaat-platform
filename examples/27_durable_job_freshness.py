from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from _shared import demo_settings, emit

from refaat.application.runtime import Refaat, RefaatProject, RuntimeBuilder
from refaat.application.worker import DurableWorker
from refaat.domain.audit import AuditEvent
from refaat.domain.authz import CanonicalAccessPolicy, Principal, durable_entitlement_digest
from refaat.domain.models import DocumentInput

QUERY = "What is the production rollback approval control for checkout releases?"

# Body-level tells. Each marker appears only in one document body, so their absence from an answer,
# a citation quote, or any audit event field proves no forbidden evidence text leaked.
ALPHA_MARKER = "peridot"  # only in the authorized research-a document body
FORBIDDEN_MARKER = "cobalt"  # only in the forbidden research-b document body
GROUP_A = "research-a"
GROUP_B = "research-b"

ALPHA_TEXT = "Research-a onboarding handbook, codename peridot. " + " ".join(
    f"onboarding note {index}." for index in range(60)
)
BRAVO_TEXT = (
    "Codename cobalt production rollback approval control: checkout releases require a signed "
    "production rollback plan and a production approval before any checkout release rollback. "
    + " ".join(f"production rollback approval detail {index}." for index in range(60))
)

DOCUMENTS = {
    "alpha.md": (GROUP_A, ALPHA_TEXT),
    "bravo.md": (GROUP_B, BRAVO_TEXT),
}

ALICE = Principal(id="alice", tenant="merck", groups=frozenset({GROUP_A}))

# The redaction contract: none of these ever appear in an audit event or the persisted snapshot.
SECRETS = (QUERY, ALPHA_MARKER, FORBIDDEN_MARKER, GROUP_A, GROUP_B)


class Directory:
    """A minimal PrincipalDirectory the worker calls to re-resolve the enqueue-time principal."""

    def __init__(self, principal: Principal | None) -> None:
        self.principal = principal

    def refresh(self, principal_id: str) -> Principal | None:
        del principal_id
        return self.principal


class CapturingSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        self.events.append(event)


def _event_strings(event: AuditEvent) -> list[str]:
    strings: list[str] = []
    for value in event.to_json().values():
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, list):
            strings.extend(str(item) for item in value)
    return strings


def _no_secret(events: list[AuditEvent]) -> bool:
    return all(secret not in text for event in events for text in _event_strings(event) for secret in SECRETS)


def _seed(runtime: Refaat) -> tuple[str, str, dict[str, str]]:
    project: RefaatProject = runtime.project(workspace="merck", workspace_name="Merck", corpus="research")
    for source_id, (group, content) in DOCUMENTS.items():
        runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id=source_id,
                title=source_id,
                media_type="text/markdown",
                content=content.encode(),
                metadata={"acl": {"classification": "confidential", "allow_groups": [group]}},
            ),
        )
    doc_ids = {
        str(doc["source_id"]): str(doc["id"])
        for doc in runtime.store.list_documents(project.workspace_id, project.corpus_id)
    }
    return project.workspace_id, project.corpus_id, doc_ids


def _drive_to_terminal(runtime: Refaat, workspace_id: str, job_id: str) -> dict[str, Any]:
    worker = DurableWorker(runtime)
    for _ in range(25):
        job = runtime.store.get_job(workspace_id, job_id)
        if str(job["status"]) in ("succeeded", "failed"):
            return dict(job)
        if not worker.run_once():
            break
    return dict(runtime.store.get_job(workspace_id, job_id))


def main() -> None:
    with TemporaryDirectory(prefix="refaat-freshness-") as directory:
        root = Path(directory)
        expected_scope = durable_entitlement_digest(ALICE, CanonicalAccessPolicy())

        # (A) FRESH: the directory re-resolves alice with the same research-a entitlements captured at
        # enqueue, so the worker admits the durable job and runs it under the refreshed principal.
        fresh_sink = CapturingSink()
        fresh_root = root / "fresh"
        fresh_root.mkdir()
        fresh_runtime = (
            RuntimeBuilder(demo_settings(fresh_root))
            .with_access_policy(CanonicalAccessPolicy())
            .with_principal_directory(Directory(ALICE))
            .with_audit_sink(fresh_sink)
            .build()
        )
        try:
            workspace_id, corpus_id, doc_ids = _seed(fresh_runtime)
            job_id = fresh_runtime.enqueue_rlm(workspace_id, corpus_id, QUERY, principal=ALICE)

            # The identity snapshot is written server-side into the job payload, never from a client
            # body: its scope digest equals the digest the server computes for alice under the policy.
            enqueued = fresh_runtime.store.get_job(workspace_id, job_id)
            snapshot = enqueued["payload"].get("_identity_snapshot") or {}
            snapshot_values = [str(value) for value in snapshot.values()]
            snapshot_captured = (
                snapshot.get("principal_id") == "alice" and snapshot.get("auth_scope_hash") == expected_scope
            )
            snapshot_redacted = bool(snapshot) and all(
                marker not in value
                for value in snapshot_values
                for marker in (ALPHA_MARKER, FORBIDDEN_MARKER, GROUP_A, GROUP_B)
            )

            fresh_job = _drive_to_terminal(fresh_runtime, workspace_id, job_id)
            fresh_admitted = str(fresh_job["status"]) == "succeeded"
            citations = fresh_job["result"]["citations"] if fresh_admitted else []
            cited_docs = {str(citation["document_id"]) for citation in citations}
            fresh_answer = str(fresh_job["result"]["answer"]) if fresh_admitted else ""

            reresolved = [event for event in fresh_sink.events if event.event_type == "identity.reresolved"]
            fresh_reresolved_event = (
                len(reresolved) == 1
                and reresolved[0].capability_name == "fresh"
                and reresolved[0].principal_id == "alice"
            )
            cites_only_authorized = bool(cited_docs) and cited_docs == {doc_ids["alpha.md"]}
            answer_excludes_forbidden = FORBIDDEN_MARKER not in fresh_answer and all(
                FORBIDDEN_MARKER not in str(citation["quote"]) for citation in citations
            )
        finally:
            fresh_runtime.close()

        # (B) SCOPE CHANGED: the same enqueue-time research-a snapshot, but the directory now returns
        # alice moved to research-b. The re-resolution refuses the job and it serves no evidence.
        changed_sink = CapturingSink()
        changed_root = root / "changed"
        changed_root.mkdir()
        changed_runtime = (
            RuntimeBuilder(demo_settings(changed_root))
            .with_access_policy(CanonicalAccessPolicy())
            .with_principal_directory(Directory(Principal(id="alice", tenant="merck", groups=frozenset({GROUP_B}))))
            .with_audit_sink(changed_sink)
            .build()
        )
        try:
            workspace_id, corpus_id, _doc_ids = _seed(changed_runtime)
            job_id = changed_runtime.enqueue_rlm(workspace_id, corpus_id, QUERY, principal=ALICE)
            changed_job = _drive_to_terminal(changed_runtime, workspace_id, job_id)

            scope_changed_refused = str(changed_job["status"]) == "failed"
            refused = [event for event in changed_sink.events if event.event_type == "identity.refused"]
            scope_changed_event = len(refused) == 1 and refused[0].capability_name == "scope_changed"
            no_rlm_authorized = "rlm.authorized" not in [event.event_type for event in changed_sink.events]
            scope_changed_serves_no_evidence = no_rlm_authorized and changed_job["result"] is None
        finally:
            changed_runtime.close()

        # (C) NO POLICY CONTROL: with no policy installed the forbidden research-b document is the
        # dominant candidate for this query and IS cited, so the fresh-path exclusion above is
        # load-bearing rather than an artifact of a corpus that never surfaced the forbidden evidence.
        control_root = root / "control"
        control_root.mkdir()
        control_runtime = RuntimeBuilder(demo_settings(control_root)).build()
        try:
            workspace_id, corpus_id, doc_ids = _seed(control_runtime)
            job_id = control_runtime.enqueue_rlm(workspace_id, corpus_id, QUERY)
            control_job = _drive_to_terminal(control_runtime, workspace_id, job_id)
            control_cited = (
                {str(citation["document_id"]) for citation in control_job["result"]["citations"]}
                if str(control_job["status"]) == "succeeded"
                else set()
            )
            forbidden_dominates_without_policy = doc_ids["bravo.md"] in control_cited
        finally:
            control_runtime.close()

        report: dict[str, Any] = {
            "mode": "deterministic",
            "snapshot_captured_server_side": snapshot_captured,
            "snapshot_is_redacted": snapshot_redacted,
            "fresh_job_admitted": fresh_admitted,
            "fresh_emits_reresolved_event": fresh_reresolved_event,
            "fresh_cites_only_authorized": cites_only_authorized,
            "fresh_answer_excludes_forbidden_marker": answer_excludes_forbidden,
            "forbidden_doc_dominates_without_policy": forbidden_dominates_without_policy,
            "scope_changed_refused": scope_changed_refused,
            "scope_changed_emits_refused_event": scope_changed_event,
            "scope_changed_serves_no_evidence": scope_changed_serves_no_evidence,
            "no_secret_in_any_event": _no_secret(fresh_sink.events + changed_sink.events),
        }
        assert all(value for key, value in report.items() if key != "mode"), report

        emit(report)


if __name__ == "__main__":
    main()
