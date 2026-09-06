from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import asdict
from datetime import datetime, timedelta
from types import TracebackType

from refaat.application.blob_gc import collect_unreferenced_blobs
from refaat.application.runtime import ProjectRef, Refaat, RefaatProject
from refaat.core.session import SessionCompactor
from refaat.core.text import stable_digest
from refaat.domain.audit import AuditEvent, emit_audit
from refaat.domain.authz import Principal, durable_entitlement_digest
from refaat.domain.enums import EvaluationStatus
from refaat.domain.erasure import ErasureStatus
from refaat.domain.freshness import (
    DEFAULT_FRESHNESS_POLICY,
    IdentitySnapshot,
    RiskClass,
    evaluate_reresolution,
    snapshot_from_payload,
)
from refaat.domain.models import JsonObject, RlmLimits
from refaat.errors import AuthorizationError, ConflictError
from refaat.errors import error_message as _error_message
from refaat.evaluation import Dataset, default_bundle
from refaat.infrastructure.audit import AuditPublisher
from refaat.infrastructure.events import OutboxPublisher
from refaat.observability import get_logger

logger = get_logger(__name__)
JobHandler = Callable[[str, JsonObject], JsonObject]


class LeaseHeartbeat(AbstractContextManager["LeaseHeartbeat"]):
    def __init__(self, runtime: Refaat, job_id: str, worker_id: str, lease_seconds: int) -> None:
        self.runtime = runtime
        self.job_id = job_id
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> LeaseHeartbeat:
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        interval = max(1.0, self.lease_seconds / 3)
        while not self._stop.wait(interval):
            if not self._beat():
                return

    def _beat(self) -> bool:
        """One heartbeat tick. Returns False only when the lease is provably lost.

        A transient store error must not kill the daemon thread and silently let
        the lease expire while the main thread runs on: swallow it, log, and keep
        beating. Only a heartbeat that succeeds and reports the lease is no longer
        owned stops the loop.
        """
        try:
            owned = self.runtime.store.heartbeat_job(self.job_id, self.worker_id, self.lease_seconds)
        except Exception:
            logger.warning("job_heartbeat_error", job_id=self.job_id)
            return True
        if not owned:
            logger.warning("job_lease_lost", job_id=self.job_id)
            return False
        return True


# Durable jobs that run retrieval or generation over evidence. They carry no principal, so under
# an active access policy they fail closed at dispatch rather than run unfiltered.
_EVIDENCE_SERVING_KINDS = frozenset(
    {"evaluation.run", "program.evaluate", "rlm.run", "optimization.run", "program.shadow_compare"}
)


class _TerminalAuthzDenial(AuthorizationError):
    """A durable evidence job whose re-resolved identity is not fresh: fail terminally, never retry."""


class DurableWorker:
    def __init__(self, runtime: Refaat) -> None:
        self.runtime = runtime
        self.settings = runtime.settings
        self.compactor = SessionCompactor(
            runtime.store, runtime.settings, classify_evidence=runtime.queries.model_router is not None
        )
        self.outbox = OutboxPublisher(runtime.store, runtime.event_sinks, worker_id=self.settings.worker_id)
        # Durable audit delivery is active only when enabled with a sink installed; otherwise audit
        # events are delivered directly at emission and there is no publisher to run.
        self.audit: AuditPublisher | None = None
        if self.settings.audit_durable and runtime.audit_delivery_sinks:
            self.audit = AuditPublisher(runtime.store, runtime.audit_delivery_sinks, worker_id=self.settings.worker_id)
        self._current_job_id: str | None = None
        self._current_principal: Principal | None = None
        self.handlers: dict[str, JobHandler] = {
            "evaluation.run": self._evaluation,
            "program.evaluate": self._evaluate_program,
            "session.compact": self._compact_session,
            "outbox.publish": self._publish_outbox,
            "audit.publish": self._publish_audit,
            "cache.purge": self._purge_cache,
            "deferred_erasure.sweep": self._sweep_deferred_erasures,
            "shadow_runs.purge": self._purge_shadow_runs,
            "quota.reap": self._reap_quota,
            "blob.gc": self._gc_blobs,
            "rlm.run": self._run_rlm,
            "optimization.run": self._run_optimization,
            "program.shadow_compare": self._shadow_compare,
            "fine_tune.run": self._run_fine_tune,
        }

    def register(self, kind: str, handler: JobHandler) -> None:
        if kind in self.handlers:
            raise ValueError(f"Job handler already registered: {kind}")
        self.handlers[kind] = handler

    def run_once(self) -> bool:
        self.runtime.store.dispatch_due_schedules()
        self._reap_abandoned()
        job = self.runtime.store.claim_job(
            self.settings.worker_id,
            self.settings.job_lease_seconds,
        )
        if job is None:
            self._publish_pending()
            return False
        job_id = str(job["id"])
        self._current_job_id = job_id
        kind = str(job["kind"])
        workspace_id = str(job["workspace_id"])
        registry = self.runtime.kill_switches
        if registry is not None and registry.engaged().freezes(workspace_id):
            # The single durable-worker freeze chokepoint, covering every job kind. Release the job
            # without counting this attempt, so a freeze does not exhaust its retries during an incident.
            logger.warning("job_denied", job_id=job_id, kind=kind)
            # run_after is now, so the job is re-claimable next poll; the worker's poll interval is the
            # natural backoff, so a sustained freeze churns at most one claim-release per poll per job.
            self.runtime.store.release_job(job_id, self.settings.worker_id, run_after=self.runtime.store.clock.now())
            return True
        handler = self.handlers.get(kind)
        if handler is None:
            self.runtime.store.fail_job(
                job_id,
                self.settings.worker_id,
                f"No handler for {kind}",
                0,
                terminal=True,
            )
            return True
        try:
            with LeaseHeartbeat(
                self.runtime,
                job_id,
                self.settings.worker_id,
                self.settings.job_lease_seconds,
            ):
                self._current_principal = self._gate_evidence_job(kind, str(job["workspace_id"]), job["payload"])
                result = handler(str(job["workspace_id"]), job["payload"])
        except _TerminalAuthzDenial as exc:
            logger.warning("job_denied", job_id=job_id, kind=kind)
            self._finalize(self.runtime.store.fail_job, job_id, _error_message(exc), 0, terminal=True)
        except Exception as exc:
            logger.exception("job_failed", job_id=job_id, kind=kind)
            delay = min(900, 2 ** min(int(job["attempts"]), 9))
            message = _error_message(exc)
            # attempts was incremented at claim, so it already counts this run; the
            # job is terminally failed once it reaches its attempt ceiling.
            job_exhausted = int(job["attempts"]) >= int(job["max_attempts"])
            failed = self._finalize(self.runtime.store.fail_job, job_id, message, delay)
            # Terminalize the dependent optimization run to FAILED only when the JOB
            # itself is terminally failed. A mid-retry attempt failure leaves the run
            # RUNNING and resumable, so a later successful attempt is not preceded by a
            # spurious FAILED a client could observe. Skip it when the lease was lost
            # (another worker owns the job now).
            if failed and job_exhausted and kind == "optimization.run":
                run_id = self._payload_run_id(job["payload"])
                if run_id is not None:
                    self.runtime.store.fail_optimization_run(str(job["workspace_id"]), run_id, message)
            if failed and job_exhausted and kind == "fine_tune.run":
                run_id = self._payload_run_id(job["payload"])
                if run_id is not None:
                    self.runtime.store.fail_fine_tune_run(str(job["workspace_id"]), run_id, message)
        else:
            self._finalize(self.runtime.store.complete_job, job_id, result)
            self._maybe_repoll_fine_tune(kind, job, result)
        finally:
            self._current_job_id = None
            self._current_principal = None
        self._publish_pending()
        return True

    def _publish_pending(self) -> None:
        # Guard the background publish so a transient store error, or a durable-audit table not yet
        # migrated, is logged and retried next tick rather than terminating run_forever.
        try:
            self.outbox.publish(limit=50)
            if self.audit is not None:
                self.audit.publish(limit=50)
        except Exception:
            logger.warning("worker_publish_failed", exc_info=True)

    def _reap_abandoned(self) -> None:
        for job in self.runtime.store.reap_abandoned_jobs():
            if str(job["kind"]) == "optimization.run":
                run_id = self._payload_run_id(job["payload"])
                if run_id is not None:
                    self.runtime.store.fail_optimization_run(
                        str(job["workspace_id"]), run_id, "Optimization job abandoned after exhausting all attempts"
                    )
            if str(job["kind"]) == "fine_tune.run":
                run_id = self._payload_run_id(job["payload"])
                if run_id is not None:
                    self.runtime.store.fail_fine_tune_run(
                        str(job["workspace_id"]), run_id, "Fine-tune job abandoned after exhausting all attempts"
                    )
        # An optimization run created but never enqueued (a crash between the two)
        # is an orphan PENDING run with no job. Re-enqueue it once it is older than
        # the grace window, so an in-flight enqueue is never reaped mid-flight.
        self.runtime.store.reenqueue_orphan_optimization_runs(self._orphan_run_grace_seconds())
        # A RUNNING fine-tune run whose next delayed poll was lost (a crash between
        # completing the poll job and enqueuing the successor) has no live job; re-drive
        # it so a long provider job still reaches a terminal state without an operator.
        self.runtime.store.reenqueue_orphan_fine_tune_runs(self._orphan_run_grace_seconds())

    def _orphan_run_grace_seconds(self) -> int:
        return max(300, self.settings.job_lease_seconds * 5)

    @staticmethod
    def _payload_run_id(payload: object) -> str | None:
        if isinstance(payload, dict) and payload.get("run_id"):
            return str(payload["run_id"])
        return None

    def _finalize(self, action: Callable[..., None], job_id: str, *args: object, **kwargs: object) -> bool:
        """Run a lease-guarded finalizer, swallowing a lost-lease conflict.

        Returns True when the write landed, False when the lease was lost, so the
        caller can skip any dependent write it must not perform without the lease.
        """
        try:
            action(job_id, self.settings.worker_id, *args, **kwargs)
        except ConflictError:
            logger.warning("job_lease_lost", job_id=job_id)
            return False
        return True

    def _gate_evidence_job(self, kind: str, workspace_id: str, payload: JsonObject) -> Principal | None:
        """Admit or refuse an evidence-serving durable job under an active access policy.

        With no policy, or for a non-evidence kind, this is a no-op returning None so the job runs
        exactly as before. Under a policy, an evidence kind with no re-resolution seam (no directory,
        or a kind other than rlm.run) keeps the blanket fail-closed refusal, retryable and unaudited
        so the unconfigured path is byte-identical. For rlm.run with a directory, the enqueue-time
        identity snapshot is re-resolved and must come back fresh: only then does the job run, under
        the refreshed principal. A required-approval verdict is refused too, since a durable evidence
        job has no approval channel here. Every staleness or mismatch denial is terminal; only a
        directory outage is retryable.
        """
        policy = self.runtime.access_policy
        if kind not in _EVIDENCE_SERVING_KINDS or policy is None:
            return None
        directory = self.runtime.principal_directory
        if directory is None or kind != "rlm.run":
            raise AuthorizationError(f"{kind} is unavailable while an access policy is active")
        now = self.runtime.store.clock.now()
        raw_snapshot = payload.get("_identity_snapshot")
        snapshot = snapshot_from_payload(raw_snapshot) if isinstance(raw_snapshot, Mapping) else None
        if snapshot is None:
            self._emit_identity_refused(workspace_id, None, "no_snapshot", now)
            raise _TerminalAuthzDenial("rlm.run denied: no identity snapshot")
        try:
            refreshed = directory.refresh(snapshot.principal_id)
        except Exception as exc:
            self._emit_identity_refused(workspace_id, snapshot, "directory_unavailable", now)
            raise AuthorizationError("identity directory unavailable") from exc
        refreshed_scope = durable_entitlement_digest(refreshed, policy) if refreshed is not None else None
        decision = evaluate_reresolution(
            snapshot, refreshed, refreshed_scope, RiskClass.READ, DEFAULT_FRESHNESS_POLICY, now
        )
        if not (decision.allowed and decision.reason == "fresh" and not decision.require_approval):
            self._emit_identity_refused(workspace_id, snapshot, decision.reason, now)
            raise _TerminalAuthzDenial(f"rlm.run denied: {decision.reason}")
        if refreshed is None:
            # A fresh verdict already implies a re-resolved principal; guard so None never reaches a run.
            self._emit_identity_refused(workspace_id, snapshot, "identity_revoked", now)
            raise _TerminalAuthzDenial("rlm.run denied: identity_revoked")
        self._emit_identity_reresolved(workspace_id, refreshed, now)
        return refreshed

    def _emit_identity_refused(
        self, workspace_id: str, snapshot: IdentitySnapshot | None, reason: str, now: datetime
    ) -> None:
        """Record a redacted identity.refused event for a denied durable evidence job.

        Best-effort: a sink outage must never downgrade a terminal denial to retryable, so this never
        raises. capability_name carries only the bounded freshness reason, never client text; the
        event carries only ids, the policy, and a timestamp. No-op with no sink or no active policy.
        """
        sink = self.runtime.audit_sink
        policy = self.runtime.access_policy
        if sink is None or policy is None:
            return
        principal_id = snapshot.principal_id if snapshot is not None else "system"
        tenant = snapshot.tenant if snapshot is not None else None
        event = AuditEvent(
            event_type="identity.refused",
            occurred_at=now,
            workspace_id=workspace_id,
            principal_id=principal_id,
            principal_kind="system",
            tenant=tenant,
            policy_id=str(getattr(policy, "name", type(policy).__name__)),
            policy_version=str(getattr(policy, "version", "")),
            capability_name=reason,
            dedup_id=stable_digest(("identity.refused", workspace_id, self._current_job_id or "", reason)),
        )
        emit_audit(sink, event, fail_closed=False, logger=logger)

    def _emit_identity_reresolved(self, workspace_id: str, principal: Principal, now: datetime) -> None:
        """Record a redacted identity.reresolved event for an admitted durable evidence job.

        Fail-closed under audit_fail_closed: a sink outage raises AuditUnavailableError so the run is
        never served unaudited, and the raise propagates to the retryable branch. capability_name is
        the fixed "fresh" reason; the event carries only ids, the policy, and a timestamp.
        """
        sink = self.runtime.audit_sink
        policy = self.runtime.access_policy
        if sink is None or policy is None:
            return
        event = AuditEvent(
            event_type="identity.reresolved",
            occurred_at=now,
            workspace_id=workspace_id,
            principal_id=principal.id,
            principal_kind=principal.kind.value,
            tenant=principal.tenant,
            policy_id=str(getattr(policy, "name", type(policy).__name__)),
            policy_version=str(getattr(policy, "version", "")),
            capability_name="fresh",
            dedup_id=stable_digest(("identity.reresolved", workspace_id, self._current_job_id or "", principal.id)),
        )
        emit_audit(sink, event, fail_closed=self.settings.audit_fail_closed, logger=logger)

    def run_forever(self, stop_event: threading.Event | None = None) -> None:
        stop = stop_event or threading.Event()
        while not stop.is_set():
            worked = self.run_once()
            if not worked:
                stop.wait(self.settings.worker_poll_seconds)

    def _evaluation(self, workspace_id: str, payload: JsonObject) -> JsonObject:
        run_id = self.runtime.evaluations.run(
            workspace_id,
            str(payload["suite_id"]),
            str(payload["corpus_id"]),
        )
        return {"run_id": run_id}

    def _evaluate_program(self, workspace_id: str, payload: JsonObject) -> JsonObject:
        cases = payload.get("cases", [])
        if not isinstance(cases, list):
            raise ValueError("Evaluation cases must be a list")
        project = RefaatProject(self.runtime, ProjectRef(workspace_id, str(payload["corpus_id"])))
        dataset = Dataset.from_records("job", cases)
        report = project.programs.get(str(payload["program"])).evaluate(dataset, default_bundle())
        return report.to_json()

    def _run_optimization(self, workspace_id: str, payload: JsonObject) -> JsonObject:
        from refaat.optimization.durable import resume_optimization_run

        run_id = str(payload["run_id"])
        run = self.runtime.store.get_optimization_run(workspace_id, run_id)
        project = RefaatProject(self.runtime, ProjectRef(workspace_id, str(run["corpus_id"])))
        # A failure propagates so the job retry engine reschedules; the run is
        # terminalized to FAILED only once the job itself is exhausted (run_once),
        # so a mid-retry failure leaves the run RUNNING and resumable.
        result = resume_optimization_run(project, run_id)
        best = result.get("best") or {}
        return {"run_id": run_id, "best_artifact_id": best.get("artifact_id")}

    def _run_fine_tune(self, workspace_id: str, payload: JsonObject) -> JsonObject:
        # A provider failure propagates so the job retry engine reschedules; the run is
        # terminalized to FAILED only once the job itself is exhausted (run_once), so a
        # mid-retry failure leaves the run resumable. An unapproved run is refused.
        return self.runtime.fine_tuning.run(workspace_id, str(payload["run_id"]))

    def _maybe_repoll_fine_tune(self, kind: str, job: JsonObject, result: object) -> None:
        # A RUNNING result is not terminal: re-enqueue a delayed poll of the same run so a long provider
        # job self-progresses to a terminal state instead of stalling until an operator re-runs it. The
        # run-scoped dedup key keeps exactly one pending poll per run.
        if kind != "fine_tune.run" or not isinstance(result, dict):
            return
        if result.get("status") != EvaluationStatus.RUNNING:
            return
        run_id = self._payload_run_id(job["payload"])
        if run_id is None:
            return
        run_after = self.runtime.store.clock.now() + timedelta(seconds=self.settings.fine_tune_poll_seconds)
        self.runtime.store.enqueue_job(
            str(job["workspace_id"]), "fine_tune.run", {"run_id": run_id}, run_after=run_after, dedup_key=run_id
        )

    def _shadow_compare(self, workspace_id: str, payload: JsonObject) -> JsonObject:
        questions = payload.get("questions", [])
        if not isinstance(questions, list):
            raise ValueError("Shadow compare questions must be a list")
        project = RefaatProject(self.runtime, ProjectRef(workspace_id, str(payload["corpus_id"])))
        program = str(payload["program"])
        environment = str(payload["environment"])
        if payload.get("deferred"):
            deadline_raw = payload.get("deadline")
            if isinstance(deadline_raw, str) and self.runtime.store.clock.now() > datetime.fromisoformat(deadline_raw):
                # The served response already returned; a comparison this stale is not worth the compute.
                return {"deferred": True, "skipped": "stale", "shadow_run_ids": []}
            served = str(payload["served"])
            bucket = int(payload["bucket"])
            compare = bool(payload.get("compare", True))
            shadow_run_ids = [
                self.runtime.deployment.run_deferred_compare(
                    project,
                    program,
                    environment,
                    query_text=str(question),
                    served=served,
                    bucket=bucket,
                    compare=compare,
                    idempotency_key=self._shadow_idempotency_key(
                        workspace_id, program, environment, index, str(question)
                    ),
                )
                for index, question in enumerate(questions)
            ]
            # Telemetry only: never return a program result, whose answer text would land in jobs.result.
            return {"deferred": True, "shadow_run_ids": [run_id for run_id in shadow_run_ids if run_id]}
        runs = [
            project.deployment.run(
                program,
                environment,
                inputs={"question": str(question)},
                allow_defer=False,
                idempotency_key=self._shadow_idempotency_key(workspace_id, program, environment, index, str(question)),
            ).to_json()
            for index, question in enumerate(questions)
        ]
        return {"runs": runs}

    def _shadow_idempotency_key(
        self, workspace_id: str, program: str, environment: str, index: int, question: str
    ) -> str:
        """Deterministic idempotency key for one shadow-compare record.

        With a real job id the (job, index) pair is unique, so an at-least-once
        redelivery of the batch upserts rather than duplicating rows, events, and
        spend. Without a job id the fallback folds a digest of the question content,
        so two direct calls with different questions at the same index never collide
        into one deduped row.
        """
        if self._current_job_id is not None:
            return f"{self._current_job_id}:{index}"
        return f"{workspace_id}:{program}:{environment}:{index}:{stable_digest([question])}"

    def _compact_session(self, workspace_id: str, payload: JsonObject) -> JsonObject:
        # This background job carries no principal, so it passes no auth_scope. The compactor
        # refuses to recompact scoped content scope-blind, so it can never re-tag one principal's
        # summary onto another's turns; a principal-driven query compacts a scoped session instead.
        compacted = self.compactor.compact_if_needed(workspace_id, str(payload["session_id"]))
        return {"compacted": compacted}

    def _run_rlm(self, workspace_id: str, payload: JsonObject) -> JsonObject:
        raw_limits = payload.get("limits", {})
        if not isinstance(raw_limits, dict):
            raise ValueError("RLM limits must be an object")
        limits = RlmLimits(
            max_depth=int(raw_limits.get("max_depth", 2)),
            max_nodes=int(raw_limits.get("max_nodes", 16)),
            branch_factor=int(raw_limits.get("branch_factor", 4)),
            leaf_token_budget=int(raw_limits.get("leaf_token_budget", 1_500)),
        )
        result = self.runtime.run_rlm(
            workspace_id,
            str(payload["corpus_id"]),
            str(payload["query"]),
            user_id=str(payload["user_id"]) if payload.get("user_id") else None,
            session_id=str(payload["session_id"]) if payload.get("session_id") else None,
            limits=limits,
            principal=self._current_principal,
        )
        return asdict(result)

    def _purge_cache(self, workspace_id: str, payload: JsonObject) -> JsonObject:
        return {
            "purged": self.runtime.store.purge_expired_cache(),
            "model_calls_purged": self.runtime.store.purge_expired_model_call_cache(),
            "outbox_purged": self.runtime.store.purge_delivered_outbox(self.settings.outbox_retention_days),
            "audit_outbox_purged": self.runtime.store.purge_delivered_audit_outbox(self.settings.outbox_retention_days),
        }

    def _purge_shadow_runs(self, workspace_id: str, payload: JsonObject) -> JsonObject:
        del workspace_id, payload
        return {
            "shadow_runs_purged": self.runtime.store.purge_expired_shadow_runs(self.settings.shadow_run_retention_days)
        }

    def _sweep_deferred_erasures(self, workspace_id: str, payload: JsonObject) -> JsonObject:
        # The at-least-once backstop for the defer conflict policy: re-drive every pending deferred
        # erasure whose covering holds have all lifted, so a lift-time execution that failed or was
        # interrupted still completes. process_deferred re-checks each hold, so a still-held document
        # stays pending. A no-op when the defer policy is not in use (no pending rows exist).
        del workspace_id, payload
        executed = 0
        for scope_workspace, scope_corpus in self.runtime.store.list_pending_deferred_erasure_scopes():
            for receipt in self.runtime.erasure.process_deferred(scope_workspace, scope_corpus):
                if receipt.status is ErasureStatus.ERASED:
                    executed += 1
        return {"deferred_erasures_executed": executed}

    def _reap_quota(self, workspace_id: str, payload: JsonObject) -> JsonObject:
        # Release leaked reservations past the lease TTL, charging their ceiling to spent, so a
        # crash between reserve and reconcile never permanently blocks a tenant's window.
        del workspace_id, payload
        if not self.runtime.store.quota_management_enabled:
            return {"quota_reclaimed": 0}
        return {"quota_reclaimed": self.runtime.store.quota_reap()}

    def _gc_blobs(self, workspace_id: str, payload: JsonObject) -> JsonObject:
        del workspace_id, payload
        reclaimed = collect_unreferenced_blobs(
            blob_store=self.runtime.ingestion.blob_store,
            store=self.runtime.store,
            now=self.runtime.store.clock.now(),
            grace_seconds=self.settings.blob_gc_grace_seconds,
        )
        return {"blobs_reclaimed": reclaimed}

    def _publish_outbox(self, workspace_id: str, payload: JsonObject) -> JsonObject:
        del workspace_id, payload
        return {"delivered": self.outbox.publish()}

    def _publish_audit(self, workspace_id: str, payload: JsonObject) -> JsonObject:
        del workspace_id, payload
        return {"delivered": self.audit.publish() if self.audit is not None else 0}
