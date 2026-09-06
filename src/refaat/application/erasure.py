"""Complete deletion (right to erasure) for a single document.

:class:`ErasureService` physically removes or re-derives every auto-derived evidence surface a
document produced, so no residue of its content survives in a queryable or model-reachable
auto-derived surface: chunks, vectors, graph entities and relations, hypothetical questions,
the RAPTOR summary tree, the answer cache, the model-call caches, session summaries and citing
turns, learned preferences, the stored blob, and the outbox. The operation is additive and does
not touch the behavior-preserving ``IngestionService.delete`` or ``replace_document``.

Fail-closed: the raw blob is deleted and its removal verified, and the exception is never
swallowed the way an ordinary delete swallows a blob-cleanup failure. Idempotent: a re-run over
an already-erased document converges on a clean ``ERASED`` receipt because every delete is a
no-op. An active legal hold (store.is_document_held) blocks erasure: by default the request is
refused and nothing is erased, and with ``legal_hold_on_conflict`` set to ``defer`` the request is
recorded durably and executed by :meth:`process_deferred` when the last hold covering the document
is lifted, re-checking the hold under the destroy lock so a deferred erasure never runs while held.

Physical deletion is the safe default because the store is plaintext at rest. Cryptographic
erasure through per-document keys is a separate at-rest-encryption concern and out of scope. The
concurrency window and the operator-curated surfaces that are flagged for review rather than
auto-erased are documented in docs/erasure.md.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from refaat.application.derivation import DerivationService
from refaat.application.raptor import RaptorTreeBuilder
from refaat.clock import Clock
from refaat.core.chunking import contextualize
from refaat.core.text import stable_digest
from refaat.domain.audit import AuditEvent, AuditSink, emit_audit
from refaat.domain.authz import AccessPolicy
from refaat.domain.erasure import ErasureReceipt, ErasureStatus
from refaat.domain.kill_switch import KillSwitchRegistry, KillSwitchScope, ensure_scope_active, ensure_workspace_active
from refaat.domain.legal_hold import validate_document_ref
from refaat.domain.models import Chunk
from refaat.errors import NotFoundError, ValidationError
from refaat.ids import new_id
from refaat.infrastructure.database.store import SQLStore
from refaat.infrastructure.model_cache import cache_identity_of, embedding_cache_key
from refaat.observability import get_logger
from refaat.ports import BlobStore, CacheStore, Embedder, ModelCallCache, SearchIndex

logger = get_logger(__name__)


def _chunk_embedding_inputs(chunks: Sequence[Chunk]) -> list[str]:
    """The exact texts the ingester embedded for these chunks, so their embedding-cache rows purge.

    Contextual retrieval embeds ``contextualize(title, heading, text)``, not the raw chunk text, so
    keying the purge on the raw text alone would strand the vector in the model-call cache. Each
    chunk's recorded ``context_strategy`` selects the form the ingester used; the raw text is always
    included too, so a config drift between ingest and erasure cannot leave a row under either form.
    """
    inputs: list[str] = []
    for chunk in chunks:
        inputs.append(chunk.text)
        if chunk.metadata.get("context_strategy") == "contextual":
            inputs.append(
                contextualize(
                    str(chunk.metadata.get("title", "")),
                    str(chunk.metadata.get("heading", "")),
                    chunk.text,
                )
            )
    return inputs


_ZERO_TALLY: dict[str, int | bool] = {
    "corpus_version": 0,
    "chunks": 0,
    "hypotheticals": 0,
    "entities": 0,
    "relations": 0,
    "raptor_rebuilt": 0,
    "query_cache_rows": 0,
    "generation_cache_rows": 0,
    "embedding_cache_rows": 0,
    "session_summaries_cleared": 0,
    "session_turns_deleted": 0,
    "preferences": 0,
    "outbox_rows": 0,
    "blob_deleted": False,
}


class ErasureService:
    def __init__(
        self,
        *,
        store: SQLStore,
        search_index: SearchIndex,
        blob_store: BlobStore,
        embedder: Embedder,
        clock: Clock,
        cache_store: CacheStore | None = None,
        model_call_cache: ModelCallCache | None = None,
        raptor_builder: RaptorTreeBuilder | None = None,
        access_policy: AccessPolicy | None = None,
        audit_sink: AuditSink | None = None,
        audit_fail_closed: bool = False,
        kill_switches: KillSwitchRegistry | None = None,
        on_conflict: str = "refuse",
        derivation_service: DerivationService | None = None,
    ) -> None:
        self.store = store
        self.kill_switches = kill_switches
        self.on_conflict = on_conflict
        self.derivation_service = derivation_service
        self.search_index = search_index
        self.blob_store = blob_store
        # The raw (unwrapped) embedder, whose cache_identity is exactly the identity the caching
        # embedder wrote model-call rows under, so erasure recomputes matching key digests.
        self.embedder = embedder
        self.clock = clock
        # Default to the store; when a backend differs, erasure sweeps both so a switch strands nothing.
        self.cache_store = cache_store if cache_store is not None else store
        self.model_call_cache = model_call_cache if model_call_cache is not None else store
        self.raptor_builder = raptor_builder
        self.access_policy = access_policy
        self.audit_sink = audit_sink
        self.audit_fail_closed = audit_fail_closed

    def erase(self, workspace_id: str, corpus_id: str, document_id: str) -> ErasureReceipt:
        ensure_workspace_active(self.kill_switches, workspace_id)
        ensure_scope_active(self.kill_switches, workspace_id, KillSwitchScope.WRITES)
        # Bound the requested document id to an id shape before any hold check or audit emission: a
        # corpus-wide hold makes any supplied id read as held, so an unvalidated free-text id would
        # otherwise reach the HELD receipt and the hold.enforced audit event's cited ids.
        validated = validate_document_ref(document_id)
        if validated is None:
            raise ValidationError("document id is required")
        document_id = validated
        request_id = new_id("erase")
        # The destroy lock is held across the whole multi-step erase and the hold recheck is
        # unconditional, so a place committed by another connection is seen and nothing is destroyed.
        with self.store.document_destroy_lock(workspace_id, corpus_id):
            if self.store.is_document_held(workspace_id, corpus_id, document_id):
                if self.on_conflict == "defer":
                    return self._defer(request_id, workspace_id, corpus_id, document_id)
                receipt = self._receipt(
                    request_id, workspace_id, corpus_id, document_id, ErasureStatus.HELD, _ZERO_TALLY
                )
                self.store.record_erasure(receipt)
                self._emit_hold_enforced(workspace_id, corpus_id, document_id)
                return receipt
            tally = dict(_ZERO_TALLY)
            try:
                self._run(workspace_id, corpus_id, document_id, tally)
                receipt = self._receipt(request_id, workspace_id, corpus_id, document_id, ErasureStatus.ERASED, tally)
            except Exception:
                # Fail-closed: record the partial outcome as FAILED for the audit trail, then
                # re-raise so the caller learns the erasure did not fully complete.
                failed = self._receipt(request_id, workspace_id, corpus_id, document_id, ErasureStatus.FAILED, tally)
                self.store.record_erasure(failed)
                raise
            self.store.record_erasure(receipt)
        self._emit_erased_audit(receipt)
        # An erase changes the committed corpus, so refresh the derivation manifest (best-effort,
        # post-commit). A no-op when no manifest service is installed; covers the deferred sweep too,
        # since process_deferred erases through this method.
        if self.derivation_service is not None:
            self.derivation_service.refresh_quietly(workspace_id, corpus_id)
        return receipt

    def _defer(self, request_id: str, workspace_id: str, corpus_id: str, document_id: str) -> ErasureReceipt:
        self.store.record_deferred_erasure(
            workspace_id, corpus_id, document_id, requested_by="system", requested_at=self.clock.now()
        )
        receipt = self._receipt(request_id, workspace_id, corpus_id, document_id, ErasureStatus.DEFERRED, _ZERO_TALLY)
        self.store.record_erasure(receipt)
        self._emit_deferred_audit(workspace_id, corpus_id, document_id)
        return receipt

    def process_deferred(self, workspace_id: str, corpus_id: str) -> list[ErasureReceipt]:
        """Execute every pending deferred erasure whose document is no longer held.

        Reuses erase(), which re-checks the hold under the destroy lock, so a document still covered by
        another active hold is never erased here and its request stays pending. A per-document failure
        (a still-active freeze, a transient store error) leaves that request pending without aborting
        the rest, so a later lift or an explicit re-run retries it.
        """
        receipts: list[ErasureReceipt] = []
        for row in self.store.list_pending_deferred_erasures(workspace_id, corpus_id):
            deferred_id = str(row["id"])
            document_id = str(row["document_id"])
            try:
                receipt = self.erase(workspace_id, corpus_id, document_id)
                if receipt.status == ErasureStatus.ERASED:
                    self.store.mark_deferred_erasure_executed(
                        deferred_id,
                        workspace_id,
                        executed_request_id=receipt.request_id,
                        executed_at=receipt.occurred_at,
                    )
            except Exception:
                logger.warning("deferred_erasure_execution_failed", deferred_id=deferred_id)
                continue
            receipts.append(receipt)
        return receipts

    @property
    def deferred_processor(self) -> Callable[[str, str], object] | None:
        """The lift hook: process_deferred under the defer policy, else None so lift stays unchanged."""
        return self.process_deferred if self.on_conflict == "defer" else None

    def _run(self, workspace_id: str, corpus_id: str, document_id: str, tally: dict[str, int | bool]) -> None:
        document = self._load_document(workspace_id, corpus_id, document_id)
        chunk_embed_inputs: list[str] = []
        chunk_ids: list[str] = []
        hypothetical_questions: list[str] = []
        blob_uri: str | None = None
        if document is not None:
            old_chunks = self.store.list_document_chunks(workspace_id, corpus_id, document_id)
            chunk_embed_inputs = _chunk_embedding_inputs(old_chunks)
            chunk_ids = [chunk.id for chunk in old_chunks]
            hypothetical_questions = self.store.list_document_hypothetical_questions(
                workspace_id, corpus_id, document_id
            )
            stats = self.store.document_stats(workspace_id, corpus_id, document_id)
            tally["chunks"] = len(old_chunks)
            tally["hypotheticals"] = len(hypothetical_questions)
            tally["entities"] = int(stats["entities"])
            tally["relations"] = int(stats["relations"])
            blob_uri = str(document["blob_uri"])
        # Captured before the RAPTOR rebuild replaces them, so their embedding-cache rows can be
        # purged too. Corpus-scoped, since a summary may have folded any document's chunks.
        old_summaries = self.store.list_raptor_summaries(workspace_id, corpus_id)

        self.search_index.delete_document(document_id)

        tally["embedding_cache_rows"] = self._purge_embedding_cache(
            [*chunk_embed_inputs, *hypothetical_questions, *old_summaries]
        )
        tally["generation_cache_rows"] = self._erase_generation_cache(workspace_id)

        # Delete the surfaces keyed on this document or its chunk ids BEFORE the row delete. The
        # chunk ids are unrecoverable once the row and its chunks are gone, so a failure in the
        # later re-derivation would otherwise strand a legacy citing turn on a resumed run.
        tally["query_cache_rows"] = self._erase_query_cache(workspace_id, corpus_id)
        tally["session_summaries_cleared"] = self.store.clear_session_summaries_for_corpus(workspace_id, corpus_id)
        tally["session_turns_deleted"] = self.store.delete_session_turns_for_document(
            workspace_id, corpus_id, document_id, chunk_ids
        )
        tally["preferences"] = self.store.delete_preferences_for_document(workspace_id, corpus_id, document_id)
        tally["outbox_rows"] = self.store.delete_outbox_for_document(document_id)

        # The raw blob is the highest-value residue, so erase and verify it BEFORE deleting the
        # document row. A blob-store failure then aborts with the row intact, so a retry re-resolves
        # blob_uri from the still-present document and completes, rather than stranding the bytes with
        # the row gone (which would make the retry skip the blob and falsely report a clean erasure).
        tally["blob_deleted"] = self._erase_blob(blob_uri)

        tally["corpus_version"] = self._delete_document_row(workspace_id, corpus_id, document_id)

        if self.raptor_builder is not None:
            tally["raptor_rebuilt"] = self.raptor_builder.rebuild(workspace_id, corpus_id)
        else:
            # RAPTOR disabled since this corpus built its tree: no builder to re-derive, so purge
            # the corpus-scoped summaries wholesale; a later re-enable rebuilds cleanly.
            self.store.replace_raptor_nodes(workspace_id, corpus_id, [])

    def _load_document(self, workspace_id: str, corpus_id: str, document_id: str) -> Any | None:
        try:
            return self.store.get_document(workspace_id, corpus_id, document_id)
        except NotFoundError:
            return None

    def _delete_document_row(self, workspace_id: str, corpus_id: str, document_id: str) -> int:
        try:
            # Route through the guarded delete so a raw, unguarded row delete is not exposed on the
            # store. erase already holds document_destroy_lock and rechecked the hold, so the guarded
            # path re-enters the reentrant lock (a no-op) and its hold recheck never refuses here (an
            # erase never reaches this point for a held document).
            return self.store.delete_document(workspace_id, corpus_id, document_id)
        except NotFoundError:
            # Already absent (a resumed or repeated erasure): treat as success and report the
            # corpus's current version so a re-run still converges on a clean receipt.
            return self.store.get_corpus(workspace_id, corpus_id).version

    def _emit_hold_enforced(self, workspace_id: str, corpus_id: str, document_id: str) -> None:
        sink = self.audit_sink
        if sink is None:
            return
        policy = self.access_policy
        occurred = self.store.clock.now()
        event = AuditEvent(
            event_type="hold.enforced",
            occurred_at=occurred,
            workspace_id=workspace_id,
            corpus_id=corpus_id,
            principal_id="system",
            principal_kind="system",
            tenant=None,
            policy_id=str(getattr(policy, "name", type(policy).__name__)) if policy is not None else "",
            policy_version=str(getattr(policy, "version", "")) if policy is not None else "",
            cited_document_ids=(document_id,),
            capability_name="document.erase",
            dedup_id=stable_digest(
                ("hold.enforced", workspace_id, corpus_id, document_id, "document.erase", occurred.isoformat())
            ),
        )
        # Best-effort, never fail-closed: the erase is already refused and the HELD ledger row is the
        # authoritative trace, so a sink outage must not crash the refusal path.
        emit_audit(sink, event, fail_closed=False, logger=logger, log_key="legal_hold_audit_sink_failed")

    def _emit_deferred_audit(self, workspace_id: str, corpus_id: str, document_id: str) -> None:
        sink = self.audit_sink
        if sink is None:
            return
        policy = self.access_policy
        occurred = self.store.clock.now()
        event = AuditEvent(
            event_type="erasure.deferred",
            occurred_at=occurred,
            workspace_id=workspace_id,
            corpus_id=corpus_id,
            principal_id="system",
            principal_kind="system",
            tenant=None,
            policy_id=str(getattr(policy, "name", type(policy).__name__)) if policy is not None else "",
            policy_version=str(getattr(policy, "version", "")) if policy is not None else "",
            cited_document_ids=(document_id,),
            capability_name="document.erase",
            dedup_id=stable_digest(
                ("erasure.deferred", workspace_id, corpus_id, document_id, "document.erase", occurred.isoformat())
            ),
        )
        # Best-effort: the deferred_erasures row and the DEFERRED ledger receipt are the authoritative
        # trace, so a sink outage must not crash the deferral path.
        emit_audit(sink, event, fail_closed=False, logger=logger, log_key="legal_hold_audit_sink_failed")

    def _purge_embedding_cache(self, texts: list[str]) -> int:
        identity = self._embedding_identity()
        if identity is None or not texts:
            return 0
        keys = [embedding_cache_key(identity, text) for text in texts]
        deleted = self.store.delete_embedding_model_cache(keys)
        if self.model_call_cache is not self.store:
            deleted += self.model_call_cache.delete_embedding_model_cache(keys)
        return deleted

    def _erase_generation_cache(self, workspace_id: str) -> int:
        deleted = self.store.delete_generation_model_cache(workspace_id)
        if self.model_call_cache is not self.store:
            deleted += self.model_call_cache.delete_generation_model_cache(workspace_id)
        return deleted

    def _erase_query_cache(self, workspace_id: str, corpus_id: str) -> int:
        deleted = self.store.delete_query_cache_for_corpus(workspace_id, corpus_id)
        if self.cache_store is not self.store:
            deleted += self.cache_store.delete_query_cache_for_corpus(workspace_id, corpus_id)
        return deleted

    def _embedding_identity(self) -> str | None:
        identity = cache_identity_of(self.embedder)
        if identity is not None:
            return identity
        # A wrapped caching embedder does not expose cache_identity directly; fall back to the
        # identity it wrote rows under so erasure still matches those rows.
        fallback = getattr(self.embedder, "embedding_cache_identity", None)
        return fallback if isinstance(fallback, str) else None

    def _erase_blob(self, blob_uri: str | None) -> bool:
        if blob_uri is None:
            return False
        self.blob_store.delete(blob_uri)
        if self._blob_present(blob_uri):
            raise ValidationError("Blob still present after erasure")
        return True

    def _blob_present(self, uri: str) -> bool:
        try:
            handle = self.blob_store.open(uri)
        except NotFoundError:
            return False
        handle.close()
        return True

    def _emit_erased_audit(self, receipt: ErasureReceipt) -> None:
        sink = self.audit_sink
        if sink is None:
            return
        policy = self.access_policy
        event = AuditEvent(
            event_type="document.erased",
            occurred_at=receipt.occurred_at,
            workspace_id=receipt.workspace_id,
            corpus_id=receipt.corpus_id,
            corpus_version=receipt.corpus_version,
            principal_id="system",
            principal_kind="system",
            tenant=None,
            policy_id=str(getattr(policy, "name", type(policy).__name__)) if policy is not None else "",
            policy_version=str(getattr(policy, "version", "")) if policy is not None else "",
            # The erased document id is an id (redaction-safe), recorded so the audit names its subject.
            cited_document_ids=(receipt.document_id,),
            dedup_id=stable_digest(
                ("document.erased", receipt.workspace_id, receipt.corpus_id, receipt.document_id, receipt.request_id)
            ),
        )
        # The redacted erasure_ledger row is the authoritative fail-closed record; the external
        # event is best-effort because the physical erasure is irreversible and already recorded,
        # so a sink outage must not raise back a completed, un-undoable erasure.
        emit_audit(sink, event, fail_closed=False, logger=logger, log_key="erasure_audit_sink_failed")

    def _receipt(
        self,
        request_id: str,
        workspace_id: str,
        corpus_id: str,
        document_id: str,
        status: ErasureStatus,
        tally: dict[str, int | bool],
    ) -> ErasureReceipt:
        return ErasureReceipt(
            request_id=request_id,
            workspace_id=workspace_id,
            corpus_id=corpus_id,
            document_id=document_id,
            status=status,
            occurred_at=self.clock.now(),
            **tally,  # type: ignore[arg-type]
        )


__all__ = ["ErasureService"]
