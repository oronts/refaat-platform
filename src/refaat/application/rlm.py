from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from datetime import datetime
from typing import cast

from refaat.application.budget import CostMeter, budget_scope, prices_from, quota_run
from refaat.application.sandbox import run_code_phase
from refaat.application.services import (
    OUTPUT_REFUSAL_MESSAGE,
    OUTPUT_REFUSAL_REASON,
    ROUTING_REFUSAL_MESSAGE,
    ROUTING_REFUSAL_REASON,
    in_shadow_execution,
    in_unserved_run,
)
from refaat.clock import Clock
from refaat.config import Settings
from refaat.core.citations import (
    build_citations,
    filter_cited_chunks,
    strip_uncited_markers,
    validate_cited_chunks,
)
from refaat.core.harness import HarnessService, system_prompt
from refaat.core.math import cosine_similarity
from refaat.core.retrieval import GRAPH_FACT_LIMIT, AdaptiveRetriever, select_graph
from refaat.core.session import SessionCompactor, stored_summary_cited_documents, tag_turn_metadata
from refaat.core.text import estimate_tokens, normalize_text, stable_digest, tokens, truncate_to_tokens
from refaat.domain.audit import AuditEvent, AuditSink, emit_audit
from refaat.domain.authz import (
    AccessPolicy,
    Principal,
    auth_scope_hash,
    authorized_document_ids,
    chunk_allowed,
    durable_entitlement_digest,
    max_classification,
    preference_scope,
    require_principal_present,
    residency_zones,
    session_summary_visible,
    summary_documents_authorized,
    summary_evidence_from_documents,
)
from refaat.domain.context_transform import ContextTransform, redact_for_egress
from refaat.domain.degradation import (
    DegradationDecision,
    DegradationMode,
    DegradationPolicy,
    DependencyClass,
    DependencyFailure,
    resolve_degradation,
)
from refaat.domain.freshness import IdentitySnapshot, snapshot_to_payload
from refaat.domain.kill_switch import KillSwitchRegistry, KillSwitchScope, ensure_scope_active, ensure_workspace_active
from refaat.domain.models import (
    Chunk,
    ContextPacket,
    GenerationRequest,
    JsonObject,
    RlmLimits,
    RlmNodeTrace,
    RlmResult,
    SearchHit,
)
from refaat.domain.output_governance import OutputContext, OutputDecision, OutputGovernor, bounded_reason
from refaat.domain.quota import QuotaContext, QuotaDimension, QuotaPolicy, ensure_within_quota
from refaat.domain.routing import ModelRouter
from refaat.domain.sandbox import CodePolicy, CodeRequest
from refaat.errors import DependencyUnavailableError, QuotaExceededError, ValidationError
from refaat.ids import new_id
from refaat.infrastructure.database.store import SQLStore
from refaat.infrastructure.generation import ExtractiveGenerator
from refaat.infrastructure.sandbox import DenyingSandbox
from refaat.observability import current_trace_id, get_logger
from refaat.ports import CodeSandbox, Embedder, Generator, GraphReadRepository

logger = get_logger(__name__)

# The honest no-answer returned when a generation provider outage is resolved to abstain under an
# installed degradation policy: empty citations, a distinguishable message, degraded set.
RLM_DEGRADED_ABSTAIN_MESSAGE = "I could not produce a grounded answer because a generation dependency was unavailable."


class _RlmDegradationAbstain(Exception):
    """Internal control-flow signal to abort a run and return an abstained answer.

    Raised from a node when a generation outage resolves to abstain, and caught once at run level.
    """

    def __init__(self, decision: DegradationDecision) -> None:
        super().__init__("rlm degradation abstain")
        self.decision = decision


def capture_durable_snapshot(principal: Principal, policy: AccessPolicy, now: datetime) -> IdentitySnapshot:
    """Capture the execution-time identity snapshot persisted on a durable rlm.run job.

    The scope digest is the injective durable_entitlement_digest, not auth_scope_hash, so the worker
    re-resolution detects a kind or attribute change too. The snapshot is redacted by construction:
    only the principal id, tenant, the scope digest, the deciding policy id and version, and a
    timezone-aware capture time, never a full principal, raw entitlements, or a secret.
    """
    return IdentitySnapshot(
        principal_id=principal.id,
        tenant=principal.tenant,
        auth_scope_hash=durable_entitlement_digest(principal, policy),
        policy_id=str(getattr(policy, "name", type(policy).__name__)),
        policy_version=str(getattr(policy, "version", "")),
        captured_at=now,
    )


@dataclass(slots=True)
class _RunState:
    query: str
    query_embedding: tuple[float, ...]
    workspace_id: str
    corpus_id: str
    user_id: str | None
    session_id: str | None
    profile: JsonObject
    supplemental: JsonObject
    graph_facts: tuple[str, ...]
    graph_boosts: dict[str, float]
    session_summary: str | None
    limits: RlmLimits
    auth_scope: str | None = None
    calls: int = 0
    max_depth_reached: int = 0
    grounding_fallbacks: int = 0
    # Resolved once at run level on the first generation outage and reused for every later node,
    # so a run's degradation verdict is consistent. degraded is set when a node served a fallback.
    degradation_decision: DegradationDecision | None = None
    degraded: bool = False


@dataclass(frozen=True, slots=True)
class _NodeOutcome:
    answer: str
    cited_chunk_ids: tuple[str, ...]
    selected_chunks: tuple[Chunk, ...]


class RecursiveContextService:
    """Bounded recursive inference over corpus chunks stored outside the model context."""

    def __init__(
        self,
        *,
        store: SQLStore,
        embedder: Embedder,
        generator: Generator,
        harness: HarnessService,
        settings: Settings,
        clock: Clock | None = None,
        sandbox: CodeSandbox | None = None,
        access_policy: AccessPolicy | None = None,
        audit_sink: AuditSink | None = None,
        model_router: ModelRouter | None = None,
        output_governor: OutputGovernor | None = None,
        context_transform: ContextTransform | None = None,
        kill_switches: KillSwitchRegistry | None = None,
        quota_policy: QuotaPolicy | None = None,
        degradation_policy: DegradationPolicy | None = None,
        strict_authorization: bool = False,
        graph_read_repository: GraphReadRepository | None = None,
    ) -> None:
        self.store = store
        self._graph_repo: GraphReadRepository = graph_read_repository if graph_read_repository is not None else store
        self.embedder = embedder
        self.generator = generator
        self.harness = harness
        self.settings = settings
        self.clock = clock
        self.kill_switches = kill_switches
        # Opt-in persisted per-scope quota seam, mirroring the query path. None by default, so an
        # RLM run is ungoverned and byte-identical.
        self.quota_policy = quota_policy
        self.access_policy = access_policy
        self.audit_sink = audit_sink
        # Opt-in data-classification model-routing seam, gating the generation egress the same
        # way as the query path. None by default, so an RLM run is byte-identical.
        self.model_router = model_router
        # Opt-in pre-model context-transform seam, redacting the evidence each RLM node egress sends
        # to the generator. None by default, so an RLM run is byte-identical. The local extractive
        # fallback keeps the original evidence so it grounds and cites real chunks.
        self.context_transform = context_transform
        # Opt-in output-governance seam. None by default, so the RLM answer is recorded and
        # returned exactly as before. When set, the governor reviews the final answer and
        # citations before the session is recorded; a denial withholds the answer, records
        # no session turn, and audits the refusal.
        self.output_governor = output_governor
        # Opt-in degradation seam for a generation provider outage at an RLM node. None by default,
        # so an outage propagates uncaught exactly as before. When set, the policy decides once per
        # run whether to serve a grounded degraded answer, abstain, or refuse.
        self.degradation_policy = degradation_policy
        # Fail closed (on by default) for an in-process RLM call that omits a principal under an active
        # policy; set strict_authorization off to opt out.
        self.strict_authorization = strict_authorization
        self.fallback = ExtractiveGenerator(max_sentences=5)
        self.compactor = SessionCompactor(store, settings, classify_evidence=model_router is not None)
        # The default denies every execution, so a runtime that injects no sandbox
        # has no code-execution capability and every RLM run is byte-identical.
        self.sandbox: CodeSandbox = sandbox or DenyingSandbox()

    def _budget_scope(self) -> AbstractContextManager[CostMeter | None]:
        return budget_scope(
            self.settings.run_token_budget,
            self.settings.run_cost_budget_usd,
            prices=prices_from(self.settings.model_prices),
        )

    def _quota_context(self, workspace_id: str, corpus_id: str, principal: Principal | None) -> QuotaContext | None:
        if self.quota_policy is None:
            return None
        if in_unserved_run():
            return None
        return QuotaContext(
            workspace_id=workspace_id,
            tenant=principal.tenant if principal is not None else None,
            principal_id=principal.id if principal is not None else None,
            principal_bearing=True,
            corpus_id=corpus_id,
            corpus_validated=True,
        )

    def _served_audit_sink(self) -> AuditSink | None:
        return None if in_unserved_run() else self.audit_sink

    def _emit_quota_audit(
        self, workspace_id: str, corpus_id: str, principal: Principal | None, dimension: QuotaDimension
    ) -> None:
        """Record one redacted quota.denied event on a quota denial. Best-effort, no-op when no
        audit sink is installed. Carries only ids, the dimension, and digests, never query text."""
        sink = self._served_audit_sink()
        if sink is None or self.clock is None:
            return
        event = AuditEvent(
            event_type="quota.denied",
            occurred_at=self.clock.now(),
            workspace_id=workspace_id,
            corpus_id=corpus_id,
            principal_id=principal.id if principal is not None else "system",
            principal_kind=principal.kind.value if principal is not None else "system",
            tenant=principal.tenant if principal is not None else None,
            policy_id=str(getattr(self.quota_policy, "cache_id", "quota") or "quota"),
            policy_version="",
            capability_name=dimension.value,
            action_fingerprint=stable_digest((workspace_id, dimension.value, principal.id if principal else "")),
            dedup_id=stable_digest((workspace_id, corpus_id, dimension.value, "quota")),
        )
        emit_audit(sink, event, fail_closed=False, logger=logger)

    def run(
        self,
        workspace_id: str,
        corpus_id: str,
        query: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        limits: RlmLimits | None = None,
        code_policy: CodePolicy | None = None,
        code_requests: Sequence[CodeRequest] = (),
        principal: Principal | None = None,
    ) -> RlmResult:
        ensure_workspace_active(self.kill_switches, workspace_id)
        # RLM output is model-generated, so a generation kill switch denies the run before any work.
        ensure_scope_active(self.kill_switches, workspace_id, KillSwitchScope.GENERATION)
        # Fail closed before quota admission and any evidence work, so a principal-less strict call is
        # denied without charging quota or reaching a model.
        require_principal_present(self.access_policy, principal, strict=self.strict_authorization)
        ctx = self._quota_context(workspace_id, corpus_id, principal)
        idem = new_id("qta") if ctx is not None else ""
        if ctx is not None:
            # Validate the corpus before counting or reserving so an unknown corpus id can never
            # mint a fresh counter row, then count this run and reserve its ceiling.
            self.store.get_corpus(workspace_id, corpus_id)
            try:
                ensure_within_quota(self.quota_policy, ctx, dimension=QuotaDimension.REQUESTS, idem=idem)
            except QuotaExceededError:
                self._emit_quota_audit(workspace_id, corpus_id, principal, QuotaDimension.REQUESTS)
                raise
        # Bound the RLM's embed and generate spend under a configured run budget; a
        # no-op scope when unset keeps the deterministic path unchanged.
        with self._budget_scope() as meter:
            try:
                with quota_run(
                    self.quota_policy,
                    ctx,
                    meter=meter,
                    token_ceiling=self.settings.run_token_budget,
                    cost_ceiling_usd=self.settings.run_cost_budget_usd,
                    idem=idem,
                ):
                    return self._run(
                        workspace_id,
                        corpus_id,
                        query,
                        user_id=user_id,
                        session_id=session_id,
                        limits=limits,
                        code_policy=code_policy,
                        code_requests=code_requests,
                        principal=principal,
                    )
            except QuotaExceededError:
                self._emit_quota_audit(workspace_id, corpus_id, principal, QuotaDimension.GENERATION_TOKENS)
                raise

    def _run(
        self,
        workspace_id: str,
        corpus_id: str,
        query: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        limits: RlmLimits | None = None,
        code_policy: CodePolicy | None = None,
        code_requests: Sequence[CodeRequest] = (),
        principal: Principal | None = None,
    ) -> RlmResult:
        normalized_query = normalize_text(query)
        if not normalized_query:
            raise ValidationError("RLM query is empty")
        corpus = self.store.get_corpus(workspace_id, corpus_id)
        ceiling = self.settings.rlm_max_input_chunks
        if ceiling is not None:
            # Bound the load in one query so there is no time-of-check window, and fail closed before the
            # full in-memory set is built. The message names only the configured limit, never the corpus
            # size, so it is not a size side channel for a narrowly scoped principal.
            loaded = self.store.list_chunks(workspace_id, corpus_id, limit=ceiling + 1)
            if len(loaded) > ceiling:
                raise ValidationError(
                    f"RLM corpus exceeds the configured input-chunk limit of {ceiling} (REFAAT_RLM_MAX_INPUT_CHUNKS)"
                )
            chunks = tuple(loaded)
        else:
            chunks = tuple(self.store.list_chunks(workspace_id, corpus_id))
        documents: list[JsonObject] | None = None
        authorized_documents: frozenset[str] | None = None
        if self.access_policy is not None and principal is not None:
            # Filter the corpus before any recursion, so an unauthorized chunk never enters a
            # sub-call, a partition, synthesis, or a citation.
            documents = self.store.list_documents(workspace_id, corpus_id)
            authorized_documents = authorized_document_ids(self.access_policy, principal, documents)
            chunks = tuple(chunk for chunk in chunks if chunk_allowed(self.access_policy, principal, chunk.metadata))
        entities, relations = select_graph(self._graph_repo, self.settings, workspace_id, corpus_id, normalized_query)
        graph_boosts, graph_facts, graph_fact_documents = AdaptiveRetriever.graph_context(
            normalized_query,
            entities,
            relations,
            authorized_documents=authorized_documents,
        )
        loaded_summary = self._load_session_summary(workspace_id, corpus_id, session_id, principal)
        session_summary, summary_cited = loaded_summary if loaded_summary is not None else (None, frozenset())
        if self.model_router is not None and chunks:
            # The RLM egress unit is every piece of evidence that reaches the model, not only the
            # chunks: the graph facts and the session summary also enter the prompt, so the router is
            # consulted once on the maximum classification across all three, before any sub-generation.
            # A denial refuses with the generic routing-refusal shape and records no session turn.
            metadatas: list[JsonObject | None] = [chunk.metadata for chunk in chunks]
            graph_ids = (
                frozenset().union(*graph_fact_documents[:GRAPH_FACT_LIMIT]) if graph_fact_documents else frozenset()
            )
            if documents is None and (graph_ids or summary_cited):
                documents = self.store.list_documents(workspace_id, corpus_id)
            if graph_ids and documents is not None:
                metadatas.extend(doc.get("metadata") for doc in documents if str(doc.get("id")) in graph_ids)
            classification = max_classification(metadatas)
            zones = residency_zones(metadatas)
            if session_summary is not None:
                # Classify a surfaced summary whether or not its provenance was recorded: an unrecorded
                # summary (None) fails closed through the helper, so it can never reach the model unrated.
                summary_class, summary_zones = summary_evidence_from_documents(summary_cited, documents or [])
                classification = max(classification, summary_class)
                zones = zones | summary_zones
            if not self.model_router.route(classification, principal, residency_zones=zones).allowed:
                refused = RlmResult(
                    answer=ROUTING_REFUSAL_MESSAGE,
                    citations=(),
                    nodes=(),
                    corpus_version=corpus.version,
                    trace_id=current_trace_id(),
                    usage={"provider": "routing_denied", "reason": ROUTING_REFUSAL_REASON},
                )
                self._emit_rlm_audit(workspace_id, corpus_id, principal, refused)
                return refused
        selected_limits = limits or RlmLimits()
        profile = self.store.get_preferences(
            workspace_id,
            corpus_id,
            preference_scope(self.access_policy, principal, user_id),
        )
        supplemental = self.harness.active(workspace_id, corpus_id)
        auth_scope = (
            auth_scope_hash(principal, self.access_policy)
            if self.access_policy is not None and principal is not None
            else None
        )
        state = _RunState(
            query=normalized_query,
            query_embedding=self.embedder.embed([normalized_query])[0],
            workspace_id=workspace_id,
            corpus_id=corpus_id,
            user_id=user_id,
            session_id=session_id,
            profile=profile,
            supplemental=supplemental,
            graph_facts=tuple(graph_facts[:GRAPH_FACT_LIMIT]),
            graph_boosts=graph_boosts,
            session_summary=session_summary,
            limits=selected_limits,
            auth_scope=auth_scope,
        )
        traces: list[RlmNodeTrace] = []
        if chunks:
            try:
                outcome = self._solve_group(
                    chunks,
                    depth=0,
                    parent_id=None,
                    reserve_after=0,
                    state=state,
                    traces=traces,
                )
            except _RlmDegradationAbstain as abstain:
                # A generation outage resolved to abstain: abort the run and return an honest
                # abstained answer, never recording a session turn for the transient outage.
                refused = self._degraded_abstain_result(corpus.version)
                self._emit_degradation_audit(workspace_id, corpus_id, principal, abstain.decision, refused)
                return refused
            cited_ids = validate_cited_chunks(outcome.cited_chunk_ids, chunks)
            # Re-resolve the cited chunks against the CURRENT store and re-authorize them. The run
            # loaded and authorized the corpus once at the start, but a concurrent replacement or ACL
            # change during the recursive run could delete a cited chunk or revoke access, so a
            # citation is kept only when its chunk still exists and is still authorized, mirroring the
            # query path. With no concurrent change this resolves to the same chunks, byte-identical.
            fresh = {chunk.id: chunk for chunk in self.store.get_chunks(workspace_id, corpus_id, cited_ids)}
            if self.access_policy is not None and principal is not None:
                fresh = {
                    chunk_id: chunk
                    for chunk_id, chunk in fresh.items()
                    if chunk_allowed(self.access_policy, principal, chunk.metadata)
                }
            resolved = [fresh[chunk_id] for chunk_id in cited_ids if chunk_id in fresh]
            resolved_ids = tuple(chunk.id for chunk in resolved)
            citations = build_citations(resolved_ids, resolved)
            answer = strip_uncited_markers(outcome.answer, resolved_ids)
        else:
            citations = ()
            answer = "I could not find grounded evidence because the selected corpus is empty."

        result = RlmResult(
            answer=answer,
            citations=citations,
            nodes=tuple(traces),
            corpus_version=corpus.version,
            trace_id=current_trace_id(),
            usage={
                "model_calls": state.calls,
                "input_chunks": len(chunks),
                "input_tokens": sum(chunk.token_count for chunk in chunks),
                "max_depth_reached": state.max_depth_reached,
                "grounding_fallbacks": state.grounding_fallbacks,
                "embedding_model": self.embedder.model_id,
                "generation_model": self.generator.model_id,
                "limits": {
                    "max_depth": selected_limits.max_depth,
                    "max_nodes": selected_limits.max_nodes,
                    "branch_factor": selected_limits.branch_factor,
                    "leaf_token_budget": selected_limits.leaf_token_budget,
                },
            },
            degraded=state.degraded,
        )
        if state.degraded and state.degradation_decision is not None:
            self._emit_degradation_audit(workspace_id, corpus_id, principal, state.degradation_decision, result)
        result = self._apply_code_phase(result, code_policy, code_requests)
        governor = self.output_governor
        if governor is not None:
            decision = self._review_output(governor, principal, workspace_id, corpus_id, result)
            if not decision.allowed:
                refused = replace(
                    result,
                    answer=OUTPUT_REFUSAL_MESSAGE,
                    citations=(),
                    usage={**result.usage, "output_denied": OUTPUT_REFUSAL_REASON},
                )
                self._emit_output_denied_audit(workspace_id, corpus_id, principal, decision, refused)
                self._emit_rlm_audit(workspace_id, corpus_id, principal, refused)
                return refused
            if decision.redacted_answer is not None:
                result = replace(result, answer=decision.redacted_answer)
        # A degraded serve during a transient outage is not recorded, mirroring the query path so the
        # session summary does not carry a fallback answer forward after the provider recovers.
        if not state.degraded:
            self._record_session(state, result)
        self._emit_rlm_audit(workspace_id, corpus_id, principal, result)
        return result

    def _review_output(
        self,
        governor: OutputGovernor,
        principal: Principal | None,
        workspace_id: str,
        corpus_id: str,
        result: RlmResult,
    ) -> OutputDecision:
        """Run the installed output governor over the RLM answer, fail-closed on error.

        A governor that raises is treated as a denial: a broken review can never let
        unreviewed output through. Only reached when a governor is installed.
        """
        try:
            return governor.review(
                OutputContext(
                    answer=result.answer,
                    citations=result.citations,
                    principal=principal,
                    workspace_id=workspace_id,
                    corpus_id=corpus_id,
                )
            )
        except Exception:
            logger.warning("output_governor_error")
            return OutputDecision(allowed=False, reason="output_governor_error")

    def _emit_output_denied_audit(
        self,
        workspace_id: str,
        corpus_id: str,
        principal: Principal | None,
        decision: OutputDecision,
        result: RlmResult,
    ) -> None:
        """Record a redacted output.denied audit event for a refused RLM answer.

        Reads self.access_policy (the RLM has no retriever) and guards self.clock, mirroring
        _emit_rlm_audit. Best-effort: the answer is already withheld, so a sink outage must
        not also crash the refusal. No-op unless a sink is installed under an active policy
        with a principal. Carries the governor's bounded reason in capability_name, never the
        answer text.
        """
        sink = self._served_audit_sink()
        policy = self.access_policy
        if sink is None or self.clock is None or policy is None or principal is None:
            return
        event = AuditEvent(
            event_type="output.denied",
            occurred_at=self.clock.now(),
            workspace_id=workspace_id,
            corpus_id=corpus_id,
            corpus_version=result.corpus_version,
            principal_id=principal.id,
            principal_kind=principal.kind.value,
            tenant=principal.tenant,
            policy_id=str(getattr(policy, "name", type(policy).__name__)),
            policy_version=str(getattr(policy, "version", "")),
            cited_document_ids=(),
            citation_count=0,
            abstained=False,
            cached=False,
            trace_id=result.trace_id,
            capability_name=bounded_reason(decision.reason),
            dedup_id=stable_digest((result.trace_id or "", workspace_id, corpus_id, principal.id)),
        )
        emit_audit(sink, event, fail_closed=False, logger=logger)

    def _emit_rlm_audit(
        self, workspace_id: str, corpus_id: str, principal: Principal | None, result: RlmResult
    ) -> None:
        sink = self._served_audit_sink()
        policy = self.access_policy
        if sink is None or self.clock is None or policy is None or principal is None:
            return
        cited_documents = tuple(sorted({citation.document_id for citation in result.citations}))
        event = AuditEvent(
            event_type="rlm.authorized",
            occurred_at=self.clock.now(),
            workspace_id=workspace_id,
            corpus_id=corpus_id,
            corpus_version=result.corpus_version,
            principal_id=principal.id,
            principal_kind=principal.kind.value,
            tenant=principal.tenant,
            policy_id=str(getattr(policy, "name", type(policy).__name__)),
            policy_version=str(getattr(policy, "version", "")),
            cited_document_ids=cited_documents,
            citation_count=len(result.citations),
            abstained=False,
            cached=False,
            trace_id=result.trace_id,
            dedup_id=stable_digest((result.trace_id or "", workspace_id, corpus_id, principal.id)),
        )
        emit_audit(sink, event, fail_closed=self.settings.audit_fail_closed, logger=logger)

    def _resolve_run_degradation(self, state: _RunState, *, grounding_survives: bool) -> DegradationDecision:
        """Resolve the run's degradation verdict once and reuse it for every later node.

        The first node to hit a generation outage decides the verdict for the whole run, so a run's
        handling is consistent. With no policy installed the verdict is the legacy refuse.
        """
        if state.degradation_decision is None:
            failure = DependencyFailure(
                dependency_class=DependencyClass.RLM_SYNTHESIS,
                grounding_survives=grounding_survives,
                provider_id=self.generator.model_id,
                reason="generation_provider_unavailable",
            )
            state.degradation_decision = resolve_degradation(self.degradation_policy, failure)
        return state.degradation_decision

    def _degraded_abstain_result(self, corpus_version: int) -> RlmResult:
        """The honest abstained answer for a run whose generation outage resolved to abstain."""
        return RlmResult(
            answer=RLM_DEGRADED_ABSTAIN_MESSAGE,
            citations=(),
            nodes=(),
            corpus_version=corpus_version,
            trace_id=current_trace_id(),
            usage={"provider": "degraded_abstain", "generation_model": self.generator.model_id},
            degraded=True,
        )

    def _emit_degradation_audit(
        self,
        workspace_id: str,
        corpus_id: str,
        principal: Principal | None,
        decision: DegradationDecision,
        result: RlmResult,
    ) -> None:
        """Record one redacted degradation.decided event when a degradation policy is installed.

        No-op when no audit sink, no clock, or no degradation policy is installed, so the default
        path emits nothing. Best-effort: a degradation is not protection-decreasing, so a sink
        outage must not also crash the served path. The event carries only the dependency class,
        mode, machine reason code, and the bounded provider id, never a query or evidence text.
        Principal and policy fields fall back to a system sentinel when authorization is off.
        """
        sink = self._served_audit_sink()
        if sink is None or self.clock is None or self.degradation_policy is None:
            return
        policy = self.access_policy
        event = AuditEvent(
            event_type="degradation.decided",
            occurred_at=self.clock.now(),
            workspace_id=workspace_id,
            corpus_id=corpus_id,
            corpus_version=result.corpus_version,
            principal_id=principal.id if principal is not None else "system",
            principal_kind=principal.kind.value if principal is not None else "system",
            tenant=principal.tenant if principal is not None else None,
            policy_id=str(getattr(policy, "name", type(policy).__name__)) if policy is not None else "",
            policy_version=str(getattr(policy, "version", "")) if policy is not None else "",
            cited_document_ids=(),
            citation_count=len(result.citations),
            abstained=result.answer == RLM_DEGRADED_ABSTAIN_MESSAGE,
            cached=False,
            trace_id=result.trace_id,
            capability_name=f"{decision.dependency_class.value}:{decision.mode.value}:{decision.reason}"[:200],
            action_fingerprint=self.generator.model_id[:128],
            dedup_id=stable_digest((result.trace_id or "", workspace_id, corpus_id, "degradation")),
        )
        emit_audit(sink, event, fail_closed=False, logger=logger, log_key="degradation_audit_sink_failed")

    def _apply_code_phase(
        self,
        result: RlmResult,
        code_policy: CodePolicy | None,
        code_requests: Sequence[CodeRequest],
    ) -> RlmResult:
        # With no policy the RLM never enters a code phase, so usage and the whole
        # result stay byte-identical to the no-sandbox run. Only an explicit policy
        # opts in, and even then the grounded answer and citations are unchanged:
        # governed computation is recorded for audit, never fabricated into evidence.
        if code_policy is None:
            return result
        steps = run_code_phase(self.sandbox, code_policy, code_requests)
        usage = {
            **result.usage,
            "code_sandbox": self.sandbox.name,
            "code_steps": [step.as_json() for step in steps],
        }
        return replace(result, usage=usage)

    def enqueue(
        self,
        workspace_id: str,
        corpus_id: str,
        query: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        limits: RlmLimits | None = None,
        principal: Principal | None = None,
    ) -> str:
        selected_limits = limits or RlmLimits()
        self.store.get_corpus(workspace_id, corpus_id)
        payload: JsonObject = {
            "corpus_id": corpus_id,
            "query": query,
            "user_id": user_id,
            "session_id": session_id,
            "limits": {
                "max_depth": selected_limits.max_depth,
                "max_nodes": selected_limits.max_nodes,
                "branch_factor": selected_limits.branch_factor,
                "leaf_token_budget": selected_limits.leaf_token_budget,
            },
        }
        # The reserved snapshot key is written server-side only, and only under an active policy with
        # a resolved principal, so the worker re-resolves the enqueue-time identity before the run.
        if principal is not None and self.access_policy is not None:
            payload["_identity_snapshot"] = snapshot_to_payload(
                capture_durable_snapshot(principal, self.access_policy, self.store.clock.now())
            )
        return self.store.enqueue_job(workspace_id, "rlm.run", payload)

    def _solve_group(
        self,
        chunks: Sequence[Chunk],
        *,
        depth: int,
        parent_id: str | None,
        reserve_after: int,
        state: _RunState,
        traces: list[RlmNodeTrace],
    ) -> _NodeOutcome:
        state.max_depth_reached = max(state.max_depth_reached, depth)
        node_id = new_id("rlmnode")
        total_tokens = sum(chunk.token_count for chunk in chunks)
        available = state.limits.max_nodes - state.calls - reserve_after
        can_split = (
            depth < state.limits.max_depth
            and total_tokens > state.limits.leaf_token_budget
            and available >= 3
            and len(chunks) > 1
        )
        route = "leaf"
        recursive_summaries: list[str] = []
        candidates: tuple[Chunk, ...]
        if can_split:
            group_count = min(
                state.limits.branch_factor,
                available - 1,
                len(chunks),
                max(2, math.ceil(total_tokens / state.limits.leaf_token_budget)),
            )
            groups = self._partition(chunks, group_count)
            child_chunks: list[Chunk] = []
            for index, group in enumerate(groups):
                future_children = len(groups) - index - 1
                child = self._solve_group(
                    group,
                    depth=depth + 1,
                    parent_id=node_id,
                    reserve_after=reserve_after + 1 + future_children,
                    state=state,
                    traces=traces,
                )
                child_chunks.extend(child.selected_chunks)
                recursive_summaries.append(f"Subproblem {index + 1}: {child.answer}")
            candidates = self._deduplicate(child_chunks)
            route = "synthesis"
        else:
            candidates = tuple(chunks)

        ranked = self._rank(candidates, state)
        selected = self._fit_budget(ranked, state.limits.leaf_token_budget)
        if not selected and candidates:
            selected = (candidates[0],)
        outcome = self._generate(
            selected,
            depth=depth,
            route=route,
            state=state,
            recursive_summaries=recursive_summaries,
        )
        traces.append(
            RlmNodeTrace(
                id=node_id,
                parent_id=parent_id,
                depth=depth,
                route=route,
                input_chunk_count=len(chunks),
                input_tokens=total_tokens,
                selected_chunk_ids=tuple(chunk.id for chunk in selected),
                cited_chunk_ids=outcome.cited_chunk_ids,
                answer_excerpt=outcome.answer[:1_000],
            )
        )
        return outcome

    def _generate(
        self,
        chunks: Sequence[Chunk],
        *,
        depth: int,
        route: str,
        state: _RunState,
        recursive_summaries: Sequence[str],
    ) -> _NodeOutcome:
        hits = tuple(
            SearchHit(
                chunk=chunk,
                score=self._score(chunk, state),
                semantic_score=max(
                    0.0,
                    cosine_similarity(state.query_embedding, chunk.embedding),
                ),
                graph_score=state.graph_boosts.get(chunk.document_id, 0.0),
                reasons=("rlm", route),
            )
            for chunk in chunks
        )
        context = ContextPacket(
            query=state.query,
            hits=hits,
            graph_facts=state.graph_facts + tuple(recursive_summaries),
            session_summary=state.session_summary,
            profile=state.profile,
            supplemental_harness=state.supplemental,
            token_count=sum(chunk.token_count for chunk in chunks),
            routes=("rlm", f"rlm:{route}", f"rlm:depth:{depth}"),
        )
        request = GenerationRequest(
            query=state.query,
            context=context,
            system_prompt=system_prompt(state.supplemental),
            workspace_id=state.workspace_id,
        )
        state.calls += 1
        # Redact the evidence this node sends to the model before the generate call. The original
        # request is kept for the extractive fallback so it grounds and cites the real chunks.
        # Byte-identical with no transform: the seam returns the same request object.
        egress = redact_for_egress(self.context_transform, request)
        try:
            generated = self.generator.generate(egress.request)
        except DependencyUnavailableError:
            decision = self._resolve_run_degradation(state, grounding_survives=bool(chunks))
            if decision.mode is DegradationMode.REFUSE:
                raise
            if decision.mode is DegradationMode.ABSTAIN:
                raise _RlmDegradationAbstain(decision) from None
            state.degraded = True
            generated = self.fallback.generate(request)
        cited = filter_cited_chunks(generated.cited_chunk_ids, chunks)
        if chunks and not cited:
            generated = self.fallback.generate(request)
            cited = filter_cited_chunks(generated.cited_chunk_ids, chunks)
            state.grounding_fallbacks += 1
        cited_set = set(cited)
        selected_chunks = tuple(chunk for chunk in chunks if chunk.id in cited_set)
        return _NodeOutcome(
            answer=generated.answer,
            cited_chunk_ids=cited,
            selected_chunks=selected_chunks or tuple(chunks[:1]),
        )

    def _rank(self, chunks: Sequence[Chunk], state: _RunState) -> tuple[Chunk, ...]:
        return tuple(
            sorted(
                chunks,
                key=lambda chunk: (-self._score(chunk, state), chunk.document_id, chunk.ordinal),
            )
        )

    def _score(self, chunk: Chunk, state: _RunState) -> float:
        query_terms = set(tokens(state.query))
        chunk_terms = set(tokens(chunk.text))
        lexical = len(query_terms & chunk_terms) / max(len(query_terms), 1)
        semantic = max(0.0, cosine_similarity(state.query_embedding, chunk.embedding))
        graph = state.graph_boosts.get(chunk.document_id, 0.0)
        return 0.60 * semantic + 0.30 * lexical + 0.10 * graph

    @staticmethod
    def _fit_budget(chunks: Sequence[Chunk], token_budget: int) -> tuple[Chunk, ...]:
        selected: list[Chunk] = []
        used = 0
        for chunk in chunks:
            if selected and used + chunk.token_count > token_budget:
                continue
            if not selected and chunk.token_count > token_budget:
                # Truncate by the token measure so a single space-free or delimiter-dense leaf chunk that
                # exceeds the budget is actually bounded; the id and checksum stay the real chunk's so a
                # citation still resolves to stored evidence.
                truncated_text = truncate_to_tokens(chunk.text, token_budget)
                selected.append(
                    Chunk(
                        id=chunk.id,
                        corpus_id=chunk.corpus_id,
                        document_id=chunk.document_id,
                        ordinal=chunk.ordinal,
                        text=truncated_text,
                        token_count=estimate_tokens(truncated_text),
                        checksum=chunk.checksum,
                        metadata=chunk.metadata,
                        embedding=chunk.embedding,
                    )
                )
                break
            selected.append(chunk)
            used += chunk.token_count
        return tuple(selected)

    @staticmethod
    def _partition(chunks: Sequence[Chunk], group_count: int) -> tuple[tuple[Chunk, ...], ...]:
        groups: list[list[Chunk]] = [[] for _ in range(group_count)]
        totals = [0] * group_count
        ordered = sorted(chunks, key=lambda chunk: (-chunk.token_count, chunk.document_id, chunk.ordinal))
        for chunk in ordered:
            index = min(range(group_count), key=lambda value: (totals[value], value))
            groups[index].append(chunk)
            totals[index] += chunk.token_count
        return tuple(tuple(group) for group in groups if group)

    @staticmethod
    def _deduplicate(chunks: Sequence[Chunk]) -> tuple[Chunk, ...]:
        values: dict[str, Chunk] = {}
        for chunk in chunks:
            values.setdefault(chunk.id, chunk)
        return tuple(values.values())

    def _load_session_summary(
        self,
        workspace_id: str,
        corpus_id: str,
        session_id: str | None,
        principal: Principal | None = None,
    ) -> tuple[str, frozenset[str] | None] | None:
        if session_id is None:
            return None
        session = self.store.get_session(workspace_id, session_id)
        if session["corpus_id"] != corpus_id:
            raise ValidationError("Session belongs to another corpus")
        summary = session.get("summary")
        if not summary:
            return None
        # A compacted summary is prior evidence bound to its writer principal: the shared gate
        # surfaces it into the RLM context only when the stored writer scope matches this reader,
        # else fail closed. With no policy the gate is a no-op and the path is unchanged; a principal-less
        # run under an active policy is denied earlier by strict authorization.
        stored_scope = cast(JsonObject, session.get("metadata") or {}).get("summary_auth_scope")
        if not session_summary_visible(stored_scope, principal, self.access_policy):
            return None
        cited = stored_summary_cited_documents(session)
        if self.access_policy is not None and principal is not None:
            documents = self.store.list_documents(workspace_id, corpus_id)
            if not summary_documents_authorized(self.access_policy, principal, cited, documents):
                return None
        # The cited documents ride along so the egress gate can classify the summary from current ACLs.
        return str(summary), cited

    def _record_session(self, state: _RunState, result: RlmResult) -> None:
        if state.session_id is None:
            return
        if in_shadow_execution():
            return
        self.store.append_session_turn(
            state.workspace_id,
            state.session_id,
            "user",
            state.query,
            tag_turn_metadata({"mode": "rlm"}, state.auth_scope),
        )
        self.store.append_session_turn(
            state.workspace_id,
            state.session_id,
            "assistant",
            result.answer,
            tag_turn_metadata(
                {
                    "mode": "rlm",
                    "citations": [citation.chunk_id for citation in result.citations],
                    # Stable across a re-ingest, so erasure still matches the turn after a replace.
                    "cited_document_ids": sorted({citation.document_id for citation in result.citations}),
                    "model_calls": result.usage.get("model_calls", 0),
                },
                state.auth_scope,
            ),
        )
        self.compactor.compact_if_needed(state.workspace_id, state.session_id, state.auth_scope)
