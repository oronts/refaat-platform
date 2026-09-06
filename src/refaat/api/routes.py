from __future__ import annotations

import hmac
import json
from collections.abc import Iterator, Mapping
from dataclasses import asdict
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Header, UploadFile, status
from fastapi.responses import StreamingResponse

from refaat.api.dependencies import (
    AdminPrincipal,
    AmbientApprovalGuard,
    Identity,
    OptionalIdentity,
    Principal,
    ProgramSurfaceGuard,
    ReadPrincipal,
    RuntimeDependency,
    WritePrincipal,
)
from refaat.api.schemas import (
    AgentCreateRequest,
    AgentHeartbeatRequest,
    AgentMessageRequest,
    ApiKeyCreateRequest,
    ApprovalGrantRequest,
    ApprovalRequestCreate,
    BootstrapWorkspaceRequest,
    BootstrapWorkspaceResponse,
    CanarySetRequest,
    CorpusCreateRequest,
    CorpusResponse,
    EngageKillSwitchRequest,
    EvaluationRunRequest,
    EvaluationSuiteRequest,
    FeedbackRequest,
    FineTuneRequestBody,
    HarnessSnapshotRequest,
    IngestResponse,
    JobCreateRequest,
    PlaceHoldRequest,
    ProgramEnvironmentRequest,
    ProgramEvaluateRequest,
    ProgramOptimizeRequest,
    ProgramRunRequest,
    QueryBody,
    QueryResultResponse,
    RlmResultResponse,
    RlmRunRequest,
    ScheduleCreateRequest,
    SessionCreateRequest,
    SkillCreateRequest,
    SkillRunRequest,
    SourceDocumentRequest,
    StatusResponse,
    TextDocumentRequest,
)
from refaat.application.approvals import ApprovalService
from refaat.application.derivation import DerivationService, compare_manifests
from refaat.application.kill_switch import KillSwitchService
from refaat.application.legal_hold import LegalHoldService
from refaat.application.runtime import ProjectRef, Refaat, RefaatProject
from refaat.application.status import StatusService
from refaat.core.text import stable_digest
from refaat.domain.audit import AuditEvent, emit_audit
from refaat.domain.authz import (
    Principal as AuthzPrincipal,
)
from refaat.domain.authz import (
    authorized_document_ids,
    durable_entitlement_digest,
)
from refaat.domain.enums import SideEffectClass
from refaat.domain.freshness import snapshot_from_payload
from refaat.domain.jobs import RESERVED_JOB_KINDS
from refaat.domain.kill_switch import KillSwitchScope
from refaat.domain.models import DocumentInput, Feedback, QueryRequest, RlmLimits
from refaat.errors import AuthenticationError, AuthorizationError, NotFoundError, ToolTimeoutError, ValidationError
from refaat.evaluation import Dataset, default_bundle
from refaat.observability import current_trace_id, get_logger
from refaat.programs.program import Program, Tunables

router = APIRouter()
logger = get_logger(__name__)


@router.get("/health/live", tags=["health"])
def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready", tags=["health"])
def ready(runtime: RuntimeDependency) -> dict[str, str]:
    runtime.store.ping()
    return {"status": "ready"}


@router.post(
    "/v1/bootstrap/workspaces",
    response_model=BootstrapWorkspaceResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["bootstrap"],
)
def bootstrap_workspace(
    body: BootstrapWorkspaceRequest,
    runtime: RuntimeDependency,
    bootstrap_token: Annotated[str | None, Header(alias="X-Bootstrap-Token")] = None,
) -> BootstrapWorkspaceResponse:
    configured = runtime.settings.bootstrap_token
    if configured is None:
        raise NotFoundError("Bootstrap endpoint is disabled")
    if bootstrap_token is None or not hmac.compare_digest(
        bootstrap_token,
        configured.get_secret_value(),
    ):
        raise AuthenticationError("Invalid bootstrap token")
    workspace_id = runtime.create_workspace(body.slug, body.name)
    key_id, raw_key = runtime.issue_api_key(
        workspace_id,
        body.key_name,
        scopes=("read", "write", "admin"),
    )
    return BootstrapWorkspaceResponse(
        workspace_id=workspace_id,
        api_key_id=key_id,
        api_key=raw_key,
    )


@router.post("/v1/api-keys", status_code=status.HTTP_201_CREATED, tags=["access"])
def create_api_key(
    body: ApiKeyCreateRequest,
    principal: AdminPrincipal,
    runtime: RuntimeDependency,
) -> dict[str, str]:
    key_id, raw = runtime.issue_api_key(
        principal.workspace_id,
        body.name,
        tuple(body.scopes),
    )
    return {"id": key_id, "api_key": raw}


@router.delete("/v1/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["access"])
def revoke_api_key(key_id: str, principal: AdminPrincipal, runtime: RuntimeDependency) -> None:
    runtime.store.revoke_api_key(principal.workspace_id, key_id)


@router.post("/v1/approvals", status_code=status.HTTP_201_CREATED, tags=["approvals"])
def grant_approval(
    body: ApprovalGrantRequest,
    principal: AdminPrincipal,
    identity: Identity,
    runtime: RuntimeDependency,
) -> dict[str, str]:
    # The approver is the server-resolved identity, never a request-body field, so a caller can
    # only approve as itself and the separation-of-duties and four-eyes checks at consume time hold.
    if identity is None:
        raise AuthorizationError("An approver identity is required to record an approval")
    service = ApprovalService(
        runtime.store,
        principal.workspace_id,
        audit_sink=runtime.audit_sink,
        audit_fail_closed=runtime.settings.audit_fail_closed,
    )
    approval_id = service.grant(
        fingerprint=body.fingerprint,
        capability_name=body.capability_name,
        approver_id=identity.id,
        approver_roles=sorted(identity.roles),
        approver_kind=identity.kind.value,
        kind="break_glass" if body.break_glass else "normal",
        reason=body.reason if body.break_glass else None,
    )
    return {"approval_id": approval_id}


@router.post("/v1/approvals/requests", status_code=status.HTTP_201_CREATED, tags=["approvals"])
def request_approval(
    body: ApprovalRequestCreate,
    principal: AdminPrincipal,
    identity: Identity,
    runtime: RuntimeDependency,
) -> dict[str, str]:
    # The requester is the server-resolved identity, never a request-body field, so a request is
    # attributed to the caller and its created_at anchor cannot be forged.
    if identity is None:
        raise AuthorizationError("A requester identity is required to record an approval request")
    created_at = runtime.store.clock.now()
    request_id = runtime.store.record_approval_request(
        principal.workspace_id,
        fingerprint=body.fingerprint,
        capability_name=body.capability_name,
        requested_by=identity.id,
        created_at=created_at,
    )
    return {"request_id": request_id, "created_at": created_at.isoformat()}


@router.get("/v1/approvals", tags=["approvals"])
def list_approvals(
    fingerprint: str,
    principal: AdminPrincipal,
    runtime: RuntimeDependency,
) -> list[dict[str, Any]]:
    service = ApprovalService(runtime.store, principal.workspace_id)
    return service.list_active(fingerprint)


@router.delete("/v1/approvals/{approval_id}", tags=["approvals"])
def cancel_approval(
    approval_id: str,
    principal: AdminPrincipal,
    identity: OptionalIdentity,
    runtime: RuntimeDependency,
) -> dict[str, bool]:
    # Cancel is authorized by the admin api key, so identity is optional here and only attributes the
    # canceller in the audit event when a resolver resolves one; it never fails the cancel.
    service = ApprovalService(
        runtime.store,
        principal.workspace_id,
        audit_sink=runtime.audit_sink,
        audit_fail_closed=runtime.settings.audit_fail_closed,
    )
    canceller_id = identity.id if identity is not None else None
    canceller_kind = identity.kind.value if identity is not None else "human"
    return {"cancelled": service.cancel(approval_id, canceller_id=canceller_id, canceller_kind=canceller_kind)}


@router.post(
    "/v1/corpora",
    response_model=CorpusResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["corpora"],
)
def create_corpus(
    body: CorpusCreateRequest,
    principal: WritePrincipal,
    runtime: RuntimeDependency,
) -> CorpusResponse:
    corpus_id = runtime.create_corpus(principal.workspace_id, body.name, body.settings)
    return CorpusResponse.model_validate(asdict(runtime.store.get_corpus(principal.workspace_id, corpus_id)))


@router.get("/v1/corpora", response_model=list[CorpusResponse], tags=["corpora"])
def list_corpora(principal: ReadPrincipal, runtime: RuntimeDependency) -> list[CorpusResponse]:
    return [
        CorpusResponse.model_validate(asdict(corpus)) for corpus in runtime.store.list_corpora(principal.workspace_id)
    ]


@router.get("/v1/corpora/{corpus_id}", response_model=CorpusResponse, tags=["corpora"])
def get_corpus(
    corpus_id: str,
    principal: ReadPrincipal,
    runtime: RuntimeDependency,
) -> CorpusResponse:
    return CorpusResponse.model_validate(asdict(runtime.store.get_corpus(principal.workspace_id, corpus_id)))


@router.get("/v1/corpora/{corpus_id}/stats", tags=["corpora"])
def corpus_stats(
    corpus_id: str,
    principal: ReadPrincipal,
    _guard: ProgramSurfaceGuard,
    runtime: RuntimeDependency,
) -> dict[str, Any]:
    return runtime.store.stats(principal.workspace_id, corpus_id)


def _require_derivation(runtime: Refaat) -> DerivationService:
    if runtime.derivation is None:
        raise NotFoundError("Derivation manifests are not enabled")
    return runtime.derivation


@router.get("/v1/corpora/{corpus_id}/derivations", tags=["corpora"])
def list_derivations(
    corpus_id: str,
    principal: ReadPrincipal,
    _guard: ProgramSurfaceGuard,
    runtime: RuntimeDependency,
) -> list[dict[str, Any]]:
    return [asdict(manifest) for manifest in _require_derivation(runtime).manifests(principal.workspace_id, corpus_id)]


@router.get("/v1/corpora/{corpus_id}/derivations/verify", tags=["corpora"])
def verify_derivation(
    corpus_id: str,
    principal: AdminPrincipal,
    _guard: ProgramSurfaceGuard,
    runtime: RuntimeDependency,
) -> dict[str, Any]:
    # AdminPrincipal, not ReadPrincipal: verify recomputes the O(corpus) rollup from committed state.
    detail = _require_derivation(runtime).verify_detail(principal.workspace_id, corpus_id)
    return {"corpus_id": corpus_id, "status": detail}


@router.get("/v1/corpora/{corpus_id}/derivations/compare", tags=["corpora"])
def compare_derivations(
    corpus_id: str,
    left: str,
    right: str,
    principal: ReadPrincipal,
    _guard: ProgramSurfaceGuard,
    runtime: RuntimeDependency,
) -> dict[str, Any]:
    service = _require_derivation(runtime)
    left_manifest = service.manifest(principal.workspace_id, corpus_id, left)
    right_manifest = service.manifest(principal.workspace_id, corpus_id, right)
    if left_manifest is None or right_manifest is None:
        raise NotFoundError("Both derivation keys must exist to compare")
    return compare_manifests(left_manifest, right_manifest)


@router.get("/v1/corpora/{corpus_id}/derivations/{derivation_key}", tags=["corpora"])
def get_derivation(
    corpus_id: str,
    derivation_key: str,
    principal: ReadPrincipal,
    _guard: ProgramSurfaceGuard,
    runtime: RuntimeDependency,
) -> dict[str, Any]:
    manifest = _require_derivation(runtime).manifest(principal.workspace_id, corpus_id, derivation_key)
    if manifest is None:
        raise NotFoundError("No derivation manifest for that key")
    return asdict(manifest)


@router.get("/v1/corpora/{corpus_id}/documents/{document_id}/derivation/verify", tags=["corpora"])
def verify_document_derivation(
    corpus_id: str,
    document_id: str,
    principal: AdminPrincipal,
    _guard: ProgramSurfaceGuard,
    runtime: RuntimeDependency,
) -> dict[str, Any]:
    # AdminPrincipal like the corpus verify: recomputes an O(document) fold from committed state to
    # localize whether this document's chunk set drifted.
    status = _require_derivation(runtime).verify_document(principal.workspace_id, corpus_id, document_id)
    return {"document_id": document_id, "status": status}


@router.post(
    "/v1/corpora/{corpus_id}/documents/text",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
)
def ingest_text_document(
    corpus_id: str,
    body: TextDocumentRequest,
    principal: WritePrincipal,
    runtime: RuntimeDependency,
) -> IngestResponse:
    result = runtime.ingest(
        principal.workspace_id,
        corpus_id,
        DocumentInput(
            source_id=body.source_id,
            title=body.title,
            media_type=body.media_type,
            content=body.content.encode(),
            metadata=body.metadata,
        ),
        actor=principal.api_key_id,
    )
    return IngestResponse.model_validate(asdict(result))


@router.post(
    "/v1/corpora/{corpus_id}/documents/upload",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
)
def upload_document(
    corpus_id: str,
    principal: WritePrincipal,
    runtime: RuntimeDependency,
    file: Annotated[UploadFile, File()],
    source_id: Annotated[str, Form(min_length=1, max_length=512)],
    title: Annotated[str | None, Form(max_length=512)] = None,
    metadata_json: Annotated[str, Form()] = "{}",
) -> IngestResponse:
    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError as exc:
        raise ValidationError("metadata_json must be valid JSON") from exc
    if not isinstance(metadata, dict):
        raise ValidationError("metadata_json must contain a JSON object")
    content = file.file.read()
    result = runtime.ingest(
        principal.workspace_id,
        corpus_id,
        DocumentInput(
            source_id=source_id,
            title=title or file.filename or source_id,
            media_type=file.content_type or "application/octet-stream",
            content=content,
            metadata=metadata,
        ),
        actor=principal.api_key_id,
    )
    return IngestResponse.model_validate(asdict(result))


@router.post(
    "/v1/corpora/{corpus_id}/documents/image",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
)
def ingest_image_document(
    corpus_id: str,
    principal: WritePrincipal,
    runtime: RuntimeDependency,
    file: Annotated[UploadFile, File()],
    source_id: Annotated[str, Form(min_length=1, max_length=512)],
    title: Annotated[str | None, Form(max_length=512)] = None,
    caption: Annotated[str | None, Form(max_length=2000)] = None,
    alt_text: Annotated[str | None, Form(max_length=2000)] = None,
    metadata_json: Annotated[str, Form()] = "{}",
) -> IngestResponse:
    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError as exc:
        raise ValidationError("metadata_json must be valid JSON") from exc
    if not isinstance(metadata, dict):
        raise ValidationError("metadata_json must contain a JSON object")
    content = file.file.read()
    result = runtime.ingest_image(
        principal.workspace_id,
        corpus_id,
        content=content,
        media_type=file.content_type or "application/octet-stream",
        source_id=source_id,
        title=title or file.filename or source_id,
        caption=caption,
        alt_text=alt_text,
        metadata=metadata,
        actor=principal.api_key_id,
    )
    return IngestResponse.model_validate(asdict(result))


@router.post(
    "/v1/corpora/{corpus_id}/documents/source",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
)
def ingest_source_document(
    corpus_id: str,
    body: SourceDocumentRequest,
    principal: WritePrincipal,
    runtime: RuntimeDependency,
) -> IngestResponse:
    result = runtime.ingest_source(
        principal.workspace_id,
        corpus_id,
        body.source,
        source_id=body.source_id,
        title=body.title,
        metadata=body.metadata,
        actor=principal.api_key_id,
    )
    return IngestResponse.model_validate(asdict(result))


@router.get("/v1/corpora/{corpus_id}/documents", tags=["documents"])
def list_documents(
    corpus_id: str,
    principal: ReadPrincipal,
    identity: Identity,
    runtime: RuntimeDependency,
) -> list[dict[str, Any]]:
    documents = runtime.store.list_documents(principal.workspace_id, corpus_id)
    if runtime.access_policy is not None and identity is not None:
        allowed = authorized_document_ids(runtime.access_policy, identity, documents)
        documents = [document for document in documents if str(document["id"]) in allowed]
    return documents


@router.delete(
    "/v1/corpora/{corpus_id}/documents/{document_id}",
    tags=["documents"],
)
def delete_document(
    corpus_id: str,
    document_id: str,
    principal: WritePrincipal,
    runtime: RuntimeDependency,
) -> dict[str, int]:
    version = runtime.ingestion.delete(principal.workspace_id, corpus_id, document_id)
    return {"corpus_version": version}


@router.post(
    "/v1/corpora/{corpus_id}/documents/{document_id}/erase",
    tags=["documents"],
)
def erase_document(
    corpus_id: str,
    document_id: str,
    principal: AdminPrincipal,
    runtime: RuntimeDependency,
) -> dict[str, Any]:
    receipt = runtime.erasure.erase(principal.workspace_id, corpus_id, document_id)
    return {
        "request_id": receipt.request_id,
        "workspace_id": receipt.workspace_id,
        "corpus_id": receipt.corpus_id,
        "document_id": receipt.document_id,
        "status": str(receipt.status),
        "corpus_version": receipt.corpus_version,
        "counts": receipt.counts(),
    }


def _legal_hold_service(runtime: RuntimeDependency, workspace_id: str) -> LegalHoldService:
    return LegalHoldService(
        runtime.store,
        workspace_id,
        audit_sink=runtime.audit_sink,
        audit_fail_closed=runtime.settings.audit_fail_closed,
        deferred_processor=runtime.erasure.deferred_processor,
    )


@router.post("/v1/corpora/{corpus_id}/legal-holds", tags=["legal-holds"])
def place_legal_hold(
    corpus_id: str,
    body: PlaceHoldRequest,
    principal: AdminPrincipal,
    runtime: RuntimeDependency,
) -> dict[str, Any]:
    # The actor is the authenticated admin identity, never a body field, so a client cannot spoof it.
    service = _legal_hold_service(runtime, principal.workspace_id)
    hold_id = service.place(
        corpus_id=corpus_id,
        document_id=body.document_id,
        placed_by=principal.api_key_id,
        reason=body.reason,
    )
    return {"hold_id": hold_id, "corpus_id": corpus_id, "document_id": body.document_id}


@router.get("/v1/corpora/{corpus_id}/legal-holds", tags=["legal-holds"])
def list_legal_holds(
    corpus_id: str,
    principal: AdminPrincipal,
    runtime: RuntimeDependency,
) -> dict[str, Any]:
    service = _legal_hold_service(runtime, principal.workspace_id)
    return {"holds": [asdict(record) for record in service.list_active(corpus_id)]}


@router.post("/v1/legal-holds/{hold_id}/lift", tags=["legal-holds"])
def lift_legal_hold(
    hold_id: str,
    principal: AdminPrincipal,
    runtime: RuntimeDependency,
) -> dict[str, Any]:
    service = _legal_hold_service(runtime, principal.workspace_id)
    lifted = service.lift(hold_id, lifted_by=principal.api_key_id)
    return {"hold_id": hold_id, "lifted": lifted}


def _kill_switch_service(runtime: RuntimeDependency, workspace_id: str) -> KillSwitchService:
    # The management surface is unavailable when the backend is off, so no route reads the table and the
    # off state is uniform: engage, disengage, and list all fail closed rather than serving an empty read.
    if runtime.kill_switches is None:
        raise NotFoundError("Kill switches are not enabled")
    return KillSwitchService(
        runtime.store,
        workspace_id,
        audit_sink=runtime.audit_sink,
        audit_fail_closed=runtime.settings.audit_fail_closed,
    )


def _parse_kill_switch_scope(scope: str) -> KillSwitchScope:
    try:
        return KillSwitchScope(scope)
    except ValueError as exc:
        raise ValidationError(f"Unknown kill-switch scope: {scope}") from exc


@router.post("/v1/kill-switches", status_code=status.HTTP_201_CREATED, tags=["kill-switches"])
def engage_kill_switch(
    body: EngageKillSwitchRequest,
    principal: AdminPrincipal,
    runtime: RuntimeDependency,
) -> dict[str, str]:
    # The actor is the authenticated admin identity, never a body field. A workspace admin can engage
    # only for its own workspace; a global switch and lifting a workspace freeze are operator-only (CLI),
    # since a workspace freeze darkens its own HTTP surface through get_principal.
    service = _kill_switch_service(runtime, principal.workspace_id)
    switch_id = service.engage(
        scope=_parse_kill_switch_scope(body.scope),
        engaged_by=principal.api_key_id,
        reason=body.reason,
    )
    return {"switch_id": switch_id}


@router.get("/v1/kill-switches", tags=["kill-switches"])
def list_kill_switches(
    principal: AdminPrincipal,
    runtime: RuntimeDependency,
) -> dict[str, Any]:
    service = _kill_switch_service(runtime, principal.workspace_id)
    return {"switches": [asdict(record) for record in service.list_engaged()]}


@router.post("/v1/kill-switches/{scope}/disengage", tags=["kill-switches"])
def disengage_kill_switch(
    scope: str,
    principal: AdminPrincipal,
    runtime: RuntimeDependency,
) -> dict[str, Any]:
    service = _kill_switch_service(runtime, principal.workspace_id)
    disengaged = service.disengage(
        scope=_parse_kill_switch_scope(scope),
        disengaged_by=principal.api_key_id,
    )
    return {"scope": scope, "disengaged": disengaged}


@router.get("/v1/status", tags=["status"])
def get_status(principal: AdminPrincipal, runtime: RuntimeDependency) -> StatusResponse:
    # A read-only, redacted posture view. The workspace is derived server-side from the admin key,
    # never a body or query field, and the HTTP view is workspace-scoped (operator=False), so the
    # global kill-switch sentinel and the process-wide governance inventory are never exposed here.
    # It funnels through get_principal like every /v1 route, so a frozen workspace's status darkens
    # at the edge; the operator CLI is the during-freeze surface. The service never raises, so the
    # body is always a 200 posture; only the scope guard (403) and the edge freeze (503) short-circuit.
    plane = StatusService(runtime).snapshot(workspace_id=principal.workspace_id, operator=False)
    return StatusResponse(**plane.to_json())


@router.post("/v1/corpora/{corpus_id}/retrieve", tags=["query"])
def retrieve_corpus(
    corpus_id: str,
    body: QueryBody,
    principal: ReadPrincipal,
    identity: Identity,
    runtime: RuntimeDependency,
) -> dict[str, Any]:
    context = runtime.retrieve(
        QueryRequest(
            workspace_id=principal.workspace_id,
            corpus_id=corpus_id,
            query=body.query,
            user_id=body.user_id,
            session_id=body.session_id,
            top_k=body.top_k,
            token_budget=body.token_budget,
            metadata=body.metadata,
            principal=identity,
        )
    )
    return {
        "query": context.query,
        "routes": list(context.routes),
        "graph_facts": list(context.graph_facts),
        "session_summary": context.session_summary,
        "token_count": context.token_count,
        "hits": [
            {
                "chunk_id": hit.chunk.id,
                "document_id": hit.chunk.document_id,
                "text": hit.chunk.text,
                "metadata": hit.chunk.metadata,
                "score": hit.score,
                "semantic_score": hit.semantic_score,
                "lexical_score": hit.lexical_score,
                "graph_score": hit.graph_score,
                "reasons": list(hit.reasons),
            }
            for hit in context.hits
        ],
    }


@router.post(
    "/v1/corpora/{corpus_id}/query",
    response_model=QueryResultResponse,
    tags=["query"],
)
def query_corpus(
    corpus_id: str,
    body: QueryBody,
    principal: ReadPrincipal,
    identity: Identity,
    runtime: RuntimeDependency,
) -> QueryResultResponse:
    result = runtime.query(
        QueryRequest(
            workspace_id=principal.workspace_id,
            corpus_id=corpus_id,
            query=body.query,
            user_id=body.user_id,
            session_id=body.session_id,
            top_k=body.top_k,
            token_budget=body.token_budget,
            metadata=body.metadata,
            principal=identity,
        )
    )
    return QueryResultResponse.model_validate(asdict(result))


@router.post("/v1/corpora/{corpus_id}/query/stream", tags=["query"])
def query_corpus_stream(
    corpus_id: str,
    body: QueryBody,
    principal: ReadPrincipal,
    identity: Identity,
    runtime: RuntimeDependency,
) -> StreamingResponse:
    request = QueryRequest(
        workspace_id=principal.workspace_id,
        corpus_id=corpus_id,
        query=body.query,
        user_id=body.user_id,
        session_id=body.session_id,
        top_k=body.top_k,
        token_budget=body.token_budget,
        metadata=body.metadata,
        principal=identity,
    )
    events = runtime.stream_query(request)
    first = next(events, None)

    def event_stream() -> Iterator[str]:
        if first is not None:
            yield f"event: {first['type']}\ndata: {json.dumps(first)}\n\n"
        for event in events:
            yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post(
    "/v1/corpora/{corpus_id}/rlm",
    response_model=RlmResultResponse,
    tags=["rlm"],
)
def run_rlm(
    corpus_id: str,
    body: RlmRunRequest,
    principal: WritePrincipal,
    identity: Identity,
    runtime: RuntimeDependency,
) -> RlmResultResponse:
    result = runtime.run_rlm(
        principal.workspace_id,
        corpus_id,
        body.query,
        user_id=body.user_id,
        session_id=body.session_id,
        limits=RlmLimits(**body.limits.model_dump()),
        principal=identity,
    )
    return RlmResultResponse.model_validate(asdict(result))


@router.post(
    "/v1/corpora/{corpus_id}/rlm/jobs",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["rlm"],
)
def enqueue_rlm(
    corpus_id: str,
    body: RlmRunRequest,
    principal: WritePrincipal,
    identity: Identity,
    runtime: RuntimeDependency,
) -> dict[str, str]:
    # Under a policy this durable path is available only with a PrincipalDirectory installed: the
    # snapshot captured here is re-resolved by the worker at execution time. With a policy but no
    # directory there is no re-resolution seam, so the endpoint stays fail-closed as before. The
    # captured identity is the server-resolved principal, never a request-body field.
    if runtime.access_policy is not None and runtime.principal_directory is None:
        raise AuthorizationError("This endpoint is unavailable while an access policy is active")
    job_id = runtime.enqueue_rlm(
        principal.workspace_id,
        corpus_id,
        body.query,
        user_id=body.user_id,
        session_id=body.session_id,
        limits=RlmLimits(**body.limits.model_dump()),
        principal=identity,
    )
    return {"id": job_id}


@router.post("/v1/corpora/{corpus_id}/feedback", status_code=status.HTTP_204_NO_CONTENT, tags=["query"])
def submit_feedback(
    corpus_id: str,
    body: FeedbackRequest,
    principal: WritePrincipal,
    identity: Identity,
    runtime: RuntimeDependency,
) -> None:
    runtime.submit_feedback(
        principal.workspace_id,
        corpus_id,
        Feedback(
            query_id=body.query_id,
            user_id=body.user_id,
            rating=body.rating,
            selected_chunk_ids=tuple(body.selected_chunk_ids),
            note=body.note,
            principal=identity,
        ),
    )


@router.post("/v1/sessions", status_code=status.HTTP_201_CREATED, tags=["sessions"])
def create_session(
    body: SessionCreateRequest,
    principal: WritePrincipal,
    runtime: RuntimeDependency,
) -> dict[str, str]:
    session_id = runtime.store.create_session(
        principal.workspace_id,
        body.corpus_id,
        user_id=body.user_id,
        goal=body.goal,
        metadata_value=body.metadata,
    )
    return {"id": session_id}


@router.get("/v1/sessions/{session_id}", tags=["sessions"])
def get_session(
    session_id: str,
    principal: ReadPrincipal,
    _guard: ProgramSurfaceGuard,
    runtime: RuntimeDependency,
) -> dict[str, Any]:
    session = runtime.store.get_session(principal.workspace_id, session_id)
    session["turns"] = runtime.store.list_session_turns(principal.workspace_id, session_id)
    return session


@router.post("/v1/jobs", status_code=status.HTTP_202_ACCEPTED, tags=["jobs"])
def create_job(
    body: JobCreateRequest,
    principal: WritePrincipal,
    _guard: ProgramSurfaceGuard,
    runtime: RuntimeDependency,
) -> dict[str, str]:
    if body.kind in RESERVED_JOB_KINDS:
        raise ValidationError(f"Job kind {body.kind!r} must be enqueued through its governed endpoint")
    job_id = runtime.store.enqueue_job(
        principal.workspace_id,
        body.kind,
        body.payload,
        max_attempts=body.max_attempts,
    )
    return {"id": job_id}


def _audit_obscured_read(
    runtime: Refaat,
    workspace_id: str,
    identity: AuthzPrincipal | None,
    job_id: str,
    reason: str,
) -> None:
    """Record the true reason a governed job read was refused while the client sees a uniform 404.

    Best-effort and a no-op without a sink, so the obscured read never fails on the audit path and an
    unconfigured runtime is unchanged. Redacted by construction: the reason code, a digest of the
    probed id, and the reader identity, never the job payload.
    """
    sink = runtime.audit_sink
    policy = runtime.access_policy
    if sink is None or policy is None:
        return
    # Best-effort in full: event construction (clock, policy, identity) is inside the guard too, so a
    # throwing property can never surface a distinguishing 500 from the obscured read.
    try:
        reader_id = identity.id if identity is not None else "system:unresolved"
        reader_kind = identity.kind.value if identity is not None else "system"
        tenant = identity.tenant if identity is not None else None
        event = AuditEvent(
            event_type=f"job.read.denied:{reason}",
            occurred_at=runtime.store.clock.now(),
            workspace_id=workspace_id,
            principal_id=reader_id,
            principal_kind=reader_kind,
            tenant=tenant,
            policy_id=str(getattr(policy, "name", type(policy).__name__)),
            policy_version=str(getattr(policy, "version", "")),
            dedup_id=stable_digest((workspace_id, job_id, reason, reader_id)),
        )
        emit_audit(sink, event, fail_closed=False, logger=logger)
    except Exception:
        logger.warning("obscured_read_audit_failed", reason=reason)


@router.get("/v1/jobs/{job_id}", tags=["jobs"])
def get_job(
    job_id: str,
    principal: ReadPrincipal,
    identity: OptionalIdentity,
    runtime: RuntimeDependency,
) -> dict[str, Any]:
    policy = runtime.access_policy
    if policy is None:
        return runtime.store.get_job(principal.workspace_id, job_id)
    if runtime.principal_directory is None:
        raise AuthorizationError("This endpoint is unavailable while an access policy is active")
    # Resolve identity for attribution only (never fail closed at the boundary): an unresolved identity
    # must fall into the same obscured not-found below, not surface a distinguishable 403.
    # A job read must not become an existence oracle: an existing job the caller may not read and a
    # missing id return the identical not-found the store raises, while the true reason is audited.
    try:
        job = runtime.store.get_job(principal.workspace_id, job_id)
    except NotFoundError:
        _audit_obscured_read(runtime, principal.workspace_id, identity, job_id, "not_found")
        raise
    # Only an rlm.run job carrying a server-side snapshot is principal-readable, and only by the exact
    # execution-time identity bound to the snapshot principal id and its durable entitlement digest.
    payload = job.get("payload")
    raw_snapshot = payload.get("_identity_snapshot") if isinstance(payload, Mapping) else None
    snapshot = snapshot_from_payload(raw_snapshot) if isinstance(raw_snapshot, Mapping) else None
    if (
        job.get("kind") != "rlm.run"
        or snapshot is None
        or identity is None
        or identity.id != snapshot.principal_id
        or durable_entitlement_digest(identity, policy) != snapshot.auth_scope_hash
    ):
        _audit_obscured_read(runtime, principal.workspace_id, identity, job_id, "not_authorized")
        raise NotFoundError("Job not found")
    return job


@router.post("/v1/schedules", status_code=status.HTTP_201_CREATED, tags=["schedules"])
def create_schedule(
    body: ScheduleCreateRequest,
    principal: WritePrincipal,
    _guard: ProgramSurfaceGuard,
    runtime: RuntimeDependency,
) -> dict[str, str]:
    if body.job_kind in RESERVED_JOB_KINDS:
        raise ValidationError(f"Job kind {body.job_kind!r} must be enqueued through its governed endpoint")
    schedule_id = runtime.store.create_schedule(
        principal.workspace_id,
        body.name,
        body.job_kind,
        body.payload,
        body.interval_seconds,
        enabled=body.enabled,
    )
    return {"id": schedule_id}


@router.post("/v1/agents", status_code=status.HTTP_201_CREATED, tags=["agents"])
def create_agent(
    body: AgentCreateRequest,
    principal: WritePrincipal,
    identity: Identity,
    runtime: RuntimeDependency,
) -> dict[str, str]:
    agent_id = runtime.agents.create(
        principal.workspace_id,
        body.name,
        body.kind,
        goal=body.goal,
        state=body.state,
        retained=body.retained,
        parent_agent_id=body.parent_agent_id,
        principal=identity,
    )
    return {"id": agent_id}


@router.post("/v1/agents/{agent_id}/heartbeat", status_code=status.HTTP_204_NO_CONTENT, tags=["agents"])
def heartbeat_agent(
    agent_id: str,
    body: AgentHeartbeatRequest,
    principal: WritePrincipal,
    runtime: RuntimeDependency,
) -> None:
    runtime.agents.heartbeat(principal.workspace_id, agent_id, body.state)


@router.post("/v1/agents/{sender_agent_id}/messages", status_code=status.HTTP_201_CREATED, tags=["agents"])
def send_agent_message(
    sender_agent_id: str,
    body: AgentMessageRequest,
    principal: WritePrincipal,
    identity: Identity,
    runtime: RuntimeDependency,
) -> dict[str, Any]:
    return asdict(
        runtime.agents.send(
            principal.workspace_id,
            sender_agent_id,
            body.recipient_agent_id,
            body.kind,
            body.payload,
            body.correlation_id,
            principal=identity,
        )
    )


@router.get("/v1/agents/{agent_id}/messages", tags=["agents"])
def receive_agent_messages(
    agent_id: str,
    principal: ReadPrincipal,
    identity: Identity,
    runtime: RuntimeDependency,
    limit: int = 50,
) -> list[dict[str, Any]]:
    return [
        asdict(message)
        for message in runtime.agents.receive(
            principal.workspace_id, agent_id, min(max(limit, 1), 100), principal=identity
        )
    ]


@router.post(
    "/v1/agent-messages/{message_id}/acknowledge",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["agents"],
)
def acknowledge_agent_message(
    message_id: str,
    principal: WritePrincipal,
    identity: Identity,
    runtime: RuntimeDependency,
) -> None:
    runtime.agents.acknowledge(principal.workspace_id, message_id, principal=identity)


@router.get("/v1/harness/active", tags=["harness"])
def active_harness(
    principal: ReadPrincipal,
    runtime: RuntimeDependency,
    corpus_id: str | None = None,
) -> dict[str, Any]:
    snapshot = runtime.store.active_harness(principal.workspace_id, corpus_id)
    return asdict(snapshot) if snapshot else {"active": False, "supplemental": {}}


@router.post("/v1/harness/snapshots", status_code=status.HTTP_201_CREATED, tags=["harness"])
def create_harness_snapshot(
    body: HarnessSnapshotRequest,
    principal: AdminPrincipal,
    runtime: RuntimeDependency,
) -> dict[str, Any]:
    return asdict(
        runtime.harness.create_snapshot(
            workspace_id=principal.workspace_id,
            corpus_id=body.corpus_id,
            supplemental=body.supplemental,
            evidence=body.evidence,
            activate=body.activate,
        )
    )


@router.post("/v1/harness/snapshots/{snapshot_id}/activate", tags=["harness"])
def activate_harness_snapshot(
    snapshot_id: str,
    principal: AdminPrincipal,
    runtime: RuntimeDependency,
) -> dict[str, Any]:
    return asdict(runtime.harness.activate_snapshot(principal.workspace_id, snapshot_id))


@router.post("/v1/harness/proposals/from-evaluation/{run_id}", tags=["harness"])
def propose_harness_from_evaluation(
    run_id: str,
    principal: AdminPrincipal,
    _guard: ProgramSurfaceGuard,
    runtime: RuntimeDependency,
) -> dict[str, Any]:
    run = runtime.store.get_evaluation_run(principal.workspace_id, run_id)
    return {"supplemental": runtime.harness.propose_from_metrics(run["metrics"]), "evidence": {"run_id": run_id}}


@router.post("/v1/evaluations/suites", status_code=status.HTTP_201_CREATED, tags=["evaluations"])
def create_evaluation_suite(
    body: EvaluationSuiteRequest,
    principal: WritePrincipal,
    runtime: RuntimeDependency,
) -> dict[str, str]:
    suite_id = runtime.store.create_evaluation_suite(
        principal.workspace_id,
        body.name,
        [case.model_dump() for case in body.cases],
    )
    return {"id": suite_id}


@router.post("/v1/evaluations/suites/{suite_id}/runs", tags=["evaluations"])
def run_evaluation(
    suite_id: str,
    body: EvaluationRunRequest,
    principal: WritePrincipal,
    _guard: ProgramSurfaceGuard,
    runtime: RuntimeDependency,
) -> dict[str, str]:
    if body.asynchronous:
        job_id = runtime.store.enqueue_job(
            principal.workspace_id,
            "evaluation.run",
            {"suite_id": suite_id, "corpus_id": body.corpus_id},
        )
        return {"job_id": job_id}
    return {"run_id": runtime.evaluations.run(principal.workspace_id, suite_id, body.corpus_id)}


@router.get("/v1/evaluations/runs/{run_id}", tags=["evaluations"])
def get_evaluation_run(
    run_id: str,
    principal: ReadPrincipal,
    _guard: ProgramSurfaceGuard,
    runtime: RuntimeDependency,
) -> dict[str, Any]:
    return runtime.store.get_evaluation_run(principal.workspace_id, run_id)


@router.get("/v1/skills", tags=["skills"])
def list_skills(principal: ReadPrincipal, runtime: RuntimeDependency) -> list[dict[str, Any]]:
    del principal
    return runtime.skill_registry.list()


@router.post("/v1/skills/{skill_name}/run", tags=["skills"])
def run_skill(
    skill_name: str,
    body: SkillRunRequest,
    principal: AdminPrincipal,
    identity: Identity,
    _guard: ProgramSurfaceGuard,
    _ambient: AmbientApprovalGuard,
    runtime: RuntimeDependency,
) -> dict[str, Any]:
    # Admin scoped because this runs registered skill code directly. Admission lives at the
    # SkillService.run chokepoint: an installed DirectSkillPolicy governs by capability and refuses a
    # requires_approval skill, and the skill_requests quota is charged there so the MCP surface is covered too.
    # The resolved identity scopes the skill's retrieval and its quota audit, mirroring in-process run_skill.
    return runtime.skills.run(
        skill_name,
        QueryRequest(
            workspace_id=principal.workspace_id,
            corpus_id=body.corpus_id,
            query=body.query,
            user_id=body.user_id,
            session_id=body.session_id,
            principal=identity,
        ),
        body.arguments,
    )


@router.post("/v1/skills/projects", status_code=status.HTTP_201_CREATED, tags=["skills"])
def create_skill_project(
    body: SkillCreateRequest,
    principal: AdminPrincipal,
    runtime: RuntimeDependency,
) -> dict[str, str]:
    del principal
    path = runtime.skill_creator.create(body.name, body.description)
    return {"path": str(path)}


@router.get("/v1/corpora/{corpus_id}/programs", tags=["programs"])
def list_programs(
    corpus_id: str,
    principal: ReadPrincipal,
    runtime: RuntimeDependency,
) -> list[dict[str, Any]]:
    project = RefaatProject(runtime, ProjectRef(principal.workspace_id, corpus_id))
    return project.programs.list()


def _require_side_effect_scope(program: Program, principal: Principal) -> None:
    # A program that could execute a write or external tool needs more than read scope on the HTTP surface,
    # so a read-scoped caller cannot invoke a program's self-declared side-effectful grants.
    if program.declared_side_effect().rank > SideEffectClass.READ_ONLY.rank and not (
        "write" in principal.scopes or "admin" in principal.scopes
    ):
        raise AuthorizationError("This program declares side effects and requires the 'write' scope")


@router.post("/v1/corpora/{corpus_id}/programs/{name}/run", tags=["programs"])
def run_program(
    corpus_id: str,
    name: str,
    body: ProgramRunRequest,
    principal: ReadPrincipal,
    identity: Identity,
    _ambient: AmbientApprovalGuard,
    runtime: RuntimeDependency,
) -> dict[str, Any]:
    project = RefaatProject(runtime, ProjectRef(principal.workspace_id, corpus_id))
    bound = project.programs.get(name)
    _require_side_effect_scope(bound.program, principal)
    runtime.charge_program_request(principal.workspace_id, corpus_id, identity)
    result = bound.run(question=body.question, user_id=body.user_id, session_id=body.session_id, principal=identity)
    return result.to_json()


@router.post("/v1/corpora/{corpus_id}/programs/{name}/run/stream", tags=["programs"])
def run_program_stream(
    corpus_id: str,
    name: str,
    body: ProgramRunRequest,
    principal: ReadPrincipal,
    identity: Identity,
    _ambient: AmbientApprovalGuard,
    runtime: RuntimeDependency,
) -> StreamingResponse:
    project = RefaatProject(runtime, ProjectRef(principal.workspace_id, corpus_id))
    bound = project.programs.get(name)
    _require_side_effect_scope(bound.program, principal)
    runtime.charge_program_request(principal.workspace_id, corpus_id, identity)
    events = bound.stream(question=body.question, user_id=body.user_id, session_id=body.session_id, principal=identity)

    def event_stream() -> Iterator[str]:
        # The response status is already 200 once the first byte is flushed, so a
        # mid-stream failure is surfaced as a terminal error event rather than a
        # silently truncated stream.
        try:
            for event in events:
                yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
        except ToolTimeoutError as exc:
            # Parity with the sync 409 body plus the tool name; a 409 status cannot ride an already-200
            # SSE stream, so outcome=unknown + execution_terminated + retryable=False carry the reconcile
            # signal. Redacted by construction, and the timeout raises before any answer byte is served.
            logger.warning("program_stream_tool_timeout", program=name, tool=exc.tool)
            error = {
                "type": "error",
                "code": exc.code,
                "message": str(exc),
                "outcome": exc.outcome,
                "side_effect": exc.side_effect,
                "tool": exc.tool,
                "execution_terminated": exc.execution_terminated,
                "retryable": False,
                "trace_id": current_trace_id(),
            }
            yield f"event: error\ndata: {json.dumps(error)}\n\n"
        except Exception:
            logger.exception("program_stream_failed", program=name)
            error = {"type": "error", "detail": "stream failed", "trace_id": current_trace_id()}
            yield f"event: error\ndata: {json.dumps(error)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/v1/corpora/{corpus_id}/programs/{name}/evaluate", tags=["programs"])
def evaluate_program(
    corpus_id: str,
    name: str,
    body: ProgramEvaluateRequest,
    principal: ReadPrincipal,
    identity: Identity,
    _guard: ProgramSurfaceGuard,
    _ambient: AmbientApprovalGuard,
    runtime: RuntimeDependency,
) -> dict[str, Any]:
    project = RefaatProject(runtime, ProjectRef(principal.workspace_id, corpus_id))
    bound = project.programs.get(name)
    # Evaluation executes the program over every case, so a side-effectful program is gated like a run.
    _require_side_effect_scope(bound.program, principal)
    runtime.charge_program_request(principal.workspace_id, corpus_id, identity)
    dataset = Dataset.from_records("api-dataset", body.cases)
    return bound.evaluate(dataset, default_bundle()).to_json()


@router.post(
    "/v1/programs/{name}/optimizations",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["programs"],
)
def enqueue_optimization(
    name: str,
    body: ProgramOptimizeRequest,
    principal: WritePrincipal,
    identity: Identity,
    _guard: ProgramSurfaceGuard,
    runtime: RuntimeDependency,
) -> dict[str, str]:
    project = RefaatProject(runtime, ProjectRef(principal.workspace_id, body.corpus_id))
    bound = project.programs.get(name)
    _require_side_effect_scope(bound.program, principal)
    # An optimization run spawns many candidate evaluations, so cap the enqueue rate on its own dimension.
    runtime.charge_optimization_request(principal.workspace_id, body.corpus_id, identity)
    dataset = Dataset.from_records("api-dataset", body.cases)
    space = [Tunables.from_json(item) for item in body.search_space] if body.search_space is not None else None
    run_id = bound.enqueue_optimization(dataset, search_space=space, budget=body.budget)
    return {"id": run_id}


@router.get("/v1/optimizations/{run_id}", tags=["programs"])
def get_optimization_run(
    run_id: str,
    principal: ReadPrincipal,
    _guard: ProgramSurfaceGuard,
    runtime: RuntimeDependency,
) -> dict[str, Any]:
    return runtime.store.get_optimization_run(principal.workspace_id, run_id)


@router.post(
    "/v1/fine-tunes",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["fine-tuning"],
)
def request_fine_tune(
    body: FineTuneRequestBody,
    principal: AdminPrincipal,
    identity: Identity,
    runtime: RuntimeDependency,
) -> dict[str, str]:
    runtime.charge_fine_tune_request(principal.workspace_id, identity)
    dataset = Dataset.from_records("api-fine-tune", body.cases)
    run_id = runtime.fine_tuning.request(
        principal.workspace_id,
        dataset,
        base_model=body.base_model,
        hyperparameters=body.hyperparameters,
        requester=principal.api_key_id,
    )
    return {"id": run_id}


@router.post("/v1/fine-tunes/{run_id}/approve", tags=["fine-tuning"])
def approve_fine_tune(
    run_id: str,
    principal: AdminPrincipal,
    runtime: RuntimeDependency,
) -> dict[str, Any]:
    # The approver is the authenticated server-resolved identity, never a client-supplied value, so
    # four-eyes cannot be forged; the service also refuses a requester approving their own run.
    return runtime.fine_tuning.approve(principal.workspace_id, run_id, principal.api_key_id)


@router.post(
    "/v1/fine-tunes/{run_id}/enqueue",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["fine-tuning"],
)
def enqueue_fine_tune(
    run_id: str,
    principal: AdminPrincipal,
    runtime: RuntimeDependency,
) -> dict[str, str]:
    # The fine-tune lifecycle is charged once at the request admission, so enqueuing an already
    # requested and approved run is not re-charged (it could otherwise strand an approved run).
    return {"job_id": runtime.fine_tuning.enqueue(principal.workspace_id, run_id)}


@router.get("/v1/fine-tunes/{run_id}", tags=["fine-tuning"])
def get_fine_tune(
    run_id: str,
    principal: ReadPrincipal,
    runtime: RuntimeDependency,
) -> dict[str, Any]:
    return runtime.store.get_fine_tune_run(principal.workspace_id, run_id)


@router.post("/v1/corpora/{corpus_id}/programs/{name}/promote", tags=["programs"])
def promote_program(
    corpus_id: str,
    name: str,
    body: ProgramEnvironmentRequest,
    principal: AdminPrincipal,
    runtime: RuntimeDependency,
) -> dict[str, Any]:
    project = RefaatProject(runtime, ProjectRef(principal.workspace_id, corpus_id))
    artifact = project.programs.get(name).program.artifact()
    project.registry.save(artifact)
    return project.registry.promote(name, body.environment, artifact.artifact_id)


@router.post("/v1/corpora/{corpus_id}/programs/{name}/approve", tags=["programs"])
def approve_program(
    corpus_id: str,
    name: str,
    body: ProgramEnvironmentRequest,
    principal: AdminPrincipal,
    runtime: RuntimeDependency,
) -> dict[str, Any]:
    project = RefaatProject(runtime, ProjectRef(principal.workspace_id, corpus_id))
    artifact = project.programs.get(name).program.artifact()
    project.registry.save(artifact)
    return project.registry.approve(name, body.environment, artifact.artifact_id, principal.api_key_id)


@router.get("/v1/programs/{name}/deployments/{environment}", tags=["programs"])
def resolve_program(
    name: str,
    environment: str,
    principal: ReadPrincipal,
    runtime: RuntimeDependency,
) -> dict[str, Any]:
    return runtime.registry.resolve(principal.workspace_id, name, environment)


@router.post("/v1/programs/{name}/rollback", tags=["programs"])
def rollback_program(
    name: str,
    body: ProgramEnvironmentRequest,
    principal: AdminPrincipal,
    runtime: RuntimeDependency,
) -> dict[str, Any]:
    return runtime.registry.rollback(principal.workspace_id, name, body.environment)


@router.post("/v1/corpora/{corpus_id}/programs/{name}/canary", tags=["programs"])
def set_canary(
    corpus_id: str,
    name: str,
    body: CanarySetRequest,
    principal: AdminPrincipal,
    runtime: RuntimeDependency,
) -> dict[str, Any]:
    # Route through the guarded deployment so a canary is refused for a program whose unserved run
    # would leak side effects, matching the programmatic surface; the run-time backstop still holds.
    project = RefaatProject(runtime, ProjectRef(principal.workspace_id, corpus_id))
    return project.deployment.set_canary(
        name, body.environment, body.candidate_artifact_id, percent=body.percent, compare=body.compare
    )


@router.delete("/v1/corpora/{corpus_id}/programs/{name}/canary", tags=["programs"])
def clear_canary(
    corpus_id: str,
    name: str,
    environment: str,
    principal: AdminPrincipal,
    runtime: RuntimeDependency,
) -> dict[str, Any]:
    del corpus_id
    return runtime.registry.clear_canary(principal.workspace_id, name, environment)


@router.get("/v1/programs/{name}/canary/{environment}", tags=["programs"])
def canary_status(
    name: str,
    environment: str,
    principal: ReadPrincipal,
    runtime: RuntimeDependency,
) -> dict[str, Any]:
    return {
        "canary": runtime.registry.current_canary(principal.workspace_id, name, environment),
        "shadow_runs": runtime.store.shadow_runs(principal.workspace_id, name, environment),
    }


@router.post("/v1/corpora/{corpus_id}/programs/{name}/deployments/{environment}/run", tags=["programs"])
def run_deployment(
    corpus_id: str,
    name: str,
    environment: str,
    body: ProgramRunRequest,
    principal: ReadPrincipal,
    identity: Identity,
    _ambient: AmbientApprovalGuard,
    runtime: RuntimeDependency,
) -> dict[str, Any]:
    project = RefaatProject(runtime, ProjectRef(principal.workspace_id, corpus_id))
    _require_side_effect_scope(project.programs.get(name).program, principal)
    runtime.charge_program_request(principal.workspace_id, corpus_id, identity)
    result = project.deployment.run(
        name,
        environment,
        inputs={"question": body.question},
        user_id=body.user_id,
        session_id=body.session_id,
        principal=identity,
    )
    return result.to_json()
