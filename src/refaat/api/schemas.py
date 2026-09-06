from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BootstrapWorkspaceRequest(BaseModel):
    slug: str = Field(min_length=3, max_length=96, pattern=r"^[a-z0-9][a-z0-9-]+$")
    name: str = Field(min_length=1, max_length=200)
    key_name: str = Field(default="bootstrap-admin", min_length=1, max_length=120)


class BootstrapWorkspaceResponse(BaseModel):
    workspace_id: str
    api_key_id: str
    api_key: str


class CorpusCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    settings: dict[str, Any] = Field(default_factory=dict)


class CorpusResponse(BaseModel):
    id: str
    workspace_id: str
    name: str
    version: int
    settings: dict[str, Any]
    created_at: datetime


class TextDocumentRequest(BaseModel):
    source_id: str = Field(min_length=1, max_length=512)
    title: str = Field(min_length=1, max_length=512)
    content: str = Field(min_length=1)
    media_type: str = Field(default="text/plain", max_length=160)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceDocumentRequest(BaseModel):
    source: str = Field(min_length=1, max_length=4_096)
    source_id: str | None = Field(default=None, max_length=512)
    title: str | None = Field(default=None, max_length=512)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    document_id: str
    corpus_id: str
    corpus_version: int
    chunks: int
    entities: int
    relations: int
    changed: bool
    checksum: str


class ApprovalGrantRequest(BaseModel):
    # A fingerprint is a one-way sha256 hex digest and a capability name is a bounded identifier, so a
    # client cannot smuggle plaintext into the audit trail through either field. The break-glass reason
    # is bounded and stored on the approval row only; it never enters an audit event.
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    capability_name: str = Field(pattern=r"^[A-Za-z0-9._:-]{1,64}$")
    break_glass: bool = False
    reason: str | None = Field(default=None, max_length=500)


class ApprovalRequestCreate(BaseModel):
    # A fingerprint is a one-way sha256 hex digest and a capability name is a bounded identifier, so
    # no client free text reaches the escalation decision path. The requester is the server-resolved
    # identity, never a body field.
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    capability_name: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9._:-]+$")


class PlaceHoldRequest(BaseModel):
    # placed_by is the server-resolved admin identity, never a body field, so a client cannot spoof
    # the actor. document_id None places a corpus-wide hold; a present value is id-shaped so free text
    # can never reach the audit trail. The reason lives on the row only and never enters an event.
    document_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")
    reason: str | None = Field(default=None, max_length=2000)


class EngageKillSwitchRequest(BaseModel):
    # engaged_by is the server-resolved admin identity, never a body field. The scope is a bounded
    # lowercase identifier the route maps to a known KillSwitchScope. The reason lives on the row only.
    scope: str = Field(pattern=r"^[a-z_]{1,32}$")
    reason: str | None = Field(default=None, max_length=2000)


class QueryBody(BaseModel):
    query: str = Field(min_length=1, max_length=32_000)
    user_id: str | None = Field(default=None, max_length=200)
    session_id: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=100)
    token_budget: int | None = Field(default=None, ge=128, le=100_000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CitationResponse(BaseModel):
    chunk_id: str
    document_id: str
    label: str
    quote: str


class QueryResultResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    routes: list[str]
    cached: bool
    corpus_version: int
    trace_id: str
    usage: dict[str, Any]
    abstained: bool = False
    degraded: bool = False


class RlmLimitsRequest(BaseModel):
    max_depth: int = Field(default=2, ge=0, le=6)
    max_nodes: int = Field(default=16, ge=1, le=128)
    branch_factor: int = Field(default=4, ge=2, le=16)
    leaf_token_budget: int = Field(default=1_500, ge=64, le=100_000)


class RlmRunRequest(BaseModel):
    query: str = Field(min_length=1, max_length=32_000)
    user_id: str | None = Field(default=None, max_length=200)
    session_id: str | None = None
    limits: RlmLimitsRequest = Field(default_factory=RlmLimitsRequest)


class RlmNodeResponse(BaseModel):
    id: str
    parent_id: str | None
    depth: int
    route: str
    input_chunk_count: int
    input_tokens: int
    selected_chunk_ids: list[str]
    cited_chunk_ids: list[str]
    answer_excerpt: str


class RlmResultResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    nodes: list[RlmNodeResponse]
    corpus_version: int
    trace_id: str
    usage: dict[str, Any]
    degraded: bool = False


class FeedbackRequest(BaseModel):
    query_id: str | None = None
    user_id: str | None = Field(default=None, max_length=200)
    rating: Literal[-1, 0, 1]
    selected_chunk_ids: list[str] = Field(default_factory=list, max_length=20)
    note: str | None = Field(default=None, max_length=4_000)


class SessionCreateRequest(BaseModel):
    corpus_id: str
    user_id: str | None = Field(default=None, max_length=200)
    goal: str | None = Field(default=None, max_length=4_000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class JobCreateRequest(BaseModel):
    kind: str = Field(min_length=1, max_length=160)
    payload: dict[str, Any] = Field(default_factory=dict)
    max_attempts: int = Field(default=5, ge=1, le=20)


class ScheduleCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    job_kind: str = Field(min_length=1, max_length=160)
    payload: dict[str, Any] = Field(default_factory=dict)
    interval_seconds: int = Field(ge=1, le=31_536_000)
    enabled: bool = True


class AgentCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    kind: str = Field(min_length=1, max_length=120)
    goal: str | None = Field(default=None, max_length=8_000)
    state: dict[str, Any] = Field(default_factory=dict)
    retained: bool = True
    parent_agent_id: str | None = None


class AgentHeartbeatRequest(BaseModel):
    state: dict[str, Any] | None = None


class AgentMessageRequest(BaseModel):
    recipient_agent_id: str
    kind: str = Field(min_length=1, max_length=120)
    payload: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str | None = Field(default=None, max_length=128)


class HarnessSnapshotRequest(BaseModel):
    corpus_id: str | None = None
    supplemental: dict[str, Any]
    evidence: dict[str, Any]
    activate: bool = False


class EvaluationCase(BaseModel):
    query: str = Field(min_length=1, max_length=32_000)
    expected_terms: list[str] = Field(default_factory=list)
    expected_document_ids: list[str] = Field(default_factory=list)


class EvaluationSuiteRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    cases: list[EvaluationCase] = Field(min_length=1, max_length=10_000)


class EvaluationRunRequest(BaseModel):
    corpus_id: str
    asynchronous: bool = True


class SkillRunRequest(BaseModel):
    corpus_id: str
    query: str = Field(min_length=1, max_length=32_000)
    user_id: str | None = Field(default=None, max_length=200)
    session_id: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)


class SkillCreateRequest(BaseModel):
    name: str
    description: str = Field(min_length=1, max_length=500)


class ApiKeyCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    scopes: list[str] = Field(default_factory=lambda: ["read", "write"])

    @field_validator("scopes")
    @classmethod
    def validate_scopes(cls, value: list[str]) -> list[str]:
        allowed = {"read", "write", "admin"}
        if not value or not set(value) <= allowed:
            raise ValueError("scopes must contain only read, write, or admin")
        return sorted(set(value))


class ProgramRunRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8_192)
    user_id: str | None = Field(default=None, max_length=200)
    session_id: str | None = Field(default=None, max_length=64)


class ProgramEvaluateRequest(BaseModel):
    cases: list[dict[str, Any]] = Field(min_length=1)


class ProgramOptimizeRequest(BaseModel):
    corpus_id: str = Field(min_length=1, max_length=64)
    cases: list[dict[str, Any]] = Field(min_length=1)
    search_space: list[dict[str, Any]] | None = None
    budget: int | None = Field(default=None, ge=1)


class ProgramEnvironmentRequest(BaseModel):
    environment: str = Field(min_length=1, max_length=64)


class FineTuneRequestBody(BaseModel):
    cases: list[dict[str, Any]] = Field(min_length=1)
    base_model: str = Field(min_length=1, max_length=200)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)


class CanarySetRequest(BaseModel):
    environment: str = Field(min_length=1, max_length=64)
    candidate_artifact_id: str = Field(min_length=1, max_length=96)
    percent: int = Field(default=0, ge=0, le=100)
    compare: bool = True


class SubsystemStatusResponse(BaseModel):
    name: str
    state: str
    configured: bool
    detail: dict[str, Any] = Field(default_factory=dict)


class StatusResponse(BaseModel):
    # by_alias serialization keeps the wire key "schema" while avoiding the reserved attribute name.
    model_config = ConfigDict(populate_by_name=True)

    generated_at: datetime
    scope: str
    workspace_id: str | None = None
    build: dict[str, Any]
    migration: dict[str, Any] = Field(alias="schema")
    subsystems: list[SubsystemStatusResponse]
