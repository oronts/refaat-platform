from __future__ import annotations

import mimetypes
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import SecretStr

from refaat.application.agents import AgentService
from refaat.application.approvals import ApprovalService
from refaat.application.composition import (
    OMISSION_REASONS,
    OmittedComponent,
    ReproducedComponent,
    RuntimeComposition,
    ShadowComposition,
)
from refaat.application.deployment import BoundDeployment, DeploymentRouter
from refaat.application.derivation import DerivationService
from refaat.application.fine_tuning import BoundFineTuning, FineTuningService
from refaat.application.legal_hold import LegalHoldService
from refaat.application.quota_charge import charge_served_request
from refaat.application.raptor import RaptorTreeBuilder
from refaat.application.registry import BoundRegistry, RegistryService
from refaat.application.rlm import RecursiveContextService
from refaat.application.status import StatusService
from refaat.clock import Clock, SystemClock
from refaat.config import Settings, get_settings
from refaat.core.chunking import BUILTIN_CHUNKING_STRATEGIES
from refaat.core.harness import HarnessService
from refaat.core.retrieval import AdaptiveRetriever
from refaat.core.text import STOPWORDS_BY_LANGUAGE, _primary_subtag, stable_digest
from refaat.domain.audit import AuditEvent, AuditSink, emit_audit
from refaat.domain.authz import (
    AccessPolicy,
    Classification,
    EntitlementScopeResolver,
    Principal,
    PrincipalDirectory,
    PrincipalResolver,
    require_principal_present,
)
from refaat.domain.context_transform import ContextTransform
from refaat.domain.degradation import (
    DegradationPolicy,
    RefuseAllDegradationPolicy,
    SafeDefaultsDegradationPolicy,
)
from refaat.domain.egress import AllowlistEgressPolicy, EgressPolicy, allowlist_from_config
from refaat.domain.erasure import ErasureReceipt
from refaat.domain.governance import DirectSkillPolicy
from refaat.domain.kill_switch import KillSwitchRegistry, ensure_workspace_active
from refaat.domain.models import (
    ContextPacket,
    Demonstration,
    DocumentInput,
    Feedback,
    HarnessSnapshot,
    IngestResult,
    JsonObject,
    QueryRequest,
    QueryResponse,
    RlmLimits,
    RlmResult,
)
from refaat.domain.output_governance import OutputGovernor
from refaat.domain.plugins import ObservedPlugin, PluginDecision, PluginDescriptor
from refaat.domain.quota import QuotaDimension, QuotaPolicy, quota_rules_from_config
from refaat.domain.resilience import CircuitBreakerConfig
from refaat.domain.routing import (
    CompositeModelRouter,
    ModelRouter,
    ProviderResidency,
    ResidencyPolicy,
    ResidencyRouter,
    ResidencyRule,
    RetentionTerm,
)
from refaat.domain.sandbox import CodePolicy, CodeRequest
from refaat.domain.status import AdminStatus
from refaat.errors import ConfigurationError, ValidationError
from refaat.infrastructure.audit import DurableAuditSink
from refaat.infrastructure.blob import LocalBlobStore, S3BlobStore
from refaat.infrastructure.connectors import ConnectorRegistry, FileConnector, HttpConnector
from refaat.infrastructure.context_transform import PatternRedactor
from refaat.infrastructure.database.store import SQLStore
from refaat.infrastructure.embeddings import DeterministicEmbedder, OpenAICompatibleEmbedder
from refaat.infrastructure.events import LoggingEventSink
from refaat.infrastructure.extractors import ExtractorRegistry
from refaat.infrastructure.fine_tuning import DeterministicFineTuner, OpenAIFineTuningProvider
from refaat.infrastructure.generation import ExtractiveGenerator, OpenAICompatibleGenerator
from refaat.infrastructure.graph import CooccurrenceGraphExtractor, OpenAICompatibleGraphExtractor
from refaat.infrastructure.hype import DeterministicHypotheticalGenerator, ModelHypotheticalGenerator
from refaat.infrastructure.judging import GeneratorJudge
from refaat.infrastructure.kill_switch import DatabaseKillSwitchRegistry
from refaat.infrastructure.model_cache import CachingEmbedder, CachingGenerator, cache_identity_of
from refaat.infrastructure.output_governance import PatternDenylistGovernor
from refaat.infrastructure.plugins import PLUGIN_PIN_GROUPS as _PLUGIN_PIN_GROUPS
from refaat.infrastructure.plugins import PluginObserver, PluginRegistry
from refaat.infrastructure.query_transform import (
    DeterministicHydeQueryTransform,
    DeterministicMultiQueryTransform,
    DeterministicQueryTransform,
    DeterministicRewriteQueryTransform,
    DeterministicStepBackQueryTransform,
    ModelHydeQueryTransform,
    ModelMultiQueryTransform,
    ModelRewriteQueryTransform,
    ModelStepBackQueryTransform,
)
from refaat.infrastructure.quota import DatabaseQuotaPolicy
from refaat.infrastructure.reranking import CrossEncoderReranker, LexicalReranker
from refaat.infrastructure.resilience import (
    InMemoryCircuitBreaker,
    build_circuit_breaker,
    wrap_embedder_breaker,
    wrap_generator_breaker,
)
from refaat.infrastructure.sandbox import DenyingSandbox, SafeExpressionSandbox, SubprocessCodeSandbox
from refaat.infrastructure.search import (
    LocalSearchIndex,
    PgVectorSearchIndex,
    ReconcilableIndex,
    SearchIndexFactory,
    VectorReconcileSink,
)
from refaat.infrastructure.secrets import EnvSecretProvider
from refaat.infrastructure.summarization import DeterministicExtractiveSummarizer, ModelSummarizer
from refaat.observability import configure_logging, configure_telemetry, get_logger
from refaat.ports import (
    BlobStore,
    CacheStore,
    ChunkingStrategy,
    CodeSandbox,
    Connector,
    CrossEncoderScorer,
    DocumentExtractor,
    Embedder,
    EventSink,
    FineTuningProvider,
    Generator,
    GraphExtractor,
    GraphReadRepository,
    HypotheticalQuestionGenerator,
    ImageCaptioner,
    JudgeModel,
    ModelCallCache,
    QueryTransform,
    Reranker,
    SearchIndex,
    SecretProvider,
    Skill,
    Summarizer,
)
from refaat.skills import SkillCreator, SkillRegistry, SkillService

from .erasure import ErasureService
from .services import (
    AuthenticationService,
    EvaluationService,
    FeedbackService,
    IngestionService,
    QueryService,
)

if TYPE_CHECKING:
    from refaat.programs.registry import ProgramRegistry

logger = get_logger(__name__)

_PLUGIN_HASH_PREFIX = 16


def _parse_plugin_pins(raw: Mapping[str, Mapping[str, str]]) -> dict[str, PluginDescriptor]:
    pins: dict[str, PluginDescriptor] = {}
    for key, spec in raw.items():
        if ":" not in key:
            raise ConfigurationError(f"plugin pin key {key!r} must be 'group:name'")
        group, name = key.split(":", 1)
        if group not in _PLUGIN_PIN_GROUPS:
            raise ConfigurationError(f"plugin pin key {key!r} has an unknown entry-point group {group!r}")
        if not name:
            raise ConfigurationError(f"plugin pin key {key!r} must include an entry-point name")
        try:
            descriptor = PluginDescriptor(
                group=group,
                name=name,
                value=str(spec["value"]),
                dist_name=str(spec["dist_name"]),
                dist_version=str(spec["dist_version"]),
                code_hash=str(spec["code_hash"]),
            )
        except KeyError as exc:
            raise ConfigurationError(f"plugin pin {key!r} is missing the required field {exc.args[0]!r}") from exc
        except ValueError as exc:
            raise ConfigurationError(f"plugin pin {key!r} is invalid: {exc}") from exc
        pins[key] = descriptor
    return pins


def _close_resources(resources: tuple[object, ...] | list[object], closed: set[int] | None = None) -> None:
    # Dedup close by object identity across owners: one object registered as both an
    # event sink and a connector is owned by the connector registry too, so the shared
    # already-closed set is threaded into that registry's close to guarantee no object
    # is closed twice.
    closed = set() if closed is None else closed
    for resource in reversed(resources):
        identity = id(resource)
        if identity in closed:
            continue
        closed.add(identity)
        try:
            if isinstance(resource, ConnectorRegistry):
                resource.close(closed)
                continue
            close = getattr(resource, "close", None)
            if callable(close):
                close()
        except Exception:
            logger.exception(
                "runtime_resource_close_failed",
                resource_type=type(resource).__name__,
            )


def _composition_slot_present(value: object) -> bool:
    """A captured refused slot counts as present when it holds an injected instance.

    The four plugin-group slots are tuples that are present only when non-empty; every other slot is
    an optional single object present when not None.
    """

    if isinstance(value, tuple):
        return len(value) > 0
    return value is not None


_MEDIA_TYPES = {
    ".bash": "text/x-shellscript",
    ".c": "text/x-c",
    ".cc": "text/x-c++src",
    ".cfg": "text/plain",
    ".conf": "text/plain",
    ".cpp": "text/x-c++src",
    ".css": "text/css",
    ".csv": "text/csv",
    ".gql": "application/graphql",
    ".go": "text/x-go",
    ".graphql": "application/graphql",
    ".h": "text/x-c",
    ".hpp": "text/x-c++src",
    ".ini": "text/plain",
    ".java": "text/x-java-source",
    ".js": "text/javascript",
    ".json": "application/json",
    ".jsonl": "application/json",
    ".jsx": "text/jsx",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".mjs": "text/javascript",
    ".php": "text/x-php",
    ".py": "text/x-python",
    ".pyi": "text/x-python",
    ".rb": "text/x-ruby",
    ".rs": "text/x-rust",
    ".sh": "text/x-shellscript",
    ".sql": "text/x-sql",
    ".toml": "application/toml",
    ".ts": "text/typescript",
    ".tsx": "text/tsx",
    ".txt": "text/plain",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".zsh": "text/x-shellscript",
}

_SPECIAL_MEDIA_TYPES = {
    "dockerfile": "text/plain",
    "makefile": "text/plain",
    "procfile": "text/plain",
}

_IGNORED_DIRECTORIES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "site",
        "vendor",
    }
)


@dataclass(frozen=True, slots=True)
class ProjectRef:
    workspace_id: str
    corpus_id: str


class RefaatProject:
    def __init__(self, runtime: Refaat, reference: ProjectRef) -> None:
        self.runtime = runtime
        self.reference = reference
        self._program_registry: ProgramRegistry | None = None

    @property
    def programs(self) -> ProgramRegistry:
        if self._program_registry is None:
            from refaat.programs.registry import ProgramRegistry

            self._program_registry = ProgramRegistry(self)
        return self._program_registry

    @property
    def registry(self) -> BoundRegistry:
        return BoundRegistry(self.runtime.registry, self.workspace_id)

    @property
    def deployment(self) -> BoundDeployment:
        return BoundDeployment(self)

    @property
    def fine_tuning(self) -> BoundFineTuning:
        return BoundFineTuning(self.runtime.fine_tuning, self.workspace_id)

    @property
    def judge_model(self) -> JudgeModel | None:
        return self.runtime.judge_model

    @property
    def context_transform(self) -> ContextTransform | None:
        """The runtime's generation-egress redactor, or None when unconfigured.

        Threaded into a governed program's own model egress (the corrective-RAG judge) so its judge
        model sees redacted evidence too. None means no redaction, byte-identical to before.
        """
        return self.runtime.queries.context_transform

    @property
    def workspace_id(self) -> str:
        return self.reference.workspace_id

    @property
    def corpus_id(self) -> str:
        return self.reference.corpus_id

    def ingest(
        self,
        source: str | Path,
        *,
        recursive: bool = True,
        metadata: JsonObject | None = None,
        actor: str | None = None,
    ) -> list[IngestResult]:
        path = Path(source)
        if not path.exists():
            raise ValidationError(f"Ingestion path does not exist: {path}")
        paths = list(self._files(path, recursive=recursive))
        if not paths:
            raise ValidationError("Ingestion path contains no supported documents")
        base = path if path.is_dir() else path.parent
        results: list[IngestResult] = []
        for item in paths:
            media_type = self._media_type(item)
            relative = item.relative_to(base).as_posix()
            results.append(
                self.runtime.ingest(
                    self.workspace_id,
                    self.corpus_id,
                    DocumentInput(
                        source_id=relative,
                        title=item.stem.replace("_", " ").strip() or item.name,
                        media_type=media_type,
                        content=item.read_bytes(),
                        metadata={"source_path": relative, **(metadata or {})},
                    ),
                    actor=actor,
                )
            )
        return results

    def ingest_source(
        self,
        source: str,
        *,
        source_id: str | None = None,
        title: str | None = None,
        metadata: JsonObject | None = None,
        actor: str | None = None,
    ) -> IngestResult:
        return self.runtime.ingest_source(
            self.workspace_id,
            self.corpus_id,
            source,
            source_id=source_id,
            title=title,
            metadata=metadata,
            actor=actor,
        )

    def ingest_image(
        self,
        *,
        content: bytes,
        media_type: str,
        source_id: str,
        title: str | None = None,
        caption: str | None = None,
        alt_text: str | None = None,
        metadata: JsonObject | None = None,
        actor: str | None = None,
    ) -> IngestResult:
        return self.runtime.ingest_image(
            self.workspace_id,
            self.corpus_id,
            content=content,
            media_type=media_type,
            source_id=source_id,
            title=title,
            caption=caption,
            alt_text=alt_text,
            metadata=metadata,
            actor=actor,
        )

    def ask(
        self,
        query: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        top_k: int | None = None,
        token_budget: int | None = None,
        fusion: str | None = None,
        rrf_k: int | None = None,
        autocut_enabled: bool | None = None,
        autocut_threshold: float | None = None,
        query_expansion: bool | None = None,
        max_query_variants: int | None = None,
        instructions: str | None = None,
        demonstrations: tuple[Demonstration, ...] = (),
        response_schema: JsonObject | None = None,
        router_id: str | None = None,
        metadata: JsonObject | None = None,
        principal: Principal | None = None,
    ) -> QueryResponse:
        return self.runtime.query(
            QueryRequest(
                workspace_id=self.workspace_id,
                corpus_id=self.corpus_id,
                query=query,
                user_id=user_id,
                session_id=session_id,
                top_k=top_k,
                token_budget=token_budget,
                fusion=fusion,
                rrf_k=rrf_k,
                autocut_enabled=autocut_enabled,
                autocut_threshold=autocut_threshold,
                query_expansion=query_expansion,
                max_query_variants=max_query_variants,
                instructions=instructions,
                demonstrations=demonstrations,
                response_schema=response_schema,
                router_id=router_id,
                metadata=metadata or {},
                principal=principal,
            )
        )

    def retrieve(
        self,
        query: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        top_k: int | None = None,
        token_budget: int | None = None,
        fusion: str | None = None,
        rrf_k: int | None = None,
        autocut_enabled: bool | None = None,
        autocut_threshold: float | None = None,
        query_expansion: bool | None = None,
        max_query_variants: int | None = None,
        metadata: JsonObject | None = None,
        principal: Principal | None = None,
    ) -> ContextPacket:
        return self.runtime.retrieve(
            QueryRequest(
                workspace_id=self.workspace_id,
                corpus_id=self.corpus_id,
                query=query,
                user_id=user_id,
                session_id=session_id,
                top_k=top_k,
                token_budget=token_budget,
                fusion=fusion,
                rrf_k=rrf_k,
                autocut_enabled=autocut_enabled,
                autocut_threshold=autocut_threshold,
                query_expansion=query_expansion,
                max_query_variants=max_query_variants,
                metadata=metadata or {},
                principal=principal,
            )
        )

    def rlm(
        self,
        query: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        limits: RlmLimits | None = None,
        code_policy: CodePolicy | None = None,
        code_requests: Sequence[CodeRequest] = (),
        principal: Principal | None = None,
    ) -> RlmResult:
        return self.runtime.run_rlm(
            self.workspace_id,
            self.corpus_id,
            query,
            user_id=user_id,
            session_id=session_id,
            limits=limits,
            code_policy=code_policy,
            code_requests=code_requests,
            principal=principal,
        )

    def enqueue_rlm(
        self,
        query: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        limits: RlmLimits | None = None,
        principal: Principal | None = None,
    ) -> str:
        return self.runtime.enqueue_rlm(
            self.workspace_id,
            self.corpus_id,
            query,
            user_id=user_id,
            session_id=session_id,
            limits=limits,
            principal=principal,
        )

    def create_session(
        self,
        *,
        user_id: str | None = None,
        goal: str | None = None,
        metadata: JsonObject | None = None,
    ) -> str:
        ensure_workspace_active(self.runtime.kill_switches, self.workspace_id)
        return self.runtime.store.create_session(
            self.workspace_id,
            self.corpus_id,
            user_id=user_id,
            goal=goal,
            metadata_value=metadata or {},
        )

    def list_documents(self) -> list[JsonObject]:
        # Document metadata is stored evidence, so a strict deployment denies an unfiltered in-process
        # listing (no principal to scope it); the HTTP route resolves an identity and filters instead.
        self.runtime.require_principal(None)
        return self.runtime.store.list_documents(self.workspace_id, self.corpus_id)

    def stats(self) -> JsonObject:
        # Corpus counts are stored evidence metadata, so a strict deployment denies an unscoped
        # in-process read, consistent with list_documents.
        self.runtime.require_principal(None)
        return self.runtime.store.stats(self.workspace_id, self.corpus_id)

    def delete_document(self, document_id: str) -> int:
        return self.runtime.ingestion.delete(self.workspace_id, self.corpus_id, document_id)

    def erase(self, document_id: str) -> ErasureReceipt:
        """Completely erase a document and every auto-derived evidence surface it produced."""
        return self.runtime.erasure.erase(self.workspace_id, self.corpus_id, document_id)

    def submit_feedback(
        self,
        *,
        rating: Literal[-1, 0, 1],
        user_id: str | None = None,
        selected_chunk_ids: tuple[str, ...] = (),
        note: str | None = None,
        query_id: str | None = None,
        principal: Principal | None = None,
    ) -> None:
        self.runtime.submit_feedback(
            self.workspace_id,
            self.corpus_id,
            Feedback(
                query_id=query_id,
                user_id=user_id,
                rating=rating,
                selected_chunk_ids=selected_chunk_ids,
                note=note,
                principal=principal,
            ),
        )

    @property
    def approvals(self) -> ApprovalService:
        return ApprovalService(
            self.runtime.store,
            self.workspace_id,
            audit_sink=self.runtime.audit_sink,
            audit_fail_closed=self.runtime.settings.audit_fail_closed,
        )

    @property
    def legal_holds(self) -> LegalHoldService:
        if not self.runtime.store.legal_hold_management_enabled:
            # The ergonomic accessor is only meaningful when the management surface is enabled. Fail
            # fast with a clear configuration error rather than hand back a service whose writes would
            # be refused. An existing hold is still enforced on every runtime regardless of this flag.
            raise ConfigurationError(
                "the legal-hold management surface is not enabled on this runtime; "
                "wire with_legal_hold_registry() to place or lift holds"
            )
        return LegalHoldService(
            self.runtime.store,
            self.workspace_id,
            audit_sink=self.runtime.audit_sink,
            audit_fail_closed=self.runtime.settings.audit_fail_closed,
            deferred_processor=self.runtime.erasure.deferred_processor,
        )

    def run_skill(
        self,
        name: str,
        query: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        arguments: JsonObject | None = None,
        principal: Principal | None = None,
    ) -> JsonObject:
        return self.runtime.skills.run(
            name,
            QueryRequest(
                workspace_id=self.workspace_id,
                corpus_id=self.corpus_id,
                query=query,
                user_id=user_id,
                session_id=session_id,
                principal=principal,
            ),
            arguments,
        )

    def create_evaluation_suite(self, name: str, cases: list[JsonObject]) -> str:
        return self.runtime.store.create_evaluation_suite(self.workspace_id, name, cases)

    def run_evaluation(self, suite_id: str) -> JsonObject:
        run_id = self.runtime.evaluations.run(self.workspace_id, suite_id, self.corpus_id)
        return self.runtime.store.get_evaluation_run(self.workspace_id, run_id)

    def enqueue_evaluation(self, suite_id: str) -> str:
        return self.runtime.store.enqueue_job(
            self.workspace_id,
            "evaluation.run",
            {"suite_id": suite_id, "corpus_id": self.corpus_id},
        )

    def active_harness(self) -> JsonObject:
        return self.runtime.harness.active(self.workspace_id, self.corpus_id)

    def create_harness_snapshot(
        self,
        *,
        supplemental: JsonObject,
        evidence: JsonObject,
        activate: bool = False,
    ) -> HarnessSnapshot:
        return self.runtime.harness.create_snapshot(
            workspace_id=self.workspace_id,
            corpus_id=self.corpus_id,
            supplemental=supplemental,
            evidence=evidence,
            activate=activate,
        )

    def activate_harness_snapshot(self, snapshot_id: str) -> HarnessSnapshot:
        return self.runtime.harness.activate_snapshot(self.workspace_id, snapshot_id)

    def _files(self, path: Path, *, recursive: bool) -> Iterator[Path]:
        if path.is_file():
            if self._is_supported(path):
                yield path
            return
        iterator = path.rglob("*") if recursive else path.glob("*")
        for candidate in sorted(iterator):
            if not candidate.is_file() or candidate.is_symlink():
                continue
            if any(part in _IGNORED_DIRECTORIES for part in candidate.parts):
                continue
            if any(part.startswith(".") for part in candidate.parts):
                continue
            if candidate.stat().st_size > self.runtime.settings.connector_max_bytes:
                continue
            if self._is_supported(candidate):
                yield candidate

    def _is_supported(self, path: Path) -> bool:
        return self._media_type(path) in self.runtime.ingestion.extractors.media_types

    @staticmethod
    def _media_type(path: Path) -> str:
        special = _SPECIAL_MEDIA_TYPES.get(path.name.casefold())
        if special:
            return special
        return _MEDIA_TYPES.get(path.suffix.casefold()) or (
            mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        )


class Refaat:
    settings: Settings
    store: SQLStore
    authentication: AuthenticationService
    ingestion: IngestionService
    erasure: ErasureService
    queries: QueryService
    derivation: DerivationService | None
    feedback: FeedbackService
    evaluations: EvaluationService
    registry: RegistryService
    deployment: DeploymentRouter
    fine_tuning: FineTuningService
    harness: HarnessService
    agents: AgentService
    rlm: RecursiveContextService
    skills: SkillService
    skill_registry: SkillRegistry
    skill_creator: SkillCreator
    connectors: ConnectorRegistry
    event_sinks: tuple[EventSink, ...]
    judge_model: JudgeModel | None
    access_policy: AccessPolicy | None
    principal_resolver: PrincipalResolver | None
    principal_directory: PrincipalDirectory | None
    entitlement_scope_resolver: EntitlementScopeResolver | None
    secret_provider: SecretProvider | None
    kill_switches: KillSwitchRegistry | None
    quota_policy: QuotaPolicy | None
    degradation_policy: DegradationPolicy | None
    egress_policy: EgressPolicy | None
    composition: RuntimeComposition
    _audit_sink: AuditSink | None
    _audit_delivery_sinks: tuple[AuditSink, ...]
    _resources: tuple[object, ...]
    _closed: bool

    def __init__(self, settings: Settings | None = None) -> None:
        built = RuntimeBuilder(settings).build()
        self.__dict__.update(built.__dict__)

    @classmethod
    def _from_services(
        cls,
        *,
        settings: Settings,
        store: SQLStore,
        authentication: AuthenticationService,
        ingestion: IngestionService,
        erasure: ErasureService,
        queries: QueryService,
        feedback: FeedbackService,
        evaluations: EvaluationService,
        registry: RegistryService,
        deployment: DeploymentRouter,
        fine_tuning: FineTuningService,
        harness: HarnessService,
        agents: AgentService,
        rlm: RecursiveContextService,
        skills: SkillService,
        skill_registry: SkillRegistry,
        skill_creator: SkillCreator,
        connectors: ConnectorRegistry,
        event_sinks: tuple[EventSink, ...],
        judge_model: JudgeModel | None,
        access_policy: AccessPolicy | None,
        principal_resolver: PrincipalResolver | None,
        principal_directory: PrincipalDirectory | None,
        entitlement_scope_resolver: EntitlementScopeResolver | None,
        secret_provider: SecretProvider | None,
        derivation_service: DerivationService | None = None,
        audit_sink: AuditSink | None,
        kill_switches: KillSwitchRegistry | None = None,
        quota_policy: QuotaPolicy | None = None,
        degradation_policy: DegradationPolicy | None = None,
        egress_policy: EgressPolicy | None = None,
        audit_delivery_sinks: tuple[AuditSink, ...] = (),
        composition: RuntimeComposition,
        resources: tuple[object, ...],
    ) -> Self:
        runtime = cls.__new__(cls)
        runtime.settings = settings
        runtime.store = store
        runtime.authentication = authentication
        runtime.ingestion = ingestion
        runtime.erasure = erasure
        runtime.queries = queries
        runtime.feedback = feedback
        runtime.evaluations = evaluations
        runtime.registry = registry
        runtime.deployment = deployment
        runtime.fine_tuning = fine_tuning
        runtime.harness = harness
        runtime.agents = agents
        runtime.rlm = rlm
        runtime.skills = skills
        runtime.skill_registry = skill_registry
        runtime.skill_creator = skill_creator
        runtime.connectors = connectors
        runtime.event_sinks = event_sinks
        runtime.judge_model = judge_model
        runtime.access_policy = access_policy
        runtime.principal_resolver = principal_resolver
        runtime.principal_directory = principal_directory
        runtime.entitlement_scope_resolver = entitlement_scope_resolver
        runtime.secret_provider = secret_provider
        runtime.derivation = derivation_service
        runtime.kill_switches = kill_switches
        runtime.quota_policy = quota_policy
        runtime.degradation_policy = degradation_policy
        runtime.egress_policy = egress_policy
        runtime.composition = composition
        runtime._audit_sink = audit_sink
        runtime._audit_delivery_sinks = audit_delivery_sinks
        runtime._resources = resources
        runtime._closed = False
        return runtime

    @property
    def audit_sink(self) -> AuditSink | None:
        return self._audit_sink

    @property
    def audit_delivery_sinks(self) -> tuple[AuditSink, ...]:
        """The real audit sinks a worker's AuditPublisher delivers to under durable audit. Empty
        unless durable audit is enabled with a sink installed; the emission sink then persists
        events for the publisher to deliver here."""
        return self._audit_delivery_sinks

    def status(self, *, workspace_id: str | None = None, operator: bool = False) -> AdminStatus:
        """A redacted, read-only posture snapshot of every governance and operational subsystem.

        Purely additive: it reads the composed seams and existing tables, writes nothing, and folds
        into no artifact or cache identity. A workspace scope reports its own state; the operator
        scope adds the process-wide governance inventory.
        """
        return StatusService(self).snapshot(workspace_id=workspace_id, operator=operator)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def create_workspace(self, slug: str, name: str) -> str:
        return self.store.create_workspace(slug, name).id

    def issue_api_key(
        self,
        workspace_id: str,
        name: str,
        scopes: tuple[str, ...] = ("read", "write", "admin"),
    ) -> tuple[str, str]:
        return self.authentication.issue(workspace_id, name, scopes)

    def create_corpus(
        self,
        workspace_id: str,
        name: str,
        settings: JsonObject | None = None,
    ) -> str:
        return self.store.create_corpus(workspace_id, name, settings).id

    def project(
        self,
        *,
        workspace: str = "default",
        corpus: str = "default",
        workspace_name: str | None = None,
        corpus_settings: JsonObject | None = None,
    ) -> RefaatProject:
        workspace_value = self.store.find_workspace_by_slug(workspace)
        if workspace_value is None:
            workspace_id = self.create_workspace(
                workspace,
                workspace_name or workspace.replace("-", " ").title(),
            )
        else:
            workspace_id = workspace_value.id
        corpus_value = self.store.find_corpus_by_name(workspace_id, corpus)
        if corpus_value is None:
            corpus_id = self.create_corpus(workspace_id, corpus, corpus_settings)
        else:
            corpus_id = corpus_value.id
        return RefaatProject(self, ProjectRef(workspace_id, corpus_id))

    def ingest(
        self,
        workspace_id: str,
        corpus_id: str,
        document: DocumentInput,
        *,
        actor: str | None = None,
    ) -> IngestResult:
        return self.ingestion.ingest(workspace_id, corpus_id, document, actor=actor)

    def ingest_source(
        self,
        workspace_id: str,
        corpus_id: str,
        source: str,
        *,
        source_id: str | None = None,
        title: str | None = None,
        metadata: JsonObject | None = None,
        actor: str | None = None,
    ) -> IngestResult:
        return self.ingestion.ingest_source(
            workspace_id,
            corpus_id,
            source,
            source_id=source_id,
            title=title,
            metadata=metadata,
            actor=actor,
        )

    def ingest_image(
        self,
        workspace_id: str,
        corpus_id: str,
        *,
        content: bytes,
        media_type: str,
        source_id: str,
        title: str | None = None,
        caption: str | None = None,
        alt_text: str | None = None,
        metadata: JsonObject | None = None,
        actor: str | None = None,
    ) -> IngestResult:
        return self.ingestion.ingest_image(
            workspace_id,
            corpus_id,
            content=content,
            media_type=media_type,
            source_id=source_id,
            title=title,
            caption=caption,
            alt_text=alt_text,
            metadata=metadata,
            actor=actor,
        )

    def require_principal(self, principal: Principal | None) -> None:
        """Fail closed under strict authorization when a principal is missing and a policy is active.

        The single in-process guard for surfaces that do not pass through the query, retrieve, or RLM
        entry, so a strict deployment denies a principal-less document listing or program run too.
        """
        require_principal_present(
            self.queries.retriever.access_policy, principal, strict=self.queries.retriever.strict_authorization
        )

    def charge_program_request(self, workspace_id: str, corpus_id: str, principal: Principal | None) -> None:
        # Charged at the served HTTP program routes, not inside BoundProgram.run, because that primitive
        # is reused by evaluation, optimization, the deployment shadow, and program-skills.
        self._charge_served(workspace_id, corpus_id, principal, QuotaDimension.PROGRAM_REQUESTS)

    def charge_optimization_request(self, workspace_id: str, corpus_id: str, principal: Principal | None) -> None:
        self._charge_served(workspace_id, corpus_id, principal, QuotaDimension.OPTIMIZATION_REQUESTS)

    def charge_fine_tune_request(self, workspace_id: str, principal: Principal | None) -> None:
        # Fine-tune carries no corpus, so it passes None: a workspace or principal rule governs it.
        self._charge_served(workspace_id, None, principal, QuotaDimension.FINE_TUNE_REQUESTS)

    def _charge_served(
        self, workspace_id: str, corpus_id: str | None, principal: Principal | None, dimension: QuotaDimension
    ) -> None:
        charge_served_request(
            quota_policy=self.quota_policy,
            kill_switches=self.kill_switches,
            store=self.store,
            audit_sink=self._audit_sink,
            workspace_id=workspace_id,
            corpus_id=corpus_id,
            principal=principal,
            dimension=dimension,
        )

    def query(self, request: QueryRequest) -> QueryResponse:
        return self.queries.query(request)

    def stream_query(self, request: QueryRequest) -> Iterator[dict[str, Any]]:
        return self.queries.stream_query(request)

    def retrieve(self, request: QueryRequest) -> ContextPacket:
        return self.skills.prepare(request)

    def run_rlm(
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
        return self.rlm.run(
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

    def enqueue_rlm(
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
        return self.rlm.enqueue(
            workspace_id,
            corpus_id,
            query,
            user_id=user_id,
            session_id=session_id,
            limits=limits,
            principal=principal,
        )

    def submit_feedback(
        self,
        workspace_id: str,
        corpus_id: str,
        feedback: Feedback,
    ) -> None:
        self.feedback.submit(workspace_id, corpus_id, feedback)

    def close(self) -> None:
        if self._closed:
            return
        try:
            _close_resources(self._resources)
        finally:
            self.store.close()
            self._closed = True


_RETENTION_BY_NAME: dict[str, RetentionTerm] = {
    "zero": RetentionTerm.ZERO,
    "transient": RetentionTerm.TRANSIENT,
    "limited": RetentionTerm.LIMITED,
    "retained": RetentionTerm.RETAINED,
}


def _parse_retention(value: object) -> RetentionTerm | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigurationError("A residency data-retention term must be a string")
    term = _RETENTION_BY_NAME.get(value.strip().lower())
    if term is None:
        raise ConfigurationError(f"Unknown residency data-retention term {value!r}")
    return term


def _parse_classification_name(name: str) -> Classification:
    try:
        return Classification[name.strip().upper()]
    except KeyError as exc:
        raise ConfigurationError(f"Unknown classification {name!r} in the residency policy") from exc


def _residency_string_set(value: object, *, field: str) -> frozenset[str]:
    if not isinstance(value, (list, tuple)):
        raise ConfigurationError(f"Residency rule field {field!r} must be a list of strings")
    return frozenset(str(item).strip() for item in value if str(item).strip())


def _residency_zone_set(value: object) -> frozenset[str]:
    if not isinstance(value, (list, tuple)):
        raise ConfigurationError("Residency rule field 'allowed_zones' must be a list of strings")
    zones: set[str] = set()
    for item in value:
        # Base str methods are called unbound so a str subclass cannot override isprintable or strip
        # to smuggle a sentinel, and str.strip yields a genuine base str; any control char is dropped.
        text = str(item)
        if str.isprintable(text):
            stripped = str.strip(text)
            if stripped:
                zones.add(stripped)
    return frozenset(zones)


def _parse_residency_rule(classification: Classification, config: object) -> ResidencyRule:
    if not isinstance(config, Mapping):
        raise ConfigurationError(f"Residency rule for {classification.name} must be a mapping")
    retention = _parse_retention(config.get("max_retention"))
    if retention is None:
        raise ConfigurationError(f"Residency rule for {classification.name} requires a max_retention term")
    purposes_raw = config.get("allowed_purposes")
    allowed_purposes = None if purposes_raw is None else _residency_string_set(purposes_raw, field="allowed_purposes")
    zones_raw = config.get("allowed_zones")
    allowed_zones = None if zones_raw is None else _residency_zone_set(zones_raw)
    return ResidencyRule(
        allowed_regions=_residency_string_set(config.get("allowed_regions"), field="allowed_regions"),
        max_retention=retention,
        allowed_purposes=allowed_purposes,
        allowed_zones=allowed_zones,
    )


class RuntimeBuilder:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._custom_connectors: list[Connector] = []
        self._custom_extractors: list[DocumentExtractor] = []
        self._custom_skills: list[Skill] = []
        self._custom_event_sinks: list[EventSink] = []
        self._custom_embedder: Embedder | None = None
        self._custom_generator: Generator | None = None
        self._custom_graph_extractor: GraphExtractor | None = None
        self._custom_stopwords: dict[str, frozenset[str]] | None = None
        self._custom_reranker: Reranker | None = None
        self._access_policy: AccessPolicy | None = None
        self._principal_resolver: PrincipalResolver | None = None
        self._principal_directory: PrincipalDirectory | None = None
        self._entitlement_scope_resolver: EntitlementScopeResolver | None = None
        self._secret_provider: SecretProvider | None = None
        self._kill_switch_registry: KillSwitchRegistry | None = None
        self._quota_policy: QuotaPolicy | None = None
        self._egress_policy: EgressPolicy | None = None
        self._legal_hold_registry = False
        self._audit_sink: AuditSink | None = None
        self._model_router: ModelRouter | None = None
        self._output_governor: OutputGovernor | None = None
        self._context_transform: ContextTransform | None = None
        self._degradation_policy: DegradationPolicy | None = None
        self._direct_skill_policy: DirectSkillPolicy | None = None
        self._custom_cross_encoder_scorer: CrossEncoderScorer | None = None
        self._custom_query_transform: QueryTransform | None = None
        self._custom_judge: JudgeModel | None = None
        self._custom_hype_generator: HypotheticalQuestionGenerator | None = None
        self._custom_raptor_builder: RaptorTreeBuilder | None = None
        self._custom_fine_tuning_provider: FineTuningProvider | None = None
        self._custom_image_captioner: ImageCaptioner | None = None
        self._custom_multimodal_generator: Generator | None = None
        self._custom_code_sandbox: CodeSandbox | None = None
        self._custom_chunking_strategy: ChunkingStrategy | None = None
        self._custom_search_index: SearchIndex | SearchIndexFactory | None = None
        self._custom_cache_store: CacheStore | None = None
        self._custom_model_call_cache: ModelCallCache | None = None
        self._custom_graph_read_repository: GraphReadRepository | None = None
        self._circuit_breaker: InMemoryCircuitBreaker | None = None
        self._resolved_circuit_breaker: InMemoryCircuitBreaker | None = None
        self._plugin_pins_override: dict[str, PluginDescriptor] | None = None
        # Ids of production adapters a shadow build reproduced by reference: the shadow evaluates over
        # them but must not close their shared client, so build() skips owning their teardown.
        self._borrowed_component_ids: set[int] = set()

    def with_connector(self, connector: Connector) -> Self:
        self._custom_connectors.append(connector)
        return self

    def with_extractor(self, extractor: DocumentExtractor) -> Self:
        self._custom_extractors.append(extractor)
        return self

    def with_skill(self, skill: Skill) -> Self:
        self._custom_skills.append(skill)
        return self

    def with_event_sink(self, sink: EventSink) -> Self:
        self._custom_event_sinks.append(sink)
        return self

    def with_embedder(self, embedder: Embedder) -> Self:
        self._custom_embedder = embedder
        return self

    def with_generator(self, generator: Generator) -> Self:
        self._custom_generator = generator
        return self

    def with_graph_extractor(self, graph_extractor: GraphExtractor) -> Self:
        self._custom_graph_extractor = graph_extractor
        return self

    def with_stopwords(self, stopwords_by_language: Mapping[str, Iterable[str]]) -> Self:
        """Override or extend the co-occurrence graph extractor's stopword registry.

        Each entry overrides a shipped language ("en", "de") or adds a new one
        ("it", "pt"). Codes are normalized to their primary subtag and words are
        casefolded to match the tokenizer. This is additive: with no injection and
        no ``stopwords_by_language`` setting the extractor uses the module default
        and every unconfigured corpus is byte-identical.
        """
        self._custom_stopwords = {
            _primary_subtag(code): frozenset(word.casefold() for word in words)
            for code, words in stopwords_by_language.items()
        }
        return self

    def with_reranker(self, reranker: Reranker) -> Self:
        self._custom_reranker = reranker
        return self

    def with_access_policy(self, access_policy: AccessPolicy) -> Self:
        """Install the Policy Decision Point that scopes evidence to a request's principal.

        Enforcement activates for any request that carries a principal. With strict authorization
        on (the default), a request under this policy that carries no principal is denied; set
        strict_authorization off for the trusted-in-process behavior where it runs unfiltered. The
        whole runtime with no policy installed is byte-identical to the unauthorized default.
        """
        self._access_policy = access_policy
        return self

    def with_principal_resolver(self, resolver: PrincipalResolver) -> Self:
        """Install the resolver that turns trusted server-side context into a principal.

        The HTTP surface uses it to resolve identity from trusted context only, never from a
        client-supplied body. With no access policy installed the HTTP path carries no principal and
        is byte-identical to the unauthorized default; with a policy installed, a request that
        resolves to no identity is denied at the boundary, so a policy without a resolver denies every
        request.
        """
        self._principal_resolver = resolver
        return self

    def with_principal_directory(self, directory: PrincipalDirectory) -> Self:
        """Install the directory that re-resolves a principal by id at execution time.

        A governed approval consume uses it to re-resolve each approver's current roles, so an
        approver who lost the required role no longer satisfies the approval. With no directory the
        consume uses the grant-time role snapshot, byte-identical to before.
        """
        self._principal_directory = directory
        return self

    def with_entitlement_scope_resolver(self, resolver: EntitlementScopeResolver) -> Self:
        """Install the resolver that maps a principal's entitlements to the scopes it effectively holds.

        Tool governance then admits a program's self-declared grant only when the calling principal
        actually holds its required scopes. With no resolver installed, governance is byte-identical to
        before: a program's declared grants are not intersected with the caller's scopes.
        """
        self._entitlement_scope_resolver = resolver
        return self

    def with_secret_provider(self, provider: SecretProvider) -> Self:
        """Install the provider that resolves capability-declared secrets at tool-execution time.

        A governed tool whose capability declares required_secrets receives the scoped values only
        through this provider, revealed at the side effect and scrubbed after. With no provider
        installed no secret is ever resolved, so the governed loop is byte-identical; a governed step
        whose capability declares a required secret then fails closed rather than running unenforced.
        """
        self._secret_provider = provider
        return self

    def with_legal_hold_registry(self) -> Self:
        """Enable the legal-hold management write surface (place, lift, list) on this runtime.

        Enforcement of an existing hold is state-driven and unconditional: every destructive path
        consults the legal_holds table on every runtime, so this flag does NOT gate enforcement, only
        the ability to record and lift holds. With it unset the runtime enforces any hold that exists
        but cannot place one, and is byte-identical to the default when no hold rows exist."""
        self._legal_hold_registry = True
        return self

    def with_kill_switch_registry(self, registry: KillSwitchRegistry) -> Self:
        """Install a custom incident kill-switch registry whose engaged() snapshot gates each guard.

        The default (kill_switch_backend="none") installs nothing, so every guard short-circuits and
        the runtime is byte-identical. kill_switch_backend="database" installs the store-backed
        registry. This overrides both with an injected registry, for example a Redis-backed one."""
        self._kill_switch_registry = registry
        return self

    def with_quota_policy(self, policy: QuotaPolicy) -> Self:
        """Install a custom persisted per-scope quota policy that gates admission and model spend.

        The default (quota_backend="none") installs nothing, so every guard short-circuits and the
        runtime is byte-identical. quota_backend="database" installs the store-backed policy over the
        configured rule snapshot. This overrides both with an injected policy, for example a
        Redis-backed one."""
        self._quota_policy = policy
        return self

    def with_egress_policy(self, policy: EgressPolicy) -> Self:
        """Install a custom egress policy governing which outbound destinations the runtime reaches.

        The default (egress_policy="none") installs nothing, so every outbound seam's chokepoint is a
        no-op and the runtime is byte-identical. egress_policy="allowlist" installs the default-deny
        AllowlistEgressPolicy over the configured allowlist. This overrides both with an injected
        policy, for example one backed by a central policy service."""
        self._egress_policy = policy
        return self

    def with_audit_sink(self, sink: AuditSink) -> Self:
        """Install the sink that receives redacted audit events for authorized queries.

        With no sink installed no audit events are produced, so the runtime is byte-identical
        to the default.
        """
        self._audit_sink = sink
        return self

    def with_model_router(self, router: ModelRouter) -> Self:
        """Install the data-classification model router that gates generation.

        Before generation both the query and the RLM pipelines derive the evidence
        classification from the retrieved ACL tags and ask the router whether to proceed; a
        denial returns a governed no-generation response. An injected router composes with a
        config-driven residency router. With no router installed and no residency policy the
        pipelines are byte-identical to the default, and the model never decides the classification.
        """
        self._model_router = router
        return self

    def with_output_governor(self, governor: OutputGovernor) -> Self:
        """Install the output governor that reviews a produced answer after generation.

        The query and RLM pipelines review the final answer plus its resolved citations
        before returning or caching it, and refuse (a governed no-output response, never
        cached) or redact on the governor's verdict. The final-answer streaming path buffers
        so no un-recallable answer token is emitted before the verdict; intermediate agentic
        events (a governed ReAct run's tool steps) are outside this seam. With no governor
        installed the pipelines are byte-identical to the default, and the model never decides
        its own review.
        """
        self._output_governor = governor
        return self

    def with_context_transform(self, transform: ContextTransform) -> Self:
        """Install the pre-model context transform that redacts evidence before generation.

        The query and RLM pipelines rewrite the retrieved evidence, graph facts, and session summary
        a generation model would see, replacing configured spans with a placeholder after the routing
        gate and before the generate or stream call; the local extractive fallback keeps the original
        evidence so it grounds and cites real chunks, and a citation still resolves to the original
        stored chunk. It covers the generation model's evidence input only, not embeddings,
        fine-tuning, ingestion-time HyPE or RAPTOR or captioning, or evaluation-time judging. With no
        transform installed the pipelines are byte-identical to the default.
        """
        self._context_transform = transform
        return self

    def with_degradation_policy(self, policy: DegradationPolicy) -> Self:
        """Install the degradation policy that governs a generation provider outage.

        On a hard generation-provider outage at the query or RLM answer seam, the policy decides
        whether to serve a grounded, cited degraded answer, abstain, or refuse. An injected policy
        wins over the config selector. With no policy installed the outage propagates uncaught, so
        the runtime is byte-identical to the default (the outage surfaces as a 503).
        """
        self._degradation_policy = policy
        return self

    def with_direct_skill_policy(self, policy: DirectSkillPolicy) -> Self:
        """Install the admission policy for the direct skill-run surface.

        A skill named by a capability in the policy is admitted against its scopes, side-effect
        ceiling, evidence classification, and payload bound before it runs directly, and refused
        outright when the capability requires approval. A skill with no declared capability, and the
        whole surface with no policy installed, are byte-identical to the ungoverned default.
        """
        self._direct_skill_policy = policy
        return self

    def with_cross_encoder_scorer(self, scorer: CrossEncoderScorer) -> Self:
        self._custom_cross_encoder_scorer = scorer
        return self

    def with_query_transform(self, query_transform: QueryTransform) -> Self:
        self._custom_query_transform = query_transform
        return self

    def with_judge_model(self, judge: JudgeModel) -> Self:
        self._custom_judge = judge
        return self

    def with_hype_generator(self, hype_generator: HypotheticalQuestionGenerator) -> Self:
        self._custom_hype_generator = hype_generator
        return self

    def with_chunking_strategy(self, strategy: ChunkingStrategy) -> Self:
        self._custom_chunking_strategy = strategy
        return self

    def with_search_index(self, index: SearchIndex | SearchIndexFactory) -> Self:
        self._custom_search_index = index
        return self

    def with_cache_store(self, cache_store: CacheStore) -> Self:
        self._custom_cache_store = cache_store
        return self

    def with_model_call_cache(self, cache: ModelCallCache) -> Self:
        self._custom_model_call_cache = cache
        return self

    def with_graph_read_repository(self, repository: GraphReadRepository) -> Self:
        self._custom_graph_read_repository = repository
        return self

    def with_circuit_breaker(self, breaker: InMemoryCircuitBreaker) -> Self:
        self._circuit_breaker = breaker
        return self

    def with_raptor_builder(self, raptor_builder: RaptorTreeBuilder) -> Self:
        self._custom_raptor_builder = raptor_builder
        return self

    def with_fine_tuning_provider(self, provider: FineTuningProvider) -> Self:
        self._custom_fine_tuning_provider = provider
        return self

    def with_image_captioner(self, captioner: ImageCaptioner) -> Self:
        self._custom_image_captioner = captioner
        return self

    def with_multimodal_generator(self, generator: Generator) -> Self:
        self._custom_multimodal_generator = generator
        return self

    def with_code_sandbox(self, sandbox: CodeSandbox) -> Self:
        self._custom_code_sandbox = sandbox
        return self

    def with_plugin_pins(self, pins: Mapping[str, Mapping[str, str]]) -> Self:
        """Pin each allowlisted entry point to an exact target, distribution, and code hash.

        A pin is keyed ``group:name`` and carries ``value``, ``dist_name``, ``dist_version``, and
        ``code_hash``. When any pin is set, every allowlisted entry point across the plugin groups
        must be pinned, and a discovered entry point loads only when it matches its pin. With no
        pins installed and none in settings, discovery is byte-identical to the name-only allowlist.
        """
        self._plugin_pins_override = _parse_plugin_pins(pins)
        return self

    def _plugin_pins(self) -> dict[str, PluginDescriptor]:
        if self._plugin_pins_override is not None:
            return self._plugin_pins_override
        return _parse_plugin_pins(self.settings.plugin_pins)

    def _verify_plugin_pins_total(self, pins: dict[str, PluginDescriptor]) -> None:
        allowlist = set(self.settings.plugin_allowlist)
        if not allowlist:
            return
        missing: list[str] = []
        for group in _PLUGIN_PIN_GROUPS:
            for point in entry_points(group=group):
                if point.name not in allowlist:
                    continue
                key = f"{group}:{point.name}"
                if key not in pins:
                    missing.append(key)
        if missing:
            raise ConfigurationError(
                "plugin pinning is configured but these allowlisted entry points have no pin: "
                + ", ".join(sorted(set(missing)))
            )

    def _plugin_audit_observer(self, clock: SystemClock) -> PluginObserver:
        sink = self._audit_sink
        fail_closed = self.settings.audit_fail_closed

        def observer(observed: ObservedPlugin, decision: PluginDecision) -> None:
            loaded = decision.allowed
            fingerprint = (
                observed.code_hash[:_PLUGIN_HASH_PREFIX]
                if loaded and observed.code_hash is not None
                else decision.reason
            )
            event_type = "plugin.loaded" if loaded else "plugin.rejected"
            event = AuditEvent(
                event_type=event_type,
                occurred_at=clock.now(),
                workspace_id="",
                principal_id="system",
                principal_kind="system",
                tenant=None,
                policy_id="",
                policy_version="",
                capability_name=f"{observed.group}:{observed.name}",
                action_fingerprint=fingerprint,
                dedup_id=stable_digest((event_type, observed.group, observed.name, fingerprint)),
            )
            emit_audit(sink, event, fail_closed=fail_closed, logger=logger, log_key="plugin_audit_sink_failed")

        return observer

    def _capture_composition(self) -> RuntimeComposition:
        """Snapshot every injected component so a shadow build can reproduce or refuse each one."""

        return RuntimeComposition(
            embedder=self._custom_embedder,
            generator=self._custom_generator,
            graph_extractor=self._custom_graph_extractor,
            reranker=self._custom_reranker,
            cross_encoder_scorer=self._custom_cross_encoder_scorer,
            query_transform=self._custom_query_transform,
            judge=self._custom_judge,
            hype_generator=self._custom_hype_generator,
            chunking_strategy=self._custom_chunking_strategy,
            stopwords=self._custom_stopwords,
            search_index=self._custom_search_index,
            cache_store=self._custom_cache_store,
            model_call_cache=self._custom_model_call_cache,
            graph_read_repository=self._custom_graph_read_repository,
            circuit_breaker=self._circuit_breaker,
            raptor_builder=self._custom_raptor_builder,
            audit_sink=self._audit_sink,
            model_router=self._model_router,
            access_policy=self._access_policy,
            principal_resolver=self._principal_resolver,
            principal_directory=self._principal_directory,
            entitlement_scope_resolver=self._entitlement_scope_resolver,
            secret_provider=self._secret_provider,
            kill_switch_registry=self._kill_switch_registry,
            quota_policy=self._quota_policy,
            egress_policy=self._egress_policy,
            output_governor=self._output_governor,
            context_transform=self._context_transform,
            degradation_policy=self._degradation_policy,
            direct_skill_policy=self._direct_skill_policy,
            image_captioner=self._custom_image_captioner,
            multimodal_generator=self._custom_multimodal_generator,
            fine_tuning_provider=self._custom_fine_tuning_provider,
            code_sandbox=self._custom_code_sandbox,
            event_sinks=tuple(self._custom_event_sinks),
            connectors=tuple(self._custom_connectors),
            extractors=tuple(self._custom_extractors),
            skills=tuple(self._custom_skills),
            plugin_pins=self._plugin_pins_override,
        )

    def reproduce_from(self, composition: RuntimeComposition) -> ShadowComposition:
        """Reproduce the safe injected components onto this builder and refuse the unsafe ones.

        Stateless model and text components are re-applied so the shadow evaluates over the same
        composition; the embedder, generator, and graph extractor are borrowed by reference so the
        shadow never closes their shared production client. Stateful backends, store-bound derivation
        services, governance components, and side-effecting adapters are refused and reported.
        """

        reproduced: list[ReproducedComponent] = []
        if composition.embedder is not None:
            # Borrow the raw inner of an injected caching wrapper, never the wrapper: the shadow re-wraps
            # against its own sql cache, so a production cache client can never receive a shadow write.
            embedder = (
                composition.embedder._inner
                if isinstance(composition.embedder, CachingEmbedder)
                else composition.embedder
            )
            self.with_embedder(embedder)
            self._borrowed_component_ids.add(id(embedder))
            reproduced.append(ReproducedComponent("embedder"))
        if composition.generator is not None:
            generator = (
                composition.generator._inner
                if isinstance(composition.generator, CachingGenerator)
                else composition.generator
            )
            self.with_generator(generator)
            self._borrowed_component_ids.add(id(generator))
            reproduced.append(ReproducedComponent("generator"))
        if composition.graph_extractor is not None:
            self.with_graph_extractor(composition.graph_extractor)
            self._borrowed_component_ids.add(id(composition.graph_extractor))
            reproduced.append(ReproducedComponent("graph_extractor"))
        if composition.reranker is not None:
            self.with_reranker(composition.reranker)
            reproduced.append(ReproducedComponent("reranker"))
        if composition.cross_encoder_scorer is not None:
            self.with_cross_encoder_scorer(composition.cross_encoder_scorer)
            reproduced.append(ReproducedComponent("cross_encoder_scorer"))
        if composition.query_transform is not None:
            self.with_query_transform(composition.query_transform)
            reproduced.append(ReproducedComponent("query_transform"))
        if composition.judge is not None:
            self.with_judge_model(composition.judge)
            reproduced.append(ReproducedComponent("judge"))
        if composition.hype_generator is not None:
            self.with_hype_generator(composition.hype_generator)
            reproduced.append(ReproducedComponent("hype_generator"))
        if composition.chunking_strategy is not None:
            self.with_chunking_strategy(composition.chunking_strategy)
            reproduced.append(ReproducedComponent("chunking_strategy"))
        if composition.stopwords is not None:
            self.with_stopwords(composition.stopwords)
            reproduced.append(ReproducedComponent("stopwords"))
        if composition.plugin_pins is not None:
            self._plugin_pins_override = composition.plugin_pins
            reproduced.append(ReproducedComponent("plugin_pins"))

        omitted: list[OmittedComponent] = []
        for name, value in (
            ("search_index", composition.search_index),
            ("cache_store", composition.cache_store),
            ("model_call_cache", composition.model_call_cache),
            ("graph_read_repository", composition.graph_read_repository),
            ("circuit_breaker", composition.circuit_breaker),
            ("raptor_builder", composition.raptor_builder),
            ("audit_sink", composition.audit_sink),
            ("model_router", composition.model_router),
            ("access_policy", composition.access_policy),
            ("principal_resolver", composition.principal_resolver),
            ("principal_directory", composition.principal_directory),
            ("entitlement_scope_resolver", composition.entitlement_scope_resolver),
            ("secret_provider", composition.secret_provider),
            ("kill_switch_registry", composition.kill_switch_registry),
            ("quota_policy", composition.quota_policy),
            ("egress_policy", composition.egress_policy),
            ("output_governor", composition.output_governor),
            ("context_transform", composition.context_transform),
            ("degradation_policy", composition.degradation_policy),
            ("direct_skill_policy", composition.direct_skill_policy),
            ("image_captioner", composition.image_captioner),
            ("multimodal_generator", composition.multimodal_generator),
            ("fine_tuning_provider", composition.fine_tuning_provider),
            ("code_sandbox", composition.code_sandbox),
            ("event_sinks", composition.event_sinks),
            ("connectors", composition.connectors),
            ("extractors", composition.extractors),
            ("skills", composition.skills),
        ):
            if _composition_slot_present(value):
                omitted.append(OmittedComponent(name, OMISSION_REASONS[name]))
        return ShadowComposition(reproduced=tuple(reproduced), omitted=tuple(omitted))

    def build(self) -> Refaat:
        self.settings.validate_for_startup()
        if self._principal_resolver is not None and self._access_policy is None:
            # A resolver with no access policy resolves identity but enforces nothing: the retrieval
            # policy-enforcement point never runs and per-request audit never emits, so evidence is
            # served unfiltered. That is an unenforced-authorization misconfiguration, not a deployment.
            raise ConfigurationError(
                "a principal resolver requires an access policy; install one with with_access_policy"
            )
        if self.settings.audit_fail_closed and self._audit_sink is None:
            # Fail-closed audit with no sink would let every governed operation (query, RLM, approval,
            # legal hold) proceed unaudited, defeating the fail-closed contract.
            raise ConfigurationError("audit_fail_closed requires an audit sink; install one with with_audit_sink")
        configure_logging(self.settings)
        clock = SystemClock()
        resources: list[object] = []
        # Register injected event sinks for cleanup before any other build step, so a
        # failure anywhere in build (including plugin discovery) still closes a sink that
        # owns a network client.
        resources.extend(self._custom_event_sinks)
        store = SQLStore(self.settings.database_url, clock, lock_max_waiters=self.settings.legal_hold_lock_max_waiters)
        try:
            if self.settings.auto_create_schema:
                store.initialize()
            else:
                store.ping()

            # One resolved policy threads into every outbound-adapter factory, so no seam can be
            # built ungoverned. None (the default) yields byte-identical adapters everywhere.
            egress_policy = self._effective_egress_policy()
            blob_store = self._blob_store(egress_policy)
            resources.append(blob_store)
            # Default to the store (byte-identical); a Redis or injected backend is tracked for teardown.
            cache_store = self._cache_store(store)
            if cache_store is not store:
                resources.append(cache_store)
            model_call_cache = self._model_call_cache(store)
            if model_call_cache is not store:
                resources.append(model_call_cache)
            # Resolve the breaker before wrapping so _wrap_* can install it innermost (before caching);
            # None (the default, or a shadow build) leaves every adapter unwrapped and byte-identical.
            self._resolved_circuit_breaker = self._effective_circuit_breaker()
            # A wrapper owns its inner adapter's close, so only the top object is
            # tracked: appending both would close the inner adapter twice.
            raw_embedder = self._embedder(egress_policy)
            raw_generator = self._generator(egress_policy)
            # A shadow build borrows a reproduced production adapter by reference: it evaluates over it
            # but must not close the shared production client, so a borrowed raw adapter is not tracked
            # and a borrowed caching wrapper keeps cleanup but stops owning the inner's close. With no
            # borrowed ids (the default) each branch collapses to the plain append below, byte-identical.
            embedder = self._wrap_embedder(raw_embedder, model_call_cache)
            if id(raw_embedder) in self._borrowed_component_ids:
                # Only a wrapper the shadow constructed may be de-owned; a borrowed object returned
                # verbatim (a raw adapter, or a caching wrapper the caller injected) is left untouched.
                if embedder is not raw_embedder and isinstance(embedder, CachingEmbedder):
                    embedder._close_inner = False
                    resources.append(embedder)
            else:
                resources.append(embedder)
            generator = self._wrap_generator(raw_generator, model_call_cache)
            # One adapter filling both roles gets two distinct wrappers, each of which
            # would close the shared inner: let only the embedder wrapper own that close.
            if (
                id(raw_embedder) == id(raw_generator)
                and isinstance(embedder, CachingEmbedder)
                and isinstance(generator, CachingGenerator)
            ):
                generator._close_inner = False
            if id(raw_generator) in self._borrowed_component_ids:
                if generator is not raw_generator and isinstance(generator, CachingGenerator):
                    generator._close_inner = False
                    resources.append(generator)
            else:
                resources.append(generator)
            graph = self._graph_extractor(egress_policy)
            if id(graph) not in self._borrowed_component_ids:
                resources.append(graph)
            search = self._search_index(store, embedder)
            resources.append(search)
            plugin_pins = self._plugin_pins()
            if plugin_pins:
                self._verify_plugin_pins_total(plugin_pins)
            plugins = PluginRegistry(
                set(self.settings.plugin_allowlist),
                pins=plugin_pins or None,
                observer=self._plugin_audit_observer(clock) if plugin_pins else None,
            )
            extractors = self._extractors(plugins)
            connectors = self._connectors(plugins, resources, egress_policy)
            event_sinks = self._event_sinks(plugins, resources)
            if isinstance(search, ReconcilableIndex):
                reconcile = VectorReconcileSink(search, store)
                resources.append(reconcile)
                event_sinks = (*event_sinks, reconcile)
            chunker = self._chunking_strategy(plugins)
            # The registry primes here, so a bad prime fails startup; enabling management only opens
            # the engage/disengage write surface, not enforcement.
            kill_switches = self._effective_kill_switch_registry(store, clock)
            if kill_switches is not None:
                store.enable_kill_switch_management()
            # The policy primes its rule snapshot here (a bad prime fails startup); enabling
            # management only opens the housekeeping reaper surface, not enforcement.
            quota_policy = self._effective_quota_policy(store, clock)
            harness = HarnessService(store, kill_switches=kill_switches)
            hype_generator = self._hype_generator(generator)
            raptor_builder = self._raptor_builder(generator, embedder, store)
            graph_repo = self._graph_read_repository(store)
            retriever = AdaptiveRetriever(
                store,
                search,
                self.settings,
                reranker=self._reranker(),
                embedder=embedder,
                query_transform=self._query_transform(generator),
                hype_cache_id=hype_generator.provider_id if hype_generator is not None else None,
                raptor_cache_id=raptor_builder.provider_id if raptor_builder is not None else None,
                access_policy=self._access_policy,
                source_trust_weight=self.settings.source_trust_weight,
                classify_evidence=self._effective_model_router() is not None,
                strict_authorization=self.settings.strict_authorization,
                chunking_expansion_mode=getattr(chunker, "expansion_mode", None),
                graph_read_repository=graph_repo,
            )
            # Durable audit: when enabled with a sink installed, emissions persist through the
            # DurableAuditSink and a worker's AuditPublisher delivers to the real sink held for it.
            # Off by default, so the emission path stays the directly-delivering installed sink. This
            # sits above IngestionService so its hold.enforced events use the same durable sink.
            audit_sink = self._audit_sink
            audit_delivery_sinks: tuple[AuditSink, ...] = ()
            if self.settings.audit_durable and self._audit_sink is not None:
                audit_delivery_sinks = (self._audit_sink,)
                audit_sink = DurableAuditSink(store)
            # Enforcement of an existing hold is unconditional; this flag only opts into the
            # management write surface (place/lift). A runtime with no management surface still
            # enforces every hold in the legal_holds table, so it starts fine with holds present.
            if self._legal_hold_registry:
                store.enable_legal_hold_management()
            derivation_service = (
                DerivationService(
                    store,
                    self.settings,
                    embedder,
                    search_backend=type(search).__name__,
                    graph_backend=type(graph).__name__,
                    chunking_strategy_id=getattr(chunker, "strategy_id", self.settings.chunking_strategy),
                )
                if self.settings.derivation_manifest_enabled
                else None
            )
            ingestion = IngestionService(
                store=store,
                blob_store=blob_store,
                extractors=extractors,
                chunker=chunker,
                embedder=embedder,
                graph_extractor=graph,
                search_index=search,
                connectors=connectors,
                settings=self.settings,
                contextual_retrieval=self.settings.contextual_retrieval,
                hype_generator=hype_generator,
                raptor_builder=raptor_builder,
                image_captioner=self._image_captioner(resources, egress_policy),
                audit_sink=audit_sink,
                access_policy=self._access_policy,
                kill_switches=kill_switches,
                quota_policy=quota_policy,
                derivation_service=derivation_service,
            )
            # One shared, stateless governor, model router, and context transform instance for both
            # the query and RLM paths, so the egress gates cannot diverge.
            output_governor = self._effective_output_governor()
            model_router = self._effective_model_router()
            context_transform = self._effective_context_transform()
            degradation_policy = self._effective_degradation_policy()
            erasure = ErasureService(
                store=store,
                search_index=search,
                blob_store=blob_store,
                # The raw embedder's cache_identity is exactly the identity the caching embedder
                # wrote model-call rows under, so erasure recomputes matching key digests.
                embedder=raw_embedder,
                clock=clock,
                cache_store=cache_store,
                model_call_cache=model_call_cache,
                raptor_builder=raptor_builder,
                access_policy=self._access_policy,
                audit_sink=audit_sink,
                audit_fail_closed=self.settings.audit_fail_closed,
                kill_switches=kill_switches,
                on_conflict=self.settings.legal_hold_on_conflict,
                derivation_service=derivation_service,
            )
            queries = QueryService(
                store=store,
                embedder=embedder,
                generator=generator,
                retriever=retriever,
                harness=harness,
                settings=self.settings,
                clock=clock,
                blob_store=blob_store,
                cache_store=cache_store,
                multimodal=self._multimodal_active(),
                audit_sink=audit_sink,
                model_router=model_router,
                output_governor=output_governor,
                context_transform=context_transform,
                kill_switches=kill_switches,
                quota_policy=quota_policy,
                degradation_policy=degradation_policy,
                strict_authorization=self.settings.strict_authorization,
            )
            authentication = AuthenticationService(
                store,
                self.settings.api_key_pepper.get_secret_value(),
            )
            feedback = FeedbackService(
                store,
                access_policy=self._access_policy,
                kill_switches=kill_switches,
                strict_authorization=self.settings.strict_authorization,
            )
            evaluations = EvaluationService(store, queries)
            registry = RegistryService(store)
            deployment = DeploymentRouter(store, registry, self.settings)
            fine_tuning = FineTuningService(
                store, registry, self._fine_tuning_provider(resources, egress_policy), kill_switches=kill_switches
            )
            agents = AgentService(store, access_policy=self._access_policy)
            skill_registry = SkillRegistry(plugins)
            for skill in self._custom_skills:
                skill_registry.register(skill)
            skills = SkillService(
                registry=skill_registry,
                store=store,
                embedder=embedder,
                retriever=retriever,
                harness=harness,
                settings=self.settings,
                direct_skill_policy=self._direct_skill_policy,
                kill_switches=kill_switches,
                quota_policy=quota_policy,
                audit_sink=audit_sink,
            )
            rlm = RecursiveContextService(
                store=store,
                embedder=embedder,
                generator=generator,
                harness=harness,
                settings=self.settings,
                clock=clock,
                sandbox=self._code_sandbox(),
                access_policy=self._access_policy,
                audit_sink=audit_sink,
                model_router=model_router,
                output_governor=output_governor,
                context_transform=context_transform,
                kill_switches=kill_switches,
                quota_policy=quota_policy,
                degradation_policy=degradation_policy,
                strict_authorization=self.settings.strict_authorization,
                graph_read_repository=graph_repo,
            )
            skill_creator = SkillCreator(self.settings.skill_root)
            configure_telemetry(self.settings, engine=store.engine)
            return Refaat._from_services(
                settings=self.settings,
                store=store,
                derivation_service=derivation_service,
                authentication=authentication,
                ingestion=ingestion,
                erasure=erasure,
                queries=queries,
                feedback=feedback,
                evaluations=evaluations,
                registry=registry,
                deployment=deployment,
                fine_tuning=fine_tuning,
                harness=harness,
                agents=agents,
                rlm=rlm,
                skills=skills,
                skill_registry=skill_registry,
                skill_creator=skill_creator,
                connectors=connectors,
                event_sinks=event_sinks,
                judge_model=self._judge(generator),
                access_policy=self._access_policy,
                principal_resolver=self._principal_resolver,
                principal_directory=self._principal_directory,
                entitlement_scope_resolver=self._entitlement_scope_resolver,
                secret_provider=self._effective_secret_provider(),
                kill_switches=kill_switches,
                quota_policy=quota_policy,
                degradation_policy=degradation_policy,
                egress_policy=egress_policy,
                audit_sink=audit_sink,
                audit_delivery_sinks=audit_delivery_sinks,
                composition=self._capture_composition(),
                resources=tuple(resources),
            )
        except Exception:
            _close_resources(resources)
            try:
                store.close()
            except Exception:
                logger.exception("runtime_store_close_failed_during_build")
            raise

    def _chunking_strategy(self, plugins: PluginRegistry) -> ChunkingStrategy:
        if self._custom_chunking_strategy is not None:
            return self._custom_chunking_strategy
        name = self.settings.chunking_strategy
        builtin = BUILTIN_CHUNKING_STRATEGIES.get(name)
        if builtin is not None:
            return builtin(self.settings)
        discovered = plugins.discover("refaat.chunkers")
        plugin = discovered.get(name)
        if plugin is not None:
            strategy: ChunkingStrategy = plugin() if isinstance(plugin, type) else plugin
            return strategy
        available = ", ".join(sorted({*BUILTIN_CHUNKING_STRATEGIES, *discovered}))
        raise ConfigurationError(f"Unknown chunking strategy {name!r}. Available: {available}")

    def _extractors(self, plugins: PluginRegistry) -> ExtractorRegistry:
        registry = ExtractorRegistry()
        for plugin in plugins.discover("refaat.extractors").values():
            extractor: DocumentExtractor = plugin() if isinstance(plugin, type) else plugin
            registry.register(extractor)
        for extractor in self._custom_extractors:
            registry.register(extractor)
        return registry

    def _connectors(
        self, plugins: PluginRegistry, resources: list[object], egress_policy: EgressPolicy | None
    ) -> ConnectorRegistry:
        max_bytes = self.settings.connector_max_bytes
        registry = ConnectorRegistry(
            (
                FileConnector(self.settings.connector_file_root, max_bytes=max_bytes),
                HttpConnector(
                    scheme="https",
                    max_bytes=max_bytes,
                    allow_private_hosts=self.settings.allow_private_connector_hosts,
                    timeout=self.settings.request_timeout_seconds,
                    egress_policy=egress_policy,
                ),
                HttpConnector(
                    scheme="http",
                    max_bytes=max_bytes,
                    allow_private_hosts=self.settings.allow_private_connector_hosts,
                    timeout=self.settings.request_timeout_seconds,
                    egress_policy=egress_policy,
                ),
            )
        )
        # Register the partially-built registry for cleanup before registration can
        # fail, so build()'s outer cleanup owns its close and threads the shared
        # already-closed set into it. A dual-role object that is also an injected event
        # sink (already in resources) is therefore closed exactly once even when
        # registration raises here.
        resources.append(registry)
        for plugin in plugins.discover("refaat.connectors").values():
            connector: Connector = plugin() if isinstance(plugin, type) else plugin
            registry.register(connector)
        for connector in self._custom_connectors:
            registry.register(connector)
        return registry

    def _event_sinks(self, plugins: PluginRegistry, resources: list[object]) -> tuple[EventSink, ...]:
        # Register each constructed sink for cleanup the moment it is built, so a later
        # sink constructor that raises still lets build()'s cleanup close the earlier
        # sinks (a sink can own a network client). The injected custom sinks are already
        # in resources (registered at the start of build), so they are only appended to
        # the returned tuple here to avoid a double close.
        sinks: list[EventSink] = []
        logging_sink = LoggingEventSink()
        resources.append(logging_sink)
        sinks.append(logging_sink)
        for plugin in plugins.discover("refaat.event_sinks").values():
            sink: EventSink = plugin() if isinstance(plugin, type) else plugin
            resources.append(sink)
            sinks.append(sink)
        sinks.extend(self._custom_event_sinks)
        return tuple(sinks)

    def _blob_store(self, egress_policy: EgressPolicy | None = None) -> BlobStore:
        if self.settings.blob_backend == "local":
            return LocalBlobStore(self.settings.blob_root)
        if not self.settings.s3_bucket:
            raise ConfigurationError("REFAAT_S3_BUCKET is required for the S3 blob backend")
        return S3BlobStore(
            bucket=self.settings.s3_bucket,
            endpoint_url=self.settings.s3_endpoint_url,
            region=self.settings.s3_region,
            access_key_id=(
                self.settings.s3_access_key_id.get_secret_value() if self.settings.s3_access_key_id else None
            ),
            secret_access_key=(
                self.settings.s3_secret_access_key.get_secret_value() if self.settings.s3_secret_access_key else None
            ),
            egress_policy=egress_policy,
        )

    def _require_redis_url(self) -> str:
        url = self.settings.redis_url
        if url is None:
            raise ConfigurationError("REFAAT_REDIS_URL is required when a Redis cache backend is selected")
        return url

    def _graph_read_repository(self, store: SQLStore) -> GraphReadRepository:
        # An injected repo wins; otherwise the composite store is the default, so the path is byte-identical.
        return self._custom_graph_read_repository if self._custom_graph_read_repository is not None else store

    def _cache_store(self, store: SQLStore) -> CacheStore:
        if self._custom_cache_store is not None:
            return self._custom_cache_store
        if self.settings.answer_cache_backend == "redis":
            from refaat.infrastructure.redis_cache import RedisCacheStore, build_redis_client, ping_or_warn

            client = build_redis_client(self._require_redis_url(), self.settings.redis_timeout_seconds)
            ping_or_warn(client)
            return RedisCacheStore(client, clock=store.clock)
        return store

    def _model_call_cache(self, store: SQLStore) -> ModelCallCache:
        if self._custom_model_call_cache is not None:
            return self._custom_model_call_cache
        if self.settings.model_call_cache_backend == "redis":
            from refaat.infrastructure.redis_cache import RedisModelCallCache, build_redis_client, ping_or_warn

            client = build_redis_client(self._require_redis_url(), self.settings.redis_timeout_seconds)
            ping_or_warn(client)
            return RedisModelCallCache(client, clock=store.clock)
        return store

    def _embedder(self, egress_policy: EgressPolicy | None = None) -> Embedder:
        if self._custom_embedder is not None:
            return self._custom_embedder
        if self.settings.embedding_provider == "deterministic":
            return DeterministicEmbedder(
                dimensions=self.settings.embedding_dimensions,
                model_id=self.settings.embedding_model,
            )
        return OpenAICompatibleEmbedder(
            base_url=self.settings.embedding_base_url or self.settings.openai_base_url,
            api_key=self._model_api_key(self.settings.embedding_api_key),
            model=self.settings.embedding_model,
            dimensions=self.settings.embedding_dimensions,
            send_dimensions=self.settings.embedding_send_dimensions,
            retries=self.settings.model_max_retries,
            retry_base_delay=self.settings.model_retry_base_delay_seconds,
            timeout=self.settings.request_timeout_seconds,
            egress_policy=egress_policy,
        )

    def _generator(self, egress_policy: EgressPolicy | None = None) -> Generator:
        if self._custom_generator is not None:
            return self._custom_generator
        if self._custom_multimodal_generator is not None:
            return self._custom_multimodal_generator
        if self.settings.multimodal_generation == "openai_vision":
            return self._build_vision_generator(egress_policy)
        if self.settings.generation_provider == "extractive":
            return ExtractiveGenerator()
        return OpenAICompatibleGenerator(
            base_url=self.settings.generation_base_url or self.settings.openai_base_url,
            api_key=self._model_api_key(self.settings.generation_api_key),
            model=self.settings.generation_model,
            retries=self.settings.model_max_retries,
            retry_base_delay=self.settings.model_retry_base_delay_seconds,
            timeout=self.settings.request_timeout_seconds,
            egress_policy=egress_policy,
        )

    def _build_vision_generator(self, egress_policy: EgressPolicy | None = None) -> Generator:
        # Imported lazily so the offline gate never loads the vision adapter, and
        # constructed only when a multimodal model is configured.
        from refaat.infrastructure.multimodal import OpenAIVisionGenerator

        return OpenAIVisionGenerator(
            base_url=self.settings.multimodal_base_url
            or self.settings.generation_base_url
            or self.settings.openai_base_url,
            api_key=self._model_api_key(self.settings.multimodal_api_key),
            model=self.settings.multimodal_model,
            retries=self.settings.model_max_retries,
            retry_base_delay=self.settings.model_retry_base_delay_seconds,
            timeout=self.settings.request_timeout_seconds,
            egress_policy=egress_policy,
        )

    def _multimodal_active(self) -> bool:
        return self._custom_multimodal_generator is not None or self.settings.multimodal_generation == "openai_vision"

    def _image_captioner(self, resources: list[object], egress_policy: EgressPolicy | None) -> ImageCaptioner | None:
        if self._custom_image_captioner is not None:
            return self._custom_image_captioner
        if self.settings.image_captioner != "openai_vision":
            return None
        # Imported lazily so the offline gate never loads the vision adapter.
        from refaat.infrastructure.multimodal import OpenAIVisionCaptioner

        captioner = OpenAIVisionCaptioner(
            base_url=self.settings.multimodal_base_url
            or self.settings.generation_base_url
            or self.settings.openai_base_url,
            api_key=self._model_api_key(self.settings.multimodal_api_key),
            model=self.settings.multimodal_model,
            retries=self.settings.model_max_retries,
            retry_base_delay=self.settings.model_retry_base_delay_seconds,
            timeout=self.settings.request_timeout_seconds,
            egress_policy=egress_policy,
        )
        resources.append(captioner)
        return captioner

    def _budget_configured(self) -> bool:
        return self.settings.run_token_budget is not None or self.settings.run_cost_budget_usd is not None

    def _require_priced_metered_adapter(self, model_id: str, *, role: str, require_output: bool) -> None:
        """Fail closed when a metered adapter's model carries no usable price.

        The startup guard checks the settings-configured models, but an injected
        adapter (or one paired with the extractive provider) bypasses it: its real
        ``model_id`` is unknown to the settings. Under a cost budget an unpriced
        metered adapter accrues zero cost and silently defeats the ceiling, so a
        missing or all-zero price record is rejected here at wrap time.
        """
        if self.settings.run_cost_budget_usd is None:
            return
        record = self.settings.model_prices.get(model_id)
        input_rate = float(record.get("input_per_1k", 0.0)) if record else 0.0
        output_rate = float(record.get("output_per_1k", 0.0)) if record else 0.0
        if input_rate <= 0.0 or (require_output and output_rate <= 0.0):
            raise ConfigurationError(
                f"A run cost budget requires a positive REFAAT_MODEL_PRICES entry for the metered {role} "
                f"model {model_id!r}; an unpriced metered adapter would silently defeat the ceiling"
            )

    def _wrap_embedder(self, embedder: Embedder, model_call_cache: ModelCallCache) -> Embedder:
        # Never wrap the deterministic offline stack, and never double-wrap.
        if isinstance(embedder, (DeterministicEmbedder, CachingEmbedder)):
            return embedder
        # The breaker sits innermost, before caching, and only over a real provider endpoint; it is
        # transparent to the cache identity, so an unconfigured breaker leaves the adapter byte-identical.
        if self._resolved_circuit_breaker is not None:
            embedder = wrap_embedder_breaker(
                embedder, self._resolved_circuit_breaker, count_429=self.settings.circuit_breaker_count_429
            )
        # Caching is a positive opt-in: only an adapter exposing cache_identity is
        # ever cached, so a custom deterministic adapter cannot be served stale.
        caching = self.settings.model_call_cache_enabled and cache_identity_of(embedder) is not None
        metering = self._budget_configured()
        if not caching and not metering:
            return embedder
        if metering:
            self._require_priced_metered_adapter(embedder.model_id, role="embedding", require_output=False)
        cache = model_call_cache if caching else None
        return CachingEmbedder(embedder, cache, ttl_seconds=self.settings.cache_ttl_seconds)

    def _wrap_generator(self, generator: Generator, model_call_cache: ModelCallCache) -> Generator:
        if isinstance(generator, (ExtractiveGenerator, CachingGenerator)):
            return generator
        if self._resolved_circuit_breaker is not None:
            generator = wrap_generator_breaker(
                generator, self._resolved_circuit_breaker, count_429=self.settings.circuit_breaker_count_429
            )
        caching = self.settings.model_call_cache_enabled and cache_identity_of(generator) is not None
        metering = self._budget_configured()
        if not caching and not metering:
            return generator
        if metering:
            self._require_priced_metered_adapter(generator.model_id, role="generation", require_output=True)
        cache = model_call_cache if caching else None
        return CachingGenerator(generator, cache, ttl_seconds=self.settings.cache_ttl_seconds)

    def _graph_extractor(self, egress_policy: EgressPolicy | None = None) -> GraphExtractor:
        if self._custom_graph_extractor is not None:
            return self._custom_graph_extractor
        if self.settings.graph_provider == "cooccurrence":
            return CooccurrenceGraphExtractor(
                max_entities=self.settings.graph_max_entities,
                min_token_length=self.settings.graph_min_token_length,
                default_language=self.settings.default_language,
                stopwords_by_language=self._effective_stopwords(),
            )
        return OpenAICompatibleGraphExtractor(
            base_url=self.settings.generation_base_url or self.settings.openai_base_url,
            api_key=self._model_api_key(self.settings.generation_api_key),
            model=self.settings.generation_model,
            retries=self.settings.model_max_retries,
            retry_base_delay=self.settings.model_retry_base_delay_seconds,
            timeout=self.settings.request_timeout_seconds,
            egress_policy=egress_policy,
        )

    def _effective_stopwords(self) -> Mapping[str, frozenset[str]] | None:
        # Return None when nothing is configured so the extractor resolves the module
        # default and every unconfigured corpus stays byte-identical. Only when a
        # project supplies a setting or a builder injection is a merged registry built,
        # layering the override onto the shipped per-language defaults.
        settings_registry = self.settings.stopwords_by_language
        if settings_registry is None and self._custom_stopwords is None:
            return None
        merged: dict[str, frozenset[str]] = dict(STOPWORDS_BY_LANGUAGE)
        if settings_registry is not None:
            for code, words in settings_registry.items():
                merged[_primary_subtag(code)] = frozenset(word.casefold() for word in words)
        if self._custom_stopwords is not None:
            merged.update(self._custom_stopwords)
        return merged

    def _reranker(self) -> Reranker | None:
        if self._custom_reranker is not None:
            return self._custom_reranker
        if self.settings.reranker_provider == "lexical":
            return LexicalReranker()
        if self.settings.reranker_provider == "cross_encoder":
            if self._custom_cross_encoder_scorer is None:
                raise ConfigurationError(
                    "reranker_provider='cross_encoder' requires a scorer injected via "
                    "RuntimeBuilder.with_cross_encoder_scorer(...) or a reranker via with_reranker(...)"
                )
            return CrossEncoderReranker(self._custom_cross_encoder_scorer)
        return None

    def _hype_generator(self, generator: Generator) -> HypotheticalQuestionGenerator | None:
        if self._custom_hype_generator is not None:
            return self._custom_hype_generator
        if not self.settings.hype_enabled:
            return None
        deterministic = DeterministicHypotheticalGenerator()
        if self.settings.generation_provider == "openai_compatible":
            return ModelHypotheticalGenerator(generator, deterministic)
        return deterministic

    def _raptor_builder(self, generator: Generator, embedder: Embedder, store: SQLStore) -> RaptorTreeBuilder | None:
        if self._custom_raptor_builder is not None:
            return self._custom_raptor_builder
        if not self.settings.raptor_enabled:
            return None
        deterministic = DeterministicExtractiveSummarizer()
        summarizer: Summarizer = (
            ModelSummarizer(generator, deterministic)
            if self.settings.generation_provider == "openai_compatible"
            else deterministic
        )
        return RaptorTreeBuilder(store=store, embedder=embedder, summarizer=summarizer)

    def _fine_tuning_provider(self, resources: list[object], egress_policy: EgressPolicy | None) -> FineTuningProvider:
        if self._custom_fine_tuning_provider is not None:
            return self._custom_fine_tuning_provider
        if self.settings.fine_tuning_provider == "openai":
            provider = OpenAIFineTuningProvider(
                base_url=self.settings.fine_tuning_base_url or self.settings.openai_base_url,
                api_key=self._model_api_key(self.settings.fine_tuning_api_key),
                retries=self.settings.model_max_retries,
                retry_base_delay=self.settings.model_retry_base_delay_seconds,
                timeout=self.settings.request_timeout_seconds,
                egress_policy=egress_policy,
            )
            resources.append(provider)
            return provider
        return DeterministicFineTuner()

    def _judge(self, generator: Generator) -> JudgeModel | None:
        if self._custom_judge is not None:
            return self._custom_judge
        if self.settings.judge_provider == "generator":
            return GeneratorJudge(generator)
        return None

    def _effective_output_governor(self) -> OutputGovernor | None:
        """The output governor to install: an injected one wins, else auto-install the
        reference denylist governor only when patterns are configured. An empty patterns
        list installs nothing, so byte-identity holds; a directly-injected governor is used
        as given, so an empty-denylist governor a caller injects on purpose is honored."""
        if self._output_governor is not None:
            return self._output_governor
        if self.settings.output_denylist_patterns:
            return PatternDenylistGovernor(
                self.settings.output_denylist_patterns,
                mode=self.settings.output_governance_mode,
            )
        return None

    def _effective_degradation_policy(self) -> DegradationPolicy | None:
        """The degradation policy to install: an injected one wins, else the config selector.

        The default ("none") installs nothing, so a generation outage propagates and both answer
        paths are byte-identical. "refuse_all" installs the fail-closed reference policy;
        "safe_defaults" installs the grounded-or-abstain reference policy.
        """
        if self._degradation_policy is not None:
            return self._degradation_policy
        selector = self.settings.degradation_policy
        if selector == "refuse_all":
            return RefuseAllDegradationPolicy()
        if selector == "safe_defaults":
            return SafeDefaultsDegradationPolicy()
        return None

    def _effective_context_transform(self) -> ContextTransform | None:
        """The context transform to install: an injected one wins, else auto-install the reference
        pattern redactor only when egress patterns are configured. An empty patterns list installs
        nothing, so byte-identity holds; a directly-injected transform is used as given."""
        if self._context_transform is not None:
            return self._context_transform
        if self.settings.egress_redaction_patterns:
            return PatternRedactor(
                self.settings.egress_redaction_patterns,
                placeholder=self.settings.egress_redaction_placeholder,
                fields=tuple(self.settings.egress_redaction_fields),
            )
        return None

    def _residency_router(self) -> ModelRouter | None:
        """Build the config-driven residency router, or None when unconfigured.

        None or an empty ``residency_policy`` installs nothing, so the generation path stays
        byte-identical. When configured, the active provider tags come from the settings and are
        matched against the per-classification rules for the derived evidence classification.
        """
        policy_config = self.settings.residency_policy
        if not policy_config:
            return None
        rules: dict[Classification, ResidencyRule] = {}
        for name, rule_config in policy_config.items():
            classification = _parse_classification_name(name)
            rules[classification] = _parse_residency_rule(classification, rule_config)
        provider = ProviderResidency(
            provider=self.settings.generation_model,
            region=self.settings.residency_provider_region,
            retention=_parse_retention(self.settings.residency_provider_retention),
            purpose=self.settings.residency_provider_purpose,
        )
        return ResidencyRouter(ResidencyPolicy(rules), provider)

    def _effective_model_router(self) -> ModelRouter | None:
        """The model router to install, mirroring ``_effective_output_governor``.

        An injected router (``with_model_router``) and a config-driven residency router compose:
        with neither, the generation path is byte-identical; either alone is used as given; both
        compose into a ``CompositeModelRouter`` (the injected ceiling plus residency). The shared
        instance gates both the query and the RLM egress.
        """
        residency = self._residency_router()
        injected = self._model_router
        if residency is None:
            return injected
        if injected is None:
            return residency
        return CompositeModelRouter((injected, residency))

    def _effective_secret_provider(self) -> SecretProvider | None:
        # An injected provider wins; otherwise a settings selector may build the reference env-var
        # provider. The default "none" builds nothing, so the runtime is byte-identical.
        if self._secret_provider is not None:
            return self._secret_provider
        if self.settings.tool_secret_provider == "env":
            return EnvSecretProvider(prefix=self.settings.tool_secret_env_prefix)
        return None

    def _effective_kill_switch_registry(self, store: SQLStore, clock: Clock) -> KillSwitchRegistry | None:
        # An injected registry wins; otherwise "database" builds the store-backed one. The default
        # "none" builds nothing, so every guard short-circuits and the runtime is byte-identical.
        if self._kill_switch_registry is not None:
            return self._kill_switch_registry
        if self.settings.kill_switch_backend == "database":
            return DatabaseKillSwitchRegistry(
                store,
                clock,
                cache_ttl_seconds=self.settings.kill_switch_cache_ttl_seconds,
                stale_deny_seconds=self.settings.kill_switch_stale_deny_seconds,
            )
        return None

    def _effective_circuit_breaker(self) -> InMemoryCircuitBreaker | None:
        # An injected breaker wins; otherwise "in_memory" builds one. The default "none" builds nothing,
        # so no adapter is breaker-wrapped and the runtime is byte-identical.
        if self._circuit_breaker is not None:
            return self._circuit_breaker
        if self.settings.circuit_breaker_backend == "in_memory":
            return build_circuit_breaker(
                CircuitBreakerConfig(
                    failure_threshold=self.settings.circuit_breaker_failure_threshold,
                    reset_timeout_seconds=self.settings.circuit_breaker_reset_timeout_seconds,
                    half_open_deadline_seconds=self.settings.circuit_breaker_half_open_deadline_seconds,
                    count_429=self.settings.circuit_breaker_count_429,
                )
            )
        return None

    def _effective_quota_policy(self, store: SQLStore, clock: Clock) -> QuotaPolicy | None:
        # An injected policy wins; otherwise "database" builds the store-backed one over the config
        # rule snapshot. The default "none" builds nothing, so every guard short-circuits and the
        # runtime is byte-identical.
        if self._quota_policy is not None:
            # An injected store-backed policy still needs the reaper enabled so a leaked reservation
            # self-heals; a non-store policy manages its own housekeeping.
            if isinstance(self._quota_policy, DatabaseQuotaPolicy):
                store.enable_quota_management()
            return self._quota_policy
        if self.settings.quota_backend == "database":
            policy = DatabaseQuotaPolicy(
                store,
                clock,
                rules=quota_rules_from_config(self.settings.quota_rules),
                lease_seconds=self.settings.quota_reservation_lease_seconds,
            )
            store.enable_quota_management()
            return policy
        return None

    def _effective_egress_policy(self) -> EgressPolicy | None:
        # An injected policy wins; otherwise "allowlist" builds the default-deny AllowlistEgressPolicy
        # over the config snapshot. The default "none" builds nothing, so every seam's chokepoint is a
        # no-op and the runtime is byte-identical. A configured-but-malformed allowlist raises here
        # (allowlist_from_config), so it never silently degrades to an ungoverned runtime.
        if self._egress_policy is not None:
            return self._egress_policy
        if self.settings.egress_policy == "allowlist":
            return AllowlistEgressPolicy(allowlist_from_config(self.settings.egress_allowlist))
        return None

    def _code_sandbox(self) -> CodeSandbox:
        if self._custom_code_sandbox is not None:
            return self._custom_code_sandbox
        if self.settings.code_sandbox == "safe_expression":
            return SafeExpressionSandbox()
        if self.settings.code_sandbox == "subprocess":
            return SubprocessCodeSandbox()
        return DenyingSandbox()

    def _query_transform(self, generator: Generator) -> QueryTransform:
        if self._custom_query_transform is not None:
            return self._custom_query_transform
        provider = self.settings.query_transform_provider
        if provider == "sub_query":
            return DeterministicQueryTransform()
        model_backed = self.settings.generation_provider == "openai_compatible"
        if provider == "hyde":
            return ModelHydeQueryTransform(generator) if model_backed else DeterministicHydeQueryTransform()
        if provider == "query_rewrite":
            return ModelRewriteQueryTransform(generator) if model_backed else DeterministicRewriteQueryTransform()
        if provider == "multi_query":
            return ModelMultiQueryTransform(generator) if model_backed else DeterministicMultiQueryTransform()
        return ModelStepBackQueryTransform(generator) if model_backed else DeterministicStepBackQueryTransform()

    def _search_index(self, store: SQLStore, embedder: Embedder) -> SearchIndex:
        if self._custom_search_index is not None:
            if isinstance(self._custom_search_index, SearchIndexFactory):
                return self._custom_search_index.build(store, embedder)
            return self._custom_search_index
        if self.settings.database_url.startswith("postgresql"):
            return PgVectorSearchIndex(
                self.settings.database_url,
                embedder.dimensions,
                store,
                provision=self.settings.auto_create_schema,
            )
        return LocalSearchIndex(store)

    def _model_api_key(self, override: SecretStr | None = None) -> str:
        key = override or self.settings.openai_api_key
        if key is None:
            raise ConfigurationError(
                "A model API key is required for the selected adapter; set REFAAT_OPENAI_API_KEY "
                "or the per-role REFAAT_EMBEDDING_API_KEY or REFAAT_GENERATION_API_KEY"
            )
        return key.get_secret_value()
