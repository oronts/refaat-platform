from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from typing import Any, cast

from refaat.application.budget import (
    CostMeter,
    budget_scope,
    prices_from,
    quota_commit_all,
    quota_release_all,
    quota_reserve_all,
    quota_run,
    resolve_stream_meter,
    use_meter,
)
from refaat.application.derivation import DerivationService
from refaat.application.raptor import RaptorTreeBuilder
from refaat.clock import Clock
from refaat.config import Settings
from refaat.core.chunking import TextChunk, contextualize
from refaat.core.citations import build_citations, extract_cited_ids, filter_cited_chunks, strip_uncited_markers
from refaat.core.guidance import guidance_digest
from refaat.core.harness import BASE_SYSTEM_PROMPT_HASH, HarnessService, system_prompt
from refaat.core.learning import PreferenceLearner
from refaat.core.math import cosine_similarity
from refaat.core.retrieval import AdaptiveRetriever, RetrievalPolicy
from refaat.core.session import SessionCompactor, stored_summary_cited_documents, tag_turn_metadata
from refaat.core.text import normalize_text, sha256_bytes, sha256_text, stable_digest, tokens
from refaat.domain.audit import AuditEvent, AuditSink, emit_audit
from refaat.domain.authz import (
    AccessPolicy,
    Classification,
    auth_scope_hash,
    chunk_allowed,
    max_classification,
    permission_fingerprint,
    preference_scope,
    require_principal_present,
    residency_zones,
    session_summary_visible,
    strip_reserved_governance_keys,
    summary_documents_authorized,
)
from refaat.domain.citation_policy import CitationPolicy, citation_policy_refuses
from refaat.domain.context_transform import ContextTransform, TransformResult, redact_for_egress
from refaat.domain.degradation import (
    DegradationDecision,
    DegradationMode,
    DegradationPolicy,
    DependencyClass,
    DependencyFailure,
    resolve_degradation,
)
from refaat.domain.kill_switch import KillSwitchRegistry, KillSwitchScope, ensure_scope_active, ensure_workspace_active
from refaat.domain.legal_hold import LegalHoldError
from refaat.domain.models import (
    CacheEntry,
    Chunk,
    Citation,
    ContextPacket,
    DocumentInput,
    Feedback,
    GenerationImage,
    GenerationRequest,
    GenerationResult,
    IngestResult,
    JsonObject,
    QueryRequest,
    QueryResponse,
)
from refaat.domain.output_governance import OutputContext, OutputDecision, OutputGovernor, bounded_reason
from refaat.domain.quota import QuotaContext, QuotaDimension, QuotaPolicy, ensure_within_quota
from refaat.domain.routing import ModelRouter
from refaat.errors import (
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    ConflictError,
    DependencyUnavailableError,
    QuotaExceededError,
    ValidationError,
    error_message,
)
from refaat.ids import new_id
from refaat.infrastructure.connectors import ConnectorRegistry
from refaat.infrastructure.database.store import SQLStore
from refaat.infrastructure.extractors import ExtractorRegistry
from refaat.infrastructure.generation import ExtractiveGenerator
from refaat.observability import current_trace_id, get_logger
from refaat.ports import (
    BlobStore,
    CacheStore,
    ChunkingStrategy,
    Embedder,
    Generator,
    GraphExtractor,
    HypotheticalQuestionGenerator,
    ImageCaptioner,
    SearchIndex,
)
from refaat.security import hash_api_key, issue_api_key, sanitize_source_url, verify_api_key

# Whole-token matching, not naive substring: a key is split on non-alphanumeric
# boundaries and a token must match exactly, so benign keys that merely contain a
# sensitive substring survive (tokenizer, secretary, credentialing_status) while the
# real secrets are stripped at any depth.
SENSITIVE_METADATA_TOKENS = frozenset(
    {
        "password",
        "passwd",
        "secret",
        "token",
        "credential",
        "credentials",
        "authorization",
        "apikey",
    }
)
# Adjacent token sequences whose sensitive word is "key" (kept out of the single-token
# set so benign compounds such as primary_key or foreign_key are not stripped).
SENSITIVE_METADATA_PHRASES = (
    ("api", "key"),
    ("access", "key"),
    ("private", "key"),
    ("aws", "secret", "access", "key"),
)
_METADATA_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")
SAFE_METADATA_KEYS = frozenset(
    {
        "token_count",
        "author",
        "authors",
        "workspace_id",
        "source_id",
        "title",
        "heading",
        "context_strategy",
        "granularity",
        "parent_id",
        "sent_start",
        "sent_end",
    }
)
logger = get_logger(__name__)


def _validate_chunk_sequence(strategy_id: str, chunks: Sequence[TextChunk]) -> None:
    # Enforce the ChunkingStrategy output contract at the ingestion boundary so a misbehaving injected
    # or plugin strategy fails with a named error rather than an opaque database integrity error or a
    # silently empty citable chunk. The built-in strategies always satisfy this, so it is inert for them.
    for index, chunk in enumerate(chunks):
        if chunk.ordinal != index:
            raise ValidationError(f"Chunking strategy {strategy_id!r} produced non-contiguous ordinals")
        if not chunk.text.strip():
            raise ValidationError(f"Chunking strategy {strategy_id!r} produced an empty chunk")
        if chunk.token_count <= 0:
            raise ValidationError(f"Chunking strategy {strategy_id!r} produced a non-positive token_count")
        if chunk.granularity == "child":
            parent_ordinal = chunk.parent_ordinal
            if (
                parent_ordinal is None
                or not 0 <= parent_ordinal < index
                or chunks[parent_ordinal].granularity != "parent"
            ):
                raise ValidationError(f"Chunking strategy {strategy_id!r} produced a child with an invalid parent")


# The honest no-answer returned when the best retrieved evidence is below the
# configured similarity floor. It carries no citations, so the citation contract
# holds trivially, and a distinguishable "abstain" route so a caller can tell it
# apart from a grounded answer. Abstention is off by default (abstain_below_score
# is 0.0), and this message is fixed so an abstain is deterministic for its scope.
ABSTAIN_MESSAGE = "I could not find information about that in this knowledge base."
ABSTAIN_ROUTE = "abstain"

# The governed no-generation response returned when the installed model router denies
# the classification of the retrieved evidence for the request principal. It carries no
# citations and a distinguishable route, and is produced before any generation call, so
# over-classified evidence is never sent to the model. Routing is off by default (no
# router installed), so this path is inert unless a ModelRouter is configured.
ROUTING_REFUSAL_MESSAGE = (
    "This request cannot be answered because the relevant evidence is not permitted for generation."
)
ROUTING_ROUTE = "routing_denied"
# The caller-facing reason is generic on purpose: the router's own reason names the evidence
# classification level, so echoing it would let a prober learn the max sensitivity of evidence
# matching a refused query. The detailed reason stays internal to the router and audit trail.
ROUTING_REFUSAL_REASON = "model_routing_policy"

# The governed no-output response returned when the installed output governor rejects a
# produced answer AFTER generation and BEFORE it is returned or cached. It carries no
# citations and a distinguishable route. The caller-facing reason is generic on purpose:
# the governor's detailed bounded code names why the output failed review, so echoing it
# would leak the review criterion; the detailed code stays internal to the audit trail,
# mirroring ROUTING_REFUSAL_REASON. Output governance is off by default (no governor
# installed), so this path is inert unless an OutputGovernor is configured.
OUTPUT_REFUSAL_MESSAGE = "This response was withheld because it did not pass output review."
OUTPUT_ROUTE = "output_denied"
OUTPUT_REFUSAL_REASON = "output_governance_policy"

# The honest no-answer returned when the configured citation policy requires grounding and the produced
# answer resolved no citations. It carries no citations and a distinguishable route. The policy is
# OPTIONAL by default, so this path is inert (a zero-citation answer is served) unless a stricter policy
# is configured. Not cached: a later generation may cite, so each query is evaluated afresh.
CITATION_REQUIRED_MESSAGE = "This request cannot be answered from the available evidence with a citation."
CITATION_ROUTE = "citation_required"
CITATION_REFUSAL_REASON = "citation_policy"


def _blank_quotes(citations: tuple[Citation, ...]) -> tuple[Citation, ...]:
    # A verbatim citation quote is stored-chunk text; when the governor redacts the answer it must not
    # survive in a quote. The chunk id and label are kept so provenance still resolves.
    return tuple(replace(citation, quote="") for citation in citations)


# The image types the multimodal ingest path accepts. An image is validated against
# this allowlist and a size bound before its bytes are stored; the content is never
# decoded or executed, only stored and referenced.
IMAGE_MEDIA_TYPES = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})
# The most images loaded from the blob store and attached to one generation request.
MAX_GENERATION_IMAGES = 8


def _with_egress_provenance(usage: JsonObject, egress: TransformResult | None) -> JsonObject:
    """Fold generation-egress redaction provenance into a response's usage, or leave it unchanged.

    Provenance carries only opaque pattern ids and counts (and the dropped-image count), never a raw
    pattern or matched text. Returns the usage unchanged when no transform ran, or the transform
    redacted nothing and dropped no image, so the no-transform path is byte-identical.
    """
    if egress is None or (not egress.redactions and egress.images_dropped == 0):
        return usage
    provenance: JsonObject = {}
    if egress.redactions:
        provenance["redactions"] = [{"pattern_id": item.pattern_id, "count": item.count} for item in egress.redactions]
    if egress.images_dropped:
        provenance["images_dropped"] = egress.images_dropped
    return {**usage, "egress_redaction": provenance}


@dataclass(frozen=True, slots=True)
class _ImageIngest:
    """Carries an image ingest's resolved descriptor and media type into ``_ingest_once``.

    When present, the descriptor (an honest deterministic string or a real caption)
    is used as the document text instead of an extractor, and every produced chunk is
    stamped with the stored image blob key and media type so it becomes a citable
    image-descriptor chunk.
    """

    descriptor: str
    media_type: str


class AuthenticationService:
    def __init__(self, store: SQLStore, pepper: str) -> None:
        self.store = store
        self.pepper = pepper

    def issue(self, workspace_id: str, name: str, scopes: tuple[str, ...]) -> tuple[str, str]:
        raw, prefix = issue_api_key()
        digest = hash_api_key(raw, self.pepper)
        key_id = self.store.create_api_key(workspace_id, name, prefix, digest, scopes)
        return key_id, raw

    def authenticate(self, raw: str) -> tuple[str, frozenset[str], str]:
        prefix = raw[:14]
        candidate = self.store.get_api_key_candidate(prefix)
        if candidate is None or not verify_api_key(raw, str(candidate["digest"]), self.pepper):
            raise AuthenticationError("Invalid API key")
        self.store.touch_api_key(str(candidate["id"]))
        scopes = frozenset(str(value) for value in candidate["scopes"])
        return str(candidate["workspace_id"]), scopes, str(candidate["id"])


class IngestionService:
    def __init__(
        self,
        *,
        store: SQLStore,
        blob_store: BlobStore,
        extractors: ExtractorRegistry,
        chunker: ChunkingStrategy,
        embedder: Embedder,
        graph_extractor: GraphExtractor,
        search_index: SearchIndex,
        connectors: ConnectorRegistry,
        settings: Settings,
        contextual_retrieval: bool = False,
        hype_generator: HypotheticalQuestionGenerator | None = None,
        raptor_builder: RaptorTreeBuilder | None = None,
        image_captioner: ImageCaptioner | None = None,
        audit_sink: AuditSink | None = None,
        access_policy: AccessPolicy | None = None,
        kill_switches: KillSwitchRegistry | None = None,
        quota_policy: QuotaPolicy | None = None,
        derivation_service: DerivationService | None = None,
    ) -> None:
        self.store = store
        self.kill_switches = kill_switches
        self.quota_policy = quota_policy
        self.blob_store = blob_store
        self.extractors = extractors
        self.chunker = chunker
        self.embedder = embedder
        self.graph_extractor = graph_extractor
        self.search_index = search_index
        self.connectors = connectors
        self.settings = settings
        self.contextual_retrieval = contextual_retrieval
        self.hype_generator = hype_generator
        self.raptor_builder = raptor_builder
        self.image_captioner = image_captioner
        self.audit_sink = audit_sink
        self.access_policy = access_policy
        self.derivation_service = derivation_service

    def _refresh_derivation(self, workspace_id: str, corpus_id: str) -> None:
        # No-op when no service is installed; the service swallows and logs a refresh failure so it never
        # breaks the ingest or delete that just committed.
        if self.derivation_service is not None:
            self.derivation_service.refresh_quietly(workspace_id, corpus_id)

    def _charge_ingest_quota(self, workspace_id: str, corpus_id: str) -> None:
        # Count one ingest against its own dimension, so a tenant cannot exhaust embedding resources
        # through unbounded ingestion, and a query-request quota keeps its meaning unchanged. The
        # corpus is validated first, so a corpus-scoped rule never mints a counter row for a bogus id;
        # ingestion carries no request principal, so a principal-scoped rule skips rather than denies.
        # A no-op when no quota policy is installed.
        if self.quota_policy is None:
            return
        self.store.get_corpus(workspace_id, corpus_id)
        ctx = QuotaContext(
            workspace_id=workspace_id,
            corpus_id=corpus_id,
            corpus_validated=True,
            principal_bearing=False,
        )
        try:
            ensure_within_quota(self.quota_policy, ctx, dimension=QuotaDimension.INGEST_REQUESTS, idem=new_id("qta"))
        except QuotaExceededError:
            self._emit_ingest_quota_audit(workspace_id, corpus_id)
            raise

    def _budget_scope(self) -> AbstractContextManager[CostMeter | None]:
        return budget_scope(
            self.settings.run_token_budget,
            self.settings.run_cost_budget_usd,
            prices=prices_from(self.settings.model_prices),
        )

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
        ensure_workspace_active(self.kill_switches, workspace_id)
        ensure_scope_active(self.kill_switches, workspace_id, KillSwitchScope.WRITES)
        content, media_type, connector_metadata = self.connectors.fetch(source)
        display_source = sanitize_source_url(source, drop_query=True)
        parsed_name = display_source.rstrip("/").rsplit("/", 1)[-1] or display_source
        return self.ingest(
            workspace_id,
            corpus_id,
            DocumentInput(
                source_id=source_id or sanitize_source_url(source, drop_query=False),
                title=title or parsed_name,
                media_type=media_type,
                content=content,
                metadata={**connector_metadata, **(metadata or {})},
            ),
            actor=actor,
        )

    def ingest(
        self, workspace_id: str, corpus_id: str, document: DocumentInput, *, actor: str | None = None
    ) -> IngestResult:
        """Ingest a document, retrying once if a concurrent first-write won the source.

        ``_ingest_once`` raises ``ConflictError`` (from ``replace_document``) when a
        racing worker inserted the same (corpus, source) first. Its own compensation
        has already removed the loser's staged blob and restored the search index, so
        the single retry re-reads the now-existing winner row and takes the clean
        UPDATE branch. The offline single-process path never races, so it runs
        ``_ingest_once`` exactly once and is behavior-preserving.

        The run budget scope bounds the embed spend (document chunks plus any HyPE
        or RAPTOR summary embeddings) when a budget is configured; with no budget it
        is a no-op scope, so the deterministic path is unchanged.
        """
        ensure_workspace_active(self.kill_switches, workspace_id)
        ensure_scope_active(self.kill_switches, workspace_id, KillSwitchScope.WRITES)
        self._charge_ingest_quota(workspace_id, corpus_id)
        with self._budget_scope():
            try:
                result = self._ingest_once(workspace_id, corpus_id, document, actor=actor)
            except ConflictError:
                result = self._ingest_once(workspace_id, corpus_id, document, actor=actor)
        self._refresh_derivation(workspace_id, corpus_id)
        return result

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
        """Ingest an image as a citable document backed by a stored descriptor chunk.

        The image bytes are validated and stored in the blob store, and the ingested
        chunk carries the descriptor text plus the blob key (``image_ref``) and media
        type. Offline the descriptor is honest and deterministic: a supplied caption or
        alt-text when given, otherwise a descriptor derived from the image bytes digest
        and the title, never a fabricated description of pixels. A configured vision
        captioner may write the caption instead and degrades to the deterministic
        descriptor on failure. The descriptor is embedded and retrieved like any chunk,
        so the image competes in normal retrieval and is cited exactly like text.
        """
        ensure_workspace_active(self.kill_switches, workspace_id)
        ensure_scope_active(self.kill_switches, workspace_id, KillSwitchScope.WRITES)
        self._charge_ingest_quota(workspace_id, corpus_id)
        self._validate_image(content, media_type)
        resolved_title = (title or source_id).strip()
        descriptor, provenance = self._image_descriptor(
            content,
            media_type=media_type,
            title=resolved_title,
            caption=caption,
            alt_text=alt_text,
        )
        document = DocumentInput(
            source_id=source_id,
            title=title or source_id,
            media_type=media_type,
            content=content,
            metadata={**(metadata or {}), "modality": "image", "descriptor_source": provenance},
        )
        image = _ImageIngest(descriptor=descriptor, media_type=media_type)
        with self._budget_scope():
            try:
                result = self._ingest_once(workspace_id, corpus_id, document, image=image, actor=actor)
            except ConflictError:
                result = self._ingest_once(workspace_id, corpus_id, document, image=image, actor=actor)
        self._refresh_derivation(workspace_id, corpus_id)
        return result

    def _validate_image(self, content: bytes, media_type: str) -> None:
        if media_type not in IMAGE_MEDIA_TYPES:
            raise ValidationError(f"Unsupported image media type: {media_type}")
        if not content:
            raise ValidationError("Image content is empty")
        if len(content) > self.settings.image_max_bytes:
            raise ValidationError("Image exceeds the configured size limit")

    def _image_descriptor(
        self,
        content: bytes,
        *,
        media_type: str,
        title: str,
        caption: str | None,
        alt_text: str | None,
    ) -> tuple[str, str]:
        supplied = (caption or alt_text or "").strip()
        if supplied:
            return supplied, "provided"
        if self.image_captioner is not None:
            try:
                generated = self.image_captioner.caption(content, media_type=media_type, title=title).strip()
            except DependencyUnavailableError:
                generated = ""
            if generated:
                return generated, f"model:{self.image_captioner.provider_id}"
        return self._deterministic_descriptor(content, title), "deterministic"

    @staticmethod
    def _deterministic_descriptor(content: bytes, title: str) -> str:
        """An honest, model-free descriptor: the image bytes digest plus the title.

        Never a fabricated visual description of pixels the offline stack cannot see;
        it is a stable, reproducible label so re-ingesting an unchanged image yields
        an identical descriptor and the write stays idempotent.
        """
        short_digest = sha256_bytes(content)[:12]
        if title:
            return f"image asset {short_digest}, {title}"
        return f"image asset {short_digest}"

    def _ingest_once(
        self,
        workspace_id: str,
        corpus_id: str,
        document: DocumentInput,
        *,
        image: _ImageIngest | None = None,
        actor: str | None = None,
    ) -> IngestResult:
        corpus = self.store.get_corpus(workspace_id, corpus_id)
        if not document.content:
            raise ValidationError("Document content is empty")
        checksum = sha256_bytes(document.content)
        existing = self.store.get_document_by_source(workspace_id, corpus_id, document.source_id)
        # An ACL-only change keeps the content checksum identical, so the no-op fast path also
        # requires an unchanged permission fingerprint; a changed ACL must re-tag and re-version.
        permissions_unchanged = existing is not None and permission_fingerprint(
            existing["metadata"]
        ) == permission_fingerprint(document.metadata)
        if (
            existing is not None
            and existing["checksum"] == checksum
            and existing["status"] == "ready"
            and permissions_unchanged
        ):
            stats = self.store.document_stats(workspace_id, corpus_id, str(existing["id"]))
            return IngestResult(
                document_id=str(existing["id"]),
                corpus_id=corpus_id,
                corpus_version=corpus.version,
                chunks=int(stats["chunks"]),
                entities=int(stats["entities"]),
                relations=int(stats["relations"]),
                changed=False,
                checksum=checksum,
            )

        if (
            existing is not None
            and existing["checksum"] != checksum
            and self.store.is_document_held(workspace_id, corpus_id, str(existing["id"]))
        ):
            self._emit_hold_enforced(workspace_id, corpus_id, str(existing["id"]), "ingest.overwrite")
            raise LegalHoldError("Document is under legal hold")
        document_id = str(existing["id"]) if existing is not None else new_id("doc")
        blob_key = f"{workspace_id}/{corpus_id}/{document_id}/{checksum}"
        blob_uri = self.blob_store.put(blob_key, document.content, media_type=document.media_type)
        try:
            if image is not None:
                text = normalize_text(image.descriptor)
                extracted_metadata: JsonObject = {}
                if not text:
                    raise ValidationError("Image produced no descriptor")
            else:
                extracted = self.extractors.extract(document.media_type, document.content, document.metadata)
                text = normalize_text(extracted.text)
                # Governance keys are caller-owned; a plugin extractor cannot override the caller's ACL.
                extracted_metadata = strip_reserved_governance_keys(extracted.metadata)
            chunk_values = list(
                self.chunker.chunk(text, metadata={**document.metadata, **extracted_metadata}, settings=self.settings)
            )
            if not chunk_values:
                raise ValidationError("Document produced no indexable chunks")
            _validate_chunk_sequence(getattr(self.chunker, "strategy_id", "custom"), chunk_values)
            context_strategy = "contextual" if self.contextual_retrieval else "flat"
            embed_inputs = [
                contextualize(document.title, value.heading, value.text) if self.contextual_retrieval else value.text
                for value in chunk_values
            ]
            embeddings = self.embedder.embed(embed_inputs)
            if len(embeddings) != len(chunk_values):
                raise ConfigurationError("Embedder returned a different number of vectors")
            metadata = self._safe_metadata({**document.metadata, **extracted_metadata})
            image_ref = blob_uri if image is not None else None
            chunk_media_type = image.media_type if image is not None else None
            chunks = tuple(
                Chunk(
                    id=self._chunk_id(document_id, checksum, value.ordinal),
                    corpus_id=corpus_id,
                    document_id=document_id,
                    ordinal=value.ordinal,
                    text=value.text,
                    token_count=value.token_count,
                    checksum=sha256_text(value.text),
                    metadata={
                        **metadata,
                        "workspace_id": workspace_id,
                        "source_id": document.source_id,
                        "title": document.title,
                        "heading": value.heading,
                        "context_strategy": context_strategy,
                        **self._structural_metadata(document_id, checksum, value),
                    },
                    embedding=embedding,
                    image_ref=image_ref,
                    media_type=chunk_media_type,
                )
                for value, embedding in zip(chunk_values, embeddings, strict=True)
            )
            language = document.metadata.get("language")
            graph = self.graph_extractor.extract(
                corpus_id=corpus_id,
                document_id=document_id,
                text=text,
                language=language if isinstance(language, str) else None,
            )
            hypothetical_rows = self._hypothetical_rows(workspace_id, chunks)
            overwrite = existing is not None
            # An overwrite runs its whole mutate (the store write, the search index write, and the
            # RAPTOR rebuild) under the corpus destroy lock, so an interleaved erase can neither be
            # resurrected by a stale-id insert nor have this ingest write its vectors back after the
            # erase removed the row. A brand-new document has no such race and takes no outer lock.
            mutate_guard = self.store.document_destroy_lock(workspace_id, corpus_id) if overwrite else nullcontext()
            with mutate_guard:
                entity_term_rows: list[dict[str, object]] = []
                if self.settings.graph_term_index_enabled:
                    entity_term_rows = [
                        {
                            "id": f"et_{stable_digest((entity.id, term), length=48)}",
                            "entity_id": entity.id,
                            "workspace_id": workspace_id,
                            "corpus_id": entity.corpus_id,
                            "document_id": entity.document_id,
                            "term": term,
                        }
                        for entity in graph.entities
                        for term in dict.fromkeys(tokens(entity.normalized_name))
                    ]
                version = self.store.replace_document(
                    workspace_id=workspace_id,
                    corpus_id=corpus_id,
                    document_id=document_id,
                    source_id=document.source_id,
                    title=document.title,
                    media_type=document.media_type,
                    checksum=checksum,
                    blob_uri=blob_uri,
                    document_metadata=metadata,
                    expect_existing=overwrite,
                    actor=actor,
                    maintain_derivation_digest=self.derivation_service is not None,
                    chunk_rows=[
                        {
                            "id": chunk.id,
                            "workspace_id": workspace_id,
                            "corpus_id": chunk.corpus_id,
                            "document_id": chunk.document_id,
                            "ordinal": chunk.ordinal,
                            "text": chunk.text,
                            "token_count": chunk.token_count,
                            "checksum": chunk.checksum,
                            "metadata": chunk.metadata,
                            "embedding": list(chunk.embedding),
                            "embedding_model": self.embedder.model_id,
                            "image_ref": chunk.image_ref,
                            "media_type": chunk.media_type,
                        }
                        for chunk in chunks
                    ],
                    entity_rows=[
                        {
                            "id": entity.id,
                            "workspace_id": workspace_id,
                            "corpus_id": entity.corpus_id,
                            "document_id": entity.document_id,
                            "name": entity.name,
                            "normalized_name": entity.normalized_name,
                            "kind": entity.kind,
                            "metadata": entity.metadata,
                        }
                        for entity in graph.entities
                    ],
                    relation_rows=[
                        {
                            "id": relation.id,
                            "workspace_id": workspace_id,
                            "corpus_id": relation.corpus_id,
                            "document_id": relation.document_id,
                            "source_entity_id": relation.source_entity_id,
                            "target_entity_id": relation.target_entity_id,
                            "kind": relation.kind,
                            "weight": relation.weight,
                            "metadata": relation.metadata,
                        }
                        for relation in graph.relations
                    ],
                    hypothetical_rows=hypothetical_rows,
                    entity_term_rows=entity_term_rows,
                )
                # Under the same lock as the store write for an overwrite, so an interleaved erase
                # cannot slot between the row write and the vector write.
                self.search_index.replace_document(document_id, chunks)
                if self.raptor_builder is not None:
                    self.raptor_builder.rebuild(workspace_id, corpus_id)
        except LegalHoldError:
            # A place committed between the early gate and here (a genuine race); the store is the
            # atomic authority and refused it. Audit the enforcement, then propagate. The staged blob
            # is left for the deferred blob.gc sweep, as below.
            self._emit_hold_enforced(workspace_id, corpus_id, document_id, "ingest.overwrite")
            raise
        except Exception:
            # The staged blob is left for the deferred blob.gc sweep, never deleted
            # inline. A concurrent identical re-ingest computes the same content
            # addressed key and may have staged and be about to commit against it, so
            # an inline delete of a freshly written key could strand that winner. The
            # sweep's grace window keeps a just-written key and re-checks liveness
            # before reclaiming a genuinely orphaned one.
            raise
        # A superseded blob is left in place for the deferred blob.gc sweep to
        # reclaim; deleting it inline would race a concurrent revert to the old
        # content, which recreates and re-references this content-addressed key.
        return IngestResult(
            document_id=document_id,
            corpus_id=corpus_id,
            corpus_version=version,
            chunks=len(chunks),
            entities=len(graph.entities),
            relations=len(graph.relations),
            changed=True,
            checksum=checksum,
        )

    def delete(self, workspace_id: str, corpus_id: str, document_id: str) -> int:
        ensure_workspace_active(self.kill_switches, workspace_id)
        ensure_scope_active(self.kill_switches, workspace_id, KillSwitchScope.WRITES)
        # Hold the corpus destroy lock across the whole read-decide-mutate. Re-reading the document
        # under the lock means an interleaved erase that removed it is seen before the search index is
        # touched, so a stale restore can never resurrect erased vectors; store.delete_document
        # re-enters the same lock and does the atomic hold check and row delete.
        with self.store.document_destroy_lock(workspace_id, corpus_id):
            document = self.store.get_document(workspace_id, corpus_id, document_id)
            old_chunks = self.store.list_document_chunks(workspace_id, corpus_id, document_id)
            self.search_index.delete_document(document_id)
            try:
                version = self.store.delete_document(workspace_id, corpus_id, document_id)
            except LegalHoldError:
                if old_chunks:
                    self.search_index.replace_document(document_id, old_chunks)
                self._emit_hold_enforced(workspace_id, corpus_id, document_id, "document.delete")
                raise
            except Exception:
                if old_chunks:
                    self.search_index.replace_document(document_id, old_chunks)
                raise
            if self.raptor_builder is not None:
                self.raptor_builder.rebuild(workspace_id, corpus_id)
            try:
                self.blob_store.delete(str(document["blob_uri"]))
            except Exception:
                logger.warning("deleted_document_blob_cleanup_failed", document_id=document_id)
            self._refresh_derivation(workspace_id, corpus_id)
            return version

    def _emit_hold_enforced(self, workspace_id: str, corpus_id: str, document_id: str, capability_name: str) -> None:
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
            capability_name=capability_name,
            dedup_id=stable_digest(
                ("hold.enforced", workspace_id, corpus_id, document_id, capability_name, occurred.isoformat())
            ),
        )
        # Best-effort, never fail-closed: the destroy is already refused and safe, so a sink outage
        # must not also crash the refusal path. The human reason is never carried (row-only).
        emit_audit(sink, event, fail_closed=False, logger=logger, log_key="legal_hold_audit_sink_failed")

    def _emit_ingest_quota_audit(self, workspace_id: str, corpus_id: str) -> None:
        sink = self.audit_sink
        if sink is None:
            return
        dimension = QuotaDimension.INGEST_REQUESTS.value
        event = AuditEvent(
            event_type="quota.denied",
            occurred_at=self.store.clock.now(),
            workspace_id=workspace_id,
            corpus_id=corpus_id,
            principal_id="system",
            principal_kind="system",
            tenant=None,
            policy_id=str(getattr(self.quota_policy, "cache_id", "quota") or "quota"),
            policy_version="",
            capability_name=dimension,
            action_fingerprint=stable_digest((workspace_id, dimension, corpus_id)),
            dedup_id=stable_digest((workspace_id, corpus_id, dimension, "quota")),
        )
        emit_audit(sink, event, fail_closed=False, logger=logger)

    def _hypothetical_rows(self, workspace_id: str, chunks: Sequence[Chunk]) -> list[JsonObject]:
        """Deterministic HyPE rows: one embedded hypothetical question per generated line.

        Empty (so the ``chunk_hypotheticals`` table stays empty and ingestion is
        byte-identical) unless a generator is injected. The row id is derived from
        the deterministic chunk id, the question ordinal, and the question text, so
        re-ingesting an unchanged document reproduces identical ids and the write
        stays idempotent. The questions are embedded in the same space as the
        chunks, in one batched call, so a query can match question-to-question.
        """
        if self.hype_generator is None:
            return []
        pending: list[tuple[Chunk, int, str]] = []
        for chunk in chunks:
            for ordinal, question in enumerate(self.hype_generator.questions(chunk.text, workspace_id=workspace_id)):
                pending.append((chunk, ordinal, question))
        if not pending:
            return []
        embeddings = self.embedder.embed([question for _chunk, _ordinal, question in pending])
        if len(embeddings) != len(pending):
            raise ConfigurationError("Embedder returned a different number of hypothetical vectors")
        return [
            {
                "id": f"hyp_{stable_digest((chunk.id, str(ordinal), question), length=48)}",
                "workspace_id": workspace_id,
                "corpus_id": chunk.corpus_id,
                "document_id": chunk.document_id,
                "chunk_id": chunk.id,
                "ordinal": ordinal,
                "question": question,
                "embedding": list(embedding),
                "embedding_model": self.embedder.model_id,
            }
            for (chunk, ordinal, question), embedding in zip(pending, embeddings, strict=True)
        ]

    @staticmethod
    def _chunk_id(document_id: str, checksum: str, ordinal: int) -> str:
        return f"chunk_{stable_digest((document_id, checksum, str(ordinal)), length=48)}"

    @classmethod
    def _structural_metadata(cls, document_id: str, checksum: str, value: TextChunk) -> JsonObject:
        """Linkage metadata for a hierarchical chunk; empty for a flat chunk.

        Flat chunks omit the ``granularity`` key entirely so their stored rows stay
        byte-identical to the pre-hierarchical layout. A child records the id of its
        parent section (derived from the same deterministic id function) plus the
        character span into the parent text, which is what windowed retrieval widens.
        """
        if not value.granularity:
            return {}
        if value.granularity != "child":
            return {"granularity": value.granularity}
        return {
            "granularity": "child",
            "parent_id": cls._chunk_id(document_id, checksum, value.parent_ordinal or 0),
            "sent_start": value.sent_start,
            "sent_end": value.sent_end,
        }

    @classmethod
    def _safe_metadata(cls, value: JsonObject) -> JsonObject:
        return {key: cls._redact_value(item) for key, item in value.items() if not cls._is_sensitive_key(key)}

    @classmethod
    def _redact_value(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: cls._redact_value(item) for key, item in value.items() if not cls._is_sensitive_key(key)}
        # Recurse into tuples and other non-string sequences too, so a secret nested in
        # a tuple is stripped instead of surviving into the JSON output. A tuple becomes
        # a list, which is what JSON stores anyway, so nothing sensitive-free changes.
        if isinstance(value, (list, tuple)):
            return [cls._redact_value(item) for item in value]
        return value

    @staticmethod
    def _is_sensitive_key(key: Any) -> bool:
        if not isinstance(key, str):
            return False
        folded = key.casefold()
        if folded in SAFE_METADATA_KEYS:
            return False
        parts = tuple(part for part in _METADATA_TOKEN_SPLIT.split(folded) if part)
        if not parts:
            return False
        if any(part in SENSITIVE_METADATA_TOKENS for part in parts):
            return True
        return any(
            parts[start : start + len(phrase)] == phrase
            for phrase in SENSITIVE_METADATA_PHRASES
            for start in range(len(parts) - len(phrase) + 1)
        )


_SHADOW_EXECUTION: ContextVar[bool] = ContextVar("refaat_shadow_execution", default=False)


@contextmanager
def shadow_execution() -> Iterator[None]:
    """Read context but persist no durable side effect for the duration of the scope.

    A shadow or canary run under the deployment router executes the unserved
    artifact for comparison only. It must still read the session and any cached
    value, since the session summary feeds the context hash and retrieval and the
    comparison must stay apples-to-apples, but for the scope's duration it must
    surface nothing durable: no session turn, no answer-cache row, and no
    model-call cache row. This scope suppresses only the writes, never the reads.
    The served run always executes OUTSIDE this scope and writes exactly once.

    A contextvar does not cross into an executor a custom program might spawn, so
    an unserved execution that runs its own work in a separate thread or process
    escapes this scope; that is why the deployment router refuses to shadow a
    program that is not ``shadow_safe``.
    """
    token = _SHADOW_EXECUTION.set(True)
    try:
        yield
    finally:
        _SHADOW_EXECUTION.reset(token)


def in_shadow_execution() -> bool:
    """True while durable writes are suppressed: an unserved shadow or canary run, or the served
    degraded-response path that must not persist a transient outage answer."""
    return _SHADOW_EXECUTION.get()


_UNSERVED_RUN: ContextVar[bool] = ContextVar("refaat_unserved_run", default=False)


@contextmanager
def unserved_run() -> Iterator[None]:
    """An unserved deployment shadow or canary run: suppresses durable writes plus the served quota and
    audit, unlike a bare ``shadow_execution`` scope (the served degraded path) that suppresses only writes."""
    token = _UNSERVED_RUN.set(True)
    try:
        with shadow_execution():
            yield
    finally:
        _UNSERVED_RUN.reset(token)


def in_unserved_run() -> bool:
    """True while an unserved deployment shadow or canary run is executing (see ``unserved_run``)."""
    return _UNSERVED_RUN.get()


@dataclass(frozen=True, slots=True)
class QueryPrep:
    request: QueryRequest
    query: str
    corpus_version: int
    cache_scope: str
    query_hash: str
    query_embedding: tuple[float, ...]
    profile: JsonObject
    supplemental: JsonObject
    policy: RetrievalPolicy


class QueryService:
    def __init__(
        self,
        *,
        store: SQLStore,
        embedder: Embedder,
        generator: Generator,
        retriever: AdaptiveRetriever,
        harness: HarnessService,
        settings: Settings,
        clock: Clock,
        blob_store: BlobStore | None = None,
        cache_store: CacheStore | None = None,
        multimodal: bool = False,
        audit_sink: AuditSink | None = None,
        model_router: ModelRouter | None = None,
        output_governor: OutputGovernor | None = None,
        context_transform: ContextTransform | None = None,
        kill_switches: KillSwitchRegistry | None = None,
        quota_policy: QuotaPolicy | None = None,
        degradation_policy: DegradationPolicy | None = None,
        strict_authorization: bool = False,
    ) -> None:
        self.store = store
        # Default to the store so the SQL path is byte-identical.
        self.cache_store = cache_store if cache_store is not None else store
        self.embedder = embedder
        self.generator = generator
        self.retriever = retriever
        self.harness = harness
        self.settings = settings
        self.clock = clock
        self.kill_switches = kill_switches
        # Opt-in persisted per-scope quota seam. None by default, so admission and model spend are
        # ungoverned and the path is byte-identical. When set, a request counts against its quota
        # after the kill-switch freeze and before generation, and reserves then reconciles its spend.
        self.quota_policy = quota_policy
        self.compactor = SessionCompactor(store, settings, classify_evidence=model_router is not None)
        # Fail closed (on by default) for an in-process call that omits a principal under an active policy;
        # set strict_authorization off to opt out.
        self.strict_authorization = strict_authorization
        self.fallback = ExtractiveGenerator()
        self.blob_store = blob_store
        self.audit_sink = audit_sink
        # Opt-in data-classification model-routing seam. None by default, so both the
        # non-streaming and streaming paths skip the routing check and are byte-identical.
        self.model_router = model_router
        # Opt-in pre-model context-transform seam. None by default, so the evidence sent to the
        # generator is untouched and the path is byte-identical. When set, the transform redacts
        # configured spans from the evidence a generation model sees, after the routing gate and
        # before the generate/stream call; the local extractive fallback keeps the original evidence.
        self.context_transform = context_transform
        # Opt-in output-governance seam. None by default, so the answer is returned and
        # cached exactly as before and the streaming path stays incremental. When set, the
        # governor reviews the final answer and citations in _finalize before return or
        # cache, and the streaming path buffers so no un-recallable token is emitted.
        self.output_governor = output_governor
        # Load and attach image pixels only when a multimodal generator is active. Off
        # by default, so a text-only corpus and a run with no vision model attach no
        # images and are byte-identical to before.
        self.multimodal = multimodal and blob_store is not None
        # Opt-in degradation seam for a generation provider outage. None by default, so an
        # outage propagates uncaught exactly as before (the API maps it to a 503). When set,
        # the policy decides whether to serve a grounded degraded answer, abstain, or refuse.
        self.degradation_policy = degradation_policy

    def _budget_scope(self) -> AbstractContextManager[CostMeter | None]:
        return budget_scope(
            self.settings.run_token_budget,
            self.settings.run_cost_budget_usd,
            prices=prices_from(self.settings.model_prices),
        )

    def _quota_context(self, request: QueryRequest) -> QuotaContext | None:
        """The trusted quota inputs for a request, or None when no quota policy is installed.

        Built only from server-derived fields; ``corpus_validated`` is True because this is called
        after ``_prepare`` read and so validated the corpus, so an arbitrary client corpus id can
        never mint a fresh counter row.
        """
        if self.quota_policy is None:
            return None
        # An unserved deployment run must not consume the shared served quota and starve the served run.
        if in_unserved_run():
            return None
        principal = request.principal
        return QuotaContext(
            workspace_id=request.workspace_id,
            tenant=principal.tenant if principal is not None else None,
            principal_id=principal.id if principal is not None else None,
            principal_bearing=True,
            corpus_id=request.corpus_id,
            corpus_validated=True,
        )

    def _quota_run(self, ctx: QuotaContext | None, meter: CostMeter | None, idem: str) -> AbstractContextManager[None]:
        return quota_run(
            self.quota_policy,
            ctx,
            meter=meter,
            token_ceiling=self.settings.run_token_budget,
            cost_ceiling_usd=self.settings.run_cost_budget_usd,
            idem=idem,
        )

    def query(self, request: QueryRequest) -> QueryResponse:
        ensure_workspace_active(self.kill_switches, request.workspace_id)
        # A generation switch denies the whole answer path (a cached answer included); retrieval alone,
        # which generates nothing, still serves. Guarded at entry so a cache hit cannot bypass it.
        ensure_scope_active(self.kill_switches, request.workspace_id, KillSwitchScope.GENERATION)
        fail_closed = self.settings.audit_fail_closed
        with self._budget_scope() as meter:
            prep = self._prepare(request)
            ctx = self._quota_context(request)
            idem = new_id("qta") if ctx is not None else ""
            if ctx is not None:
                self._admit_request(request, ctx, idem)
            cached = self._cached(prep)
            if cached is not None:
                self._record_session(prep.request, prep.query, cached)
                self._emit_query_audit(request, cached, fail_closed=fail_closed)
                return cached
            context, generation_request = self._retrieve(prep)
            if self._should_abstain(context):
                response = self._abstained_response(prep, context)
                self._emit_query_audit(request, response, fail_closed=fail_closed)
                return response
            if self.model_router is not None:
                route = self.model_router.route(
                    self._evidence_classification(context),
                    request.principal,
                    residency_zones=self._evidence_residency_zones(context),
                )
                if not route.allowed:
                    response = self._routing_refusal(prep, context)
                    self._emit_query_audit(request, response, fail_closed=fail_closed)
                    return response
            try:
                with self._quota_run(ctx, meter, idem):
                    egress = redact_for_egress(self.context_transform, generation_request)
                    try:
                        generated = self.generator.generate(egress.request)
                    except DependencyUnavailableError:
                        return self._handle_generation_outage(
                            request, prep, context, generation_request, fail_closed=fail_closed
                        )
                    response = self._finalize(prep, context, generation_request, generated, egress=egress)
                    self._emit_query_audit(request, response, fail_closed=fail_closed)
                    return response
            except QuotaExceededError:
                self._emit_quota_audit(request, ctx, dimension=QuotaDimension.GENERATION_TOKENS)
                raise

    def _admit_request(self, request: QueryRequest, ctx: QuotaContext, idem: str) -> None:
        try:
            ensure_within_quota(self.quota_policy, ctx, dimension=QuotaDimension.REQUESTS, idem=idem)
        except QuotaExceededError:
            self._emit_quota_audit(request, ctx, dimension=QuotaDimension.REQUESTS)
            raise

    def _served_audit_sink(self) -> AuditSink | None:
        # None for an unserved deployment run only; the served degraded path (also under shadow) still audits.
        return None if in_unserved_run() else self.audit_sink

    def _emit_quota_audit(self, request: QueryRequest, ctx: QuotaContext | None, *, dimension: QuotaDimension) -> None:
        """Record one redacted quota.denied event on a quota denial.

        No-op unless an audit sink is installed and a quota policy governs the request, so the
        default path emits nothing. Best-effort: the request is already denied, so a sink outage
        must not also crash the refusal. Carries only ids, the dimension, and machine digests,
        never a query or evidence text.
        """
        sink = self._served_audit_sink()
        if sink is None or ctx is None:
            return
        principal = request.principal
        event = AuditEvent(
            event_type="quota.denied",
            occurred_at=self.clock.now(),
            workspace_id=ctx.workspace_id,
            corpus_id=request.corpus_id,
            principal_id=principal.id if principal is not None else "system",
            principal_kind=principal.kind.value if principal is not None else "system",
            tenant=ctx.tenant,
            policy_id=str(getattr(self.quota_policy, "cache_id", "quota") or "quota"),
            policy_version="",
            capability_name=dimension.value,
            action_fingerprint=stable_digest((ctx.workspace_id, dimension.value, ctx.principal_id or "")),
            dedup_id=stable_digest((request.workspace_id, request.corpus_id, dimension.value, "quota")),
        )
        emit_audit(sink, event, fail_closed=False, logger=logger)

    def _handle_generation_outage(
        self,
        request: QueryRequest,
        prep: QueryPrep,
        context: ContextPacket,
        generation_request: GenerationRequest,
        *,
        fail_closed: bool,
    ) -> QueryResponse:
        """Resolve a generation provider outage under the installed degradation policy.

        Called only from within the ``except DependencyUnavailableError`` block, so a REFUSE
        verdict re-raises the active outage with a bare ``raise``. With no policy installed the
        verdict is the legacy refuse, so the outage propagates unchanged (a 503). A degraded or
        abstained response is never written to the answer cache or session: a transient outage must
        not poison durable state, so it is built under a shadow scope that suppresses those writes.
        """
        failure, decision = self._resolve_generation_outage(context)
        if decision.mode is DegradationMode.REFUSE:
            raise
        if decision.mode is DegradationMode.ABSTAIN:
            with shadow_execution():
                response = self._abstained_response(prep, context)
        else:
            generated = self.fallback.generate(generation_request)
            with shadow_execution():
                response = self._finalize(prep, context, generation_request, generated, degraded=True)
        self._emit_degradation_audit(request, failure, decision, response)
        self._emit_query_audit(request, response, fail_closed=fail_closed)
        return response

    def _resolve_generation_outage(self, context: ContextPacket) -> tuple[DependencyFailure, DegradationDecision]:
        """Describe a generation provider outage and resolve it under the installed policy.

        Grounding survives when retrieved hits are present, so the extractive fallback can produce a
        cited answer. With no policy installed the verdict is the legacy refuse.
        """
        failure = DependencyFailure(
            dependency_class=DependencyClass.GENERATION,
            grounding_survives=bool(context.hits),
            provider_id=self.generator.model_id,
            reason="generation_provider_unavailable",
        )
        return failure, resolve_degradation(self.degradation_policy, failure)

    def _degraded_stream_events(
        self,
        request: QueryRequest,
        prep: QueryPrep,
        context: ContextPacket,
        generation_request: GenerationRequest,
        failure: DependencyFailure,
        decision: DegradationDecision,
        meter: CostMeter | None,
    ) -> Iterator[dict[str, Any]]:
        """Emit the streaming events for a degraded or abstained generation outage.

        Reached only for a non-refuse verdict resolved before the first token was emitted. The
        degraded or abstained response is built under a shadow scope so the transient outage never
        writes the answer cache or session, then one delta and the authoritative final event follow.
        """
        if decision.mode is DegradationMode.ABSTAIN:
            with shadow_execution():
                response = self._abstained_response(prep, context)
        else:
            with use_meter(meter):
                generated = self.fallback.generate(generation_request)
            with shadow_execution():
                response = self._finalize(prep, context, generation_request, generated, degraded=True)
        self._emit_degradation_audit(request, failure, decision, response)
        self._emit_query_audit(request, response, fail_closed=self.settings.audit_fail_closed)
        yield {"type": "delta", "text": response.answer}
        yield self._final_event(response, replaced=decision.mode is DegradationMode.SERVE_DEGRADED)

    def _emit_degradation_audit(
        self,
        request: QueryRequest,
        failure: DependencyFailure,
        decision: DegradationDecision,
        response: QueryResponse,
    ) -> None:
        """Record one redacted degradation.decided event when a degradation policy is installed.

        No-op when no audit sink or no degradation policy is installed, so the default path emits
        nothing. Best-effort, never fail-closed: a degradation is not protection-decreasing, so a
        sink outage must not also crash the served path. The event carries only the dependency
        class, mode, machine reason code, and the bounded provider id, never a query or evidence
        text. Principal and policy fields fall back to a system sentinel when authorization is off.
        """
        sink = self._served_audit_sink()
        if sink is None or self.degradation_policy is None:
            return
        policy = self.retriever.access_policy
        principal = request.principal
        event = AuditEvent(
            event_type="degradation.decided",
            occurred_at=self.clock.now(),
            workspace_id=request.workspace_id,
            corpus_id=request.corpus_id,
            corpus_version=response.corpus_version,
            principal_id=principal.id if principal is not None else "system",
            principal_kind=principal.kind.value if principal is not None else "system",
            tenant=principal.tenant if principal is not None else None,
            policy_id=str(getattr(policy, "name", type(policy).__name__)) if policy is not None else "",
            policy_version=str(getattr(policy, "version", "")) if policy is not None else "",
            cited_document_ids=(),
            citation_count=len(response.citations),
            abstained=response.abstained,
            cached=False,
            trace_id=response.trace_id,
            capability_name=f"{decision.dependency_class.value}:{decision.mode.value}:{decision.reason}"[:200],
            action_fingerprint=(failure.provider_id or "unknown")[:128],
            dedup_id=stable_digest((response.trace_id or "", request.workspace_id, request.corpus_id, "degradation")),
        )
        emit_audit(sink, event, fail_closed=False, logger=logger, log_key="degradation_audit_sink_failed")

    def _emit_query_audit(self, request: QueryRequest, response: QueryResponse, *, fail_closed: bool = False) -> None:
        """Record one redacted audit event for a served query when authorization is active.

        No-op unless an audit sink is installed and the request carries a principal under an
        active policy; with no sink or no policy the path is byte-identical, and a principal-less
        request under a policy is denied earlier by strict authorization. Best-effort by default: a sink
        failure is logged and swallowed so a read query still serves. Under fail_closed a sink
        failure raises so the query refuses rather than serving unaudited; the streaming path sets it
        on the before-first-byte emissions and buffers under an active policy so the terminal audit
        fires before any answer delta.
        """
        sink = self._served_audit_sink()
        policy = self.retriever.access_policy
        principal = request.principal
        if sink is None or policy is None or principal is None:
            return
        cited_documents = tuple(sorted({citation.document_id for citation in response.citations}))
        event = AuditEvent(
            event_type="query.authorized",
            occurred_at=self.clock.now(),
            workspace_id=request.workspace_id,
            corpus_id=request.corpus_id,
            corpus_version=response.corpus_version,
            principal_id=principal.id,
            principal_kind=principal.kind.value,
            tenant=principal.tenant,
            policy_id=str(getattr(policy, "name", type(policy).__name__)),
            policy_version=str(getattr(policy, "version", "")),
            cited_document_ids=cited_documents,
            citation_count=len(response.citations),
            abstained=response.abstained,
            cached=response.cached,
            trace_id=response.trace_id,
            dedup_id=stable_digest((response.trace_id or "", request.workspace_id, request.corpus_id, principal.id)),
        )
        emit_audit(sink, event, fail_closed=fail_closed, logger=logger)

    def stream_query(self, request: QueryRequest) -> Iterator[dict[str, Any]]:
        ensure_workspace_active(self.kill_switches, request.workspace_id)
        ensure_scope_active(self.kill_switches, request.workspace_id, KillSwitchScope.GENERATION)
        # A stream is driven across per-iteration context copies by the ASGI
        # threadpool, so the run meter is resolved once as a plain object and
        # re-bound transiently around each synchronous unit (never across a yield),
        # instead of one budget_scope whose ContextVar reset would straddle the
        # yields and break across the copy boundary. With no budget and no outer
        # meter the meter is None and every use_meter is a no-op, so the
        # deterministic offline stream is byte-identical.
        meter = resolve_stream_meter(
            self.settings.run_token_budget,
            self.settings.run_cost_budget_usd,
            prices=prices_from(self.settings.model_prices),
        )
        with use_meter(meter):
            prep = self._prepare(request)
            cached = self._cached(prep)
        ctx = self._quota_context(request)
        idem = new_id("qta") if ctx is not None else ""
        if ctx is not None:
            self._admit_request(request, ctx, idem)
        if cached is not None:
            with use_meter(meter):
                self._record_session(prep.request, prep.query, cached)
            self._emit_query_audit(request, cached, fail_closed=self.settings.audit_fail_closed)
            yield {"type": "delta", "text": cached.answer}
            yield self._final_event(cached)
            return
        with use_meter(meter):
            context, generation_request = self._retrieve(prep)
        if self._should_abstain(context):
            with use_meter(meter):
                response = self._abstained_response(prep, context)
            self._emit_query_audit(request, response, fail_closed=self.settings.audit_fail_closed)
            yield {"type": "delta", "text": response.answer}
            yield self._final_event(response)
            return
        if self.model_router is not None:
            route = self.model_router.route(
                self._evidence_classification(context),
                request.principal,
                residency_zones=self._evidence_residency_zones(context),
            )
            if not route.allowed:
                with use_meter(meter):
                    response = self._routing_refusal(prep, context)
                self._emit_query_audit(request, response, fail_closed=self.settings.audit_fail_closed)
                yield {"type": "delta", "text": response.answer}
                yield self._final_event(response)
                return
        # Redact the evidence sent to the model after the routing gate and before either stream
        # branch. The original generation_request is kept for _finalize (the extractive fallback
        # grounds and cites against the real evidence). Byte-identical with no transform: the seam
        # returns the same request object.
        egress = redact_for_egress(self.context_transform, generation_request)
        # Reserve the run's token and cost ceilings before any model spend; a no-op when unconfigured.
        try:
            hold = quota_reserve_all(
                self.quota_policy,
                ctx,
                token_ceiling=self.settings.run_token_budget,
                cost_ceiling_usd=self.settings.run_cost_budget_usd,
                idem=idem,
            )
        except QuotaExceededError:
            self._emit_quota_audit(request, ctx, dimension=QuotaDimension.GENERATION_TOKENS)
            raise
        committed = False
        try:
            if self._buffered_streaming(request):
                # Buffered streaming for any whole-answer decision: an output governor reviews the WHOLE
                # answer, and a citation policy may refuse a zero-citation answer, so no un-recallable token
                # may be emitted before that decision. Consume the stream without yielding, finalize (the
                # governor and the citation policy both run there), then emit exactly one delta with the
                # governed answer or the refusal, and the final event.
                try:
                    generated = self._drain_stream(egress.request, meter)
                except DependencyUnavailableError:
                    failure, decision = self._resolve_generation_outage(context)
                    if decision.mode is DegradationMode.REFUSE:
                        raise
                    yield from self._degraded_stream_events(
                        request, prep, context, generation_request, failure, decision, meter
                    )
                    return
                with use_meter(meter):
                    response = self._finalize(prep, context, generation_request, generated, egress=egress)
                # Audit before committing the quota hold, so a fail-closed audit outage releases the
                # reservation rather than charging the tenant for an answer that is never served, matching
                # the synchronous query path.
                self._emit_query_audit(request, response, fail_closed=self.settings.audit_fail_closed)
                quota_commit_all(self.quota_policy, hold, meter)
                committed = True
                yield {"type": "delta", "text": response.answer}
                yield self._final_event(response, replaced=response.answer != generated.answer)
                return
            parts: list[str] = []
            stream_usage: dict[str, Any] = {}
            yielded_any = False
            try:
                inner = self.generator.stream(egress.request)
                while True:
                    with use_meter(meter):
                        try:
                            piece = next(inner)
                        except StopIteration as stop:
                            if isinstance(stop.value, dict):
                                stream_usage = stop.value
                            break
                    parts.append(piece)
                    yielded_any = True
                    yield {"type": "delta", "text": piece}
            except DependencyUnavailableError:
                # An outage after the first token has already left cannot be retracted, so it
                # propagates as today; only an outage before any delta is degradable.
                if yielded_any:
                    raise
                failure, decision = self._resolve_generation_outage(context)
                if decision.mode is DegradationMode.REFUSE:
                    raise
                yield from self._degraded_stream_events(
                    request, prep, context, generation_request, failure, decision, meter
                )
                return
            text = "".join(parts)
            generated = GenerationResult(
                answer=text,
                cited_chunk_ids=extract_cited_ids(text),
                usage=stream_usage or {"provider": "stream"},
            )
            with use_meter(meter):
                response = self._finalize(prep, context, generation_request, generated, egress=egress)
            quota_commit_all(self.quota_policy, hold, meter)
            committed = True
            self._emit_query_audit(request, response)
            yield self._final_event(response, replaced=response.answer != text)
        finally:
            if hold is not None and not committed:
                quota_release_all(self.quota_policy, hold)

    def _buffered_streaming(self, request: QueryRequest) -> bool:
        # A whole-answer decision (output governor review, or a citation policy that may refuse a
        # zero-citation answer) cannot let an un-recallable token leave before the decision, so the stream
        # is buffered. When neither is active the stream is emitted directly, byte-identical to before.
        if self.output_governor is not None or self.settings.citation_policy != "optional":
            return True
        # Under fail-closed audit the terminal query event must fire before any byte leaves, so a sink
        # outage denies rather than serving unaudited. Buffer only when the audit would actually emit (a
        # served sink, an active policy, and a principal); otherwise the incremental path is byte-identical.
        return (
            self.settings.audit_fail_closed
            and self._served_audit_sink() is not None
            and self.retriever.access_policy is not None
            and request.principal is not None
        )

    def _drain_stream(self, generation_request: GenerationRequest, meter: CostMeter | None) -> GenerationResult:
        """Consume a streamed generation to completion WITHOUT yielding, for the buffered
        governed path. Accumulates the pieces and captures the final-chunk usage, so a
        governed stream can review the whole answer before any token reaches the caller."""
        parts: list[str] = []
        stream_usage: dict[str, Any] = {}
        inner = self.generator.stream(generation_request)
        while True:
            with use_meter(meter):
                try:
                    piece = next(inner)
                except StopIteration as stop:
                    if isinstance(stop.value, dict):
                        stream_usage = stop.value
                    break
            parts.append(piece)
        text = "".join(parts)
        return GenerationResult(
            answer=text,
            cited_chunk_ids=extract_cited_ids(text),
            usage=stream_usage or {"provider": "stream"},
        )

    def _prepare(self, request: QueryRequest) -> QueryPrep:
        # Fail closed before the cache read so an omitted principal cannot be served an unfiltered
        # cached answer either. On by default; set strict off to restore the trusted-in-process path.
        require_principal_present(self.retriever.access_policy, request.principal, strict=self.strict_authorization)
        # Local import: model_cache imports in_shadow_execution from this module, so a top-level
        # import here would be circular.
        from refaat.infrastructure.model_cache import resolved_cache_identity

        query = normalize_text(request.query)
        if not query:
            raise ValidationError("Query is empty")
        request = replace(request, query=query)
        corpus = self.store.get_corpus(request.workspace_id, request.corpus_id)
        identity_scope = self._scope(request)
        profile = self.store.get_preferences(
            request.workspace_id,
            request.corpus_id,
            preference_scope(self.retriever.access_policy, request.principal, request.user_id),
        )
        supplemental = self.harness.active(request.workspace_id, request.corpus_id)
        policy = self.retriever.resolve_policy(request, supplemental)
        session_summary = self._session_summary(request)
        context_hash = self._context_hash(
            policy=policy,
            profile=profile,
            supplemental=supplemental,
            session_summary=session_summary,
            embedding_model=self.embedder.model_id,
            generation_model=self.generator.model_id,
            reranker=(
                "none"
                if self.retriever.reranker is None
                else str(getattr(self.retriever.reranker, "cache_id", type(self.retriever.reranker).__name__))
            ),
            query_transform=(
                "none"
                if self.retriever.query_transform is None
                else str(
                    getattr(
                        self.retriever.query_transform,
                        "cache_id",
                        type(self.retriever.query_transform).__name__,
                    )
                )
            ),
            guidance=guidance_digest(request.instructions, request.demonstrations),
            response_schema=(
                sha256_text(
                    json.dumps(request.response_schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                )
                if request.response_schema is not None
                else None
            ),
            router_id=request.router_id,
            hype=self.retriever.hype_cache_id,
            raptor=self.retriever.raptor_cache_id,
            abstain_floor=self.settings.abstain_below_score,
            citation_policy=str(self.settings.citation_policy),
            source_trust_weight=self.settings.source_trust_weight,
            model_router=(str(self.model_router.cache_id) if self.model_router is not None else None),
            output_governor=(str(self.output_governor.cache_id) if self.output_governor is not None else None),
            egress_redaction=(str(self.context_transform.cache_id) if self.context_transform is not None else None),
            degradation=(str(self.degradation_policy.cache_id) if self.degradation_policy is not None else None),
            auth_scope=(
                auth_scope_hash(request.principal, self.retriever.access_policy)
                if self.retriever.access_policy is not None and request.principal is not None
                else None
            ),
            embedding_cache_identity=resolved_cache_identity(self.embedder),
            generation_cache_identity=resolved_cache_identity(self.generator),
            search_backend=(
                cache_id if isinstance(cache_id := getattr(self.retriever.index, "cache_id", None), str) else None
            ),
            graph_backend=self.retriever.graph_cache_id,
            index_generation=(
                self.store.get_active_index_generation(request.workspace_id, request.corpus_id)
                if self.settings.index_generations_enabled
                else None
            ),
        )
        return QueryPrep(
            request=request,
            query=query,
            corpus_version=corpus.version,
            cache_scope=self._cache_scope(identity_scope, context_hash),
            query_hash=sha256_text(query.casefold()),
            query_embedding=self.embedder.embed([query])[0],
            profile=profile,
            supplemental=supplemental,
            policy=policy,
        )

    def _cached(self, prep: QueryPrep) -> QueryResponse | None:
        request = prep.request
        exact = self.cache_store.get_exact_cache(
            request.workspace_id,
            request.corpus_id,
            prep.corpus_version,
            prep.cache_scope,
            prep.query_hash,
        )
        if exact is not None and self._cache_hit_authorized(request, exact.response):
            return self._served_cache_hit(request, exact.response, "cag:exact")
        for candidate in self.cache_store.list_cache_candidates(
            request.workspace_id,
            request.corpus_id,
            prep.corpus_version,
            prep.cache_scope,
            limit=50,
        ):
            if cosine_similarity(
                prep.query_embedding, candidate.query_embedding
            ) >= self.settings.semantic_cache_threshold and self._cache_hit_authorized(request, candidate.response):
                hit = self._served_cache_hit(request, candidate.response, "cag:semantic")
                if hit is not None:
                    return hit
        return None

    def _served_cache_hit(self, request: QueryRequest, response: QueryResponse, route: str) -> QueryResponse | None:
        revalidated = self._revalidate_citations(request, response)
        if not revalidated.citations and self.settings.citation_policy != "optional":
            # A hit that revalidated to zero citations (its cited chunks were deleted) is refused as a MISS
            # under a non-optional citation policy, so the pipeline regenerates and re-evaluates the policy.
            return None
        return self._cached_response(revalidated, route)

    def _cache_hit_authorized(self, request: QueryRequest, response: QueryResponse) -> bool:
        """Whether a cached answer may still be served to the request principal.

        A document ACL change can make a previously-cited document forbidden while the
        principal's entitlements (and so the cache scope) are unchanged, and the cached
        answer text still discusses it. A hit is served only when every cited chunk still
        exists and is still authorized; otherwise it is a miss. Inactive when unconfigured.
        """
        policy = self.retriever.access_policy
        principal = request.principal
        if policy is None or principal is None or not response.citations:
            return True
        cited_ids = tuple(citation.chunk_id for citation in response.citations)
        by_id = {chunk.id: chunk for chunk in self.store.get_chunks(request.workspace_id, request.corpus_id, cited_ids)}
        return all(
            chunk_id in by_id and chunk_allowed(policy, principal, by_id[chunk_id].metadata) for chunk_id in cited_ids
        )

    def _revalidate_citations(self, request: QueryRequest, response: QueryResponse) -> QueryResponse:
        """Re-resolve a cached response's citations against the currently stored chunks.

        A cache row is scoped to a corpus version, but a chunk can vanish between the
        version read in ``_prepare`` and the cache read here, a concurrent delete that
        advances the version after this request already captured the older one. Routing
        the cached citation ids through the same stored-chunk resolution the fresh
        answer uses drops any whose chunk no longer exists, so a returned citation
        always resolves to a stored chunk. A hit whose citations all still resolve is
        returned unchanged.
        """
        if not response.citations:
            return response
        cited_ids = tuple(citation.chunk_id for citation in response.citations)
        citation_chunks = self._citation_chunks(request, cited_ids)
        if len(citation_chunks) == len(response.citations):
            return response
        resolved_ids = tuple(chunk.id for chunk in citation_chunks)
        citations = build_citations(resolved_ids, citation_chunks)
        if OUTPUT_ROUTE in response.routes:
            citations = _blank_quotes(citations)
        return replace(response, answer=strip_uncited_markers(response.answer, resolved_ids), citations=citations)

    def _retrieve(self, prep: QueryPrep) -> tuple[ContextPacket, GenerationRequest]:
        context = self.retriever.retrieve(
            prep.request,
            prep.query_embedding,
            profile=prep.profile,
            supplemental_harness=prep.supplemental,
            policy=prep.policy,
        )
        generation_request = GenerationRequest(
            query=prep.query,
            context=context,
            system_prompt=system_prompt(prep.supplemental),
            instructions=prep.request.instructions,
            demonstrations=prep.request.demonstrations,
            response_schema=prep.request.response_schema,
            workspace_id=prep.request.workspace_id,
            images=self._images_for(prep.request, context),
        )
        return context, generation_request

    def _images_for(self, request: QueryRequest, context: ContextPacket) -> tuple[GenerationImage, ...]:
        """Load the pixels of retrieved image-descriptor hits for a vision generator.

        Empty unless multimodal generation is active, so the default extractive path
        never touches the blob store and a text-only request is byte-identical. Only a
        retrieved hit that carries an ``image_ref`` and media type contributes an
        image, keyed by its real stored chunk id; a failed blob read is skipped rather
        than failing the query. The offline extractive generator ignores these images
        regardless, so they change nothing until a real vision model is configured.
        """
        if not self.multimodal or self.blob_store is None:
            return ()
        images: list[GenerationImage] = []
        for hit in context.hits:
            chunk = hit.chunk
            if not chunk.image_ref or not chunk.media_type:
                continue
            try:
                with self.blob_store.open(chunk.image_ref) as handle:
                    content = handle.read()
            except Exception:
                logger.warning("multimodal_image_load_failed", chunk_id=chunk.id)
                continue
            images.append(GenerationImage(chunk_id=chunk.id, media_type=chunk.media_type, content=content))
            if len(images) >= MAX_GENERATION_IMAGES:
                break
        return tuple(images)

    def _finalize(
        self,
        prep: QueryPrep,
        context: ContextPacket,
        generation_request: GenerationRequest,
        generated: GenerationResult,
        *,
        egress: TransformResult | None = None,
        degraded: bool = False,
    ) -> QueryResponse:
        request = prep.request
        chunks = [hit.chunk for hit in context.hits]
        cited_ids = filter_cited_chunks(generated.cited_chunk_ids, chunks)
        # The primary generation already spent its tokens. If its citations do not
        # resolve we swap in the grounded extractive fallback answer, but the model
        # call still happened, so the real usage is kept rather than the fallback's
        # zero, so cost accounting reflects what was actually spent. The fallback runs
        # against the ORIGINAL generation_request (not the redacted egress request), so
        # a fallback answer stays grounded in and cites the real stored evidence.
        model_usage = generated.usage
        if chunks and not cited_ids:
            generated = self.fallback.generate(generation_request)
            cited_ids = filter_cited_chunks(generated.cited_chunk_ids, chunks)
        citation_chunks = self._citation_chunks(request, cited_ids)
        resolved_ids = tuple(chunk.id for chunk in citation_chunks)
        citations = build_citations(resolved_ids, citation_chunks)
        if citation_policy_refuses(
            CitationPolicy(self.settings.citation_policy),
            has_evidence=bool(chunks),
            has_citations=bool(citations),
        ):
            return self._citation_refusal(prep, context)
        usage: JsonObject = {
            **model_usage,
            "embedding_model": self.embedder.model_id,
            "generation_model": self.generator.model_id,
            "context_tokens": context.token_count,
            "retrieved_chunks": len(context.hits),
        }
        usage = _with_egress_provenance(usage, egress)
        response = QueryResponse(
            answer=strip_uncited_markers(generated.answer, resolved_ids),
            citations=citations,
            routes=context.routes,
            cached=False,
            corpus_version=prep.corpus_version,
            trace_id=current_trace_id(),
            usage=usage,
            degraded=degraded,
        )
        governor = self.output_governor
        if governor is not None:
            decision = self._review_output(governor, request, response)
            if not decision.allowed:
                # A denial is never cached and never a session turn: the governor (or an
                # ACL that shapes the answer) must be re-evaluated on the next query, so
                # _output_refusal returns without _persist.
                refusal = self._output_refusal(prep, context)
                self._emit_output_denied_audit(request, decision, refusal)
                return refusal
            if decision.redacted_answer is not None:
                response = replace(
                    response,
                    answer=decision.redacted_answer,
                    citations=_blank_quotes(response.citations),
                    routes=(*response.routes, OUTPUT_ROUTE),
                )
        return self._persist(prep, response)

    def _review_output(
        self, governor: OutputGovernor, request: QueryRequest, response: QueryResponse
    ) -> OutputDecision:
        """Run the installed output governor over a produced answer, fail-closed on error.

        A governor that raises is treated as a denial: a broken review can never let
        unreviewed output through. Only reached when a governor is installed.
        """
        try:
            return governor.review(
                OutputContext(
                    answer=response.answer,
                    citations=response.citations,
                    principal=request.principal,
                    workspace_id=request.workspace_id,
                    corpus_id=request.corpus_id,
                )
            )
        except Exception:
            logger.warning("output_governor_error")
            return OutputDecision(allowed=False, reason="output_governor_error")

    def _output_refusal(self, prep: QueryPrep, context: ContextPacket) -> QueryResponse:
        """Build a governed no-output response for a rejected answer, WITHOUT persisting.

        Mirrors ``_routing_refusal`` (fixed message, no citations, a distinguishable route)
        but a denial is never cached and never becomes a session turn, so it does not call
        ``_persist``.
        """
        return QueryResponse(
            answer=OUTPUT_REFUSAL_MESSAGE,
            citations=(),
            routes=(*context.routes, OUTPUT_ROUTE),
            cached=False,
            corpus_version=prep.corpus_version,
            trace_id=current_trace_id(),
            usage={
                "provider": "output_denied",
                "reason": OUTPUT_REFUSAL_REASON,
                "embedding_model": self.embedder.model_id,
                "generation_model": self.generator.model_id,
                "context_tokens": context.token_count,
                "retrieved_chunks": len(context.hits),
            },
            abstained=False,
        )

    def _citation_refusal(self, prep: QueryPrep, context: ContextPacket) -> QueryResponse:
        """Build a no-answer for a produced answer that resolved no citations under a citation policy
        requiring grounding, WITHOUT persisting: a later generation may cite, so each query is evaluated
        afresh rather than cached, mirroring the output-review refusal.
        """
        return QueryResponse(
            answer=CITATION_REQUIRED_MESSAGE,
            citations=(),
            routes=(*context.routes, CITATION_ROUTE),
            cached=False,
            corpus_version=prep.corpus_version,
            trace_id=current_trace_id(),
            usage={
                "provider": "citation_required",
                "reason": CITATION_REFUSAL_REASON,
                "embedding_model": self.embedder.model_id,
                "generation_model": self.generator.model_id,
                "context_tokens": context.token_count,
                "retrieved_chunks": len(context.hits),
            },
            abstained=False,
        )

    def _emit_output_denied_audit(
        self, request: QueryRequest, decision: OutputDecision, response: QueryResponse
    ) -> None:
        """Record a redacted output.denied audit event when authorization is active.

        Best-effort, never fail-closed: the answer is already withheld, so a sink outage
        must not also crash the refusal path. No-op unless an audit sink is installed and
        the request carries a principal under an active policy. The event carries the
        governor's bounded machine reason in capability_name and never the answer text.
        """
        sink = self._served_audit_sink()
        policy = self.retriever.access_policy
        principal = request.principal
        if sink is None or policy is None or principal is None:
            return
        event = AuditEvent(
            event_type="output.denied",
            occurred_at=self.clock.now(),
            workspace_id=request.workspace_id,
            corpus_id=request.corpus_id,
            corpus_version=response.corpus_version,
            principal_id=principal.id,
            principal_kind=principal.kind.value,
            tenant=principal.tenant,
            policy_id=str(getattr(policy, "name", type(policy).__name__)),
            policy_version=str(getattr(policy, "version", "")),
            cited_document_ids=(),
            citation_count=0,
            abstained=False,
            cached=False,
            trace_id=response.trace_id,
            capability_name=bounded_reason(decision.reason),
            dedup_id=stable_digest((response.trace_id or "", request.workspace_id, request.corpus_id, principal.id)),
        )
        emit_audit(sink, event, fail_closed=False, logger=logger)

    def _persist(self, prep: QueryPrep, response: QueryResponse) -> QueryResponse:
        entry = CacheEntry(
            id=new_id("cache"),
            workspace_id=prep.request.workspace_id,
            corpus_id=prep.request.corpus_id,
            corpus_version=prep.corpus_version,
            user_scope=prep.cache_scope,
            query=prep.query,
            query_hash=prep.query_hash,
            query_embedding=prep.query_embedding,
            response=response,
            expires_at=self.clock.now() + timedelta(seconds=self.settings.cache_ttl_seconds),
            created_at=self.clock.now(),
        )
        if not in_shadow_execution():
            self.cache_store.put_cache(entry)
        self._record_session(prep.request, prep.query, response)
        return response

    def _should_abstain(self, context: ContextPacket) -> bool:
        """True when abstention is enabled and the best evidence is below the floor.

        The floor is a similarity threshold, so it is compared against the maximum
        per-hit semantic (vector cosine) score, not the fused rank score. With
        abstain_below_score at 0.0 (the default) this short-circuits to False before
        inspecting any hit, so the pipeline is byte-identical when the feature is off.
        """
        threshold = self.settings.abstain_below_score
        if threshold <= 0.0:
            return False
        best = max((hit.semantic_score for hit in context.hits), default=0.0)
        return best < threshold

    def _abstained_response(self, prep: QueryPrep, context: ContextPacket) -> QueryResponse:
        """Build and persist an honest no-answer: no citations, a distinguishable route.

        The abstain response is deterministic for its scope, so it is cached under the
        same corpus-version-scoped answer cache as a grounded answer; a repeated
        identical query returns it from the cache with abstained still set.
        """
        response = QueryResponse(
            answer=ABSTAIN_MESSAGE,
            citations=(),
            routes=(*context.routes, ABSTAIN_ROUTE),
            cached=False,
            corpus_version=prep.corpus_version,
            trace_id=current_trace_id(),
            usage={
                "provider": "abstain",
                "embedding_model": self.embedder.model_id,
                "generation_model": self.generator.model_id,
                "context_tokens": context.token_count,
                "retrieved_chunks": len(context.hits),
            },
            abstained=True,
        )
        return self._persist(prep, response)

    @staticmethod
    def _evidence_classification(context: ContextPacket) -> Classification:
        """The maximum classification across every piece of evidence that reaches the model.

        The prompt carries the retrieved hits, the graph facts, and the session summary, each of
        which can be authorized yet higher-classified, so the ceiling is the max over all three. All
        are derived deterministically from ACL tags (see ``domain.authz.max_classification``), never
        by a model: the router only maps this value to a verdict.
        """
        return max(
            max_classification(hit.chunk.metadata for hit in context.hits),
            context.graph_evidence_classification,
            context.session_summary_classification,
        )

    @staticmethod
    def _evidence_residency_zones(context: ContextPacket) -> frozenset[str]:
        """The declared residency zones across every piece of evidence that reaches the model.

        Mirrors :meth:`_evidence_classification`: the hits, the graph facts, and the session summary
        all reach the model, so the egress zone gate unions all three. Untagged or malformed evidence
        contributes a sentinel that no operator zone set admits, so a zone-gating rule fails closed.
        """
        return (
            residency_zones(hit.chunk.metadata for hit in context.hits)
            | context.graph_evidence_zones
            | context.session_summary_zones
        )

    def _routing_refusal(self, prep: QueryPrep, context: ContextPacket) -> QueryResponse:
        """Build and persist a governed no-generation response for a denied route.

        Mirrors ``_abstained_response`` but carries the fixed routing refusal message
        and a distinguishable route, with no citations, so a refusal is deterministic
        for its scope and never sends evidence to the model.
        """
        response = QueryResponse(
            answer=ROUTING_REFUSAL_MESSAGE,
            citations=(),
            routes=(*context.routes, ROUTING_ROUTE),
            cached=False,
            corpus_version=prep.corpus_version,
            trace_id=current_trace_id(),
            usage={
                "provider": "routing_denied",
                "reason": ROUTING_REFUSAL_REASON,
                "embedding_model": self.embedder.model_id,
                "generation_model": self.generator.model_id,
                "context_tokens": context.token_count,
                "retrieved_chunks": len(context.hits),
            },
            abstained=False,
        )
        return self._persist(prep, response)

    def _citation_chunks(
        self,
        request: QueryRequest,
        cited_ids: tuple[str, ...],
    ) -> list[Chunk]:
        """Resolve cited ids to their STORED chunk rows, dropping any that are missing.

        Context compression and hierarchical expansion rewrite or widen the
        in-memory hit text the generator sees, but a citation.quote must stay a
        verbatim contiguous excerpt of the stored corpus chunk. Only the stored rows
        are trusted here: a cited id whose row cannot be read is dropped rather than
        quoted from the in-memory (possibly expanded, compressed, or synthesized)
        chunk, so a quote can never come from text that is not in the corpus.
        """
        if not cited_ids:
            return []
        by_id = {chunk.id: chunk for chunk in self.store.get_chunks(request.workspace_id, request.corpus_id, cited_ids)}
        resolved = [by_id[chunk_id] for chunk_id in cited_ids if chunk_id in by_id]
        access_policy = self.retriever.access_policy
        principal = request.principal
        if access_policy is not None and principal is not None:
            # Defense in depth: a citation can never resolve to a chunk the principal
            # may not see, even if one slipped past the retrieval enforcement point.
            resolved = [chunk for chunk in resolved if chunk_allowed(access_policy, principal, chunk.metadata)]
        return resolved

    @staticmethod
    def _final_event(response: QueryResponse, *, replaced: bool = False) -> dict[str, Any]:
        return {
            "type": "final",
            "answer": response.answer,
            "citations": [asdict(citation) for citation in response.citations],
            "routes": list(response.routes),
            "cached": response.cached,
            "corpus_version": response.corpus_version,
            "trace_id": response.trace_id,
            "usage": response.usage,
            "abstained": response.abstained,
            "replaced": replaced,
        }

    def _session_summary(self, request: QueryRequest) -> str | None:
        if request.session_id is None:
            return None
        session = self.store.get_session(request.workspace_id, request.session_id)
        if session["corpus_id"] != request.corpus_id:
            raise ValidationError("Session belongs to another corpus")
        summary = session.get("summary")
        if not summary:
            return None
        stored_scope = cast(JsonObject, session.get("metadata") or {}).get("summary_auth_scope")
        policy = self.retriever.access_policy
        if not session_summary_visible(stored_scope, request.principal, policy):
            return None
        if policy is not None and request.principal is not None:
            documents = self.store.list_documents(request.workspace_id, request.corpus_id)
            cited = stored_summary_cited_documents(session)
            if not summary_documents_authorized(policy, request.principal, cited, documents):
                return None
        return str(summary)

    @staticmethod
    def _context_hash(
        *,
        policy: RetrievalPolicy,
        profile: JsonObject,
        supplemental: JsonObject,
        session_summary: str | None,
        embedding_model: str,
        generation_model: str,
        reranker: str,
        query_transform: str,
        guidance: str | None,
        response_schema: str | None,
        router_id: str | None = None,
        hype: str | None = None,
        raptor: str | None = None,
        abstain_floor: float = 0.0,
        citation_policy: str = "optional",
        source_trust_weight: float = 0.0,
        model_router: str | None = None,
        output_governor: str | None = None,
        egress_redaction: str | None = None,
        degradation: str | None = None,
        auth_scope: str | None = None,
        embedding_cache_identity: str | None = None,
        generation_cache_identity: str | None = None,
        search_backend: str | None = None,
        graph_backend: str | None = None,
        index_generation: str | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "base_prompt_hash": BASE_SYSTEM_PROMPT_HASH,
            "embedding_model": embedding_model,
            "generation_model": generation_model,
            "policy": policy.as_json(),
            "profile": profile,
            "reranker": reranker,
            "query_transform": query_transform,
            "session_summary": session_summary,
            "supplemental_harness": supplemental,
        }
        if guidance is not None:
            payload["guidance"] = guidance
        if response_schema is not None:
            payload["response_schema"] = response_schema
        if router_id is not None:
            payload["router_id"] = router_id
        if hype is not None:
            payload["hype"] = hype
        if raptor is not None:
            payload["raptor"] = raptor
        # An abstain decision depends on the floor, so an enabled floor scopes the
        # answer cache to that exact value: a cached abstain (or a grounded answer
        # that a different floor would have refused) can never be served under
        # another floor, and the default 0.0 folds nothing so the disabled path
        # stays byte-identical to before the feature existed.
        if abstain_floor > 0.0:
            payload["abstain_floor"] = abstain_floor
        # A non-optional citation policy can refuse an answer that would otherwise be served, and never
        # caches its refusal, so it scopes the answer cache to that policy: an answer cached under a
        # permissive policy is never served under a stricter one. The default "optional" folds nothing so
        # the disabled path stays byte-identical.
        if citation_policy != "optional":
            payload["citation_policy"] = citation_policy
        # The source-trust down-rank reorders candidates before the top_k, so a different
        # weight can change which chunks reach the context and the answer. An enabled weight
        # scopes the answer cache to that exact value; the default 0.0 folds nothing so the
        # disabled path stays byte-identical to before the feature existed.
        if source_trust_weight > 0.0:
            payload["source_trust_weight"] = source_trust_weight
        # Installing, removing, or changing the model router changes the routing verdict,
        # so its identity scopes the cache; folded in only when a router is active, so the
        # no-router path stays byte-identical.
        if model_router is not None:
            payload["model_router"] = model_router
        # Installing, removing, or changing the output governor changes the served answer
        # (a refusal, a redaction, or nothing), and a cache hit returns before the governor
        # runs, so the governor identity scopes the cache. Folded in only when a governor is
        # active, so the no-governor path stays byte-identical.
        if output_governor is not None:
            payload["output_governor"] = output_governor
        # Installing, removing, or changing the generation-egress redactor changes the evidence the
        # model sees and so the produced answer, and a cache hit returns before generation, so the
        # redactor identity scopes the cache. Folded in only when a transform is active, so the
        # no-transform path stays byte-identical.
        if egress_redaction is not None:
            payload["egress_redaction"] = egress_redaction
        # Installing, removing, or changing the degradation policy changes what a generation
        # outage serves (a degraded answer, an abstain, or a refuse), and a cache hit returns
        # before generation, so the policy identity scopes the cache. Folded in only when a policy
        # is active, so the no-policy path stays byte-identical.
        if degradation is not None:
            payload["degradation"] = degradation
        # A permission change on the principal misses old entries; folded in only when active.
        if auth_scope is not None:
            payload["auth_scope"] = auth_scope
        # Two providers can share a bare model name (a proxy or a re-pointed endpoint), and one name
        # can serve text or vision, so the adapter cache identity (endpoint, dimensions, mode) scopes
        # the answer cache alongside the model id. Folded in only when the adapter exposes one, so the
        # deterministic offline stack (no cache identity) stays byte-identical.
        if embedding_cache_identity is not None:
            payload["embedding_cache_identity"] = embedding_cache_identity
        if generation_cache_identity is not None:
            payload["generation_cache_identity"] = generation_cache_identity
        # An injected backend can select different evidence, so its cache_id scopes the cache; the
        # built-in indexes expose none, so the default path stays byte-identical.
        if search_backend is not None:
            payload["search_backend"] = search_backend
        # An injected graph read backend can change the graph facts and boosts, so its cache_id scopes
        # the cache; the default store exposes none, so the default path stays byte-identical.
        if graph_backend is not None:
            payload["graph_backend"] = graph_backend
        # A physical-only generation flip does not bump corpus_version, so the active generation's
        # content identity scopes the cache; folded only when active, so the default stays byte-identical.
        if index_generation is not None:
            payload["index_generation"] = index_generation
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return sha256_text(canonical)

    @staticmethod
    def _cache_scope(identity_scope: str, context_hash: str) -> str:
        kind = identity_scope.partition(":")[0]
        identity_hash = sha256_text(identity_scope)[:24]
        return f"{kind}:{identity_hash}:context:{context_hash[:32]}"

    def _record_session(self, request: QueryRequest, query: str, response: QueryResponse) -> None:
        if not request.session_id:
            return
        if in_shadow_execution():
            return
        session = self.store.get_session(request.workspace_id, request.session_id)
        if session["corpus_id"] != request.corpus_id:
            raise ValidationError("Session belongs to another corpus")
        access_policy = self.retriever.access_policy
        auth_scope = (
            auth_scope_hash(request.principal, access_policy)
            if access_policy is not None and request.principal is not None
            else None
        )
        self.store.append_session_turn(
            request.workspace_id,
            request.session_id,
            "user",
            query,
            tag_turn_metadata(request.metadata, auth_scope),
        )
        self.store.append_session_turn(
            request.workspace_id,
            request.session_id,
            "assistant",
            response.answer,
            tag_turn_metadata(
                {
                    "citations": [citation.chunk_id for citation in response.citations],
                    # Stable across a re-ingest, so erasure still matches the turn after a replace.
                    "cited_document_ids": sorted({citation.document_id for citation in response.citations}),
                },
                auth_scope,
            ),
        )
        self.compactor.compact_if_needed(request.workspace_id, request.session_id, auth_scope)

    @staticmethod
    def _scope(request: QueryRequest) -> str:
        # Under an active policy the scope binds to the authenticated principal, so a
        # client-supplied user_id or session_id cannot select another principal's learned
        # profile or cache. Byte-identical with no policy; a principal-less request under a
        # policy is denied earlier by strict authorization.
        if request.principal is not None:
            if request.session_id:
                return f"session:{request.principal.id}:{request.session_id}"
            return f"user:{request.principal.id}"
        if request.session_id:
            return f"session:{request.session_id}"
        if request.user_id:
            return f"user:{request.user_id}"
        return "anonymous"

    @staticmethod
    def _cached_response(response: QueryResponse, route: str) -> QueryResponse:
        return replace(
            response,
            routes=(route, *response.routes),
            cached=True,
            trace_id=current_trace_id(),
            usage={**response.usage, "cache": route},
        )


class EvaluationService:
    def __init__(self, store: SQLStore, query_service: QueryService) -> None:
        self.store = store
        self.query_service = query_service

    def run(self, workspace_id: str, suite_id: str, corpus_id: str) -> str:
        ensure_workspace_active(self.query_service.kill_switches, workspace_id)
        # Evaluation runs queries with no principal, so it fails closed under an active policy
        # rather than serve unfiltered evidence; this covers the in-process, HTTP, and worker
        # reach paths at once. Inert when no policy is installed.
        if self.query_service.retriever.access_policy is not None:
            raise AuthorizationError("Evaluation is unavailable while an access policy is active")
        suite = self.store.get_evaluation_suite(workspace_id, suite_id)
        cases = suite["cases"]
        if not isinstance(cases, list) or not cases:
            raise ValidationError("Evaluation suite has no cases")
        run_id = self.store.create_evaluation_run(workspace_id, suite_id, corpus_id)
        results: list[JsonObject] = []
        answer_hits = 0
        citation_hits = 0
        grounded = 0
        try:
            for index, case_value in enumerate(cases):
                if not isinstance(case_value, dict):
                    raise ValidationError(f"Evaluation case {index} is invalid")
                query = str(case_value.get("query", "")).strip()
                expected_terms = [str(value).casefold() for value in case_value.get("expected_terms", [])]
                expected_documents = {str(value) for value in case_value.get("expected_document_ids", [])}
                response = self.query_service.query(
                    QueryRequest(
                        workspace_id=workspace_id,
                        corpus_id=corpus_id,
                        query=query,
                        user_id=f"evaluation:{run_id}",
                    )
                )
                answer_ok = all(term in response.answer.casefold() for term in expected_terms)
                cited_documents = {citation.document_id for citation in response.citations}
                citation_ok = not expected_documents or expected_documents <= cited_documents
                grounded_ok = bool(response.citations) or not response.answer.strip()
                answer_hits += int(answer_ok)
                citation_hits += int(citation_ok)
                grounded += int(grounded_ok)
                results.append(
                    {
                        "index": index,
                        "query": query,
                        "answer_ok": answer_ok,
                        "citation_ok": citation_ok,
                        "grounded": grounded_ok,
                        "cited_documents": sorted(cited_documents),
                        "routes": list(response.routes),
                    }
                )
            count = len(cases)
            metrics: JsonObject = {
                "case_count": count,
                "answer_hit_rate": answer_hits / count,
                "citation_recall": citation_hits / count,
                "grounded_rate": grounded / count,
            }
            self.store.finish_evaluation_run(
                workspace_id,
                run_id,
                metrics=metrics,
                results=results,
            )
        except Exception as exc:
            self.store.finish_evaluation_run(
                workspace_id,
                run_id,
                metrics={},
                results=results,
                error=error_message(exc),
            )
            raise
        return run_id


class FeedbackService:
    def __init__(
        self,
        store: SQLStore,
        access_policy: AccessPolicy | None = None,
        kill_switches: KillSwitchRegistry | None = None,
        strict_authorization: bool = False,
    ) -> None:
        self.kill_switches = kill_switches
        self.access_policy = access_policy
        self.strict_authorization = strict_authorization
        self.learner = PreferenceLearner(store, access_policy)

    def submit(self, workspace_id: str, corpus_id: str, feedback: Feedback) -> None:
        ensure_workspace_active(self.kill_switches, workspace_id)
        ensure_scope_active(self.kill_switches, workspace_id, KillSwitchScope.WRITES)
        # The per-chunk ACL check is skipped when principal is None, so strict must deny here.
        require_principal_present(self.access_policy, feedback.principal, strict=self.strict_authorization)
        self.learner.learn(workspace_id, corpus_id, feedback)
