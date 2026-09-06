from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Protocol, runtime_checkable

from refaat.core.sparse import BM25Params
from refaat.domain.fine_tuning import FineTuneRequest, FineTuneResult
from refaat.domain.models import (
    BlobEntry,
    CacheEntry,
    Chunk,
    ContextPacket,
    Entity,
    ExtractedDocument,
    GenerationRequest,
    GenerationResult,
    GraphExtraction,
    JsonObject,
    JudgeVerdict,
    MetadataFilter,
    Relation,
    SearchHit,
    ToolCallDelta,
    ToolProposal,
    ToolSpec,
    TransformedQuery,
)
from refaat.domain.sandbox import SandboxLimits, SandboxResult
from refaat.domain.secrets import SecretScope, SecretValue

if TYPE_CHECKING:
    from refaat.config import Settings
    from refaat.core.chunking import TextChunk


class Embedder(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]: ...


class Generator(Protocol):
    @property
    def model_id(self) -> str: ...

    def generate(self, request: GenerationRequest) -> GenerationResult: ...

    def stream(self, request: GenerationRequest) -> Iterator[str]: ...


@runtime_checkable
class CacheIdentified(Protocol):
    """Opt-in marker exposed by a real, cacheable model adapter.

    ``cache_identity`` fingerprints the exact provider serving a call (its base
    URL and model, plus dimensions for an embedder), so a cached value can never
    be served across two backends that happen to share a bare model name. The
    built-in deterministic adapters do not expose it, so wrapping is a positive
    opt-in: only an adapter that returns a stable identity is ever cached, and a
    custom deterministic adapter cannot be silently cached and served stale after
    a config change.
    """

    @property
    def cache_identity(self) -> str: ...


class ToolPlanner(Protocol):
    """Proposes the next tool to call in a governed ReAct loop.

    ``plan`` returns a :class:`ToolProposal` (tool plus arguments) or ``None`` to
    stop. The deterministic keyword planner and a model planner both satisfy this
    one seam, so the governed loop body is planner-agnostic.
    """

    def tools(self) -> tuple[str, ...]: ...

    def plan(self, query: str, used: frozenset[str]) -> ToolProposal | None: ...


class ToolCallingGenerator(Protocol):
    """Optional sibling of :class:`Generator` for models that can choose a tool.

    Kept separate from :class:`Generator` on purpose: the offline extractive
    generator cannot honestly choose tools and must not be forced to carry a
    stub. Only a function-calling provider implements this Protocol.
    """

    @property
    def model_id(self) -> str: ...

    def propose_tool_call(self, query: str, tools: Sequence[ToolSpec], used: frozenset[str]) -> ToolProposal | None: ...


class StreamingToolCallingGenerator(Protocol):
    """Streaming sibling of :class:`ToolCallingGenerator`.

    Kept separate from :class:`ToolCallingGenerator` and :class:`Generator` for
    the same honesty reason: only a real function-calling provider that can
    stream tool-call fragments implements this. ``stream_tool_call`` yields
    provider-neutral :class:`ToolCallDelta` fragments; the neutral reducer, not
    the adapter, decides when the call is complete.
    """

    @property
    def model_id(self) -> str: ...

    def stream_tool_call(
        self, query: str, tools: Sequence[ToolSpec], used: frozenset[str]
    ) -> Iterator[ToolCallDelta]: ...


class GraphExtractor(Protocol):
    @property
    def model_id(self) -> str: ...

    def extract(
        self, *, corpus_id: str, document_id: str, text: str, language: str | None = None
    ) -> GraphExtraction: ...


class JudgeModel(Protocol):
    """An LLM-as-judge that scores an answer against its cited evidence.

    Kept separate from :class:`Generator` on purpose: the offline extractive
    generator cannot honestly judge and must not be forced to carry a stub. Only
    an injected model implements this seam, so evaluation degrades to the
    deterministic Faithfulness proxy when no judge is configured.
    """

    @property
    def model_id(self) -> str: ...

    def assess(self, query: str, answer: str, evidence: Sequence[str]) -> JudgeVerdict: ...


class Reranker(Protocol):
    def rerank(self, query: str, hits: Sequence[SearchHit]) -> tuple[SearchHit, ...]: ...


class CrossEncoderScorer(Protocol):
    """Injected model seam for a cross-encoder reranker.

    Keeps the cross-encoder model library out of core: the real scorer, a
    sentence-transformers cross-encoder or a hosted rerank endpoint, is supplied
    by the caller. ``score`` returns one relevance score per document, aligned
    with the input order.
    """

    @property
    def model_id(self) -> str: ...

    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]: ...


class QueryTransform(Protocol):
    @property
    def cache_id(self) -> str: ...

    def expand(self, query: str, max_variants: int, *, workspace_id: str = "") -> tuple[TransformedQuery, ...]: ...


class ChunkingStrategy(Protocol):
    """Splits a normalized document body into ordered chunks at ingestion.

    ``chunk`` runs once per document and proposes only the text, ordinal, and
    structure of each chunk; the pipeline mints every chunk id and checksum from
    ``(document_id, checksum, ordinal)``, so a strategy can never forge an id or
    collide with a reserved namespace. ``strategy_id`` is a stable name. A strategy
    that emits parent and child chunks may also expose an optional ``expansion_mode``
    property returning ``"parent_child"`` or ``"sentence_window"`` so retrieval expands
    its structure at query time; an omitted property means ``"flat"`` (no expansion).
    The three built-in strategies and any injected domain splitter satisfy this one
    seam, and contextual embedding, HyPE, and RAPTOR compose on the output without
    knowing which strategy ran.
    """

    @property
    def strategy_id(self) -> str: ...

    def chunk(self, text: str, *, metadata: JsonObject, settings: Settings) -> Sequence[TextChunk]: ...


class HypotheticalQuestionGenerator(Protocol):
    """Produces the hypothetical questions a chunk answers, for HyPE indexing.

    ``questions`` runs at ingestion, once per chunk; each returned question is
    embedded in the shared vector space and stored so a query can match
    question-to-question at retrieval. ``provider_id`` is a stable identity folded
    into the answer-cache key. The deterministic templated generator and a
    model-backed generator both satisfy this one seam.
    """

    @property
    def provider_id(self) -> str: ...

    def questions(self, chunk_text: str, *, workspace_id: str = "") -> Sequence[str]: ...


class Summarizer(Protocol):
    """Summarizes a cluster of child texts into one node summary for a RAPTOR tree.

    ``summarize`` runs at ingestion, once per internal tree node; the returned
    summary is embedded in the shared vector space and stored so a query can match a
    high-level node and resolve down to its real leaf chunks. The call sees passage
    text only, never a stored chunk id, so a summary can never carry or forge a
    citation. ``provider_id`` is a stable identity folded into the answer-cache key.
    The deterministic extractive summarizer and a model-backed summarizer both
    satisfy this one seam.
    """

    @property
    def provider_id(self) -> str: ...

    def summarize(self, texts: Sequence[str], *, workspace_id: str = "") -> str: ...


class ImageCaptioner(Protocol):
    """Produces a descriptor caption for an image at ingestion time.

    Kept as a seam, not a hard dependency: offline there is no captioner and the
    ingestion service builds an honest deterministic descriptor from the image bytes
    digest and the provided title, never a fabricated description of pixels it cannot
    see. Only a real multimodal model implements this Protocol, and it is imported
    lazily and constructed only when configured, so the gate never touches it. A
    failed call degrades to the deterministic descriptor. ``provider_id`` is a stable
    identity folded into the descriptor provenance.
    """

    @property
    def provider_id(self) -> str: ...

    def caption(self, content: bytes, *, media_type: str, title: str) -> str: ...


class FineTuningProvider(Protocol):
    """Produces a fine-tuned-model reference from an approved training request.

    Kept as a seam, not a hard dependency: the offline default is a deterministic
    provider that records the request and returns a content-addressed reference
    without training or a network call, so the whole gate runs with no external
    resource. A real provider that actually trains a model is used only when
    configured and is never exercised in the gate. ``provider_id`` is a stable
    identity folded into the run's provenance.
    """

    @property
    def provider_id(self) -> str: ...

    def submit(self, request: FineTuneRequest) -> FineTuneResult: ...


class SecretProvider(Protocol):
    """Resolves named, capability-scoped secrets at tool-execution time.

    The default is no provider installed: no secret is ever resolved and the governed loop is
    byte-identical. A reference env-var or static-map provider serves tests and simple deployments;
    a real Vault, AWS Secrets Manager, or KMS provider plugs into this same port and uses the scope
    for per-tenant least privilege. The model never influences a name here: names come only from the
    registered ToolCapability.required_secrets, never from model output or a tool argument.
    """

    @property
    def provider_id(self) -> str: ...

    def available(self, names: frozenset[str], *, scope: SecretScope) -> frozenset[str]:
        """The subset of names this provider can resolve for this scope, without materializing values."""
        ...

    def resolve(self, name: str, *, scope: SecretScope) -> SecretValue | None:
        """Materialize one secret, or None when it cannot be resolved (fail-closed at the call site)."""
        ...


class Resolver(Protocol):
    def resolve(self, host: str, port: int) -> tuple[str, ...]: ...


class DocumentExtractor(Protocol):
    @property
    def media_types(self) -> frozenset[str]: ...

    def extract(self, content: bytes, *, metadata: JsonObject) -> ExtractedDocument: ...


class BlobStore(Protocol):
    def put(self, key: str, content: bytes, *, media_type: str) -> str: ...

    def open(self, uri: str) -> BinaryIO: ...

    def delete(self, uri: str) -> None: ...

    def list_blobs(self) -> Iterator[BlobEntry]: ...

    def modified_at(self, uri: str) -> datetime | None: ...


class SearchIndex(Protocol):
    def replace_document(self, document_id: str, chunks: Sequence[Chunk]) -> None: ...

    def delete_document(self, document_id: str) -> None: ...

    def semantic_search(
        self,
        *,
        workspace_id: str,
        corpus_id: str,
        query_embedding: tuple[float, ...],
        limit: int,
        metadata_filter: MetadataFilter | None = None,
        allowed_document_ids: frozenset[str] | None = None,
    ) -> list[SearchHit]: ...

    def lexical_search(
        self,
        *,
        workspace_id: str,
        corpus_id: str,
        query: str,
        limit: int,
        metadata_filter: MetadataFilter | None = None,
        params: BM25Params | None = None,
        allowed_document_ids: frozenset[str] | None = None,
    ) -> list[SearchHit]: ...


class EventSink(Protocol):
    def emit(self, topic: str, payload: JsonObject, *, event_id: str | None = None) -> None: ...


class Connector(Protocol):
    @property
    def scheme(self) -> str: ...

    def fetch(self, source: str) -> tuple[bytes, str, JsonObject]: ...


class Skill(Protocol):
    name: str
    description: str

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject: ...


class CacheStore(Protocol):
    def get_exact_cache(
        self, workspace_id: str, corpus_id: str, version: int, user_scope: str, query_hash: str
    ) -> CacheEntry | None: ...

    def list_cache_candidates(
        self, workspace_id: str, corpus_id: str, version: int, user_scope: str, limit: int
    ) -> list[CacheEntry]: ...

    def put_cache(self, entry: CacheEntry) -> None: ...

    def delete_query_cache_for_corpus(self, workspace_id: str, corpus_id: str) -> int: ...


class GraphReadRepository(Protocol):
    """The read-only graph seam behind the query and RLM hot paths; the composite store is the default.

    An adapter may expose an optional ``cache_id`` str; when present it folds into the answer-cache
    identity, since graph reads change results. The default store exposes none, so the path stays
    byte-identical. graph_neighborhood must never under-return a name graph_context would match.
    """

    def graph_index_present(self, workspace_id: str, corpus_id: str) -> bool: ...

    def graph_neighborhood(
        self,
        workspace_id: str,
        corpus_id: str,
        *,
        term_candidates: frozenset[str],
        name_substrings: frozenset[str],
    ) -> tuple[list[Entity], list[Relation]]: ...

    def list_graph(self, workspace_id: str, corpus_id: str) -> tuple[list[Entity], list[Relation]]: ...

    def max_entity_name_length(self, workspace_id: str, corpus_id: str) -> int: ...


class FileGuard(Protocol):
    def resolve(self, root: Path, candidate: str) -> Path: ...


class ModelCallCache(Protocol):
    """Content-addressed cache for provider round-trips, keyed by backend plus input.

    A hit returns byte-identical output to a miss because decode is deterministic
    (temperature 0). The generation key includes the provider ``cache_identity``
    (base URL plus model) and is scoped by ``workspace_id`` so a tenant can only
    ever read rows it wrote, and two backends that share a bare model name never
    collide. Embeddings are content-addressed and backend-scoped but not
    workspace-scoped on purpose: an embedding of identical text under the same
    backend is genuinely identical, so a cross-tenant hit leaks no vector value a
    tenant could not compute itself. The one residual signal is existence: a
    faster, zero-charge hit lets a tenant infer that some other tenant previously
    embedded that exact text. This same-text membership side channel is an accepted
    tradeoff for a shared embedding cache; scope embeddings by workspace, like
    generation rows, if membership inference is in the threat model.
    """

    def get_model_call(self, workspace_id: str, model_id: str, cache_key: str) -> JsonObject | None: ...

    def put_model_call(
        self, workspace_id: str, model_id: str, cache_key: str, payload: JsonObject, ttl_seconds: int
    ) -> None: ...

    def get_embedding(self, model_id: str, cache_key: str) -> tuple[float, ...] | None: ...

    def put_embedding(self, model_id: str, cache_key: str, vector: tuple[float, ...], ttl_seconds: int) -> None: ...

    def delete_generation_model_cache(self, workspace_id: str) -> int: ...

    def delete_embedding_model_cache(self, key_digests: Sequence[str]) -> int: ...


class CodeSandbox(Protocol):
    """An isolation boundary that may execute one caller-supplied computation.

    The shipped default refuses every call and executes nothing, so code
    execution is a positive opt-in: only when an operator installs a sandbox
    whose ``executes`` is true, and a policy admits the request, does any code
    run. ``execute`` sees only the supplied ``inputs`` mapping and returns a
    :class:`SandboxResult`; it never touches the network, the filesystem, or any
    host state beyond what a real isolation adapter grants inside its own bound.
    """

    @property
    def name(self) -> str: ...

    @property
    def executes(self) -> bool: ...

    def execute(self, code: str, inputs: JsonObject, limits: SandboxLimits) -> SandboxResult: ...
