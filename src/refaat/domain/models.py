from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from refaat.domain.authz import Classification, Principal
from refaat.domain.enums import SideEffectClass

if TYPE_CHECKING:
    from refaat.domain.governance import ToolCapability

JsonObject = dict[str, Any]


def _type_tagged_key(value: Any) -> tuple[str, str]:
    return type(value).__name__, repr(value)


@dataclass(frozen=True, slots=True)
class MetadataPredicate:
    field: str
    values: tuple[Any, ...]

    def matches(self, metadata: Mapping[str, Any]) -> bool:
        if self.field not in metadata:
            return False
        actual = metadata[self.field]
        return any(type(actual) is type(candidate) and actual == candidate for candidate in self.values)

    def as_json(self) -> JsonObject:
        return {
            "field": self.field,
            "values": sorted(self.values, key=_type_tagged_key),
        }


@dataclass(frozen=True, slots=True)
class MetadataFilter:
    predicates: tuple[MetadataPredicate, ...]

    def matches(self, metadata: Mapping[str, Any]) -> bool:
        return all(predicate.matches(metadata) for predicate in self.predicates)

    def as_json(self) -> JsonObject:
        return {
            "predicates": [
                predicate.as_json() for predicate in sorted(self.predicates, key=lambda predicate: predicate.field)
            ]
        }

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any] | None) -> MetadataFilter | None:
        if not spec:
            return None
        predicates: list[MetadataPredicate] = []
        for field_name in sorted(spec):
            raw = spec[field_name]
            values = tuple(raw) if isinstance(raw, (list, tuple)) else (raw,)
            if not values:
                continue
            predicates.append(MetadataPredicate(field=field_name, values=values))
        if not predicates:
            return None
        return cls(predicates=tuple(predicates))


@dataclass(frozen=True, slots=True)
class Workspace:
    id: str
    slug: str
    name: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Corpus:
    id: str
    workspace_id: str
    name: str
    version: int
    settings: JsonObject
    created_at: datetime


@dataclass(frozen=True, slots=True)
class DocumentInput:
    source_id: str
    title: str
    media_type: str
    content: bytes
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    text: str
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BlobEntry:
    uri: str
    modified_at: datetime


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    corpus_id: str
    document_id: str
    ordinal: int
    text: str
    token_count: int
    checksum: str
    metadata: JsonObject
    embedding: tuple[float, ...]
    # An image-descriptor chunk carries the blob key of the stored image bytes plus
    # the image media type; both are None for an ordinary text chunk, so a text-only
    # corpus stores rows and constructs objects byte-identically to before. The
    # descriptor text in ``text`` is what is embedded, retrieved, and cited; a
    # multimodal generator uses ``image_ref`` to load the pixels, but the citation is
    # always this stored chunk, never the raw image.
    image_ref: str | None = None
    media_type: str | None = None


@dataclass(frozen=True, slots=True)
class RaptorNode:
    """One internal node of a persisted RAPTOR summary tree.

    ``id`` uses the reserved ``raptor_summary_`` prefix, a namespace a real chunk id
    (always ``chunk_``) can never take, so a node id can never collide with a stored
    chunk and is provably uncitable: it is absent from the chunks table, so citation
    resolution drops it. ``leaf_chunk_ids`` records the real stored leaf chunks this
    node covers, which is how a summary hit maps back to real chunks at query time.
    """

    id: str
    corpus_id: str
    level: int
    parent_node_id: str | None
    summary: str
    embedding: tuple[float, ...]
    child_node_ids: tuple[str, ...]
    leaf_chunk_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Entity:
    id: str
    corpus_id: str
    document_id: str
    name: str
    normalized_name: str
    kind: str
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Relation:
    id: str
    corpus_id: str
    document_id: str
    source_entity_id: str
    target_entity_id: str
    kind: str
    weight: float
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GraphExtraction:
    entities: tuple[Entity, ...]
    relations: tuple[Relation, ...]


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk: Chunk
    score: float
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    graph_score: float = 0.0
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TransformedQuery:
    """A single retrieval variant produced by a QueryTransform.

    kind is one of "original", "subquery" (a decomposed clause), "expansion"
    (a keyword-focused rewrite), "hypothetical" (a HyDE pseudo-answer), "rewrite"
    (a model-restated query), or "stepback" (a broader concept query). The
    runtime retrieves each variant and RRF-merges the per-variant rankings; a
    single "original" variant is the byte-identical single-query path.

    embed_text, when set, is the text whose embedding replaces the semantic
    search vector for this variant while text still drives lexical search. It is
    the HyDE seam: the hypothetical answer influences the embedding target only
    and is never lexically searched, retrieved, or cited. When embed_text is None
    the embedding target is text, so every existing construction stays
    byte-identical.
    """

    text: str
    kind: str
    embed_text: str | None = None


@dataclass(frozen=True, slots=True)
class Citation:
    chunk_id: str
    document_id: str
    label: str
    quote: str


@dataclass(frozen=True, slots=True)
class Demonstration:
    """A labeled question and answer pair shown to a provider as a format example.

    Demonstrations are illustrative only: they steer answer shape, never evidence.
    Any citation-shaped token in a demonstration is rewritten to a reserved
    example sentinel before it can reach a provider, so no real corpus chunk id
    is ever borrowed from a demonstration to forge a citation.
    """

    question: str
    answer: str

    def as_json(self) -> JsonObject:
        return {"question": self.question, "answer": self.answer}


@dataclass(frozen=True, slots=True)
class QueryRequest:
    workspace_id: str
    corpus_id: str
    query: str
    user_id: str | None = None
    session_id: str | None = None
    top_k: int | None = None
    token_budget: int | None = None
    fusion: str | None = None
    rrf_k: int | None = None
    autocut_enabled: bool | None = None
    autocut_threshold: float | None = None
    query_expansion: bool | None = None
    max_query_variants: int | None = None
    instructions: str | None = None
    demonstrations: tuple[Demonstration, ...] = ()
    response_schema: JsonObject | None = None
    router_id: str | None = None
    metadata: JsonObject = field(default_factory=dict)
    principal: Principal | None = None


@dataclass(frozen=True, slots=True)
class ContextPacket:
    query: str
    hits: tuple[SearchHit, ...]
    graph_facts: tuple[str, ...]
    session_summary: str | None
    profile: JsonObject
    supplemental_harness: JsonObject
    token_count: int
    routes: tuple[str, ...]
    # Graph facts and the session summary reach the model outside context.hits; open defaults keep a
    # policy-free packet's ceiling exactly as the hits alone imply.
    graph_evidence_classification: Classification = Classification.PUBLIC
    graph_evidence_zones: frozenset[str] = frozenset()
    session_summary_classification: Classification = Classification.PUBLIC
    session_summary_zones: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class GenerationImage:
    """The pixels of one retrieved image-descriptor chunk, carried to a vision model.

    Loaded from the blob store for a retrieved hit that has an ``image_ref``, only
    when a multimodal generator is active. ``chunk_id`` ties the image back to its
    real stored descriptor chunk, so a vision answer cites that chunk, never the raw
    image. The offline extractive generator ignores this field entirely, so a
    text-only request (``images == ()``) is byte-identical to before.
    """

    chunk_id: str
    media_type: str
    content: bytes


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    query: str
    context: ContextPacket
    system_prompt: str
    instructions: str | None = None
    demonstrations: tuple[Demonstration, ...] = ()
    response_schema: JsonObject | None = None
    workspace_id: str = ""
    images: tuple[GenerationImage, ...] = ()


@dataclass(frozen=True, slots=True)
class GenerationResult:
    answer: str
    cited_chunk_ids: tuple[str, ...]
    usage: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class QueryResponse:
    answer: str
    citations: tuple[Citation, ...]
    routes: tuple[str, ...]
    cached: bool
    corpus_version: int
    trace_id: str
    usage: JsonObject = field(default_factory=dict)
    abstained: bool = False
    # Set when the answer was served over a degraded path after a generation provider outage,
    # under an installed degradation policy. Default False, so serialization is unchanged when
    # not degraded.
    degraded: bool = False


@dataclass(frozen=True, slots=True)
class IngestResult:
    document_id: str
    corpus_id: str
    corpus_version: int
    chunks: int
    entities: int
    relations: int
    changed: bool
    checksum: str


@dataclass(frozen=True, slots=True)
class Feedback:
    query_id: str | None
    user_id: str | None
    rating: Literal[-1, 0, 1]
    selected_chunk_ids: tuple[str, ...] = ()
    note: str | None = None
    principal: Principal | None = None


@dataclass(frozen=True, slots=True)
class CacheEntry:
    id: str
    workspace_id: str
    corpus_id: str
    corpus_version: int
    user_scope: str
    query: str
    query_hash: str
    query_embedding: tuple[float, ...]
    response: QueryResponse
    expires_at: datetime
    created_at: datetime


@dataclass(frozen=True, slots=True)
class AgentMessage:
    id: str
    workspace_id: str
    sender_agent_id: str
    recipient_agent_id: str
    kind: str
    payload: JsonObject
    correlation_id: str | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class HarnessSnapshot:
    id: str
    workspace_id: str
    corpus_id: str | None
    parent_id: str | None
    base_prompt_hash: str
    supplemental: JsonObject
    evidence: JsonObject
    active: bool
    created_at: datetime


@dataclass(frozen=True, slots=True)
class RlmLimits:
    max_depth: int = 2
    max_nodes: int = 16
    branch_factor: int = 4
    leaf_token_budget: int = 1_500

    def __post_init__(self) -> None:
        if not 0 <= self.max_depth <= 6:
            raise ValueError("max_depth must be between 0 and 6")
        if not 1 <= self.max_nodes <= 128:
            raise ValueError("max_nodes must be between 1 and 128")
        if not 2 <= self.branch_factor <= 16:
            raise ValueError("branch_factor must be between 2 and 16")
        if not 64 <= self.leaf_token_budget <= 100_000:
            raise ValueError("leaf_token_budget must be between 64 and 100000")


@dataclass(frozen=True, slots=True)
class RlmNodeTrace:
    id: str
    parent_id: str | None
    depth: int
    route: str
    input_chunk_count: int
    input_tokens: int
    selected_chunk_ids: tuple[str, ...]
    cited_chunk_ids: tuple[str, ...]
    answer_excerpt: str


@dataclass(frozen=True, slots=True)
class RlmResult:
    answer: str
    citations: tuple[Citation, ...]
    nodes: tuple[RlmNodeTrace, ...]
    corpus_version: int
    trace_id: str
    usage: JsonObject = field(default_factory=dict)
    # Set when the run was served over a degraded path after a generation provider outage,
    # under an installed degradation policy. Default False, so a normal run is unchanged.
    degraded: bool = False


@dataclass(frozen=True, slots=True)
class ToolGrant:
    """A registration-time binding of a tool to the scopes and side-effect class it needs.

    Governance is a property of the grant, not of the skill, so the same skill can
    be granted differently under different policies without changing the skill.
    """

    skill_name: str
    required_scopes: tuple[str, ...] = ()
    side_effect: SideEffectClass = SideEffectClass.READ_ONLY


@dataclass(frozen=True, slots=True)
class ReActPolicy:
    """Bounds and permissions for a governed ReAct run.

    The caller holds ``granted_scopes`` and admits tools up to ``side_effect_ceiling``.
    ``max_iterations`` is a hard cap so a run always terminates.
    """

    max_iterations: int = 4
    granted_scopes: tuple[str, ...] = ()
    side_effect_ceiling: SideEffectClass = SideEffectClass.READ_ONLY
    grants: tuple[ToolGrant, ...] = ()
    capabilities: tuple[ToolCapability, ...] = ()

    def __post_init__(self) -> None:
        if not 1 <= self.max_iterations <= 32:
            raise ValueError("max_iterations must be between 1 and 32")
        grant_names = {grant.skill_name for grant in self.grants}
        seen: set[str] = set()
        for capability in self.capabilities:
            if capability.name in seen:
                raise ValueError(f"duplicate capability for tool {capability.name!r}")
            seen.add(capability.name)
            if capability.name not in grant_names:
                raise ValueError(f"capability {capability.name!r} has no matching tool grant")


@dataclass(frozen=True, slots=True)
class ReActStep:
    thought: str
    tool: str
    arguments: JsonObject
    authorized: bool
    reason: str
    observation: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolProposal:
    """A planner's proposal to call one tool with concrete arguments.

    The deterministic keyword planner proposes an empty ``arguments`` map; a
    model planner fills it from a function-calling response. Argument validation
    is applied only to model proposals, so the offline default never regresses.
    """

    tool: str
    arguments: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One entry of a tool manifest advertised to a model planner.

    ``parameters`` is a JSON-Schema object describing the tool's arguments;
    a keyword-routed tool exposes a no-required-field schema so it is never
    rejected by validation on the model path.
    """

    name: str
    description: str
    parameters: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    """One provider-neutral fragment of a streamed tool call.

    ``tool`` carries the chosen tool name once (the first non-None fragment);
    ``arguments_fragment`` is an opaque partial slice of the JSON argument
    string that is buffered and only parsed once ``done`` terminates the call,
    so a partial argument is never validated, authorized, or executed.
    """

    tool: str | None = None
    arguments_fragment: str = ""
    done: bool = False


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """A JudgeModel's assessment of one answer against its evidence.

    ``rationale`` is a diagnostic string only; it is never a citation and is
    never resolved against the corpus, so a judge can explain a score without
    ever emitting or forging evidence.
    """

    score: float
    passed: bool
    rationale: str
