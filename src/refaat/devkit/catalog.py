"""The port catalog: one row per typed Protocol in refaat.ports.

Each row records how a plugin author implements and installs an adapter for that port: the required
members, the injection tier and its seam, the shipped default, the name of the planned per-port
conformance base, and a note on any citation, cache-identity, or fail-closed rule the adapter must
uphold. The shipped universal check is refaat.conformance.structural_conformance (every declared
member present); the named per-port bases are a follow-up. The ports_catalog gate asserts this list
stays in lockstep with the live Protocols, so a new port cannot be added without a catalog row.
"""

from __future__ import annotations

from dataclasses import dataclass

# Injection tiers, grounded in the real composition root:
#   entry_point : discovered through a refaat.* entry-point group, gated by the allowlist and pins
#   builder     : injected through a RuntimeBuilder.with_* method
#   config      : selected by a settings value inside the builder, no with_* seam
#   none        : constructed internally or a marker, with no external injection seam yet
TIERS = ("entry_point", "builder", "config", "none")


@dataclass(frozen=True, slots=True)
class PortSpec:
    name: str
    members: tuple[str, ...]
    tier: str
    seam: str
    default: str
    conformance_base: str
    notes: str

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            raise ValueError(f"port {self.name}: unknown tier {self.tier!r}")
        if (self.tier == "none") != (self.seam == ""):
            raise ValueError(
                f"port {self.name}: a none-tier port must have an empty seam, and only a none-tier port may"
            )
        if list(self.members) != sorted(set(self.members)):
            raise ValueError(f"port {self.name}: members must be unique and sorted")


PORT_CATALOG: tuple[PortSpec, ...] = (
    PortSpec(
        name="Embedder",
        members=("dimensions", "embed", "model_id"),
        tier="builder",
        seam="with_embedder",
        default="refaat.infrastructure.embeddings.DeterministicEmbedder",
        conformance_base="EmbedderContract",
        notes="dimensions must be constant and equal every vector length; embed must be deterministic "
        "for identical input, and a real backend should also implement CacheIdentified.",
    ),
    PortSpec(
        name="Generator",
        members=("generate", "model_id", "stream"),
        tier="builder",
        seam="with_generator",
        default="refaat.infrastructure.generation.ExtractiveGenerator",
        conformance_base="GeneratorContract",
        notes="citation contract: an answer may cite only chunk ids present in the request evidence; an "
        "invented id must not survive.",
    ),
    PortSpec(
        name="GraphExtractor",
        members=("extract", "model_id"),
        tier="builder",
        seam="with_graph_extractor",
        default="refaat.infrastructure.graph.CooccurrenceGraphExtractor",
        conformance_base="GraphExtractorContract",
        notes="entities and relations must reference only the supplied document text.",
    ),
    PortSpec(
        name="JudgeModel",
        members=("assess", "model_id"),
        tier="builder",
        seam="with_judge_model",
        default="deterministic faithfulness proxy",
        conformance_base="JudgeModelContract",
        notes="a score is advisory and never rewrites a grounded answer or a citation.",
    ),
    PortSpec(
        name="Reranker",
        members=("rerank",),
        tier="builder",
        seam="with_reranker",
        default="refaat.infrastructure.reranking.LexicalReranker",
        conformance_base="RerankerContract",
        notes="reranking reorders candidates only; it must not introduce a chunk that was not retrieved.",
    ),
    PortSpec(
        name="CrossEncoderScorer",
        members=("model_id", "score"),
        tier="builder",
        seam="with_cross_encoder_scorer",
        default="injected scorer, no offline default",
        conformance_base="CrossEncoderScorerContract",
        notes="score is a pure relevance signal over (query, passage); model_id folds into the cache id.",
    ),
    PortSpec(
        name="QueryTransform",
        members=("cache_id", "expand"),
        tier="builder",
        seam="with_query_transform",
        default="deterministic transform catalog",
        conformance_base="QueryTransformContract",
        notes="a variant query is never retrieved or cited; cache_id folds into the answer-cache identity.",
    ),
    PortSpec(
        name="HypotheticalQuestionGenerator",
        members=("provider_id", "questions"),
        tier="builder",
        seam="with_hype_generator",
        default="deterministic templated generator",
        conformance_base="HypotheticalQuestionGeneratorContract",
        notes="a hypothetical question is embedded and matched but is never a chunk and never cited; "
        "provider_id folds into the answer-cache identity.",
    ),
    PortSpec(
        name="ImageCaptioner",
        members=("caption", "provider_id"),
        tier="builder",
        seam="with_image_captioner",
        default="honest deterministic descriptor when unconfigured",
        conformance_base="ImageCaptionerContract",
        notes="a caption becomes a stored descriptor chunk; the raw image bytes are never cited.",
    ),
    PortSpec(
        name="FineTuningProvider",
        members=("provider_id", "submit"),
        tier="builder",
        seam="with_fine_tuning_provider",
        default="refaat.infrastructure.fine_tuning.DeterministicFineTuner",
        conformance_base="FineTuningProviderContract",
        notes="a run is approval-gated and its model is an immutable un-promoted artifact.",
    ),
    PortSpec(
        name="SecretProvider",
        members=("available", "provider_id", "resolve"),
        tier="builder",
        seam="with_secret_provider",
        default="no provider unless configured",
        conformance_base="SecretProviderContract",
        notes="a secret is supplied at execution and must never enter an artifact, a cache key, or an audit event.",
    ),
    PortSpec(
        name="CodeSandbox",
        members=("execute", "executes", "name"),
        tier="builder",
        seam="with_code_sandbox",
        default="refaat.infrastructure.sandbox.DenyingSandbox",
        conformance_base="CodeSandboxContract",
        notes="deny by default; an executing sandbox must bound resources and never let a result become a citation.",
    ),
    PortSpec(
        name="ChunkingStrategy",
        members=("chunk", "strategy_id"),
        tier="entry_point",
        seam="refaat.chunkers",
        default="refaat.core.chunking.AdaptiveChunkingStrategy",
        conformance_base="ChunkingStrategyContract",
        notes="propose only text, ordinal, and structure; the pipeline mints every id and checksum, so a "
        "strategy cannot forge a chunk id. Also injectable via with_chunking_strategy.",
    ),
    PortSpec(
        name="DocumentExtractor",
        members=("extract", "media_types"),
        tier="entry_point",
        seam="refaat.extractors",
        default="built-in text and PDF extractors",
        conformance_base="DocumentExtractorContract",
        notes="convert bytes to text and metadata for the declared media types.",
    ),
    PortSpec(
        name="Connector",
        members=("fetch", "scheme"),
        tier="entry_point",
        seam="refaat.connectors",
        default="refaat.infrastructure.connectors.FileConnector",
        conformance_base="ConnectorContract",
        notes="outbound fetches must route through the egress policy; a hand-rolled network call bypasses governance.",
    ),
    PortSpec(
        name="EventSink",
        members=("emit",),
        tier="entry_point",
        seam="refaat.event_sinks",
        default="refaat.infrastructure.events.LoggingEventSink",
        conformance_base="EventSinkContract",
        notes="delivery is at least once, so a sink must be idempotent on the event id.",
    ),
    PortSpec(
        name="Skill",
        members=("description", "name", "run"),
        tier="entry_point",
        seam="refaat.skills",
        default="the built-in corpus_health skill",
        conformance_base="SkillContract",
        notes="a skill receives the canonical context packet and explicit JSON arguments; an external "
        "skill routes egress through the policy.",
    ),
    PortSpec(
        name="BlobStore",
        members=("delete", "list_blobs", "modified_at", "open", "put"),
        tier="config",
        seam="blob_backend",
        default="refaat.infrastructure.blob.LocalBlobStore",
        conformance_base="BlobStoreContract",
        notes="put then open must round-trip the exact bytes; selected by blob_backend, no with_* seam yet.",
    ),
    PortSpec(
        name="SearchIndex",
        members=("delete_document", "lexical_search", "replace_document", "semantic_search"),
        tier="builder",
        seam="with_search_index",
        default="refaat.infrastructure.search.LocalSearchIndex",
        conformance_base="refaat.conformance.search_index_conformance",
        notes="every returned hit must resolve to a stored chunk in scope, the ACL and metadata pushdown "
        "must fail closed on both the semantic and the lexical path, a transport or partial-shard error "
        "must raise rather than fall back to a scan, and a deleted document must never resurface. The "
        "built-in Local and pgvector backends are selected by database_url; a third backend is injected "
        "via with_search_index, as an index or a factory whose build(store, embedder) receives the "
        "runtime store for id-to-chunk hydration.",
    ),
    PortSpec(
        name="GraphReadRepository",
        members=("graph_index_present", "graph_neighborhood", "list_graph", "max_entity_name_length"),
        tier="builder",
        seam="with_graph_read_repository",
        default="refaat.infrastructure.database.store.SQLStore",
        conformance_base="refaat.conformance.graph_read_conformance",
        notes="read-only graph seam behind the query and RLM hot paths. graph_neighborhood must never "
        "under-return a name graph_context would match (over-return is harmless, graph_context re-filters). "
        "An adapter may expose an optional cache_id that folds into the answer-cache identity since graph "
        "reads change results; the default composite store exposes none, so the path is byte-identical.",
    ),
    PortSpec(
        name="CacheIdentified",
        members=("cache_identity",),
        tier="none",
        seam="",
        default="marker on OpenAI-compatible embedder and generator",
        conformance_base="CacheIdentifiedContract",
        notes="cache_identity must fingerprint the backend and model so two backends sharing a bare model "
        "name never collide in the answer cache; opt-in marker, no injection seam.",
    ),
    PortSpec(
        name="ToolPlanner",
        members=("plan", "tools"),
        tier="none",
        seam="",
        default="refaat.programs.react.KeywordToolPlanner",
        conformance_base="ToolPlannerContract",
        notes="plan must return only a tool named in tools() and None to stop; constructed inside the "
        "programs layer, no with_* seam.",
    ),
    PortSpec(
        name="ToolCallingGenerator",
        members=("model_id", "propose_tool_call"),
        tier="none",
        seam="",
        default="a real function-calling provider only",
        conformance_base="ToolCallingGeneratorContract",
        notes="propose only from the provided tool specs; kept separate from Generator so the offline "
        "generator never fakes a tool choice.",
    ),
    PortSpec(
        name="StreamingToolCallingGenerator",
        members=("model_id", "stream_tool_call"),
        tier="none",
        seam="",
        default="a real streaming function-calling provider only",
        conformance_base="StreamingToolCallingGeneratorContract",
        notes="a partial argument fragment is opaque until the terminating delta; only a real provider implements it.",
    ),
    PortSpec(
        name="Summarizer",
        members=("provider_id", "summarize"),
        tier="none",
        seam="",
        default="refaat.infrastructure.summarization.DeterministicExtractiveSummarizer",
        conformance_base="SummarizerContract",
        notes="a summary node is embedded and matched but resolves to real leaf chunks and is never "
        "cited; injected through the raptor builder, no direct with_* seam.",
    ),
    PortSpec(
        name="Resolver",
        members=("resolve",),
        tier="none",
        seam="",
        default="system DNS resolver",
        conformance_base="ResolverContract",
        notes="used only to pin an outbound host to a validated address for SSRF defense.",
    ),
    PortSpec(
        name="FileGuard",
        members=("resolve",),
        tier="none",
        seam="",
        default="the built-in path guard",
        conformance_base="FileGuardContract",
        notes="resolve must reject a path that escapes the configured root.",
    ),
    PortSpec(
        name="CacheStore",
        members=("delete_query_cache_for_corpus", "get_exact_cache", "list_cache_candidates", "put_cache"),
        tier="builder",
        seam="with_cache_store",
        default="the relational store",
        conformance_base="CacheStoreContract",
        notes="the version-aware answer cache; the erasure member must delete a corpus's cached answers or "
        "right-to-erasure silently leaves them behind. Byte-identical when unset (the store is the default).",
    ),
    PortSpec(
        name="ModelCallCache",
        members=(
            "delete_embedding_model_cache",
            "delete_generation_model_cache",
            "get_embedding",
            "get_model_call",
            "put_embedding",
            "put_model_call",
        ),
        tier="builder",
        seam="with_model_call_cache",
        default="the relational store",
        conformance_base="ModelCallCacheContract",
        notes="a content-addressed cache for provider round-trips; a hit must be byte-identical to a miss and "
        "the erasure members must purge a workspace's generations and an erased text's embeddings.",
    ),
)

_BY_NAME = {spec.name: spec for spec in PORT_CATALOG}

_TIER_HEADING = {
    "entry_point": "Entry-point plugins",
    "builder": "Builder injection",
    "config": "Config selection",
    "none": "No injection seam yet",
}
_TIER_INTRO = {
    "entry_point": "Discovered through the named `refaat.*` entry-point group, gated by "
    "`REFAAT_PLUGIN_ALLOWLIST` and optional code-hash pins. Also injectable in-process on the builder.",
    "builder": "Injected in-process through the named `RuntimeBuilder` method.",
    "config": "Selected by the named settings value inside the builder. No `with_*` seam today.",
    "none": "Constructed internally or a marker. No external injection seam today.",
}


def port_names() -> frozenset[str]:
    return frozenset(_BY_NAME)


def get_port(name: str) -> PortSpec | None:
    return _BY_NAME.get(name)


def render_markdown() -> str:
    """Render the ports-catalog reference page from the catalog, so the doc cannot drift from it."""
    lines = [
        "# Ports catalog",
        "",
        "Every replaceable concern in Refaat is a typed Protocol port. This page is the reference for "
        "each one: what a plugin implements, how it is injected, and the invariant its adapter must "
        "uphold. The ports_catalog gate keeps this catalog in lockstep with the live ports, so a new "
        "port cannot ship without an entry here.",
        "",
    ]
    for tier in TIERS:
        rows = sorted((s for s in PORT_CATALOG if s.tier == tier), key=lambda x: x.name)
        if not rows:
            continue
        lines += [f"## {_TIER_HEADING[tier]}", "", _TIER_INTRO[tier], ""]
        lines += ["| Port | Members | Seam | Default | Invariant |", "|---|---|---|---|---|"]
        for s in rows:
            members = ", ".join(f"`{m}`" for m in s.members)
            seam = f"`{s.seam}`" if s.seam else "n/a"
            lines.append(f"| `{s.name}` | {members} | {seam} | {s.default} | {s.notes} |")
        lines.append("")
    lines += [
        "## Authoring an adapter",
        "",
        "Implement the port's members, register it (an entry-point group for a packaged plugin, or a "
        "`with_*` method for an in-process object), add its name to the allowlist, and pin it. See "
        "[Authoring a plugin](../guides/authoring-a-plugin.md) and [Extending Refaat](../extending.md).",
        "",
    ]
    return "\n".join(lines)
