from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from refaat.domain.models import JsonObject

if TYPE_CHECKING:
    from refaat.domain.plugins import PluginDescriptor
    from refaat.ports import (
        ChunkingStrategy,
        CrossEncoderScorer,
        Embedder,
        Generator,
        GraphExtractor,
        HypotheticalQuestionGenerator,
        JudgeModel,
        QueryTransform,
        Reranker,
    )


@dataclass(frozen=True, slots=True)
class ReproducedComponent:
    """A production-injected component the shadow rebuilt with the same injected instance."""

    name: str


@dataclass(frozen=True, slots=True)
class OmittedComponent:
    """A production-injected component the shadow refused to reproduce, with the reason why."""

    name: str
    reason: str


@dataclass(frozen=True, slots=True)
class ShadowComposition:
    """What a shadow runtime reproduced from the production composition and what it refused.

    Emitted alongside a shadow optimization so a caller can see exactly which injected components
    reached the throwaway corpus and which were deliberately left out, rather than silently dropped.
    """

    reproduced: tuple[ReproducedComponent, ...] = ()
    omitted: tuple[OmittedComponent, ...] = ()

    def is_empty(self) -> bool:
        return not self.reproduced and not self.omitted

    def to_json(self) -> JsonObject:
        return {
            "reproduced": [component.name for component in self.reproduced],
            "omitted": [{"component": component.name, "reason": component.reason} for component in self.omitted],
        }


@dataclass(frozen=True, slots=True)
class RuntimeComposition:
    """A snapshot of every component injected into a RuntimeBuilder at build time.

    The reproducible slots are stateless, corpus-agnostic model and text components a shadow runtime
    can safely rebuild over a throwaway corpus. The refused-but-captured slots are stateful backends,
    store-bound derivation services, governance components, and side-effecting adapters the shadow must
    never run; they are captured only so the shadow can report that it left them out.
    """

    embedder: Embedder | None = None
    generator: Generator | None = None
    graph_extractor: GraphExtractor | None = None
    reranker: Reranker | None = None
    cross_encoder_scorer: CrossEncoderScorer | None = None
    query_transform: QueryTransform | None = None
    judge: JudgeModel | None = None
    hype_generator: HypotheticalQuestionGenerator | None = None
    chunking_strategy: ChunkingStrategy | None = None
    stopwords: dict[str, frozenset[str]] | None = None

    search_index: object | None = None
    cache_store: object | None = None
    model_call_cache: object | None = None
    graph_read_repository: object | None = None
    circuit_breaker: object | None = None
    raptor_builder: object | None = None
    audit_sink: object | None = None
    model_router: object | None = None
    access_policy: object | None = None
    principal_resolver: object | None = None
    principal_directory: object | None = None
    entitlement_scope_resolver: object | None = None
    secret_provider: object | None = None
    kill_switch_registry: object | None = None
    quota_policy: object | None = None
    egress_policy: object | None = None
    output_governor: object | None = None
    context_transform: object | None = None
    degradation_policy: object | None = None
    direct_skill_policy: object | None = None
    image_captioner: object | None = None
    multimodal_generator: object | None = None
    fine_tuning_provider: object | None = None
    code_sandbox: object | None = None

    event_sinks: tuple[object, ...] = ()
    connectors: tuple[object, ...] = ()
    extractors: tuple[object, ...] = ()
    skills: tuple[object, ...] = ()

    plugin_pins: dict[str, PluginDescriptor] | None = field(default=None)


_PLUGIN_GROUP_REASON = (
    "only injected instances are excluded; the shadow still constructs allowlisted plugin components from settings"
)
_GOVERNANCE_REASON = "a governance component that must not run over a throwaway shadow corpus"
_EXTERNAL_ADAPTER_REASON = "an external or side-effecting adapter"

OMISSION_REASONS: dict[str, str] = {
    "search_index": (
        "a stateful backend; reproducing a factory could write to a production cluster, and "
        "RetrievalOptimizer already carries the live index"
    ),
    "cache_store": "a stateful cache backend that would read or write production cache keys",
    "model_call_cache": "a stateful cache backend that would read or write production cache keys",
    "graph_read_repository": "a stateful graph read backend a shadow run would read production evidence through",
    "circuit_breaker": "a stateful breaker whose per-endpoint counters a shadow run would draw down and trip",
    "raptor_builder": "a store-bound derivation service that would write RAPTOR nodes into the production database",
    "audit_sink": _GOVERNANCE_REASON,
    "model_router": _GOVERNANCE_REASON,
    "access_policy": _GOVERNANCE_REASON,
    "principal_resolver": _GOVERNANCE_REASON,
    "principal_directory": _GOVERNANCE_REASON,
    "entitlement_scope_resolver": _GOVERNANCE_REASON,
    "secret_provider": _GOVERNANCE_REASON,
    "kill_switch_registry": _GOVERNANCE_REASON,
    "quota_policy": _GOVERNANCE_REASON,
    "egress_policy": _GOVERNANCE_REASON,
    "output_governor": _GOVERNANCE_REASON,
    "context_transform": _GOVERNANCE_REASON,
    "degradation_policy": _GOVERNANCE_REASON,
    "direct_skill_policy": _GOVERNANCE_REASON,
    "image_captioner": _EXTERNAL_ADAPTER_REASON,
    "multimodal_generator": _EXTERNAL_ADAPTER_REASON,
    "fine_tuning_provider": _EXTERNAL_ADAPTER_REASON,
    "code_sandbox": _EXTERNAL_ADAPTER_REASON,
    "event_sinks": _PLUGIN_GROUP_REASON,
    "connectors": _PLUGIN_GROUP_REASON,
    "extractors": _PLUGIN_GROUP_REASON,
    "skills": _PLUGIN_GROUP_REASON,
}
