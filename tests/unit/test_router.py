from __future__ import annotations

from refaat.application.services import QueryService
from refaat.core.retrieval import RetrievalPolicy
from refaat.core.router import RouterConfig, extract_features, route, route_query
from refaat.domain.authz import Classification
from refaat.domain.routing import ClassificationCeilingRouter
from refaat.programs.program import GroundedRagProgram, Tunables
from refaat.programs.router import RouterProgram

# The shipped grounded-rag artifact id. Adding Tunables.router as the last field,
# folded into as_json only when non-None, must leave this byte-identical.
PINNED_GROUNDED_RAG = "sha256:b76d73f8d956bcd38ccc9d4c24372b07b387647a189f3a9e300cded186febbb4"

# Pinned so an include-null regression would change this digest.
PINNED_CONTEXT_HASH_ROUTER_NONE = "7937b529c3dadd9ada1e5483bac13fcef58f71fcee4aa31da688b7d137df034c"

_SIMPLE = "what is rollback"
_COMPLEX = "How does Kafka connect to Postgres, and how does Redis cache the results"


def _policy() -> RetrievalPolicy:
    return RetrievalPolicy(
        top_k=8,
        candidate_limit=32,
        token_budget=600,
        semantic_weight=0.55,
        lexical_weight=0.30,
        graph_weight=0.15,
        fusion="rrf",
        rrf_k=60,
        autocut_enabled=False,
        autocut_threshold=0.5,
        metadata_filter=None,
        bm25_k1=1.2,
        bm25_b=0.75,
        query_expansion=True,
        max_query_variants=4,
        compression_enabled=False,
        compression_min_salience=0.0,
        expansion_mode="flat",
        window_radius=1,
    )


def _hash(**overrides: object) -> str:
    base: dict[str, object] = {
        "policy": _policy(),
        "profile": {},
        "supplemental": {},
        "session_summary": None,
        "embedding_model": "emb",
        "generation_model": "gen",
        "reranker": "none",
        "query_transform": "none",
        "guidance": None,
        "response_schema": None,
    }
    base.update(overrides)
    return QueryService._context_hash(**base)  # type: ignore[arg-type]


def test_feature_extraction_is_deterministic() -> None:
    first = extract_features(_COMPLEX)
    second = extract_features(_COMPLEX)
    assert first == second
    assert first.clause_count >= 2
    assert first.token_count > extract_features(_SIMPLE).token_count
    assert extract_features(_SIMPLE).entity_count == 0


def test_routing_table_separates_simple_from_complex() -> None:
    simple = route_query(_SIMPLE, RouterConfig())
    complex_route = route_query(_COMPLEX, RouterConfig())
    assert simple.label == "direct"
    assert simple.fusion == "linear"
    assert simple.query_expansion is False
    assert complex_route.label == "deep"
    assert complex_route.fusion == "rrf"
    assert complex_route.query_expansion is True
    assert simple != complex_route


def test_threshold_sensitivity_changes_route() -> None:
    features = extract_features(_COMPLEX)
    strict = RouterConfig(multi_clause_threshold=5, long_query_tokens=99, entity_threshold=99)
    assert route(features, RouterConfig()).label == "deep"
    assert route(features, strict).label == "direct"


def test_config_cache_id_stable_and_content_addressed() -> None:
    assert RouterConfig().cache_id == RouterConfig().cache_id
    assert RouterConfig().cache_id != RouterConfig(long_query_tokens=8).cache_id
    assert RouterConfig().cache_id.startswith("router:")


def test_grounded_artifact_id_byte_identical_for_router_none() -> None:
    assert GroundedRagProgram().artifact().artifact_id == PINNED_GROUNDED_RAG
    assert Tunables().as_json() == {
        "top_k": None,
        "token_budget": None,
        "minimum_citation_coverage": 0.0,
    }
    assert "router" not in Tunables().as_json()


def test_non_none_router_changes_artifact_id() -> None:
    routed = GroundedRagProgram().with_tunables(Tunables(router=RouterConfig().as_json()))
    assert routed.artifact().artifact_id != PINNED_GROUNDED_RAG
    assert routed.tunables.as_json()["router"] == RouterConfig().as_json()


def test_router_definition_extra_merges_base_pinning() -> None:
    program = RouterProgram().with_response_schema({"type": "object", "properties": {}})
    extra = program._definition_extra()
    assert extra["router"]["kind"] == "router"
    assert extra["router"]["inner"].startswith("sha256:")
    assert extra["router"]["config"] == RouterConfig().cache_id
    # The response schema folds into the router's identity via the always-applied validators fold, so a
    # schema-enforced router is a distinct artifact from a bare one.
    assert program.artifact().artifact_id != RouterProgram().artifact().artifact_id


def test_context_hash_router_none_is_byte_identical() -> None:
    assert _hash() == PINNED_CONTEXT_HASH_ROUTER_NONE
    assert _hash(router_id=None) == PINNED_CONTEXT_HASH_ROUTER_NONE
    assert _hash(model_router=None) == PINNED_CONTEXT_HASH_ROUTER_NONE


def test_context_hash_router_identity_changes_scope() -> None:
    assert _hash(router_id="router:abc123") != _hash()
    assert _hash(router_id="router:abc123") != _hash(router_id="router:def456")


def test_context_hash_model_router_identity_changes_scope() -> None:
    assert _hash(model_router="ceiling:PUBLIC:") != _hash()
    assert _hash(model_router="ceiling:PUBLIC:") != _hash(model_router="ceiling:INTERNAL:")


def test_ceiling_router_cache_id_is_stable_and_ceiling_sensitive() -> None:
    public = ClassificationCeilingRouter(ceiling=Classification.PUBLIC)
    assert public.cache_id == ClassificationCeilingRouter(ceiling=Classification.PUBLIC).cache_id
    assert public.cache_id != ClassificationCeilingRouter(ceiling=Classification.INTERNAL).cache_id
    assert public.cache_id != ClassificationCeilingRouter(ceiling=Classification.PUBLIC, model_id="local").cache_id
