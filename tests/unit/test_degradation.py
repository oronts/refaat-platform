"""The pure degradation decision core and the conditional context-hash fold.

RefuseAll refuses every dependency class. SafeDefaults serves a marked grounded fallback when
grounding survives and abstains otherwise, so it never serves an ungrounded answer. A None policy
resolves to the legacy refuse, matching today's uncaught-outage behavior. The decision constructors
carry the right mode and marking. The QueryService context-hash fold is byte-identical when no
policy is installed and rescopes the answer cache when one is.
"""

from __future__ import annotations

from refaat.application.services import QueryService
from refaat.core.retrieval import RetrievalPolicy
from refaat.domain.degradation import (
    DegradationDecision,
    DegradationMode,
    DependencyClass,
    DependencyFailure,
    RefuseAllDegradationPolicy,
    SafeDefaultsDegradationPolicy,
    resolve_degradation,
)

# The base context hash with no optional fold, pinned in tests/unit/test_router.py. The degradation
# fold must leave it byte-identical when no policy is installed.
PINNED_CONTEXT_HASH = "7937b529c3dadd9ada1e5483bac13fcef58f71fcee4aa31da688b7d137df034c"


def _grounded(dependency_class: DependencyClass) -> DependencyFailure:
    return DependencyFailure(dependency_class=dependency_class, grounding_survives=True, provider_id="gen-v1")


def _ungrounded(dependency_class: DependencyClass) -> DependencyFailure:
    return DependencyFailure(dependency_class=dependency_class, grounding_survives=False, provider_id="gen-v1")


def test_refuse_all_refuses_every_dependency_class() -> None:
    policy = RefuseAllDegradationPolicy()
    for dependency_class in (DependencyClass.GENERATION, DependencyClass.RLM_SYNTHESIS):
        for failure in (_grounded(dependency_class), _ungrounded(dependency_class)):
            decision = policy.decide(failure)
            assert decision.mode is DegradationMode.REFUSE
            assert decision.dependency_class is dependency_class
            assert decision.marked is False


def test_safe_defaults_serves_degraded_when_grounding_survives() -> None:
    policy = SafeDefaultsDegradationPolicy()
    for dependency_class in (DependencyClass.GENERATION, DependencyClass.RLM_SYNTHESIS):
        decision = policy.decide(_grounded(dependency_class))
        assert decision.mode is DegradationMode.SERVE_DEGRADED
        assert decision.marked is True
        assert decision.dependency_class is dependency_class


def test_safe_defaults_abstains_when_grounding_does_not_survive() -> None:
    policy = SafeDefaultsDegradationPolicy()
    for dependency_class in (DependencyClass.GENERATION, DependencyClass.RLM_SYNTHESIS):
        decision = policy.decide(_ungrounded(dependency_class))
        assert decision.mode is DegradationMode.ABSTAIN
        assert decision.marked is False


def test_resolve_degradation_none_is_the_legacy_refuse() -> None:
    for dependency_class in (DependencyClass.GENERATION, DependencyClass.RLM_SYNTHESIS):
        decision = resolve_degradation(None, _grounded(dependency_class))
        assert decision.mode is DegradationMode.REFUSE
        assert decision.dependency_class is dependency_class
        # Even a surviving-grounding failure refuses when no policy is installed, matching today.
        assert resolve_degradation(None, _ungrounded(dependency_class)).mode is DegradationMode.REFUSE


def test_resolve_degradation_delegates_to_the_installed_policy() -> None:
    failure = _grounded(DependencyClass.GENERATION)
    assert resolve_degradation(SafeDefaultsDegradationPolicy(), failure).mode is DegradationMode.SERVE_DEGRADED
    assert resolve_degradation(RefuseAllDegradationPolicy(), failure).mode is DegradationMode.REFUSE


def test_decision_constructors_carry_the_right_mode_and_marking() -> None:
    served = DegradationDecision.serve_degraded(DependencyClass.GENERATION, marked=True, reason="grounded_fallback")
    assert served.mode is DegradationMode.SERVE_DEGRADED
    assert served.marked is True
    assert served.reason == "grounded_fallback"

    abstained = DegradationDecision.abstain(DependencyClass.RLM_SYNTHESIS, "no_grounding")
    assert abstained.mode is DegradationMode.ABSTAIN
    assert abstained.marked is False

    refused = DegradationDecision.refuse(DependencyClass.GENERATION, "refuse_all")
    assert refused.mode is DegradationMode.REFUSE
    assert refused.marked is False


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


def test_context_hash_is_byte_identical_when_no_degradation_policy() -> None:
    assert _hash() == PINNED_CONTEXT_HASH
    assert _hash(degradation=None) == PINNED_CONTEXT_HASH


def test_context_hash_degradation_identity_changes_scope() -> None:
    assert _hash(degradation="safe-defaults") != _hash()
    assert _hash(degradation="safe-defaults") != _hash(degradation="refuse-all")
