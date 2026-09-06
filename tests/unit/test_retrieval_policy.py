from __future__ import annotations

import json
from typing import Any, cast

from refaat.config import Settings
from refaat.core.retrieval import AdaptiveRetriever
from refaat.domain.models import QueryRequest
from refaat.optimization import fusion_search_space
from refaat.programs import GroundedRagProgram


def _retriever(settings: Settings) -> AdaptiveRetriever:
    return AdaptiveRetriever(cast(Any, None), cast(Any, None), settings)


def _request(**overrides: Any) -> QueryRequest:
    return QueryRequest(workspace_id="w", corpus_id="c", query="q", **overrides)


def test_as_json_carries_the_fusion_and_autocut_defaults(settings: Settings) -> None:
    policy = _retriever(settings).resolve_policy(_request(), {})
    data = policy.as_json()
    assert data["fusion"] == "rrf"
    assert data["rrf_k"] == 60
    assert data["autocut_enabled"] is False
    assert data["autocut_threshold"] == 0.5


def test_request_overrides_beat_harness_and_default(settings: Settings) -> None:
    request = _request(fusion="linear", rrf_k=10, autocut_enabled=True, autocut_threshold=0.7)
    harness = {"retrieval": {"fusion": "rrf", "rrf_k": 99, "autocut_enabled": False}}
    policy = _retriever(settings).resolve_policy(request, harness)
    assert policy.fusion == "linear"
    assert policy.rrf_k == 10
    assert policy.autocut_enabled is True
    assert policy.autocut_threshold == 0.7


def test_harness_overrides_apply_when_the_request_is_silent(settings: Settings) -> None:
    harness = {"retrieval": {"fusion": "linear", "rrf_k": 25, "autocut_enabled": True, "autocut_threshold": 0.9}}
    policy = _retriever(settings).resolve_policy(_request(), harness)
    assert policy.fusion == "linear"
    assert policy.rrf_k == 25
    assert policy.autocut_enabled is True
    assert policy.autocut_threshold == 0.9


def test_invalid_knobs_clamp_or_fall_back(settings: Settings) -> None:
    request = _request(fusion="bogus", rrf_k=0, autocut_threshold=2.0)
    policy = _retriever(settings).resolve_policy(request, {})
    assert policy.fusion == "rrf"
    assert policy.rrf_k == 1
    assert policy.autocut_threshold == 1.0


def test_as_json_carries_query_expansion_defaults(settings: Settings) -> None:
    data = _retriever(settings).resolve_policy(_request(), {}).as_json()
    assert data["query_expansion"] is True
    assert data["max_query_variants"] == 4


def test_query_expansion_precedence_request_then_harness_then_default(settings: Settings) -> None:
    request = _request(query_expansion=False, max_query_variants=2)
    harness = {"retrieval": {"query_expansion": True, "max_query_variants": 7}}
    policy = _retriever(settings).resolve_policy(request, harness)
    assert policy.query_expansion is False
    assert policy.max_query_variants == 2

    harness_only = _retriever(settings).resolve_policy(_request(), harness)
    assert harness_only.query_expansion is True
    assert harness_only.max_query_variants == 7


def test_max_query_variants_clamps_to_bounds(settings: Settings) -> None:
    policy = _retriever(settings).resolve_policy(_request(max_query_variants=99), {})
    assert policy.max_query_variants == 8
    floor = _retriever(settings).resolve_policy(_request(max_query_variants=0), {})
    assert floor.max_query_variants == 1


def test_as_json_carries_compression_defaults(settings: Settings) -> None:
    data = _retriever(settings).resolve_policy(_request(), {}).as_json()
    assert data["compression_enabled"] is False
    assert data["compression_min_salience"] == 0.0


def test_compression_harness_overrides_apply_and_clamp(settings: Settings) -> None:
    harness = {"retrieval": {"compression_enabled": True, "compression_min_salience": 2.0}}
    policy = _retriever(settings).resolve_policy(_request(), harness)
    assert policy.compression_enabled is True
    assert policy.compression_min_salience == 1.0


def test_fusion_search_space_yields_distinct_artifacts() -> None:
    space = fusion_search_space()
    assert len(space) == 4
    payloads = {json.dumps(tunables.as_json(), sort_keys=True) for tunables in space}
    artifacts = {GroundedRagProgram().with_tunables(tunables).artifact().artifact_id for tunables in space}
    assert len(payloads) == 4
    assert len(artifacts) == 4
