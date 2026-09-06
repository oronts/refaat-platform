import pytest

from refaat.core.harness import HarnessService
from refaat.errors import ValidationError


def test_harness_proposal_requires_evidence_volume(runtime, workspace_id, corpus_id) -> None:
    harness = HarnessService(runtime.store)
    with pytest.raises(ValidationError):
        harness.propose_from_metrics({"case_count": 3, "citation_recall": 0.2})

    proposal = harness.propose_from_metrics({"case_count": 12, "citation_recall": 0.5, "answer_hit_rate": 0.7})
    assert proposal["retrieval"]["top_k"] == 16


def test_harness_rejects_unbounded_state(runtime, workspace_id, corpus_id) -> None:
    with pytest.raises(ValidationError):
        runtime.harness.create_snapshot(
            workspace_id=workspace_id,
            corpus_id=corpus_id,
            supplemental={"tools": {"shell": True}},
            evidence={"run_id": "eval_1"},
        )

    snapshot = runtime.harness.create_snapshot(
        workspace_id=workspace_id,
        corpus_id=corpus_id,
        supplemental={"retrieval": {"semantic_weight": 0.6, "top_k": 10}},
        evidence={"run_id": "eval_1"},
        activate=True,
    )
    assert snapshot.active
    assert runtime.harness.active(workspace_id, corpus_id)["retrieval"]["top_k"] == 10


def test_harness_accepts_new_retrieval_knobs(runtime, workspace_id, corpus_id) -> None:
    snapshot = runtime.harness.create_snapshot(
        workspace_id=workspace_id,
        corpus_id=corpus_id,
        supplemental={
            "retrieval": {"fusion": "linear", "rrf_k": 25, "autocut_enabled": True, "autocut_threshold": 0.6}
        },
        evidence={"run_id": "eval_2"},
        activate=True,
    )
    assert snapshot.active
    active = runtime.harness.active(workspace_id, corpus_id)["retrieval"]
    assert active["fusion"] == "linear"
    assert active["rrf_k"] == 25


def test_harness_accepts_filters_and_bm25_knobs(runtime, workspace_id, corpus_id) -> None:
    snapshot = runtime.harness.create_snapshot(
        workspace_id=workspace_id,
        corpus_id=corpus_id,
        supplemental={
            "retrieval": {
                "filters": {"category": "security", "tier": [1, 2]},
                "bm25_k1": 1.5,
                "bm25_b": 0.4,
            }
        },
        evidence={"run_id": "eval_bm25"},
        activate=True,
    )
    assert snapshot.active
    active = runtime.harness.active(workspace_id, corpus_id)["retrieval"]
    assert active["filters"]["category"] == "security"
    assert active["bm25_k1"] == 1.5


def test_harness_accepts_query_expansion_knobs(runtime, workspace_id, corpus_id) -> None:
    snapshot = runtime.harness.create_snapshot(
        workspace_id=workspace_id,
        corpus_id=corpus_id,
        supplemental={"retrieval": {"query_expansion": False, "max_query_variants": 6}},
        evidence={"run_id": "eval_expand"},
        activate=True,
    )
    assert snapshot.active
    active = runtime.harness.active(workspace_id, corpus_id)["retrieval"]
    assert active["query_expansion"] is False
    assert active["max_query_variants"] == 6


def test_harness_accepts_compression_knobs(runtime, workspace_id, corpus_id) -> None:
    snapshot = runtime.harness.create_snapshot(
        workspace_id=workspace_id,
        corpus_id=corpus_id,
        supplemental={"retrieval": {"compression_enabled": True, "compression_min_salience": 0.3}},
        evidence={"run_id": "eval_compress"},
        activate=True,
    )
    assert snapshot.active
    active = runtime.harness.active(workspace_id, corpus_id)["retrieval"]
    assert active["compression_enabled"] is True
    assert active["compression_min_salience"] == 0.3


def test_harness_accepts_window_radius(runtime, workspace_id, corpus_id) -> None:
    for radius in (0, 8):
        snapshot = runtime.harness.create_snapshot(
            workspace_id=workspace_id,
            corpus_id=corpus_id,
            supplemental={"retrieval": {"window_radius": radius}},
            evidence={"run_id": f"eval_window_{radius}"},
            activate=True,
        )
        assert snapshot.active
        assert runtime.harness.active(workspace_id, corpus_id)["retrieval"]["window_radius"] == radius


@pytest.mark.parametrize(
    "retrieval",
    [
        {"fusion": "bogus"},
        {"window_radius": 9},
        {"window_radius": -1},
        {"window_radius": 1.5},
        {"window_radius": True},
        {"expansion_mode": "parent_child"},
        {"compression_enabled": "yes"},
        {"compression_min_salience": True},
        {"compression_min_salience": 1.5},
        {"compression_min_salience": -0.2},
        {"rrf_k": 0},
        {"rrf_k": True},
        {"autocut_enabled": "yes"},
        {"autocut_threshold": 2.0},
        {"autocut_threshold": True},
        {"filters": "not-an-object"},
        {"filters": [1, 2]},
        {"filters": {"category": {"nested": "object"}}},
        {"bm25_k1": 6.0},
        {"bm25_k1": -1.0},
        {"bm25_k1": True},
        {"bm25_b": 2.0},
        {"bm25_b": True},
        {"query_expansion": "yes"},
        {"max_query_variants": 0},
        {"max_query_variants": 9},
        {"max_query_variants": True},
    ],
)
def test_harness_rejects_invalid_retrieval_knobs(runtime, workspace_id, corpus_id, retrieval) -> None:
    with pytest.raises(ValidationError):
        runtime.harness.create_snapshot(
            workspace_id=workspace_id,
            corpus_id=corpus_id,
            supplemental={"retrieval": retrieval},
            evidence={"run_id": "eval_3"},
        )
