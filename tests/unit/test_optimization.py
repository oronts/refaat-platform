from __future__ import annotations

from refaat.optimization import (
    Candidate,
    OptimizationResult,
    context_packing_search_space,
    default_search_space,
    retrieval_search_space,
)


def test_search_space_is_cartesian_product() -> None:
    space = retrieval_search_space(top_k=(4, 8), token_budget=(None, 200), minimum_citation_coverage=(0.0,))
    assert len(space) == 4
    assert {tunables.top_k for tunables in space} == {4, 8}
    assert {tunables.token_budget for tunables in space} == {None, 200}
    assert len(default_search_space()) == 4


def test_candidate_and_result_serialize() -> None:
    candidate = Candidate(
        tunables={"top_k": 4},
        artifact_id="sha256:x",
        hard_constraints_passed=True,
        score=0.9,
        aggregate={"grounded": 1.0},
        mean_tokens=128.0,
    )
    result = OptimizationResult(
        program="p",
        dataset_version="sha256:d",
        candidate_count=1,
        baseline=candidate,
        best=candidate,
        improvement=0.0,
        candidates=(candidate,),
    )
    payload = result.to_json()
    assert payload["best"]["artifact_id"] == "sha256:x"
    assert payload["candidate_count"] == 1
    assert payload["candidates"][0]["score"] == 0.9
    assert payload["candidates"][0]["mean_tokens"] == 128.0
    assert payload["pareto_front"] == []


def test_context_packing_search_space_varies_token_budget() -> None:
    space = context_packing_search_space(top_k=(4,), token_budget=(256, 512, None))
    assert {tunables.token_budget for tunables in space} == {256, 512, None}
    assert all(tunables.top_k == 4 for tunables in space)


def test_pareto_front_keeps_non_dominated_candidates() -> None:
    from refaat.optimization.candidates import pareto_front

    def candidate(score: float, tokens: float, *, ok: bool = True) -> Candidate:
        return Candidate(
            tunables={},
            artifact_id=f"sha256:{score}-{tokens}",
            hard_constraints_passed=ok,
            score=score,
            aggregate={},
            mean_tokens=tokens,
        )

    best_quality = candidate(0.9, 100.0)
    cheap = candidate(0.7, 50.0)
    dominated = candidate(0.6, 120.0)
    non_compliant = candidate(0.95, 80.0, ok=False)

    front = {c.artifact_id for c in pareto_front([best_quality, cheap, dominated, non_compliant])}
    assert best_quality.artifact_id in front
    assert cheap.artifact_id in front
    assert dominated.artifact_id not in front
    assert non_compliant.artifact_id not in front
