from __future__ import annotations

import pytest

from refaat.application.runtime import Refaat
from refaat.domain.models import DocumentInput
from refaat.evaluation import Dataset, default_bundle
from refaat.evaluation.dataset import EvaluationCase
from refaat.optimization import (
    assemble_result,
    build_ranked_result,
    evaluate_candidate,
    fusion_search_space,
    retrieval_search_space,
    select_demonstrations,
)
from refaat.optimization.candidates import Candidate
from refaat.programs.program import Tunables

_DEMOS = select_demonstrations(
    (
        EvaluationCase(id="a", query="What is a release?", reference_answer="A release is a shipped version."),
        EvaluationCase(id="b", query="What is an incident?", reference_answer="An incident is a Sev1 event."),
    ),
    2,
)

_ROUND_TRIP_CASES = [
    *retrieval_search_space(top_k=(2, 8)),
    *fusion_search_space(),
    Tunables(top_k=6, instructions="Answer precisely.", demonstrations=_DEMOS),
    Tunables(router={"long_query_tokens": 30, "simple_top_k": 3, "complex_top_k": 9}),
    Tunables(top_k=4, token_budget=512, minimum_citation_coverage=0.5, autocut_enabled=True, autocut_threshold=0.2),
]


@pytest.mark.parametrize("tunables", _ROUND_TRIP_CASES)
def test_tunables_from_json_round_trips(tunables: Tunables) -> None:
    restored = Tunables.from_json(tunables.as_json())
    assert restored == tunables
    assert restored.as_json() == tunables.as_json()


# to_records / from_records is a faithful inverse for an empty id and reference,
# so a resumed run recomputes the identical content-addressed dataset version.
def test_dataset_round_trips_empty_id_and_reference() -> None:
    dataset = Dataset(
        name="drift",
        cases=(
            EvaluationCase(id="", query="what is a release?", reference_answer=""),
            EvaluationCase(id="b", query="what is an incident?", reference_answer="A Sev1 event."),
        ),
    )
    restored = Dataset.from_records(dataset.name, dataset.to_records())
    assert restored.cases == dataset.cases
    assert restored.version == dataset.version


def test_candidate_from_json_round_trips() -> None:
    candidate = Candidate(
        tunables={"top_k": 8, "token_budget": None, "minimum_citation_coverage": 0.0},
        artifact_id="sha256:abc",
        hard_constraints_passed=True,
        score=0.75,
        aggregate={"answer_contains": 1.0, "citation_recall": 0.5},
        mean_tokens=42.0,
    )
    assert Candidate.from_json(candidate.to_json()) == candidate


def _seed(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    for index in range(4):
        runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id=f"doc{index}.txt",
                title=f"Doc {index}",
                media_type="text/plain",
                content=(
                    f"Runbook {index}: a production release requires sign-off before deploy; "
                    "escalate a Sev1 incident to on-call within fifteen minutes."
                ).encode(),
            ),
        )


def test_extracted_helpers_reproduce_build_ranked_result(runtime: Refaat) -> None:
    _seed(runtime)
    project = runtime.project(workspace="acme", corpus="handbook")
    dataset = Dataset.from_records(
        "guard",
        [
            {"query": "production release sign-off", "expected_terms": ["release"]},
            {"query": "incident escalation", "expected_terms": ["incident"]},
        ],
    )
    space = retrieval_search_space(top_k=(2, 6, 12))
    bundle = default_bundle()
    program = project.programs.get("grounded-rag").program

    expected = build_ranked_result(project, program, space, dataset, bundle)
    candidates = [evaluate_candidate(project, program, tunables, dataset, bundle) for tunables in space]
    manual = assemble_result(program.name, dataset.version, candidates)

    assert manual.to_json() == expected.to_json()
