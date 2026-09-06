from __future__ import annotations

from refaat.application.runtime import Refaat
from refaat.domain.models import DocumentInput
from refaat.evaluation import Dataset
from refaat.evaluation.dataset import EvaluationCase
from refaat.optimization import LabeledFewShotOptimizer, select_demonstrations
from refaat.optimization.candidates import build_ranked_result_budgeted as _shared_helper
from refaat.programs.program import GroundedRagProgram, Tunables


def test_select_demonstrations_id_sorted_sanitized_skips_unlabeled() -> None:
    cases = (
        EvaluationCase(id="c", query="Q-c", reference_answer="answer c [chunk:real_c]"),
        EvaluationCase(id="a", query="Q-a", reference_answer="answer a"),
        EvaluationCase(id="b", query="Q-b", reference_answer=None),
    )
    demos = select_demonstrations(cases, 3)
    assert [demo.question for demo in demos] == ["Q-a", "Q-c"]
    assert "real_c" not in demos[1].answer
    assert "[chunk:example]" in demos[1].answer


def test_select_demonstrations_respects_k() -> None:
    cases = tuple(
        EvaluationCase(id=f"case-{index}", query=f"Q{index}", reference_answer=f"A{index}") for index in range(5)
    )
    assert len(select_demonstrations(cases, 2)) == 2


def test_tunables_key_presence_for_guidance() -> None:
    assert "instructions" not in Tunables().as_json()
    assert "demonstrations" not in Tunables().as_json()
    guided = Tunables(
        instructions="be brief",
        demonstrations=select_demonstrations((EvaluationCase(id="a", query="q", reference_answer="a"),), 1),
    )
    payload = guided.as_json()
    assert payload["instructions"] == "be brief"
    assert payload["demonstrations"] == [{"question": "q", "answer": "a"}]


def _ingest(project) -> None:
    for index in range(4):
        project.runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id=f"doc{index}.txt",
                title=f"Doc {index}",
                media_type="text/plain",
                content=f"Runbook {index}: escalate a Sev1 incident to on-call within fifteen minutes.".encode(),
            ),
        )


def _labeled_dataset() -> Dataset:
    return Dataset.from_records(
        "fewshot",
        [
            {"query": "incident escalation path", "expected_terms": ["incident"], "reference_answer": "Escalate."},
            {"query": "on-call response window", "expected_terms": ["minutes"], "reference_answer": "Fifteen minutes."},
            {"query": "sev1 handling", "expected_terms": ["sev1"], "reference_answer": "Handle sev1."},
            {"query": "runbook location", "expected_terms": ["runbook"], "reference_answer": "In the runbook."},
        ],
    )


def test_optimizer_distinct_artifacts_heldout_and_leaves_program_unchanged(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    _ingest(project)
    program = GroundedRagProgram()
    before = program.tunables

    result = LabeledFewShotOptimizer(project).optimize(program, _labeled_dataset(), shots=(1, 2))

    assert result.candidate_count == 2
    assert len({candidate.artifact_id for candidate in result.candidates}) == 2
    assert all(candidate.hard_constraints_passed for candidate in result.candidates)
    assert result.pareto_front
    assert program.tunables is before
    assert program.tunables.instructions is None
    assert program.tunables.demonstrations == ()
    # Held-out: the two demonstration cases are excluded, two remain scored.
    assert result.dataset_version != _labeled_dataset().version


def test_token_axis_is_load_bearing_by_shot_count(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    _ingest(project)

    result = LabeledFewShotOptimizer(project).optimize(GroundedRagProgram(), _labeled_dataset(), shots=(1, 2))

    by_shots = {len(candidate.tunables["demonstrations"]): candidate.mean_tokens for candidate in result.candidates}
    assert by_shots[2] > by_shots[1]


def test_both_optimizers_share_the_rank_helper() -> None:
    from refaat.optimization import fewshot, optimizer

    assert fewshot.build_ranked_result_budgeted is _shared_helper
    assert optimizer.build_ranked_result_budgeted is _shared_helper
