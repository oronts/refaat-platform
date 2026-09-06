from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from _shared import build_demo_runtime, emit, load_evaluation_cases, seed_demo_project


def main() -> None:
    with TemporaryDirectory(prefix="refaat-evaluation-") as directory:
        root = Path(directory)
        with build_demo_runtime(root) as runtime:
            project = seed_demo_project(runtime)
            baseline = project.create_harness_snapshot(
                supplemental={
                    "retrieval": {
                        "semantic_weight": 0.55,
                        "lexical_weight": 0.30,
                        "graph_weight": 0.15,
                        "top_k": 10,
                    }
                },
                evidence={"reason": "documented example baseline"},
                activate=True,
            )
            suite_id = project.create_evaluation_suite(
                "production-readiness-regression",
                load_evaluation_cases(),
            )
            evaluation = project.run_evaluation(suite_id)
            metrics = evaluation["metrics"]
            proposal = runtime.harness.propose_from_metrics(metrics)
            improved = project.create_harness_snapshot(
                supplemental=proposal,
                evidence={
                    "evaluation_run_id": evaluation["id"],
                    "metrics": metrics,
                },
                activate=True,
            )
            improved_state = project.active_harness()
            rolled_back = project.activate_harness_snapshot(baseline.id)

            assert metrics["case_count"] == 10
            assert proposal
            assert improved_state == proposal
            assert rolled_back.id == baseline.id
            assert project.active_harness() == baseline.supplemental

            emit(
                {
                    "suite_id": suite_id,
                    "run_id": evaluation["id"],
                    "metrics": metrics,
                    "proposal": proposal,
                    "improved_snapshot_id": improved.id,
                    "rollback_snapshot_id": rolled_back.id,
                    "active_after_rollback": project.active_harness(),
                }
            )


if __name__ == "__main__":
    main()
