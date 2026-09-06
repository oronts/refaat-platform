from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from _shared import build_demo_runtime, copy_demo_documents, emit, seed_demo_project

from refaat.application.worker import DurableWorker
from refaat.evaluation import Dataset
from refaat.optimization import retrieval_search_space


def main() -> None:
    with TemporaryDirectory(prefix="refaat-durable-optimize-") as directory:
        root = Path(directory)
        documents = copy_demo_documents(root / "documents")
        with build_demo_runtime(root) as runtime:
            project = seed_demo_project(runtime, documents)
            dataset = Dataset.from_records(
                "northstar-durable-optimization",
                [
                    {"query": "What is required before a production release?", "expected_terms": ["release"]},
                    {"query": "How is a production incident escalated?", "expected_terms": ["incident"]},
                ],
            )
            search_space = retrieval_search_space(top_k=(4, 8, 12))

            synchronous = project.programs.get("grounded-rag").optimize(dataset, search_space=search_space)

            run_id = project.programs.get("grounded-rag").enqueue_optimization(dataset, search_space=search_space)
            worked = DurableWorker(runtime).run_once()
            run = runtime.store.get_optimization_run(project.workspace_id, run_id)
            if run["status"] != "succeeded":
                raise SystemExit("Durable optimization run did not succeed")
            if run["result"] != synchronous.to_json():
                raise SystemExit("Durable optimization result diverged from the synchronous run")

            candidates = runtime.store.list_optimization_candidates(run_id)
            registered_best = runtime.registry.get(project.workspace_id, str(run["best_artifact_id"]))

            emit(
                {
                    "run_id": run_id,
                    "worked": worked,
                    "run_status": run["status"],
                    "candidate_total": run["candidate_total"],
                    "candidate_count": len(candidates),
                    "best_artifact_id": run["best_artifact_id"],
                    "matches_synchronous": run["result"] == synchronous.to_json(),
                    "registered_best_program": registered_best["program"],
                }
            )


if __name__ == "__main__":
    main()
