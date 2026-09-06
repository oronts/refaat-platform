from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from _shared import build_demo_runtime, copy_demo_documents, emit, seed_demo_project

from refaat.application.worker import DurableWorker
from refaat.evaluation import Dataset


def main() -> None:
    with TemporaryDirectory(prefix="refaat-eval-job-") as directory:
        root = Path(directory)
        documents = copy_demo_documents(root / "documents")
        with build_demo_runtime(root) as runtime:
            project = seed_demo_project(runtime, documents)
            dataset = Dataset.from_records(
                "northstar-regression",
                [
                    {"query": "What is required before a production release?", "expected_terms": ["release"]},
                    {"query": "How is a production incident escalated?", "expected_terms": ["incident"]},
                ],
            )
            job_id = project.programs.get("grounded-rag").enqueue_evaluation(dataset)
            worked = DurableWorker(runtime).run_once()
            job = runtime.store.get_job(project.workspace_id, job_id)
            if job["status"] != "succeeded" or not job["result"]["hard_constraints_passed"]:
                raise SystemExit("Durable evaluation job did not succeed")
            emit(
                {
                    "job_id": job_id,
                    "worked": worked,
                    "job_status": job["status"],
                    "program": job["result"]["program"],
                    "case_count": job["result"]["case_count"],
                    "hard_constraints_passed": job["result"]["hard_constraints_passed"],
                }
            )


if __name__ == "__main__":
    main()
