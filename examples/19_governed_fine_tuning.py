from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from _shared import build_demo_runtime, copy_demo_documents, emit, seed_demo_project

from refaat.application.fine_tuning import fine_tune_artifact, fine_tune_program_name
from refaat.application.worker import DurableWorker
from refaat.errors import NotFoundError, ValidationError
from refaat.evaluation import Dataset


def main() -> None:
    with TemporaryDirectory(prefix="refaat-fine-tune-") as directory:
        root = Path(directory)
        documents = copy_demo_documents(root / "documents")
        with build_demo_runtime(root) as runtime:
            project = seed_demo_project(runtime, documents)
            dataset = Dataset.from_records(
                "northstar-fine-tune",
                [
                    {"query": "What is required before a production release?", "reference_answer": "A checklist."},
                    {"query": "How is a production incident escalated?", "reference_answer": "Through on-call."},
                ],
            )

            run_id = project.fine_tuning.request(
                dataset,
                base_model="refaat-base-small",
                hyperparameters={"epochs": 3},
                requester="founder",
            )
            created = project.fine_tuning.get(run_id)

            refused_without_approval = False
            try:
                project.fine_tuning.run(run_id)
            except ValidationError:
                refused_without_approval = True

            project.fine_tuning.approve(run_id, approver="lead-reviewer")
            job_id = project.fine_tuning.enqueue(run_id)
            worked = DurableWorker(runtime).run_once()

            run = project.fine_tuning.get(run_id)
            model_ref = str(run["result_model_ref"])
            program_name = fine_tune_program_name(str(run["base_model"]))
            artifact = fine_tune_artifact(
                base_model=str(run["base_model"]),
                dataset_version=str(run["dataset_version"]),
                hyperparameters=dict(run["hyperparameters"]),
                model_ref=model_ref,
            )
            registered = runtime.registry.get(project.workspace_id, artifact.artifact_id)

            not_auto_promoted = False
            try:
                runtime.registry.resolve(project.workspace_id, program_name, "production")
            except NotFoundError:
                not_auto_promoted = True

            emit(
                {
                    "run_id": run_id,
                    "job_id": job_id,
                    "worked": worked,
                    "created_status": created["status"],
                    "created_approved": created["approved"],
                    "refused_without_approval": refused_without_approval,
                    "run_status": run["status"],
                    "approved_by": run["approved_by"],
                    "model_ref": model_ref,
                    "model_ref_deterministic": model_ref.startswith("ft:refaat:"),
                    "provenance_trained": run["provenance"].get("trained"),
                    "provenance_dataset_version": run["provenance"].get("dataset_version"),
                    "provenance_recorded": bool(run["provenance"].get("dataset_version")),
                    "artifact_registered": registered["program"] == program_name,
                    "not_auto_promoted": not_auto_promoted,
                }
            )


if __name__ == "__main__":
    main()
