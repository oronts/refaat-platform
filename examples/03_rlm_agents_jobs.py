from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from _shared import demo_settings, emit, seed_demo_project

from refaat import RlmLimits, RuntimeBuilder
from refaat.application.worker import DurableWorker


def main() -> None:
    with TemporaryDirectory(prefix="refaat-durable-") as directory:
        root = Path(directory)
        settings = demo_settings(root)
        with RuntimeBuilder(settings).build() as runtime:
            project = seed_demo_project(runtime)
            result = project.rlm(
                "Compare release, incident, security, privacy, and customer-onboarding controls.",
                limits=RlmLimits(
                    max_depth=2,
                    max_nodes=16,
                    branch_factor=3,
                    leaf_token_budget=128,
                ),
            )
            planner_id = runtime.agents.create(
                project.workspace_id,
                "readiness-planner",
                "planner",
                goal="Prepare the readiness review",
            )
            reviewer_id = runtime.agents.create(
                project.workspace_id,
                "evidence-reviewer",
                "reviewer",
                goal="Validate cited controls",
                parent_agent_id=planner_id,
            )
            message = runtime.agents.send(
                project.workspace_id,
                planner_id,
                reviewer_id,
                "review.requested",
                {"question": "Validate release and incident controls"},
                correlation_id=result.trace_id,
            )
            mailbox = runtime.agents.receive(project.workspace_id, reviewer_id)
            runtime.agents.acknowledge(project.workspace_id, message.id)

            job_id = project.enqueue_rlm(
                "Summarize the complete production-readiness evidence.",
                limits=RlmLimits(max_depth=1, max_nodes=8, branch_factor=3, leaf_token_budget=160),
            )
            worker = DurableWorker(runtime)
            assert worker.run_once()
            job = runtime.store.get_job(project.workspace_id, job_id)

            assert result.citations
            assert result.usage["model_calls"] > 1
            assert result.usage["max_depth_reached"] > 0
            assert mailbox[0].id == message.id
            assert str(job["status"]) == "succeeded"

        with RuntimeBuilder(settings).build() as restarted:
            persisted_job = restarted.store.get_job(project.workspace_id, job_id)
            persisted_agent = restarted.store.get_agent(project.workspace_id, reviewer_id)
            assert str(persisted_job["status"]) == "succeeded"
            assert persisted_agent["goal"] == "Validate cited controls"

            emit(
                {
                    "rlm_nodes": len(result.nodes),
                    "rlm_usage": result.usage,
                    "citations": [citation.label for citation in result.citations],
                    "mailbox_message": mailbox[0],
                    "background_job_status": str(persisted_job["status"]),
                    "persisted_agent": {
                        "id": persisted_agent["id"],
                        "goal": persisted_agent["goal"],
                        "parent_agent_id": persisted_agent["parent_agent_id"],
                    },
                }
            )


if __name__ == "__main__":
    main()
