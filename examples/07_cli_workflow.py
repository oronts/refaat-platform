from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from _shared import DOCUMENTS_ROOT, EVALUATION_CASES_PATH, emit, run_cli


def main() -> None:
    with TemporaryDirectory(prefix="refaat-cli-") as directory:
        root = Path(directory)
        doctor = run_cli(root, "doctor")
        workspace = run_cli(root, "init", "--slug", "northstar-cli", "--name", "Northstar CLI")
        workspace_id = str(workspace["workspace_id"])
        corpus = run_cli(root, "corpus", "create", workspace_id, "operations")
        corpus_id = str(corpus["id"])
        ingestion = run_cli(
            root,
            "ingest",
            workspace_id,
            corpus_id,
            str(DOCUMENTS_ROOT / "02_release_policy.md"),
        )
        answer = run_cli(
            root,
            "query",
            workspace_id,
            corpus_id,
            "What evidence is required before a production release?",
        )
        recursive = run_cli(
            root,
            "rlm",
            workspace_id,
            corpus_id,
            "Summarize the release controls.",
            "--max-depth",
            "1",
            "--max-nodes",
            "4",
            "--branch-factor",
            "2",
            "--leaf-token-budget",
            "128",
        )
        suite = run_cli(
            root,
            "evaluation",
            "create",
            workspace_id,
            "cli-regression",
            str(EVALUATION_CASES_PATH),
        )
        evaluation = run_cli(
            root,
            "evaluation",
            "run",
            workspace_id,
            str(suite["suite_id"]),
            corpus_id,
        )
        skills = run_cli(root, "skill", "list")

        assert doctor["status"] == "ok"
        assert ingestion["changed"]
        assert answer["citations"]
        assert recursive["nodes"]
        assert evaluation["metrics"]["case_count"] == 10
        assert any(skill["name"] == "corpus_health" for skill in skills)

        emit(
            {
                "doctor": doctor,
                "workspace_id": workspace_id,
                "corpus_id": corpus_id,
                "document_id": ingestion["document_id"],
                "query_citations": len(answer["citations"]),
                "rlm_nodes": len(recursive["nodes"]),
                "evaluation_metrics": evaluation["metrics"],
                "skills": skills,
            }
        )


if __name__ == "__main__":
    main()
