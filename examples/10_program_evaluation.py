from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from _shared import build_demo_runtime, copy_demo_documents, emit, seed_demo_project

from refaat.evaluation import Dataset, default_bundle


def main() -> None:
    with TemporaryDirectory(prefix="refaat-eval-") as directory:
        root = Path(directory)
        documents = copy_demo_documents(root / "documents")
        with build_demo_runtime(root) as runtime:
            project = seed_demo_project(runtime, documents)
            dataset = Dataset.from_records(
                "northstar-regression",
                [
                    {"query": "What evidence is required before a production release?", "expected_terms": ["release"]},
                    {"query": "How is a production incident escalated?", "expected_terms": ["incident"]},
                    {"query": "How does the platform protect customer data?", "expected_terms": ["security"]},
                ],
            )
            report = project.programs.get("grounded-rag").evaluate(dataset, default_bundle())
            if not report.hard_constraints_passed:
                raise SystemExit("Evaluation did not satisfy its hard constraints")
            emit(
                {
                    "program": report.program,
                    "dataset_version": report.dataset_version,
                    "case_count": report.case_count,
                    "hard_constraints_passed": report.hard_constraints_passed,
                    "score": round(report.score, 3),
                    "aggregate": {name: round(value, 3) for name, value in report.aggregate.items()},
                }
            )


if __name__ == "__main__":
    main()
