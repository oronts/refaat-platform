from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from _shared import build_demo_runtime, copy_demo_documents, emit, seed_demo_project

from refaat.evaluation import Dataset
from refaat.optimization import retrieval_search_space


def main() -> None:
    with TemporaryDirectory(prefix="refaat-optimize-") as directory:
        root = Path(directory)
        documents = copy_demo_documents(root / "documents")
        with build_demo_runtime(root) as runtime:
            project = seed_demo_project(runtime, documents)
            dataset = Dataset.from_records(
                "northstar-optimization",
                [
                    {"query": "What is required before a production release?", "expected_terms": ["release"]},
                    {"query": "How is a production incident escalated?", "expected_terms": ["incident"]},
                ],
            )
            result = project.programs.get("grounded-rag").optimize(
                dataset, search_space=retrieval_search_space(top_k=(4, 8, 12))
            )
            emit(
                {
                    "program": result.program,
                    "dataset_version": result.dataset_version,
                    "candidate_count": result.candidate_count,
                    "improvement": round(result.improvement, 3),
                    "best": {
                        "tunables": result.best.tunables,
                        "score": round(result.best.score, 3),
                        "artifact_id": result.best.artifact_id,
                    },
                }
            )


if __name__ == "__main__":
    main()
