from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from _shared import build_demo_runtime, copy_demo_documents, emit, seed_demo_project

from refaat.programs import GroundedRagProgram, InputField, Signature, Tunables


class OperationsQuestion(Signature):
    question: str = InputField(description="An operations question answered from the handbook.")


class OperationsAssistant(GroundedRagProgram):
    name = "operations-assistant"
    input_signature = OperationsQuestion


def main() -> None:
    with TemporaryDirectory(prefix="refaat-programs-") as directory:
        root = Path(directory)
        documents = copy_demo_documents(root / "documents")
        with build_demo_runtime(root) as runtime:
            project = seed_demo_project(runtime, documents)
            assistant = project.programs.register(
                OperationsAssistant(tunables=Tunables(top_k=6, minimum_citation_coverage=0.5))
            )
            question = "How does the incident escalation process work?"
            first = assistant.run(question=question)
            cached = assistant.run(question=question)
            if not first.grounded or not first.citations:
                raise SystemExit("Program answer was not grounded in retrieved evidence")
            emit(
                {
                    "program": first.program,
                    "artifact_id": first.artifact_id,
                    "grounded": first.grounded,
                    "citations": len(first.citations),
                    "routes": list(first.routes),
                    "second_cached": cached.cached,
                    "programs": [item["program"] for item in project.programs.list()],
                }
            )


if __name__ == "__main__":
    main()
