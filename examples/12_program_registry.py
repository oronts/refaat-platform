from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from _shared import build_demo_runtime, copy_demo_documents, emit, seed_demo_project

from refaat.programs import GroundedRagProgram, Tunables


def main() -> None:
    with TemporaryDirectory(prefix="refaat-registry-") as directory:
        root = Path(directory)
        documents = copy_demo_documents(root / "documents")
        with build_demo_runtime(root) as runtime:
            project = seed_demo_project(runtime, documents)
            registry = project.registry
            base = GroundedRagProgram().artifact()
            tuned = GroundedRagProgram(tunables=Tunables(top_k=4)).artifact()
            base_id = registry.save(base)
            tuned_id = registry.save(tuned)
            registry.approve("grounded-rag", "production", base_id)
            registry.promote("grounded-rag", "production", base_id)
            registry.approve("grounded-rag", "production", tuned_id)
            registry.promote("grounded-rag", "production", tuned_id)
            resolved_tuned = registry.resolve("grounded-rag", "production")["definition_hash"] == tuned.definition_hash
            rolled = registry.rollback("grounded-rag", "production")
            resolved_base = registry.resolve("grounded-rag", "production")["definition_hash"] == base.definition_hash
            if not (resolved_tuned and resolved_base and rolled["artifact_id"] == base_id):
                raise SystemExit("Registry promotion or rollback did not resolve correctly")
            emit(
                {
                    "base_artifact": base_id,
                    "tuned_artifact": tuned_id,
                    "resolved_tuned_after_promote": resolved_tuned,
                    "rolled_back_to": rolled["artifact_id"],
                    "resolved_base_after_rollback": resolved_base,
                    "history_length": len(registry.history("grounded-rag", "production")),
                }
            )


if __name__ == "__main__":
    main()
