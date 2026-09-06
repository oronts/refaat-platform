from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from _shared import build_demo_runtime, copy_demo_documents, emit, seed_demo_project


def main() -> None:
    with TemporaryDirectory(prefix="refaat-local-") as directory:
        root = Path(directory)
        documents = copy_demo_documents(root / "documents")
        with build_demo_runtime(root) as runtime:
            project = seed_demo_project(runtime, documents)
            context = project.retrieve(
                "How are the Release Manager and Security Team involved in a Production Release?",
                top_k=6,
            )
            first = project.ask("What evidence is required before a production release?")
            cached = project.ask("What evidence is required before a production release?")

            release_policy = documents / "02_release_policy.md"
            release_policy.write_text(
                release_policy.read_text(encoding="utf-8")
                + "\n\nA deployment ticket must identify the on-call owner.\n",
                encoding="utf-8",
            )
            update = project.ingest(release_policy)[0]
            refreshed = project.ask("What evidence is required before a production release?")

            assert project.stats()["documents"] == 6
            assert context.graph_facts
            assert first.citations
            assert cached.cached
            assert cached.routes[0] == "cag:exact"
            assert update.changed
            assert refreshed.corpus_version > cached.corpus_version
            assert not refreshed.cached

            emit(
                {
                    "stats": project.stats(),
                    "retrieval_routes": context.routes,
                    "graph_facts": context.graph_facts[:5],
                    "first_cached": first.cached,
                    "second_cached": cached.cached,
                    "cache_route": cached.routes[0],
                    "version_before_update": cached.corpus_version,
                    "version_after_update": refreshed.corpus_version,
                    "citation_labels": [citation.label for citation in refreshed.citations],
                }
            )


if __name__ == "__main__":
    main()
