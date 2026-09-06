from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from _shared import build_demo_runtime, emit, seed_demo_project


def main() -> None:
    with TemporaryDirectory(prefix="refaat-memory-") as directory:
        root = Path(directory)
        with build_demo_runtime(root) as runtime:
            project = seed_demo_project(runtime)
            session_id = project.create_session(
                user_id="alice",
                goal="Prepare an enterprise production-readiness review",
            )
            release = project.ask(
                "What evidence is required before a production release?",
                user_id="alice",
                session_id=session_id,
            )
            project.ask(
                "Who coordinates a Severity One Incident?",
                user_id="alice",
                session_id=session_id,
            )
            project.ask(
                "Where must Personal Data remain?",
                user_id="alice",
                session_id=session_id,
            )
            selected = tuple(citation.chunk_id for citation in release.citations[:1])
            project.submit_feedback(
                rating=1,
                user_id="alice",
                selected_chunk_ids=selected,
                note="This release-policy evidence is especially useful.",
            )
            personalized = project.retrieve(
                "Which controls should Alice review before deployment?",
                user_id="alice",
            )
            session = runtime.store.get_session(project.workspace_id, session_id)
            turns = runtime.store.list_session_turns(project.workspace_id, session_id)
            profile = runtime.store.get_preferences(
                project.workspace_id,
                project.corpus_id,
                "user:alice",
            )

            assert release.citations
            assert session["summary"]
            assert len(turns) == runtime.settings.retained_session_turns
            assert profile
            assert "profile" in personalized.routes
            assert any("profile" in hit.reasons for hit in personalized.hits)

            emit(
                {
                    "session_id": session_id,
                    "goal": session["goal"],
                    "summary": session["summary"],
                    "retained_turns": len(turns),
                    "profile": profile,
                    "personalized_routes": personalized.routes,
                    "preferred_hits": [
                        {
                            "document_id": hit.chunk.document_id,
                            "reasons": hit.reasons,
                        }
                        for hit in personalized.hits
                        if "profile" in hit.reasons
                    ],
                }
            )


if __name__ == "__main__":
    main()
