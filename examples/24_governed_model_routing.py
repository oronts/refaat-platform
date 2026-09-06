from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from _shared import demo_settings, emit

from refaat.application.runtime import Refaat, RefaatProject, RuntimeBuilder
from refaat.application.services import ROUTING_REFUSAL_MESSAGE, ROUTING_ROUTE
from refaat.domain.authz import Classification
from refaat.domain.models import DocumentInput
from refaat.domain.routing import ClassificationCeilingRouter

QUERY = "checkout production release rollback plan"
MARKER = "BLUEJAY"

PUBLIC_DOCUMENT = DocumentInput(
    source_id="public.md",
    title="Public release note",
    media_type="text/markdown",
    content=b"The public checkout production release requires an approved rollback plan for every deploy.",
    metadata={"acl": {"classification": "public"}},
)

# The marker appears only in the confidential body, so its presence in an answer would
# prove over-classified evidence reached the model.
CONFIDENTIAL_DOCUMENT = DocumentInput(
    source_id="secret.md",
    title="Confidential release note",
    media_type="text/markdown",
    content=(
        b"The confidential checkout production release rollback plan is codenamed BLUEJAY and gates every deploy."
    ),
    metadata={"acl": {"classification": "confidential"}},
)


def _ingest(runtime: Refaat, project: RefaatProject, document: DocumentInput) -> None:
    runtime.ingest(project.workspace_id, project.corpus_id, document)


def _stored_chunk_ids(runtime: Refaat, project: RefaatProject) -> set[str]:
    return {chunk.id for chunk in runtime.store.list_chunks(project.workspace_id, project.corpus_id)}


def main() -> None:
    with TemporaryDirectory(prefix="refaat-model-routing-") as directory:
        root = Path(directory)
        routed_root = root / "routed"
        unrouted_root = root / "unrouted"
        routed_root.mkdir()
        unrouted_root.mkdir()

        routed = (
            RuntimeBuilder(demo_settings(routed_root))
            .with_model_router(ClassificationCeilingRouter(ceiling=Classification.INTERNAL))
            .build()
        )
        try:
            public_project = routed.project(workspace="acme", corpus="public")
            _ingest(routed, public_project, PUBLIC_DOCUMENT)
            public = public_project.ask(QUERY)
            public_stored = _stored_chunk_ids(routed, public_project)

            confidential_project = routed.project(workspace="acme", corpus="confidential")
            _ingest(routed, confidential_project, CONFIDENTIAL_DOCUMENT)
            confidential = confidential_project.ask(QUERY)
        finally:
            routed.close()

        # The control proves the refusal is router-caused: the same confidential evidence,
        # queried on a runtime with no model router, is answered normally and surfaces the marker.
        control_runtime = RuntimeBuilder(demo_settings(unrouted_root)).build()
        try:
            control_project = control_runtime.project(workspace="acme", corpus="confidential")
            _ingest(control_runtime, control_project, CONFIDENTIAL_DOCUMENT)
            control = control_project.ask(QUERY)
        finally:
            control_runtime.close()

        public_generated = ROUTING_ROUTE not in public.routes and public.answer != ROUTING_REFUSAL_MESSAGE
        public_cites = bool(public.citations)
        public_resolve = bool(public.citations) and all(
            citation.chunk_id in public_stored for citation in public.citations
        )

        confidential_refused = ROUTING_ROUTE in confidential.routes and confidential.answer == ROUTING_REFUSAL_MESSAGE
        refusal_generic = (
            confidential.answer == ROUTING_REFUSAL_MESSAGE and "confidential" not in confidential.answer.casefold()
        )
        control_generates = (
            ROUTING_ROUTE not in control.routes
            and control.answer != ROUTING_REFUSAL_MESSAGE
            and MARKER.casefold() in control.answer.casefold()
        )

        report: dict[str, object] = {
            "mode": "deterministic",
            "ceiling": Classification.INTERNAL.name.casefold(),
            "public_answer_generated": public_generated,
            "public_answer_cites": public_cites,
            "public_citations_resolve": public_resolve,
            "confidential_refused": confidential_refused,
            "confidential_citations_empty": confidential.citations == (),
            "confidential_marker_absent_from_answer": MARKER.casefold() not in confidential.answer.casefold(),
            "refusal_message_is_generic": refusal_generic,
            "control_unrouted_confidential_generates": control_generates,
        }

        assert report["public_answer_generated"], report
        assert report["public_answer_cites"], report
        assert report["public_citations_resolve"], report
        assert report["confidential_refused"], report
        assert report["confidential_citations_empty"], report
        assert report["confidential_marker_absent_from_answer"], report
        assert report["refusal_message_is_generic"], report
        assert report["control_unrouted_confidential_generates"], report

        emit(report)


if __name__ == "__main__":
    main()
