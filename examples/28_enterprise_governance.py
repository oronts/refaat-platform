from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from _shared import demo_settings, emit

from refaat.application.runtime import Refaat, RefaatProject, RuntimeBuilder
from refaat.application.services import ROUTING_REFUSAL_MESSAGE, ROUTING_ROUTE
from refaat.domain.audit import AuditEvent
from refaat.domain.authz import CanonicalAccessPolicy, Classification, Principal
from refaat.domain.models import DocumentInput, QueryResponse
from refaat.domain.routing import ClassificationCeilingRouter

QUERY = "production release rollback plan approval"

# Each marker appears only in one document body, so its presence in an answer proves that
# document's evidence reached the model.
MARKERS = {"alpha": "peridot", "bravo": "cobalt", "secret": "bluejay"}

# Two per-group confidential runbooks in a shared corpus, isolated by ACL. Both sit at or below the
# model-routing ceiling, so an authorized principal gets a generated, cited answer over its own.
SHARED_DOCS = {
    "alpha": DocumentInput(
        source_id="alpha.md",
        title="Team A runbook",
        media_type="text/markdown",
        content=b"The team A production release rollback plan is codenamed peridot and gates every deploy.",
        metadata={"doc": "alpha", "acl": {"classification": "confidential", "allow_groups": ["team-a"]}},
    ),
    "bravo": DocumentInput(
        source_id="bravo.md",
        title="Team B runbook",
        media_type="text/markdown",
        content=b"The team B production release rollback plan is codenamed cobalt and gates every deploy.",
        metadata={"doc": "bravo", "acl": {"classification": "confidential", "allow_groups": ["team-b"]}},
    ),
}

# A restricted runbook alice is authorized to read, but that sits above the routing ceiling, so the
# router refuses to send it to the model even though authorization granted the evidence.
RESTRICTED_DOC = DocumentInput(
    source_id="secret.md",
    title="Restricted runbook",
    media_type="text/markdown",
    content=b"The restricted production release rollback plan is codenamed bluejay and gates every deploy.",
    metadata={"doc": "secret", "acl": {"classification": "restricted", "allow_groups": ["team-a"]}},
)

ALICE = Principal(id="alice", tenant="acme", groups=frozenset({"team-a"}))
BOB = Principal(id="bob", tenant="acme", groups=frozenset({"team-b"}))

SECRETS = (QUERY, *MARKERS.values())


class CapturingSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        self.events.append(event)


def _event_strings(event: AuditEvent) -> list[str]:
    strings: list[str] = []
    for value in event.to_json().values():
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, list):
            strings.extend(str(item) for item in value)
    return strings


def _no_secret(events: list[AuditEvent]) -> bool:
    return all(secret not in text for event in events for text in _event_strings(event) for secret in SECRETS)


def _cited_docs(response: QueryResponse, chunk_marker: dict[str, str]) -> set[str]:
    return {chunk_marker[citation.chunk_id] for citation in response.citations}


def _chunk_markers(runtime: Refaat, project: RefaatProject) -> dict[str, str]:
    return {
        chunk.id: str(chunk.metadata.get("doc"))
        for chunk in runtime.store.list_chunks(project.workspace_id, project.corpus_id)
    }


def _seed(runtime: Refaat, corpus: str, documents: list[DocumentInput]) -> RefaatProject:
    project = runtime.project(workspace="acme", corpus=corpus)
    for document in documents:
        runtime.ingest(project.workspace_id, project.corpus_id, document)
    return project


def main() -> None:
    with TemporaryDirectory(prefix="refaat-enterprise-") as directory:
        root = Path(directory)
        governed_root = root / "governed"
        unrouted_root = root / "unrouted"
        plain_root = root / "plain"
        for path in (governed_root, unrouted_root, plain_root):
            path.mkdir()

        # One runtime carries the whole governance stack: authorization, a classification ceiling on
        # generation, and an audit sink. The seams compose on the same evidence and cache pipeline.
        sink = CapturingSink()
        governed = (
            RuntimeBuilder(demo_settings(governed_root))
            .with_access_policy(CanonicalAccessPolicy())
            .with_model_router(ClassificationCeilingRouter(ceiling=Classification.CONFIDENTIAL))
            .with_audit_sink(sink)
            .build()
        )
        try:
            shared = _seed(governed, "shared", list(SHARED_DOCS.values()))
            shared_marker = _chunk_markers(governed, shared)
            restricted = _seed(governed, "restricted", [RESTRICTED_DOC])

            alice_shared = shared.ask(QUERY, top_k=20, principal=ALICE)
            bob_shared = shared.ask(QUERY, top_k=20, principal=BOB)
            alice_restricted = restricted.ask(QUERY, top_k=20, principal=ALICE)
            bob_restricted = restricted.ask(QUERY, top_k=20, principal=BOB)

            alice_cited = _cited_docs(alice_shared, shared_marker)
            bob_cited = _cited_docs(bob_shared, shared_marker)
            query_events = [event for event in sink.events if event.event_type == "query.authorized"]
        finally:
            governed.close()

        # Control one: no model router. Alice, still authorized for the restricted evidence, now gets a
        # generated answer that surfaces the restricted marker, proving the refusal was router-caused.
        unrouted = RuntimeBuilder(demo_settings(unrouted_root)).with_access_policy(CanonicalAccessPolicy()).build()
        try:
            unrouted_restricted = _seed(unrouted, "restricted", [RESTRICTED_DOC])
            control = unrouted_restricted.ask(QUERY, top_k=20, principal=ALICE)
        finally:
            unrouted.close()

        # Control two: no policy installed. A no-principal query sees both team runbooks, proving the
        # per-group isolation was policy-caused, not an artifact of the corpus.
        plain = RuntimeBuilder(demo_settings(plain_root)).build()
        try:
            plain_shared = _seed(plain, "shared", list(SHARED_DOCS.values()))
            plain_context = plain_shared.retrieve(QUERY, top_k=20)
            plain_docs = {str(hit.chunk.metadata.get("doc")) for hit in plain_context.hits}
        finally:
            plain.close()

        alice_refused = ROUTING_ROUTE in alice_restricted.routes and alice_restricted.answer == ROUTING_REFUSAL_MESSAGE
        control_generates = (
            ROUTING_ROUTE not in control.routes and MARKERS["secret"].casefold() in control.answer.casefold()
        )

        report: dict[str, Any] = {
            "mode": "deterministic",
            "ceiling": Classification.CONFIDENTIAL.name.casefold(),
            # Authorization isolates per-group evidence on the generation path.
            "alice_cites_only_alpha": alice_cited == {"alpha"},
            "bob_cites_only_bravo": bob_cited == {"bravo"},
            "cross_principal_disjoint": bool(alice_cited) and bool(bob_cited) and alice_cited.isdisjoint(bob_cited),
            "alice_answer_excludes_bravo": MARKERS["bravo"] not in alice_shared.answer,
            "bob_answer_excludes_alpha": MARKERS["alpha"] not in bob_shared.answer,
            # Routing composes with authorization: alice MAY read the restricted evidence, and the router
            # still refuses to send it to the model.
            "alice_authorized_but_router_refuses": alice_refused,
            "refused_citations_empty": alice_restricted.citations == (),
            "refused_marker_absent": MARKERS["secret"] not in alice_restricted.answer,
            "refusal_message_is_generic": alice_refused and "restricted" not in alice_restricted.answer.casefold(),
            # Authorization, a distinct layer, denies bob the restricted evidence entirely.
            "bob_denied_restricted_evidence": bob_restricted.citations == ()
            and MARKERS["secret"] not in bob_restricted.answer,
            # Audit fires on every served decision and never carries evidence text.
            "audit_event_per_served_query": len(query_events) == 4,
            "audit_events_redacted": _no_secret(query_events),
            # Controls prove each guard is causal.
            "control_unrouted_surfaces_restricted_marker": control_generates,
            "control_no_policy_sees_both_documents": plain_docs == {"alpha", "bravo"},
        }

        assert all(value for key, value in report.items() if key not in ("mode", "ceiling")), report
        emit(report)


if __name__ == "__main__":
    main()
