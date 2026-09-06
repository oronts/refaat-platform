from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from _shared import demo_settings, emit

from refaat.application.runtime import Refaat, RefaatProject, RuntimeBuilder
from refaat.domain.authz import CanonicalAccessPolicy, Principal
from refaat.domain.models import DocumentInput, QueryResponse

QUERY = "production release rollback plan approval"

DOCUMENTS = {
    "research-a": DocumentInput(
        source_id="alpha.md",
        title="Alpha production runbook",
        media_type="text/markdown",
        content=(
            b"Project Falcon: the alpha production release requires an approved rollback plan. "
            b"The alpha rollback control is codenamed peridot and gates every deploy."
        ),
        metadata={"doc": "alpha", "acl": {"classification": "confidential", "allow_groups": ["research-a"]}},
    ),
    "research-b": DocumentInput(
        source_id="bravo.md",
        title="Bravo production runbook",
        media_type="text/markdown",
        content=(
            b"Project Osprey: the bravo production release requires an approved rollback plan. "
            b"The bravo rollback control is codenamed cobalt and gates every deploy."
        ),
        metadata={"doc": "bravo", "acl": {"classification": "confidential", "allow_groups": ["research-b"]}},
    ),
}

# A token that appears only in one document's body, so it is a body-level tell that
# the other principal's evidence never bled into an answer.
MARKER_TOKEN = {"alpha": "peridot", "bravo": "cobalt"}

ALICE = Principal(id="alice", tenant="merck", groups=frozenset({"research-a"}))
BOB = Principal(id="bob", tenant="merck", groups=frozenset({"research-b"}))


def _seed(runtime: Refaat) -> tuple[RefaatProject, dict[str, str]]:
    project = runtime.project(workspace="merck", workspace_name="Merck", corpus="research")
    for document in DOCUMENTS.values():
        runtime.ingest(project.workspace_id, project.corpus_id, document)
    chunk_marker = {
        chunk.id: str(chunk.metadata.get("doc"))
        for chunk in runtime.store.list_chunks(project.workspace_id, project.corpus_id)
    }
    return project, chunk_marker


def _retrieved_docs(project: RefaatProject, principal: Principal | None) -> set[str]:
    context = project.retrieve(QUERY, top_k=20, principal=principal)
    return {str(hit.chunk.metadata.get("doc")) for hit in context.hits}


def _cited_docs(response: QueryResponse, chunk_marker: dict[str, str]) -> set[str]:
    return {chunk_marker[citation.chunk_id] for citation in response.citations}


def _normalized_citations(response: QueryResponse, chunk_marker: dict[str, str]) -> tuple[tuple[str, str], ...]:
    """Citations keyed by stable doc marker plus verbatim quote, comparable across runtimes."""
    return tuple((chunk_marker[citation.chunk_id], citation.quote) for citation in response.citations)


def main() -> None:
    with TemporaryDirectory(prefix="refaat-authz-") as directory:
        root = Path(directory)
        policy_root = root / "policy"
        plain_root = root / "plain"
        policy_root.mkdir()
        plain_root.mkdir()

        lenient = demo_settings(policy_root).model_copy(update={"strict_authorization": False})
        policy_runtime = RuntimeBuilder(lenient).with_access_policy(CanonicalAccessPolicy()).build()
        try:
            project, chunk_marker = _seed(policy_runtime)

            alice_answer = project.ask(QUERY, top_k=20, principal=ALICE)
            bob_answer = project.ask(QUERY, top_k=20, principal=BOB)
            open_answer = project.ask(QUERY, top_k=20, principal=None)

            alice_retrieved = _retrieved_docs(project, ALICE)
            bob_retrieved = _retrieved_docs(project, BOB)
            open_retrieved = _retrieved_docs(project, None)

            alice_cited = _cited_docs(alice_answer, chunk_marker)
            bob_cited = _cited_docs(bob_answer, chunk_marker)

            stored_ids = set(chunk_marker)
            citations_resolve = all(
                citation.chunk_id in stored_ids
                for response in (alice_answer, bob_answer, open_answer)
                for citation in response.citations
            )
            open_normalized = _normalized_citations(open_answer, chunk_marker)
        finally:
            policy_runtime.close()

        # A second runtime with no policy installed is the true unfiltered baseline; over
        # the same corpus its no-principal answer must match the policy runtime's
        # no-principal answer once ids are normalized to the stable doc marker.
        plain_runtime = RuntimeBuilder(demo_settings(plain_root)).build()
        try:
            plain_project, plain_marker = _seed(plain_runtime)
            plain_answer = plain_project.ask(QUERY, top_k=20, principal=None)
            plain_context = plain_project.retrieve(QUERY, top_k=20)
            plain_retrieved = {str(hit.chunk.metadata.get("doc")) for hit in plain_context.hits}
            plain_normalized = _normalized_citations(plain_answer, plain_marker)
        finally:
            plain_runtime.close()

        report: dict[str, Any] = {
            "mode": "offline",
            "policy": CanonicalAccessPolicy.name,
            "alice_retrieves_only_alpha": alice_retrieved == {"alpha"},
            "bob_retrieves_only_bravo": bob_retrieved == {"bravo"},
            "alice_cites_only_alpha": alice_cited == {"alpha"},
            "bob_cites_only_bravo": bob_cited == {"bravo"},
            "alice_answer_excludes_bravo_body": MARKER_TOKEN["bravo"] not in alice_answer.answer,
            "bob_answer_excludes_alpha_body": MARKER_TOKEN["alpha"] not in bob_answer.answer,
            "cross_principal_disjoint": bool(alice_cited) and bool(bob_cited) and alice_cited.isdisjoint(bob_cited),
            "citations_resolve_to_stored_chunks": citations_resolve,
            "no_principal_sees_both_documents": open_retrieved == {"alpha", "bravo"},
            "no_principal_union_of_principals": open_retrieved == (alice_retrieved | bob_retrieved),
            "no_policy_runtime_sees_both": plain_retrieved == {"alpha", "bravo"},
            "no_principal_matches_no_policy_default": open_normalized == plain_normalized and bool(open_normalized),
        }

        checks = [
            report["alice_retrieves_only_alpha"],
            report["bob_retrieves_only_bravo"],
            report["alice_cites_only_alpha"],
            report["bob_cites_only_bravo"],
            report["alice_answer_excludes_bravo_body"],
            report["bob_answer_excludes_alpha_body"],
            report["cross_principal_disjoint"],
            report["citations_resolve_to_stored_chunks"],
            report["no_principal_sees_both_documents"],
            report["no_principal_union_of_principals"],
            report["no_policy_runtime_sees_both"],
            report["no_principal_matches_no_policy_default"],
        ]
        assert all(checks), report

        emit(report)


if __name__ == "__main__":
    main()
