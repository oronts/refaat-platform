"""Authorization enforcement across retrieval, citations, graph, RLM, and the cache.

The decision core is proven in tests/unit/test_authz.py; here the enforcement points are
actually applied, so an unauthorized chunk must never reach a principal's retrieval,
citations, graph facts, RLM recursion, or a cached answer.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from refaat.application.runtime import RefaatProject, RuntimeBuilder
from refaat.config import Settings
from refaat.core.session import SessionCompactor, tag_turn_metadata
from refaat.domain.authz import CanonicalAccessPolicy, Classification, Principal, auth_scope_hash
from refaat.domain.models import DocumentInput, RlmLimits
from refaat.domain.routing import ClassificationCeilingRouter

pytestmark = pytest.mark.integration

ALICE = Principal(id="alice", tenant="merck", groups=frozenset({"research-a"}))
BOB = Principal(id="bob", tenant="merck", groups=frozenset({"research-b"}))
QUERY = "checkout production release rollback plan"

DOCUMENTS: dict[str, dict[str, object]] = {
    "alpha.md": {"acl": {"classification": "confidential", "allow_groups": ["research-a"]}, "doc": "alpha"},
    "bravo.md": {"acl": {"classification": "confidential", "allow_groups": ["research-b"]}, "doc": "bravo"},
    "public.md": {"acl": {"classification": "public"}, "doc": "public"},
    "locked.md": {"acl": {"classification": "restricted", "allow_groups": ["research-x"]}, "doc": "locked"},
    "broken.md": {"acl": {"classification": "TOP_SECRET"}, "doc": "broken"},
    "untagged.md": {"doc": "untagged"},
}
ALL_DOCS = {str(meta["doc"]) for meta in DOCUMENTS.values()}


def _ingest(project: RefaatProject) -> None:
    for source_id, metadata in DOCUMENTS.items():
        content = f"The {metadata['doc']} checkout production release requires an approved rollback plan."
        project.runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id=source_id,
                title=source_id,
                media_type="text/markdown",
                content=content.encode(),
                metadata=dict(metadata),
            ),
        )


def _visible(project: RefaatProject, principal: Principal | None) -> set[str]:
    context = project.retrieve(QUERY, principal=principal, top_k=20, autocut_enabled=False)
    return {str(hit.chunk.metadata["doc"]) for hit in context.hits}


@pytest.fixture
def authz_project(settings: Settings) -> Iterator[RefaatProject]:
    # This fixture asserts the trusted-in-process behavior (principal-less serving under a policy), so it
    # pins strict_authorization off; the flipped default is exercised by test_default_strict_denies_... below.
    lenient = settings.model_copy(update={"strict_authorization": False})
    runtime = RuntimeBuilder(lenient).with_access_policy(CanonicalAccessPolicy()).build()
    try:
        project = runtime.project(workspace="merck", corpus="research")
        _ingest(project)
        yield project
    finally:
        runtime.close()


def test_alice_sees_only_her_group_and_public(authz_project: RefaatProject) -> None:
    assert _visible(authz_project, ALICE) == {"alpha", "public"}


def test_bob_sees_only_his_group_and_public(authz_project: RefaatProject) -> None:
    assert _visible(authz_project, BOB) == {"bravo", "public"}


def test_principals_never_share_restricted_evidence(authz_project: RefaatProject) -> None:
    alice = _visible(authz_project, ALICE)
    bob = _visible(authz_project, BOB)
    assert "bravo" not in alice
    assert "alpha" not in bob


def test_restricted_untagged_and_malformed_are_denied_to_everyone(authz_project: RefaatProject) -> None:
    for principal in (ALICE, BOB):
        assert {"locked", "untagged", "broken"}.isdisjoint(_visible(authz_project, principal))


def test_no_principal_is_unfiltered_even_with_a_policy_installed(authz_project: RefaatProject) -> None:
    assert _visible(authz_project, None) == ALL_DOCS


def test_a_denied_chunk_is_never_cited(authz_project: RefaatProject) -> None:
    response = authz_project.ask(QUERY, principal=ALICE)
    assert response.citations
    quotes = " ".join(citation.quote for citation in response.citations)
    for forbidden in ("bravo", "locked", "untagged", "broken"):
        assert forbidden not in quotes


def test_cache_is_isolated_per_principal(authz_project: RefaatProject) -> None:
    alice_first = authz_project.ask(QUERY, principal=ALICE)
    assert alice_first.cached is False
    bob = authz_project.ask(QUERY, principal=BOB)
    assert bob.cached is False
    assert "alpha" not in " ".join(c.quote for c in bob.citations)
    assert authz_project.ask(QUERY, principal=ALICE).cached is True


GRAPH_DOCUMENTS: dict[str, dict[str, object]] = {
    "alpha.md": {
        "acl": {"classification": "confidential", "allow_groups": ["research-a"]},
        "doc": "alpha",
        "text": "Alphacompound Alphacompound targets Alphatarget Alphatarget in the alpha research program.",
    },
    "bravo.md": {
        "acl": {"classification": "confidential", "allow_groups": ["research-b"]},
        "doc": "bravo",
        "text": "Bravocompound Bravocompound targets Bravotarget Bravotarget in the bravo research program.",
    },
}


@pytest.fixture
def graph_project(settings: Settings) -> Iterator[RefaatProject]:
    lenient = settings.model_copy(update={"strict_authorization": False})
    runtime = RuntimeBuilder(lenient).with_access_policy(CanonicalAccessPolicy()).build()
    try:
        project = runtime.project(workspace="merck", corpus="graph")
        for source_id, meta in GRAPH_DOCUMENTS.items():
            project.runtime.ingest(
                project.workspace_id,
                project.corpus_id,
                DocumentInput(
                    source_id=source_id,
                    title=source_id,
                    media_type="text/markdown",
                    content=str(meta["text"]).encode(),
                    metadata={"acl": meta["acl"], "doc": meta["doc"]},
                ),
            )
        yield project
    finally:
        runtime.close()


def _facts(project: RefaatProject, principal: Principal | None, query: str) -> str:
    return " ".join(project.retrieve(query, principal=principal, top_k=20, autocut_enabled=False).graph_facts)


def test_graph_fact_only_visible_to_authorized_principal(graph_project: RefaatProject) -> None:
    alice_facts = _facts(graph_project, ALICE, "Alphacompound Bravocompound targets")
    bob_facts = _facts(graph_project, BOB, "Alphacompound Bravocompound targets")
    assert "Bravo" not in alice_facts
    assert "Alpha" not in bob_facts


def test_graph_is_unfiltered_without_a_principal(graph_project: RefaatProject) -> None:
    facts = _facts(graph_project, None, "Alphacompound Bravocompound targets")
    assert "Alpha" in facts
    assert "Bravo" in facts


def test_graph_evidence_classification_is_counted_when_a_router_is_installed(settings: Settings) -> None:
    # A model router (or residency gate) runs independently of any access policy, and a graph fact
    # reaches the model outside context.hits, so with a router installed the retriever classifies the
    # fact's backing document even with no principal and no access policy. This closes the
    # router-without-enforcement bypass: a confidential fact raises the ceiling regardless.
    runtime = (
        RuntimeBuilder(settings).with_model_router(ClassificationCeilingRouter(ceiling=Classification.PUBLIC)).build()
    )
    try:
        project = runtime.project(workspace="merck", corpus="graph-routed")
        for source_id, meta in GRAPH_DOCUMENTS.items():
            project.runtime.ingest(
                project.workspace_id,
                project.corpus_id,
                DocumentInput(
                    source_id=source_id,
                    title=source_id,
                    media_type="text/markdown",
                    content=str(meta["text"]).encode(),
                    metadata={"acl": meta["acl"], "doc": meta["doc"]},
                ),
            )
        context = project.retrieve(
            "Alphacompound Bravocompound targets", principal=None, top_k=20, autocut_enabled=False
        )
        assert context.graph_facts
        assert context.graph_evidence_classification is Classification.CONFIDENTIAL
    finally:
        runtime.close()


def test_graph_evidence_classification_stays_default_without_a_router(graph_project: RefaatProject) -> None:
    # With no router or residency gate to consume it, the retriever skips classification and the packet
    # keeps the open default, so the ungoverned path does no extra work and does not change behavior.
    context = graph_project.retrieve("Alphacompound Alphatarget", principal=ALICE, top_k=20, autocut_enabled=False)
    assert context.graph_facts
    assert context.graph_evidence_classification is Classification.PUBLIC


def test_rlm_recursion_only_sees_authorized_evidence(authz_project: RefaatProject) -> None:
    limits = RlmLimits(max_depth=2, max_nodes=8, branch_factor=3, leaf_token_budget=400)
    alice = authz_project.rlm(QUERY, principal=ALICE, limits=limits)
    quotes = " ".join(citation.quote for citation in alice.citations)
    for forbidden in ("bravo", "locked", "untagged", "broken"):
        assert forbidden not in quotes
    bob = authz_project.rlm(QUERY, principal=BOB, limits=limits)
    assert "alpha" not in " ".join(citation.quote for citation in bob.citations)


def _reingest_alpha_acl(project: RefaatProject, allow_group: str) -> None:
    content = "The alpha checkout production release requires an approved rollback plan."
    project.runtime.ingest(
        project.workspace_id,
        project.corpus_id,
        DocumentInput(
            source_id="alpha.md",
            title="alpha.md",
            media_type="text/markdown",
            content=content.encode(),
            metadata={"acl": {"classification": "confidential", "allow_groups": [allow_group]}, "doc": "alpha"},
        ),
    )


def test_acl_only_change_takes_effect_on_fresh_query(authz_project: RefaatProject) -> None:
    assert "alpha" in _visible(authz_project, ALICE)
    _reingest_alpha_acl(authz_project, "research-b")
    assert "alpha" not in _visible(authz_project, ALICE)
    assert "alpha" in _visible(authz_project, BOB)


def _document_id(project: RefaatProject, source_id: str) -> str:
    doc = project.runtime.store.get_document_by_source(project.workspace_id, project.corpus_id, source_id)
    return str(doc["id"])


def test_summary_provenance_recorded_under_policy_without_router(authz_project: RefaatProject) -> None:
    session_id = authz_project.create_session()
    _drive_compaction(authz_project, ALICE, session_id)
    session = authz_project.runtime.store.get_session(authz_project.workspace_id, session_id)
    cited = (session.get("metadata") or {}).get("summary_cited_documents")
    assert isinstance(cited, list)
    assert _document_id(authz_project, "alpha.md") in cited


def test_summary_suppressed_when_a_cited_document_acl_tightens(authz_project: RefaatProject) -> None:
    session_id = authz_project.create_session()
    _drive_compaction(authz_project, ALICE, session_id)
    summary, routes = _summary(authz_project, ALICE, session_id)
    assert summary is not None
    assert "session" in routes
    _reingest_alpha_acl(authz_project, "research-b")
    after_summary, after_routes = _summary(authz_project, ALICE, session_id)
    assert after_summary is None
    assert "session" not in after_routes


def test_summary_still_surfaces_when_acls_are_unchanged(authz_project: RefaatProject) -> None:
    session_id = authz_project.create_session()
    _drive_compaction(authz_project, ALICE, session_id)
    summary, routes = _summary(authz_project, ALICE, session_id)
    assert summary is not None
    assert "session" in routes


def test_rlm_summary_suppressed_when_a_cited_document_acl_tightens(authz_project: RefaatProject) -> None:
    session_id = authz_project.create_session()
    _drive_compaction(authz_project, ALICE, session_id)
    ws, corpus = authz_project.workspace_id, authz_project.corpus_id
    assert authz_project.runtime.rlm._load_session_summary(ws, corpus, session_id, ALICE) is not None
    _reingest_alpha_acl(authz_project, "research-b")
    assert authz_project.runtime.rlm._load_session_summary(ws, corpus, session_id, ALICE) is None


def test_no_policy_compaction_records_no_provenance(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings).build()
    try:
        project = runtime.project(workspace="merck", corpus="research")
        _ingest(project)
        session_id = project.create_session(user_id="alice")
        for _ in range(3):
            project.ask(QUERY, session_id=session_id)
        session = runtime.store.get_session(project.workspace_id, session_id)
        assert session.get("summary")
        assert "summary_cited_documents" not in (session.get("metadata") or {})
        context = project.retrieve(QUERY, session_id=session_id, top_k=20)
        assert "session" in context.routes
    finally:
        runtime.close()


def test_acl_change_invalidates_a_previously_cached_answer(authz_project: RefaatProject) -> None:
    first = authz_project.ask(QUERY, principal=ALICE)
    assert first.cached is False
    assert "alpha" in " ".join(c.quote for c in first.citations)
    _reingest_alpha_acl(authz_project, "research-b")
    after = authz_project.ask(QUERY, principal=ALICE)
    assert "alpha" not in " ".join(c.quote for c in after.citations)


# A compacted session summary is prior evidence that must stay bound to the principal who wrote it.
# session_compaction_turns=4 and retained_session_turns=2 in the test settings, so three asks (six
# turns) force one compaction and a stored writer scope.


def _drive_compaction(project: RefaatProject, principal: Principal, session_id: str) -> None:
    for _ in range(3):
        project.ask(QUERY, principal=principal, session_id=session_id)


def _summary(project: RefaatProject, principal: Principal, session_id: str) -> tuple[str | None, tuple[str, ...]]:
    context = project.retrieve(QUERY, principal=principal, session_id=session_id, top_k=20, autocut_enabled=False)
    return context.session_summary, context.routes


def test_session_summary_is_bound_to_its_writer_principal(authz_project: RefaatProject) -> None:
    session_id = authz_project.create_session()
    _drive_compaction(authz_project, ALICE, session_id)
    alice_summary, alice_routes = _summary(authz_project, ALICE, session_id)
    assert alice_summary is not None
    assert "alpha" in alice_summary
    assert "session" in alice_routes
    bob_summary, bob_routes = _summary(authz_project, BOB, session_id)
    assert bob_summary is None
    assert "session" not in bob_routes


def test_shared_session_compaction_never_merges_cross_principal(authz_project: RefaatProject) -> None:
    session_id = authz_project.create_session()
    _drive_compaction(authz_project, ALICE, session_id)
    _drive_compaction(authz_project, BOB, session_id)
    bob_summary, _ = _summary(authz_project, BOB, session_id)
    if bob_summary is not None:
        assert "alpha" not in bob_summary
    alice_summary, _ = _summary(authz_project, ALICE, session_id)
    if alice_summary is not None:
        assert "bravo" not in alice_summary


def test_rlm_session_summary_is_bound_to_its_writer_principal(authz_project: RefaatProject) -> None:
    session_id = authz_project.create_session()
    _drive_compaction(authz_project, ALICE, session_id)
    service = authz_project.runtime.rlm  # type: ignore[attr-defined]
    assert service._load_session_summary(authz_project.workspace_id, authz_project.corpus_id, session_id, ALICE)
    assert service._load_session_summary(authz_project.workspace_id, authz_project.corpus_id, session_id, BOB) is None


def test_scope_blind_recompaction_refuses_a_scoped_session(authz_project: RefaatProject, settings: Settings) -> None:
    session_id = authz_project.create_session()
    _drive_compaction(authz_project, ALICE, session_id)
    store = authz_project.runtime.store  # type: ignore[attr-defined]
    scope_b = auth_scope_hash(BOB, CanonicalAccessPolicy())
    for index in range(6):
        store.append_session_turn(
            authz_project.workspace_id,
            session_id,
            "user",
            f"bob secret delta {index}",
            tag_turn_metadata({}, scope_b),
        )
    compactor = SessionCompactor(store, settings)
    assert compactor.compact_if_needed(authz_project.workspace_id, session_id) is False
    session = store.get_session(authz_project.workspace_id, session_id)
    assert "bob secret delta" not in str(session.get("summary") or "")


def _document_id_for(project: RefaatProject, doc: str) -> str:
    store = project.runtime.store  # type: ignore[attr-defined]
    for document in store.list_documents(project.workspace_id, project.corpus_id):
        metadata = document.get("metadata") or {}
        if str(metadata.get("doc")) == doc:
            return str(document["id"])
    raise AssertionError(f"document {doc!r} not found")


def test_summary_cited_documents_are_stored_at_compaction_under_a_router(settings: Settings) -> None:
    # With a router installed the compaction records the documents the summary folded, so a read-time
    # ceiling can recompute their classification from current ACLs.
    runtime = (
        RuntimeBuilder(settings)
        .with_access_policy(CanonicalAccessPolicy())
        .with_model_router(ClassificationCeilingRouter(ceiling=Classification.RESTRICTED))
        .build()
    )
    try:
        project = runtime.project(workspace="merck", corpus="summary-store")
        _ingest(project)
        session_id = project.create_session()
        _drive_compaction(project, ALICE, session_id)
        store = runtime.store  # type: ignore[attr-defined]
        cited = (store.get_session(project.workspace_id, session_id).get("metadata") or {}).get(
            "summary_cited_documents"
        )
        assert isinstance(cited, list)
        assert _document_id_for(project, "alpha") in cited
    finally:
        runtime.close()


def test_summary_classification_is_recomputed_onto_the_packet_under_a_router(settings: Settings) -> None:
    # The surfaced summary's classification is recomputed from its cited documents at read time, so a
    # summary over confidential evidence raises the ceiling even when the current hits are lower.
    runtime = (
        RuntimeBuilder(settings)
        .with_access_policy(CanonicalAccessPolicy())
        .with_model_router(ClassificationCeilingRouter(ceiling=Classification.RESTRICTED))
        .build()
    )
    try:
        project = runtime.project(workspace="merck", corpus="summary-routed")
        _ingest(project)
        session_id = project.create_session()
        _drive_compaction(project, ALICE, session_id)
        context = project.retrieve(QUERY, principal=ALICE, session_id=session_id, top_k=20, autocut_enabled=False)
        assert context.session_summary is not None
        assert context.session_summary_classification is Classification.CONFIDENTIAL
    finally:
        runtime.close()


def test_summary_classification_is_counted_without_an_access_policy(settings: Settings) -> None:
    # A router runs independently of any access policy, so the summary evidence is classified even with
    # no policy and no principal: the derivation is not gated on per-principal enforcement. This closes
    # the router-without-enforcement bypass for the summary path.
    runtime = (
        RuntimeBuilder(settings)
        .with_model_router(ClassificationCeilingRouter(ceiling=Classification.RESTRICTED))
        .build()
    )
    try:
        project = runtime.project(workspace="merck", corpus="summary-noacl")
        project.runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id="secret.md",
                title="secret",
                media_type="text/markdown",
                content=b"The secret checkout production release requires an approved rollback plan.",
                metadata={"acl": {"classification": "confidential"}, "doc": "secret"},
            ),
        )
        session_id = project.create_session()
        for _ in range(3):
            project.ask(QUERY, session_id=session_id)
        context = project.retrieve(QUERY, session_id=session_id, top_k=20, autocut_enabled=False)
        assert context.session_summary is not None
        assert context.session_summary_classification is Classification.CONFIDENTIAL
    finally:
        runtime.close()


def test_summary_classification_reflects_a_reclassification_after_compaction(settings: Settings) -> None:
    # The classification is recomputed from current ACLs at read time, not frozen at compaction, so
    # re-classifying a cited document UP after the summary is written raises the read-time ceiling.
    runtime = (
        RuntimeBuilder(settings)
        .with_access_policy(CanonicalAccessPolicy())
        .with_model_router(ClassificationCeilingRouter(ceiling=Classification.RESTRICTED))
        .build()
    )
    try:
        project = runtime.project(workspace="merck", corpus="summary-retag")

        def _ingest_target(classification: str) -> None:
            project.runtime.ingest(
                project.workspace_id,
                project.corpus_id,
                DocumentInput(
                    source_id="target.md",
                    title="target",
                    media_type="text/markdown",
                    content=b"The target checkout production release requires an approved rollback plan.",
                    metadata={
                        "acl": {"classification": classification, "allow_groups": ["research-a"]},
                        "doc": "target",
                    },
                ),
            )

        _ingest_target("internal")
        session_id = project.create_session()
        _drive_compaction(project, ALICE, session_id)
        _ingest_target("restricted")
        context = project.retrieve(QUERY, principal=ALICE, session_id=session_id, top_k=20, autocut_enabled=False)
        assert context.session_summary is not None
        assert context.session_summary_classification is Classification.RESTRICTED
    finally:
        runtime.close()


def test_summary_classification_stays_default_without_a_router(authz_project: RefaatProject) -> None:
    # With no router to consume it, the retriever does not classify the summary and the packet keeps the
    # open default, so the ungoverned path is unchanged and compaction stores no cited documents.
    session_id = authz_project.create_session()
    _drive_compaction(authz_project, ALICE, session_id)
    context = authz_project.retrieve(QUERY, principal=ALICE, session_id=session_id, top_k=20, autocut_enabled=False)
    assert context.session_summary is not None
    assert context.session_summary_classification is Classification.PUBLIC


def test_rlm_egress_gate_classifies_the_summary_and_fails_closed_on_a_deleted_document(settings: Settings) -> None:
    # The RLM egress gate classifies the session summary, not only the corpus chunks. A summary that
    # cited a since-deleted document is unresolved provenance, so it fails closed and the RLM refuses
    # rather than sending the summary's retained content to an under-cleared model.
    from refaat.application.services import ROUTING_REFUSAL_MESSAGE

    runtime = (
        RuntimeBuilder(settings)
        .with_access_policy(CanonicalAccessPolicy())
        .with_model_router(ClassificationCeilingRouter(ceiling=Classification.INTERNAL))
        .build()
    )
    try:
        project = runtime.project(workspace="merck", corpus="rlm-summary")

        def _ingest_doc(source_id: str, doc: str, classification: str) -> None:
            project.runtime.ingest(
                project.workspace_id,
                project.corpus_id,
                DocumentInput(
                    source_id=source_id,
                    title=doc,
                    media_type="text/markdown",
                    content=f"The {doc} checkout production release requires an approved rollback plan.".encode(),
                    metadata={"acl": {"classification": classification, "allow_groups": ["research-a"]}, "doc": doc},
                ),
            )

        _ingest_doc("target.md", "target", "internal")
        _ingest_doc("public.md", "public", "public")
        session_id = project.create_session()
        _drive_compaction(project, ALICE, session_id)
        project.delete_document(_document_id_for(project, "target"))
        result = project.rlm(QUERY, principal=ALICE, session_id=session_id)
        assert result.answer == ROUTING_REFUSAL_MESSAGE
    finally:
        runtime.close()


def test_strict_authorization_denies_an_in_process_call_without_a_principal(settings: Settings) -> None:
    # With strict authorization on, an in-process call under an active policy that omits the principal is
    # denied rather than served unfiltered, on every in-process surface: query, retrieve, RLM, the
    # document listing, and a program run (denied before its own embedding or model preprocessing). A
    # supplied principal is served normally. On by default; set strict off for the trusted-in-process path.
    from refaat.domain.models import Feedback
    from refaat.errors import AuthorizationError

    runtime = (
        RuntimeBuilder(settings.model_copy(update={"strict_authorization": True}))
        .with_access_policy(CanonicalAccessPolicy())
        .build()
    )
    try:
        project = runtime.project(workspace="merck", corpus="strict")
        _ingest(project)
        with pytest.raises(AuthorizationError):
            project.ask(QUERY)
        with pytest.raises(AuthorizationError):
            project.retrieve(QUERY)
        with pytest.raises(AuthorizationError):
            project.rlm(QUERY)
        with pytest.raises(AuthorizationError):
            project.list_documents()
        with pytest.raises(AuthorizationError):
            project.stats()
        with pytest.raises(AuthorizationError):
            project.submit_feedback(rating=1, user_id="u", selected_chunk_ids=())
        with pytest.raises(AuthorizationError):
            runtime.submit_feedback(
                project.workspace_id,
                project.corpus_id,
                Feedback(query_id=None, rating=1, user_id="u", selected_chunk_ids=()),
            )
        bound = project.programs.get("grounded-rag")
        with pytest.raises(AuthorizationError):
            bound.run(question=QUERY)
        assert project.ask(QUERY, principal=ALICE).citations
        assert bound.run(question=QUERY, principal=ALICE).grounded
    finally:
        runtime.close()


def test_default_strict_denies_a_principal_less_call(settings: Settings) -> None:
    # The default itself is now strict: with NO strict override, an in-process call under an active policy
    # that omits the principal is denied. This proves the flipped default, unlike the test above which sets
    # strict=True explicitly. It must not reuse authz_project (pinned strict=False for the trusted path).
    from refaat.errors import AuthorizationError

    runtime = RuntimeBuilder(settings).with_access_policy(CanonicalAccessPolicy()).build()
    try:
        project = runtime.project(workspace="merck", corpus="strict-default")
        _ingest(project)
        with pytest.raises(AuthorizationError):
            project.ask(QUERY)
        with pytest.raises(AuthorizationError):
            project.retrieve(QUERY)
        with pytest.raises(AuthorizationError):
            project.rlm(QUERY)
        assert project.ask(QUERY, principal=ALICE).citations
    finally:
        runtime.close()


def test_strict_authorization_is_a_no_op_without_an_access_policy(settings: Settings) -> None:
    # No access policy means nothing to enforce, so a principal-less call still runs under strict mode.
    runtime = RuntimeBuilder(settings.model_copy(update={"strict_authorization": True})).build()
    try:
        project = runtime.project(workspace="merck", corpus="strict-nopolicy")
        _ingest(project)
        assert project.ask(QUERY).answer
    finally:
        runtime.close()
