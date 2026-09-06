"""Regression suite for authorization bypasses found by the independent adversarial review.

Each test encodes a confirmed leak so it can never regress: HyPE and RAPTOR retrieval routes
must obey the ACL, the tenant boundary must hold even under a tenant-blind custom policy, and a
session summary must not cross a principal boundary.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from refaat.application.runtime import RefaatProject, RuntimeBuilder
from refaat.config import Settings
from refaat.domain.authz import (
    AccessDecision,
    AccessTag,
    CanonicalAccessPolicy,
    Principal,
    chunk_allowed,
    preference_scope,
)
from refaat.domain.models import DocumentInput, ExtractedDocument

pytestmark = pytest.mark.integration

ALICE = Principal(id="alice", tenant="merck", groups=frozenset({"research-a"}))
BOB = Principal(id="bob", tenant="merck", groups=frozenset({"research-b"}))
QUERY = "checkout production release rollback plan"

DOCS = {
    "alpha.md": ({"acl": {"classification": "confidential", "allow_groups": ["research-a"]}}, "ALPHA-ALLOWED"),
    "bravo.md": ({"acl": {"classification": "confidential", "allow_groups": ["research-b"]}}, "SECRET-BRAVO-ZEPHYR"),
}


def _ingest(project: RefaatProject) -> None:
    for source_id, (metadata, marker) in DOCS.items():
        content = f"The {marker} checkout production release requires an approved rollback plan."
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


@pytest.fixture
def hype_raptor_project(settings: Settings) -> Iterator[RefaatProject]:
    tuned = settings.model_copy(update={"hype_enabled": True, "raptor_enabled": True})
    runtime = RuntimeBuilder(tuned).with_access_policy(CanonicalAccessPolicy()).build()
    try:
        project = runtime.project(workspace="merck", corpus="research")
        _ingest(project)
        yield project
    finally:
        runtime.close()


def test_hype_and_raptor_routes_obey_the_acl(hype_raptor_project: RefaatProject) -> None:
    context = hype_raptor_project.retrieve(QUERY, principal=ALICE, top_k=20, autocut_enabled=False)
    hit_text = " ".join(hit.chunk.text for hit in context.hits)
    assert "SECRET-BRAVO-ZEPHYR" not in hit_text
    answer = hype_raptor_project.ask(QUERY, principal=ALICE).answer
    assert "SECRET-BRAVO-ZEPHYR" not in answer
    quotes = " ".join(c.quote for c in hype_raptor_project.ask(QUERY, principal=ALICE).citations)
    assert "SECRET-BRAVO-ZEPHYR" not in quotes


class _TenantBlindPolicy:
    """A custom policy that decides only on groups and never inspects the tenant."""

    name = "tenant-blind"
    version = "1"

    def decide(self, principal: Principal, tag: AccessTag) -> AccessDecision:
        allowed = bool(principal.groups & tag.allow_groups) if tag.allow_groups else True
        return AccessDecision(allowed=allowed, reason="group_only")


@pytest.fixture
def tenant_project(settings: Settings) -> Iterator[RefaatProject]:
    runtime = RuntimeBuilder(settings).with_access_policy(_TenantBlindPolicy()).build()
    try:
        project = runtime.project(workspace="corp", corpus="research")
        project.runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id="acme.md",
                title="acme.md",
                media_type="text/markdown",
                content=b"The ACME-TENANT-SECRET checkout production release requires a rollback plan.",
                metadata={"acl": {"classification": "confidential", "tenant": "acme", "allow_groups": ["research"]}},
            ),
        )
        yield project
    finally:
        runtime.close()


def test_tenant_boundary_holds_under_a_tenant_blind_custom_policy(tenant_project: RefaatProject) -> None:
    mallory = Principal(id="mallory", tenant="globex", groups=frozenset({"research"}))
    context = tenant_project.retrieve("checkout production release rollback plan", principal=mallory, top_k=20)
    assert all("ACME-TENANT-SECRET" not in hit.chunk.text for hit in context.hits)
    resident = Principal(id="dana", tenant="acme", groups=frozenset({"research"}))
    resident_hits = tenant_project.retrieve(
        "checkout production release rollback plan", principal=resident, top_k=20
    ).hits
    assert any("ACME-TENANT-SECRET" in hit.chunk.text for hit in resident_hits)


@pytest.fixture
def session_project(settings: Settings) -> Iterator[RefaatProject]:
    runtime = RuntimeBuilder(settings).with_access_policy(CanonicalAccessPolicy()).build()
    try:
        project = runtime.project(workspace="merck", corpus="research")
        _ingest(project)
        yield project
    finally:
        runtime.close()


def _write_alice_session_summary(project: RefaatProject) -> str:
    """Drive enough turns to force a compaction so a principal-bound summary exists."""
    session_id = project.create_session(user_id="alice")
    for _ in range(3):
        project.ask(QUERY, principal=ALICE, session_id=session_id)
    session = project.runtime.store.get_session(project.workspace_id, session_id)
    assert session.get("summary"), "expected the session to have compacted a summary"
    return session_id


def test_same_principal_reads_its_own_session_summary_under_a_policy(session_project: RefaatProject) -> None:
    session_id = _write_alice_session_summary(session_project)
    context = session_project.retrieve(QUERY, principal=ALICE, session_id=session_id, top_k=20)
    assert "session" in context.routes
    assert context.session_summary


def test_session_summary_does_not_cross_a_principal_boundary(session_project: RefaatProject) -> None:
    session_id = _write_alice_session_summary(session_project)
    context = session_project.retrieve(QUERY, principal=BOB, session_id=session_id, top_k=20)
    assert "session" not in context.routes
    assert "SECRET-BRAVO-ZEPHYR" not in (context.session_summary or "")
    assert "ALPHA-ALLOWED" not in (context.session_summary or "")


@pytest.fixture
def no_policy_project(settings: Settings) -> Iterator[RefaatProject]:
    runtime = RuntimeBuilder(settings).build()
    try:
        project = runtime.project(workspace="merck", corpus="research")
        _ingest(project)
        yield project
    finally:
        runtime.close()


def test_session_summary_unchanged_without_a_policy(no_policy_project: RefaatProject) -> None:
    session_id = no_policy_project.create_session(user_id="alice")
    for _ in range(3):
        no_policy_project.ask(QUERY, session_id=session_id)
    session = no_policy_project.runtime.store.get_session(no_policy_project.workspace_id, session_id)
    assert session.get("summary")
    assert "summary_auth_scope" not in (session.get("metadata") or {})
    context = no_policy_project.retrieve(QUERY, session_id=session_id, top_k=20)
    assert "session" in context.routes
    assert context.session_summary


class _AclForgingExtractor:
    """A plugin extractor that tries to overwrite the caller's governance metadata."""

    media_types = frozenset({"application/x-evil"})

    def extract(self, content: bytes, *, metadata: dict) -> ExtractedDocument:
        return ExtractedDocument(
            text=content.decode(),
            metadata={
                "acl": {"classification": "public", "allow_groups": ["everyone"]},
                "tenant": "attacker",
                "classification": "public",
                "allow_groups": ["everyone"],
                "deny_users": ["alice"],
                "page": 1,
            },
        )


def test_plugin_extractor_cannot_override_the_callers_acl(settings: Settings) -> None:
    runtime = (
        RuntimeBuilder(settings)
        .with_extractor(_AclForgingExtractor())
        .with_access_policy(CanonicalAccessPolicy())
        .build()
    )
    try:
        project = runtime.project(workspace="merck", corpus="research")
        project.runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id="report",
                title="report",
                media_type="application/x-evil",
                content=b"The confidential rollback plan requires research-a approval.",
                metadata={"acl": {"classification": "confidential", "allow_groups": ["research-a"]}},
            ),
        )
        chunks = runtime.store.list_chunks(project.workspace_id, project.corpus_id)
        assert chunks
        for chunk in chunks:
            # The caller's ACL is authoritative; the extractor's forged governance keys are gone.
            assert chunk.metadata["acl"] == {"classification": "confidential", "allow_groups": ["research-a"]}
            assert "tenant" not in chunk.metadata
            assert "allow_groups" not in chunk.metadata
            assert "deny_users" not in chunk.metadata
            # A non-governance key the extractor added still flows through.
            assert chunk.metadata["page"] == 1
            # The caller's group is admitted; the extractor's public/everyone tag never took effect.
            assert chunk_allowed(CanonicalAccessPolicy(), ALICE, chunk.metadata)
            assert not chunk_allowed(CanonicalAccessPolicy(), BOB, chunk.metadata)
    finally:
        runtime.close()


class _MetadataMutatingExtractor:
    """A plugin extractor that tries to inject a governance key by mutating the caller's metadata in place."""

    media_types = frozenset({"application/x-mutate"})

    def extract(self, content: bytes, *, metadata: dict) -> ExtractedDocument:
        metadata["acl"] = {"classification": "public", "allow_groups": ["everyone"]}
        metadata["tenant"] = "attacker"
        return ExtractedDocument(text=content.decode(), metadata={"page": 1})


def test_plugin_extractor_cannot_inject_acl_by_mutating_caller_metadata(settings: Settings) -> None:
    runtime = (
        RuntimeBuilder(settings)
        .with_extractor(_MetadataMutatingExtractor())
        .with_access_policy(CanonicalAccessPolicy())
        .build()
    )
    try:
        project = runtime.project(workspace="merck", corpus="research")
        project.runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id="report",
                title="report",
                media_type="application/x-mutate",
                content=b"The confidential rollback plan requires research-a approval.",
                metadata={"acl": {"classification": "confidential", "allow_groups": ["research-a"]}},
            ),
        )
        for chunk in runtime.store.list_chunks(project.workspace_id, project.corpus_id):
            # An in-place mutation of the caller's metadata cannot reach the stored ACL: the caller's is kept.
            assert chunk.metadata["acl"] == {"classification": "confidential", "allow_groups": ["research-a"]}
            assert "tenant" not in chunk.metadata
            assert chunk_allowed(CanonicalAccessPolicy(), ALICE, chunk.metadata)
            assert not chunk_allowed(CanonicalAccessPolicy(), BOB, chunk.metadata)
    finally:
        runtime.close()


def test_preference_scope_binds_to_the_principal_not_a_spoofed_user_id() -> None:
    from refaat.application.services import QueryService
    from refaat.domain.models import QueryRequest

    spoofed = QueryRequest(
        workspace_id="ws",
        corpus_id="c",
        query="q",
        user_id="victim",
        principal=ALICE,
    )
    # Under an active principal the scope is bound to the principal id, so a client-supplied
    # user_id cannot select another user's learned profile or cache.
    assert QueryService._scope(spoofed) == "user:alice"
    # No principal keeps the legacy user-id scope, byte-identical.
    legacy = QueryRequest(workspace_id="ws", corpus_id="c", query="q", user_id="victim")
    assert QueryService._scope(legacy) == "user:victim"


@pytest.fixture
def learning_project(settings: Settings) -> Iterator[RefaatProject]:
    runtime = RuntimeBuilder(settings).with_access_policy(CanonicalAccessPolicy()).build()
    try:
        project = runtime.project(workspace="merck", corpus="research")
        _ingest(project)
        yield project
    finally:
        runtime.close()


def _chunk_id_for(project: RefaatProject, principal: Principal, marker: str) -> str:
    context = project.retrieve(QUERY, principal=principal, top_k=20, autocut_enabled=False)
    for hit in context.hits:
        if marker in hit.chunk.text:
            return hit.chunk.id
    raise AssertionError(f"{marker} not retrievable by its owner")


def test_preference_read_does_not_leak_across_the_principal_boundary(learning_project: RefaatProject) -> None:
    project = learning_project
    alpha_id = _chunk_id_for(project, ALICE, "ALPHA-ALLOWED")
    project.submit_feedback(rating=1, user_id="alice", selected_chunk_ids=(alpha_id,), principal=ALICE)

    # Bob spoofs alice's user_id. He loads his own empty profile, so the 'profile' route never
    # fires and alice's learned document ids never surface across the boundary.
    spoofed = project.retrieve(QUERY, principal=BOB, user_id="alice", top_k=20, autocut_enabled=False)
    assert "profile" not in spoofed.routes
    assert not spoofed.profile

    # Alice, correctly identified, still reads her own learned profile: same-principal memory holds.
    own = project.retrieve(QUERY, principal=ALICE, user_id="alice", top_k=20, autocut_enabled=False)
    assert "profile" in own.routes


def test_preference_write_binds_to_the_principal_not_a_spoofed_user_id(learning_project: RefaatProject) -> None:
    project = learning_project
    bravo_id = _chunk_id_for(project, BOB, "SECRET-BRAVO-ZEPHYR")
    # Bob writes feedback while claiming to be alice. The write must land under bob's scope only.
    project.submit_feedback(rating=1, user_id="alice", selected_chunk_ids=(bravo_id,), principal=BOB)
    store = project.runtime.store
    policy = CanonicalAccessPolicy()
    assert store.get_preferences(project.workspace_id, project.corpus_id, preference_scope(policy, BOB, "alice"))
    assert not store.get_preferences(project.workspace_id, project.corpus_id, preference_scope(policy, ALICE, "alice"))


def test_feedback_on_a_forbidden_chunk_is_not_an_existence_oracle(learning_project: RefaatProject) -> None:
    project = learning_project
    bravo_id = _chunk_id_for(project, BOB, "SECRET-BRAVO-ZEPHYR")
    # Alice cannot see bravo. Feedback on its id must record nothing, so a forbidden chunk and a
    # nonexistent one are indistinguishable and the 'profile' route cannot reveal its existence.
    project.submit_feedback(rating=1, user_id="alice", selected_chunk_ids=(bravo_id,), principal=ALICE)
    store = project.runtime.store
    scope = preference_scope(CanonicalAccessPolicy(), ALICE, "alice")
    assert not store.get_preferences(project.workspace_id, project.corpus_id, scope)
    context = project.retrieve(QUERY, principal=ALICE, top_k=20, autocut_enabled=False)
    assert "profile" not in context.routes


def test_preference_scope_isolates_a_shared_id_across_tenants() -> None:
    policy = CanonicalAccessPolicy()
    atlas = Principal(id="shared-user", tenant="atlas", groups=frozenset({"finance"}))
    boreal = Principal(id="shared-user", tenant="boreal", groups=frozenset({"engineering"}))
    assert preference_scope(policy, atlas, "shared-user") != preference_scope(policy, boreal, "shared-user")
    # No policy keeps the legacy user-id scope, byte-identical.
    assert preference_scope(None, None, "shared-user") == "user:shared-user"


def test_hierarchical_expansion_reauthorizes_fetched_parents(settings: Settings) -> None:
    # A child chunk's stored parent_id is caller-influenced. A forbidden parent must not be pulled
    # in by parent_child or sentence_window expansion, which fetch and substitute the parent text
    # after the retrieval PEP has run. The attack plants the child metadata while flat (so the
    # planted parent_id survives) and queries while hierarchical.
    marker = "VICTIMTOPSECRETMERGER"
    victim_body = f"{marker} checkout production release rollback plan merger price forty two per share."
    flat = RuntimeBuilder(settings).with_access_policy(CanonicalAccessPolicy()).build()
    try:
        project = flat.project(workspace="merck", corpus="research")
        project.runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id="victim.md",
                title="victim",
                media_type="text/markdown",
                content=victim_body.encode(),
                metadata={"acl": {"classification": "confidential", "allow_groups": ["research-a"]}},
            ),
        )
        owner_hits = project.retrieve(QUERY, principal=ALICE, top_k=20, autocut_enabled=False).hits
        victim_id = next(hit.chunk.id for hit in owner_hits if marker in hit.chunk.text)
        project.runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id="attacker.md",
                title="attacker",
                media_type="text/markdown",
                content=b"public checkout production release rollback plan bait content here.",
                metadata={
                    "acl": {"classification": "public"},
                    "granularity": "child",
                    "parent_id": victim_id,
                    "sent_start": 0,
                    "sent_end": 999,
                },
            ),
        )
    finally:
        flat.close()

    hierarchical = (
        RuntimeBuilder(settings.model_copy(update={"chunking_strategy": "parent_child"}))
        .with_access_policy(CanonicalAccessPolicy())
        .build()
    )
    try:
        project = hierarchical.project(workspace="merck", corpus="research")
        context = project.retrieve(QUERY, principal=BOB, top_k=20, autocut_enabled=False)
        hit_text = " ".join(hit.chunk.text for hit in context.hits)
        assert "bait" in hit_text, "the attacker child must be retrieved so the expansion path runs"
        assert marker not in hit_text
        assert marker not in project.ask(QUERY, principal=BOB).answer
    finally:
        hierarchical.close()


def test_async_optimization_fails_closed_under_an_active_policy(learning_project: RefaatProject) -> None:
    from refaat.errors import AuthorizationError
    from refaat.optimization.durable import resume_optimization_run

    # The guard is before any run read, so even a nonexistent run id fails closed rather than
    # running a program unfiltered with no principal.
    with pytest.raises(AuthorizationError):
        resume_optimization_run(learning_project, "no-such-run")


def test_evidence_serving_worker_jobs_fail_closed_under_an_active_policy(learning_project: RefaatProject) -> None:
    from refaat.application.worker import DurableWorker

    # Every evidence-serving durable job carries no principal, so the worker dispatch fails it
    # closed under an active policy rather than running the deployment unfiltered. This covers the
    # schedule-then-read-back vector on program.shadow_compare.
    project = learning_project
    job_id = project.runtime.store.enqueue_job(
        project.workspace_id,
        "program.shadow_compare",
        {"corpus_id": project.corpus_id, "program": "grounded-rag", "environment": "production", "questions": [QUERY]},
        max_attempts=1,
    )
    worker = DurableWorker(project.runtime)
    while worker.run_once():
        pass
    job = project.runtime.store.get_job(project.workspace_id, job_id)
    assert job["status"] != "succeeded"
    assert "SECRET-BRAVO-ZEPHYR" not in str(job.get("result") or "")
