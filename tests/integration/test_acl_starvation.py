"""Authorized results must not be starved by top_k truncation in an ACL-dense corpus.

The ACL pushdown narrows the index fetch to the principal's authorized documents, and a bounded
over-fetch-and-refill backstop recovers authorized results even from an index that ignores the
pushdown. chunk_allowed stays the authoritative gate, so neither mechanism can ever surface an
unauthorized chunk; only recall is affected.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from refaat.application.runtime import Refaat, RefaatProject, RuntimeBuilder
from refaat.config import Settings
from refaat.core.retrieval import AdaptiveRetriever
from refaat.domain.authz import CanonicalAccessPolicy, Principal
from refaat.domain.models import Chunk, DocumentInput, SearchHit
from refaat.infrastructure.search import LocalSearchIndex

pytestmark = pytest.mark.integration

ALICE = Principal(id="alice", tenant="merck", groups=frozenset({"research-a"}))
QUERY = "production release rollback plan approval"
_A_ACL = {"classification": "confidential", "allow_groups": ["research-a"]}
_B_ACL = {"classification": "confidential", "allow_groups": ["research-b"]}


def _ingest_dense(project: RefaatProject, unauthorized: int) -> str:
    project.runtime.ingest(
        project.workspace_id,
        project.corpus_id,
        DocumentInput(
            source_id="authorized.md",
            title="authorized",
            media_type="text/markdown",
            content=b"The production release rollback plan needs approval from the research-a team.",
            metadata={"acl": _A_ACL, "doc": "authorized"},
        ),
    )
    for index in range(unauthorized):
        project.runtime.ingest(
            project.workspace_id,
            project.corpus_id,
            DocumentInput(
                source_id=f"other-{index}.md",
                title=f"other-{index}",
                media_type="text/markdown",
                content=b"The production release rollback plan needs approval from another team entirely.",
                metadata={"acl": _B_ACL, "doc": f"other-{index}"},
            ),
        )
    return project.corpus_id


@pytest.fixture
def dense_project(settings: Settings) -> Iterator[RefaatProject]:
    runtime = RuntimeBuilder(settings).with_access_policy(CanonicalAccessPolicy()).build()
    try:
        project = runtime.project(workspace="merck", corpus="research")
        _ingest_dense(project, unauthorized=40)
        yield project
    finally:
        runtime.close()


def test_authorized_result_present_and_no_leak_in_dense_corpus(dense_project: RefaatProject) -> None:
    context = dense_project.retrieve(QUERY, principal=ALICE, top_k=3, autocut_enabled=False)
    docs = {str(hit.chunk.metadata["doc"]) for hit in context.hits}
    assert "authorized" in docs, "the authorized doc must survive the ACL-dense truncation"
    assert not any(doc.startswith("other-") for doc in docs), "no research-b doc may reach ALICE"


def _hit(index: int, *, authorized: bool) -> SearchHit:
    acl = _A_ACL if authorized else _B_ACL
    chunk = Chunk(
        id=f"chunk_{index}",
        corpus_id="research",
        document_id=f"doc_{index}",
        ordinal=index,
        text="production release rollback plan",
        token_count=4,
        checksum=str(index),
        metadata={"acl": acl},
        embedding=(1.0, 0.0),
    )
    return SearchHit(chunk=chunk, score=1.0 - index / 1000.0)


def test_over_fetch_backstop_recovers_a_deep_authorized_result(settings: Settings) -> None:
    # An index that ignores the pushdown returns the sole authorized hit deep in the ranking, past the
    # first over-fetch window; the refill widens the fetch until it is recovered.
    runtime = RuntimeBuilder(settings).with_access_policy(CanonicalAccessPolicy()).build()
    try:
        retriever = AdaptiveRetriever(runtime.store, LocalSearchIndex(runtime.store), settings)
        pool = [_hit(index, authorized=(index == 30)) for index in range(60)]
        sizes: list[int] = []

        def fetch(size: int) -> list[SearchHit]:
            sizes.append(size)
            return pool[:size]

        result = retriever._authorized_fetch(fetch, 1, CanonicalAccessPolicy(), ALICE)
        assert [hit.chunk.id for hit in result] == ["chunk_30"]
        assert sizes[0] == settings.acl_overfetch_factor  # target 1 * factor
        assert sizes[-1] > sizes[0], "the fetch widened to recover the deep authorized hit"
        assert all(size <= settings.acl_overfetch_hard_cap for size in sizes)
    finally:
        runtime.close()


def test_over_fetch_terminates_on_an_exhausted_corpus(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings).with_access_policy(CanonicalAccessPolicy()).build()
    try:
        retriever = AdaptiveRetriever(runtime.store, LocalSearchIndex(runtime.store), settings)
        pool = [_hit(index, authorized=False) for index in range(3)]
        calls = 0

        def fetch(size: int) -> list[SearchHit]:
            nonlocal calls
            calls += 1
            return pool[:size]

        result = retriever._authorized_fetch(fetch, 5, CanonicalAccessPolicy(), ALICE)
        assert result == []  # no authorized hit exists
        assert calls == 1, "an exhausted corpus (raw < size) stops after one fetch"
    finally:
        runtime.close()


def test_ranked_slice_and_authorize_close_hype_raptor_starvation(settings: Settings) -> None:
    # The HyPE/RAPTOR routes widen the ranked slice under a policy and apply chunk_allowed before the
    # candidate_limit truncation, so an authorized hit ranked below the unauthorized ones is recovered.
    from refaat.domain.models import QueryRequest

    runtime = RuntimeBuilder(settings).with_access_policy(CanonicalAccessPolicy()).build()
    try:
        retriever = AdaptiveRetriever(
            runtime.store, LocalSearchIndex(runtime.store), settings, access_policy=CanonicalAccessPolicy()
        )
        best = {f"chunk_{i}": 1.0 - i / 1000.0 for i in range(50)}
        # No policy: the slice stays candidate_limit (byte-identical).
        assert retriever._ranked_slice(best, None, 5) == 5
        # Active policy: the whole ranked set (bounded by the cap) is considered.
        assert retriever._ranked_slice(best, frozenset({"doc"}), 5) == min(50, settings.acl_overfetch_hard_cap)

        request = QueryRequest(workspace_id="merck", corpus_id="research", query=QUERY, principal=ALICE)
        hits = [_hit(i, authorized=(i == 40)) for i in range(50)]
        # With no policy the hits pass through unchanged.
        assert retriever._authorize_and_truncate(list(hits), request, 3, None) == hits
        # Under a policy only the authorized hit survives, even though it ranks last.
        recovered = retriever._authorize_and_truncate(list(hits), request, 3, frozenset({"doc_40"}))
        assert [hit.chunk.id for hit in recovered] == ["chunk_40"]
    finally:
        runtime.close()


def test_local_index_pushdown_narrows_and_fails_closed(settings: Settings) -> None:
    runtime: Refaat = RuntimeBuilder(settings).build()
    try:
        project = runtime.project(workspace="acme", corpus="handbook")
        for name in ("alpha", "bravo"):
            runtime.ingest(
                project.workspace_id,
                project.corpus_id,
                DocumentInput(
                    source_id=f"{name}.md",
                    title=name,
                    media_type="text/markdown",
                    content=f"The {name} production release rollback plan needs approval.".encode(),
                ),
            )
        index = LocalSearchIndex(runtime.store)
        baseline = index.lexical_search(
            workspace_id=project.workspace_id, corpus_id=project.corpus_id, query=QUERY, limit=10
        )
        assert baseline, "the mixed corpus returns lexical hits"
        # None is byte-identical to the baseline call.
        with_none = index.lexical_search(
            workspace_id=project.workspace_id,
            corpus_id=project.corpus_id,
            query=QUERY,
            limit=10,
            allowed_document_ids=None,
        )
        assert [hit.chunk.id for hit in with_none] == [hit.chunk.id for hit in baseline]
        # The pushdown narrows to the allowed document.
        alpha_doc = next(hit.chunk.document_id for hit in baseline if "alpha" in hit.chunk.text)
        narrowed = index.lexical_search(
            workspace_id=project.workspace_id,
            corpus_id=project.corpus_id,
            query=QUERY,
            limit=10,
            allowed_document_ids=frozenset({alpha_doc}),
        )
        assert narrowed
        assert all(hit.chunk.document_id == alpha_doc for hit in narrowed)
        # An empty allow-set fails closed to zero rows.
        empty = index.lexical_search(
            workspace_id=project.workspace_id,
            corpus_id=project.corpus_id,
            query=QUERY,
            limit=10,
            allowed_document_ids=frozenset(),
        )
        assert empty == []
    finally:
        runtime.close()
