"""A reusable behavioral conformance contract for the SearchIndex port.

Structural conformance checks the members exist; this checks the invariants a backend must uphold so
the retriever can trust any backend equally: every returned hit resolves to a stored chunk in the
requested scope, results are ordered by descending score, an empty ACL allow-set fails closed, a
document allow-set and a metadata filter never leak, a mode a backend declares it does not support
raises rather than silently degrading, and both the semantic and the lexical path enforce all of this
(the retriever passes the ACL and metadata filters to both). The built-in Local and pgvector backends
and the reference OpenSearch backend all pass it. The caller seeds a backend through the normal
ingestion path, then passes the resulting ids here, so the contract only reads and never depends on a
write path. The case must be discriminating (its unfiltered result spans more than the one scoped
document), or the scope checks would be vacuous; the contract reports that rather than passing silently.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from refaat.conformance.contract import ConformanceViolation
from refaat.domain.search import SearchIndexCapabilities
from refaat.errors import CapabilityNotSupportedError

if TYPE_CHECKING:
    from refaat.domain.models import MetadataFilter, SearchHit
    from refaat.ports import SearchIndex


@dataclass(frozen=True, slots=True)
class SearchIndexConformanceCase:
    index: SearchIndex
    workspace_id: str
    corpus_id: str
    query_embedding: tuple[float, ...]
    all_chunk_ids: frozenset[str]
    document_id: str
    document_chunk_ids: frozenset[str]
    lexical_term: str
    metadata_filter: MetadataFilter | None = None
    metadata_expected_ids: frozenset[str] = field(default_factory=frozenset)
    foreign_scope_ids: frozenset[str] = field(default_factory=frozenset)
    # Opt-in delete verification: when a probe document is supplied the contract deletes it last and
    # proves its chunks no longer resolve, so a backend whose delete silently no-ops is caught. Default
    # None keeps the contract read-only and non-destructive for a reused case.
    delete_probe_document_id: str | None = None
    delete_probe_chunk_ids: frozenset[str] = field(default_factory=frozenset)


def _v(member: str, problem: str) -> ConformanceViolation:
    return ConformanceViolation(port="SearchIndex", member=member, problem=problem)


def _ids(hits: Sequence[SearchHit]) -> set[str]:
    return {hit.chunk.id for hit in hits}


def _descending(hits: Sequence[SearchHit]) -> bool:
    return all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))


def _call(
    member: str, note: str, fn: Callable[[], list[SearchHit]]
) -> tuple[list[SearchHit], list[ConformanceViolation]]:
    """Run one index probe; a raised exception becomes a violation so later probes still run."""
    try:
        return fn(), []
    except Exception as exc:
        return [], [_v(member, f"{note} raised {exc!r}")]


def _scope_pushdown_checks(
    member: str,
    search: Callable[..., list[SearchHit]],
    case: SearchIndexConformanceCase,
) -> list[ConformanceViolation]:
    """The ACL and metadata pushdown a backend must honor, applied identically to both search modes."""
    out: list[ConformanceViolation] = []

    denied, vs = _call(member, "an empty ACL allow-set", lambda: search(allowed_document_ids=frozenset()))
    out += vs
    if denied:
        out.append(_v(member, "an empty ACL allow-set returned hits; it must fail closed"))

    scoped, vs = _call(
        member, "a single-document allow-set", lambda: search(allowed_document_ids=frozenset({case.document_id}))
    )
    out += vs
    if leaked := _ids(scoped) - case.document_chunk_ids:
        out.append(_v(member, f"a document allow-set leaked other documents' chunks: {sorted(leaked)}"))
    if case.document_chunk_ids and not scoped:
        out.append(_v(member, "a document allow-set dropped all of that document's chunks"))

    if case.metadata_filter is not None:
        filtered, vs = _call(member, "a metadata filter", lambda: search(metadata_filter=case.metadata_filter))
        out += vs
        if leaked := _ids(filtered) - case.metadata_expected_ids:
            out.append(_v(member, f"a metadata filter leaked non-matching chunks: {sorted(leaked)}"))
        if case.metadata_expected_ids and not filtered:
            out.append(_v(member, "a metadata filter dropped all matching chunks"))
    return out


def _semantic_checks(case: SearchIndexConformanceCase) -> list[ConformanceViolation]:
    index, ws, corpus = case.index, case.workspace_id, case.corpus_id
    limit = len(case.all_chunk_ids) + len(case.foreign_scope_ids) + 5

    def search(**kwargs: Any) -> list[SearchHit]:
        return index.semantic_search(
            workspace_id=ws, corpus_id=corpus, query_embedding=case.query_embedding, limit=limit, **kwargs
        )

    out: list[ConformanceViolation] = []
    hits, vs = _call("semantic_search", "the seeded corpus", lambda: search())
    out += vs
    if not vs:
        if not hits:
            out.append(_v("semantic_search", "returned no hits for the seeded corpus"))
        if leaked := _ids(hits) - case.all_chunk_ids:
            out.append(_v("semantic_search", f"returned chunks outside the requested scope: {sorted(leaked)}"))
        if not _descending(hits):
            out.append(_v("semantic_search", "hits are not ordered by descending score"))
        if hits and not (_ids(hits) - case.document_chunk_ids):
            out.append(
                _v("semantic_search", "case is not discriminating: unfiltered results span only the scoped document")
            )
    out += _scope_pushdown_checks("semantic_search", search, case)
    return out


def _lexical_checks(case: SearchIndexConformanceCase) -> list[ConformanceViolation]:
    index, ws, corpus = case.index, case.workspace_id, case.corpus_id
    limit = len(case.all_chunk_ids) + len(case.foreign_scope_ids) + 5

    def search(**kwargs: Any) -> list[SearchHit]:
        return index.lexical_search(workspace_id=ws, corpus_id=corpus, query=case.lexical_term, limit=limit, **kwargs)

    out: list[ConformanceViolation] = []
    hits, vs = _call("lexical_search", "a seeded term", lambda: search())
    out += vs
    if not vs:
        if leaked := _ids(hits) - case.all_chunk_ids:
            out.append(_v("lexical_search", f"returned chunks outside the requested scope: {sorted(leaked)}"))
        if not (_ids(hits) & case.document_chunk_ids):
            out.append(_v("lexical_search", "the seeded term did not retrieve the seeded document"))
        if not _descending(hits):
            out.append(_v("lexical_search", "hits are not ordered by descending score"))
        if hits and not (_ids(hits) - case.document_chunk_ids):
            out.append(
                _v("lexical_search", "case is not discriminating: the seeded term matches only the scoped document")
            )
    out += _scope_pushdown_checks("lexical_search", search, case)
    return out


def _delete_verification_checks(
    case: SearchIndexConformanceCase, *, supports_semantic: bool, supports_lexical: bool
) -> list[ConformanceViolation]:
    """Prove that deleting a probe document removes its chunks from search; a no-op delete is caught.

    For an index that owns a separate copy (pgvector, OpenSearch); a store-backed index whose
    delete_document is a no-op because the store is authoritative should leave the probe unset.
    """
    probe = case.delete_probe_document_id
    if probe is None:
        return []
    index, ws, corpus = case.index, case.workspace_id, case.corpus_id
    limit = len(case.all_chunk_ids) + len(case.foreign_scope_ids) + 5
    scope = frozenset({probe})
    if supports_semantic:

        def search() -> list[SearchHit]:
            return index.semantic_search(
                workspace_id=ws,
                corpus_id=corpus,
                query_embedding=case.query_embedding,
                limit=limit,
                allowed_document_ids=scope,
            )
    elif supports_lexical:

        def search() -> list[SearchHit]:
            return index.lexical_search(
                workspace_id=ws, corpus_id=corpus, query=case.lexical_term, limit=limit, allowed_document_ids=scope
            )
    else:
        return []

    before, vs = _call("delete_document", "the probe document before deletion", search)
    if vs:
        return vs
    if not (_ids(before) & case.delete_probe_chunk_ids):
        return [
            _v("delete_document", "the probe document was not retrievable before deletion, so the check is vacuous")
        ]
    try:
        index.delete_document(probe)
    except Exception as exc:
        return [_v("delete_document", f"raised {exc!r}")]
    after, vs = _call("delete_document", "a search after deletion", search)
    if vs:
        return vs
    if survived := _ids(after) & case.delete_probe_chunk_ids:
        return [_v("delete_document", f"deleted document chunks still returned by search: {sorted(survived)}")]
    return []


def _must_raise_unsupported(member: str, call: Callable[[], object]) -> list[ConformanceViolation]:
    try:
        call()
    except CapabilityNotSupportedError:
        return []
    except Exception as exc:
        return [_v(member, f"declared unsupported must raise CapabilityNotSupportedError, raised {exc!r}")]
    return [_v(member, "declared unsupported but did not raise CapabilityNotSupportedError")]


def search_index_conformance(case: SearchIndexConformanceCase) -> list[ConformanceViolation]:
    """Return a backend's violations of the SearchIndex behavioral contract; an empty list is conformant."""
    index, ws, corpus = case.index, case.workspace_id, case.corpus_id
    limit = len(case.all_chunk_ids) + len(case.foreign_scope_ids) + 5
    caps = getattr(index, "capabilities", None)
    supports_semantic = caps.supports_semantic if isinstance(caps, SearchIndexCapabilities) else True
    supports_lexical = caps.supports_lexical if isinstance(caps, SearchIndexCapabilities) else True

    violations: list[ConformanceViolation] = []
    if supports_semantic:
        violations.extend(_semantic_checks(case))
    else:
        violations.extend(
            _must_raise_unsupported(
                "semantic_search",
                lambda: index.semantic_search(
                    workspace_id=ws, corpus_id=corpus, query_embedding=case.query_embedding, limit=limit
                ),
            )
        )
    if supports_lexical:
        violations.extend(_lexical_checks(case))
    else:
        violations.extend(
            _must_raise_unsupported(
                "lexical_search",
                lambda: index.lexical_search(workspace_id=ws, corpus_id=corpus, query=case.lexical_term, limit=limit),
            )
        )
    # Destructive, so last: deleting the probe must not disturb the read checks above.
    violations.extend(
        _delete_verification_checks(case, supports_semantic=supports_semantic, supports_lexical=supports_lexical)
    )
    return violations
