"""A reusable behavioral conformance contract for the GraphReadRepository port.

Structural conformance checks the four members exist; this checks the invariants ``select_graph`` relies
on so the query and RLM hot paths can trust any backend: ``list_graph`` returns a stable order across
calls with every relation's endpoints present, ``max_entity_name_length`` never truncates a real name,
and ``graph_neighborhood`` returns a relation-closed subset of the full graph that never under-returns an
entity graph_context would match, on BOTH of graph_context's branches, a name substring OR a token
intersection. That last guarantee is load-bearing: over-return is harmless because graph_context
re-filters, but under-return silently drops a fact. The neighborhood is only the read select_graph uses
when the index is present, so the caller must seed an indexed corpus (``graph_index_present`` true) or the
contract reports it cannot verify the term-indexed read. The case must span more than one entity and match
at least one, or the checks are vacuous, which the contract reports rather than passing silently. The
caller seeds the backend through the normal ingestion path, so the contract only reads.
"""

from __future__ import annotations

from dataclasses import dataclass

from refaat.conformance.contract import ConformanceViolation
from refaat.core.text import name_substrings, normalized_key, tokens
from refaat.ports import GraphReadRepository


@dataclass(frozen=True, slots=True)
class GraphReadConformanceCase:
    repo: GraphReadRepository
    workspace_id: str
    corpus_id: str
    query: str


def _v(member: str, problem: str) -> ConformanceViolation:
    return ConformanceViolation(port="GraphReadRepository", member=member, problem=problem)


def graph_read_conformance(case: GraphReadConformanceCase) -> list[ConformanceViolation]:
    """Return a backend's violations of the GraphReadRepository behavioral contract; empty is conformant."""
    repo, ws, corpus = case.repo, case.workspace_id, case.corpus_id
    out: list[ConformanceViolation] = []

    entities, relations = repo.list_graph(ws, corpus)
    entity_ids = {e.id for e in entities}
    relation_ids = {r.id for r in relations}

    if len(entities) < 2:
        out.append(_v("list_graph", "case is not discriminating: the corpus spans fewer than two entities"))
    again_e, again_r = repo.list_graph(ws, corpus)
    if [e.id for e in again_e] != [e.id for e in entities] or [r.id for r in again_r] != [r.id for r in relations]:
        out.append(_v("list_graph", "is not deterministic: two calls returned a different order"))
    for relation in relations:
        if relation.source_entity_id not in entity_ids or relation.target_entity_id not in entity_ids:
            out.append(_v("list_graph", f"relation {relation.id} references an entity absent from the graph"))

    try:
        present = repo.graph_index_present(ws, corpus)
    except Exception as exc:
        out.append(_v("graph_index_present", f"raised {exc!r}"))
        return out

    max_len = repo.max_entity_name_length(ws, corpus)
    longest = max((len(e.normalized_name) for e in entities), default=0)
    if max_len < longest:
        out.append(_v("max_entity_name_length", f"reported {max_len} below the longest name {longest}, truncating"))

    if not present:
        # Without the index select_graph reads list_graph, so the neighborhood is not exercised here.
        out.append(_v("graph_index_present", "index not present, so the term-indexed read is not exercised"))
        return out

    query_key = normalized_key(case.query)
    query_terms = frozenset(tokens(case.query))
    names = name_substrings(query_key, max(max_len, longest)) | {""}
    neigh_e, neigh_r = repo.graph_neighborhood(ws, corpus, term_candidates=query_terms, name_substrings=names)
    neigh_ids = {e.id for e in neigh_e}
    if leaked := neigh_ids - entity_ids:
        out.append(_v("graph_neighborhood", f"returned entities outside list_graph: {sorted(leaked)}"))
    if leaked := {r.id for r in neigh_r} - relation_ids:
        out.append(_v("graph_neighborhood", f"returned relations outside list_graph: {sorted(leaked)}"))
    for relation in neigh_r:
        if relation.source_entity_id not in neigh_ids or relation.target_entity_id not in neigh_ids:
            out.append(_v("graph_neighborhood", f"relation {relation.id} is not closed over the returned entities"))

    # Mirror graph_context's predicate: a name substring of the query key OR a token intersection.
    expected = {
        e.id for e in entities if e.normalized_name in query_key or bool(set(tokens(e.normalized_name)) & query_terms)
    }
    if missing := expected - neigh_ids:
        out.append(_v("graph_neighborhood", f"under-returned matched entities: {sorted(missing)}"))
    if not expected:
        out.append(_v("graph_neighborhood", "case is not discriminating: the query matches no entity"))
    return out
