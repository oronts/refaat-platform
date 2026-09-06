"""The opt-in term-indexed graph read is a faithful, bounded equivalent of the full scan.

Flag-off writes no entity_terms rows and reads via list_graph, so it is byte-identical. Flag-on fetches
only the query-relevant subgraph and, fed to the unchanged graph_context, never under-returns: the boosts
and the fact set match the full scan, including under an ACL filter, while a narrow query reads fewer
rows. A corpus with no term index falls back to the full scan.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import func, insert, select

from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.config import Settings
from refaat.core.retrieval import AdaptiveRetriever, select_graph
from refaat.core.text import tokens
from refaat.domain.models import DocumentInput
from refaat.ids import new_id
from refaat.infrastructure.database.schema import entity_terms

pytestmark = pytest.mark.integration

_DOCS = [
    DocumentInput(
        source_id="alpha.txt",
        title="Alpha",
        media_type="text/plain",
        content=(
            b"Northwind Logistics partnered with Berlin Freight to expand European shipping.\n\n"
            b"Northwind Logistics and Berlin Freight signed the shipping agreement."
        ),
    ),
    DocumentInput(
        source_id="beta.txt",
        title="Beta",
        media_type="text/plain",
        content=(
            b"Acme Corporation acquired Tokyo Robotics to enter the Japanese market.\n\n"
            b"Acme Corporation and Tokyo Robotics announced the merger."
        ),
    ),
]

_QUERIES = [
    "northwind logistics shipping partnership",
    "acme tokyo robotics merger",
    "chait",
    "nothing here matches anything xyzzy",
]


def _built(settings: Settings) -> Iterator[Refaat]:
    runtime = RuntimeBuilder(settings).build()
    try:
        yield runtime
    finally:
        runtime.close()


def _ingest(runtime: Refaat, ws: str, corpus: str) -> None:
    for doc in _DOCS:
        runtime.ingest(ws, corpus, doc)


def _entity_term_count(runtime: Refaat, ws: str) -> int:
    with runtime.store.engine.connect() as conn:
        return int(
            conn.execute(
                select(func.count()).select_from(entity_terms).where(entity_terms.c.workspace_id == ws)
            ).scalar_one()
        )


def test_flag_on_matches_the_full_scan_and_reads_fewer_rows(settings: Settings) -> None:
    enabled = settings.model_copy(update={"graph_term_index_enabled": True})
    runtime = next(_built(enabled))
    try:
        project = runtime.project(workspace="acme", corpus="graph")
        ws, corpus = project.workspace_id, project.corpus_id
        _ingest(runtime, ws, corpus)
        full_entities, full_relations = runtime.store.list_graph(ws, corpus)
        assert full_entities
        assert full_relations

        for query in _QUERIES:
            neigh_e, neigh_r = select_graph(runtime.store, enabled, ws, corpus, query)
            b_on, f_on, d_on = AdaptiveRetriever.graph_context(query, neigh_e, neigh_r)
            b_off, f_off, d_off = AdaptiveRetriever.graph_context(query, full_entities, full_relations)
            # No under-return: the same documents are boosted and the same fact set is emitted.
            assert b_on.keys() == b_off.keys(), query
            for doc_id in b_on:
                assert abs(b_on[doc_id] - b_off[doc_id]) < 1e-9, query
            assert set(f_on) == set(f_off), query
            assert set(d_on) == set(d_off), query

        # ACL parity: graph_context filters the subgraph exactly as it filters the full scan.
        one_doc = frozenset({full_entities[0].document_id})
        b_on, f_on, _ = AdaptiveRetriever.graph_context(
            _QUERIES[0], *select_graph(runtime.store, enabled, ws, corpus, _QUERIES[0]), authorized_documents=one_doc
        )
        b_off, f_off, _ = AdaptiveRetriever.graph_context(
            _QUERIES[0], full_entities, full_relations, authorized_documents=one_doc
        )
        assert b_on.keys() == b_off.keys()
        assert set(f_on) == set(f_off)

        # Work limit: a narrow query reads strictly less than the whole corpus graph.
        narrow_e, narrow_r = select_graph(runtime.store, enabled, ws, corpus, "northwind")
        assert len(narrow_e) + len(narrow_r) < len(full_entities) + len(full_relations)
    finally:
        runtime.close()


def test_entity_terms_mirror_the_entity_tokens_and_cascade_on_erase(settings: Settings) -> None:
    enabled = settings.model_copy(update={"graph_term_index_enabled": True})
    runtime = next(_built(enabled))
    try:
        project = runtime.project(workspace="acme", corpus="graph")
        ws, corpus = project.workspace_id, project.corpus_id
        _ingest(runtime, ws, corpus)
        entities, _ = runtime.store.list_graph(ws, corpus)
        expected = sum(len(set(tokens(entity.normalized_name))) for entity in entities)
        assert expected > 0
        assert _entity_term_count(runtime, ws) == expected
        # Re-ingest is idempotent (delete-then-insert), so the count is stable.
        runtime.ingest(ws, corpus, _DOCS[0])
        entities, _ = runtime.store.list_graph(ws, corpus)
        assert _entity_term_count(runtime, ws) == sum(len(set(tokens(e.normalized_name))) for e in entities)
    finally:
        runtime.close()


def test_flag_off_writes_no_index_and_stays_byte_identical(settings: Settings) -> None:
    runtime = next(_built(settings))
    try:
        project = runtime.project(workspace="acme", corpus="graph")
        ws, corpus = project.workspace_id, project.corpus_id
        _ingest(runtime, ws, corpus)
        assert _entity_term_count(runtime, ws) == 0
        # select_graph returns exactly list_graph when the flag is off.
        entities, relations = select_graph(runtime.store, settings, ws, corpus, _QUERIES[0])
        full_entities, full_relations = runtime.store.list_graph(ws, corpus)
        assert [e.id for e in entities] == [e.id for e in full_entities]
        assert [r.id for r in relations] == [r.id for r in full_relations]
    finally:
        runtime.close()


def test_flag_on_without_a_backfilled_index_falls_back_to_the_full_scan(settings: Settings) -> None:
    runtime = next(_built(settings))  # ingest with the flag OFF, so no entity_terms rows exist
    try:
        project = runtime.project(workspace="acme", corpus="graph")
        ws, corpus = project.workspace_id, project.corpus_id
        _ingest(runtime, ws, corpus)
        assert not runtime.store.graph_index_present(ws, corpus)
        enabled = settings.model_copy(update={"graph_term_index_enabled": True})
        entities, relations = select_graph(runtime.store, enabled, ws, corpus, _QUERIES[0])
        full_entities, full_relations = runtime.store.list_graph(ws, corpus)
        assert {e.id for e in entities} == {e.id for e in full_entities}
        assert {r.id for r in relations} == {r.id for r in full_relations}
        # A query beyond the char cap also falls back.
        capped = settings.model_copy(update={"graph_term_index_enabled": True, "graph_term_index_max_query_chars": 3})
        entities, _ = select_graph(runtime.store, capped, ws, corpus, "a much longer query than three chars")
        assert {e.id for e in entities} == {e.id for e in full_entities}
    finally:
        runtime.close()


def test_partial_index_reads_as_absent_and_falls_back(settings: Settings) -> None:
    runtime = next(_built(settings))  # flag off, so ingest writes no entity_terms rows
    try:
        project = runtime.project(workspace="acme", corpus="graph")
        ws, corpus = project.workspace_id, project.corpus_id
        _ingest(runtime, ws, corpus)
        assert not runtime.store.graph_index_present(ws, corpus)
        entities, _ = runtime.store.list_graph(ws, corpus)
        one_doc = entities[0].document_id
        with runtime.store.engine.begin() as conn:
            conn.execute(
                insert(entity_terms),
                [
                    {
                        "id": f"et_probe_{index}_{ordinal}",
                        "entity_id": entity.id,
                        "workspace_id": ws,
                        "corpus_id": corpus,
                        "document_id": entity.document_id,
                        "term": term,
                    }
                    for index, entity in enumerate(entities)
                    if entity.document_id == one_doc and entity.normalized_name
                    for ordinal, term in enumerate(dict.fromkeys(tokens(entity.normalized_name)))
                ],
            )
        # One document backfilled, the other still bare: partial, so absent and full-scan on both paths.
        assert not runtime.store.graph_index_present(ws, corpus)
        enabled = settings.model_copy(update={"graph_term_index_enabled": True})
        for query in _QUERIES:
            neigh_e, neigh_r = select_graph(runtime.store, enabled, ws, corpus, query)
            full_e, full_r = runtime.store.list_graph(ws, corpus)
            assert {e.id for e in neigh_e} == {e.id for e in full_e}, query
            assert {r.id for r in neigh_r} == {r.id for r in full_r}, query
    finally:
        runtime.close()


def test_completeness_ignores_a_cross_scope_term_row(settings: Settings) -> None:
    runtime = next(_built(settings))  # flag off, so no term rows are written anywhere
    try:
        p1 = runtime.project(workspace="acme", corpus="graph")
        ws1, c1 = p1.workspace_id, p1.corpus_id
        _ingest(runtime, ws1, c1)
        p2 = runtime.project(workspace="beta", corpus="graph2")
        ws2, c2 = p2.workspace_id, p2.corpus_id
        _ingest(runtime, ws2, c2)
        entity_one = runtime.store.list_graph(ws1, c1)[0][0]
        doc_two = runtime.store.list_graph(ws2, c2)[0][0].document_id
        with runtime.store.engine.begin() as conn:
            conn.execute(
                insert(entity_terms),
                [
                    {
                        "id": "et_cross_scope",
                        "entity_id": entity_one.id,
                        "workspace_id": ws2,
                        "corpus_id": c2,
                        "document_id": doc_two,
                        "term": "acme",
                    }
                ],
            )
        # The row references entity_one by id but is stamped to scope 2; scoped completeness ignores it,
        # so corpus 1 still reads as unindexed and select_graph full-scans (never under-returns).
        assert not runtime.store.graph_index_present(ws1, c1)
        enabled = settings.model_copy(update={"graph_term_index_enabled": True})
        for query in _QUERIES:
            neigh_e, neigh_r = select_graph(runtime.store, enabled, ws1, c1, query)
            full_e, full_r = runtime.store.list_graph(ws1, c1)
            assert {e.id for e in neigh_e} == {e.id for e in full_e}, query
            assert {r.id for r in neigh_r} == {r.id for r in full_r}, query
    finally:
        runtime.close()


def test_graph_neighborhood_returns_an_empty_name_entity(settings: Settings) -> None:
    runtime = next(_built(settings))
    try:
        workspace = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(workspace, "graph")
        document_id = new_id("doc")
        runtime.store.replace_document(
            workspace_id=workspace,
            corpus_id=corpus,
            document_id=document_id,
            source_id="empty.txt",
            title="Empty",
            media_type="text/plain",
            checksum="checksum",
            blob_uri="blob://empty",
            document_metadata={},
            chunk_rows=[],
            entity_rows=[
                {
                    "id": new_id("ent"),
                    "workspace_id": workspace,
                    "corpus_id": corpus,
                    "document_id": document_id,
                    "name": "!!!",
                    "normalized_name": "",
                    "kind": "named",
                    "metadata": {},
                }
            ],
            relation_rows=[],
        )
        # An empty normalized_name is a substring of every query; the selector adds "" so it is fetched.
        entities, _ = runtime.store.graph_neighborhood(
            workspace, corpus, term_candidates=frozenset(), name_substrings=frozenset({""})
        )
        assert [entity.normalized_name for entity in entities] == [""]
    finally:
        runtime.close()
