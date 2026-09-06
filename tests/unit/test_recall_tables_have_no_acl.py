"""Recall tables carry selectors only, never ACL vocabulary.

Which chunks are searched is a recall concern; whether a principal may see them stays on chunk.metadata
and is enforced at the query-time PEP. A future index_generations (or any *vector*/*generation* table)
that grew an ACL column would let a build capture ACLs and enforce them at rollback, so this gate turns
red the moment such a column appears in any recall structure. It is a no-op green today.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from refaat.infrastructure.database.schema import metadata
from refaat.infrastructure.opensearch_index import OpenSearchSearchIndex
from refaat.infrastructure.search import PgVectorSearchIndex

ACL_VOCAB = frozenset(
    {
        "acl",
        "classification",
        "allow_groups",
        "allow_roles",
        "deny_groups",
        "deny_users",
        "residency_zone",
        "tenant",
        "source_trust",
    }
)
SCOPE_OK = frozenset({"workspace_id", "corpus_id", "document_id", "chunk_id"})

_MIGRATIONS = Path(__file__).resolve().parents[2] / "migrations" / "versions"


def _acl_words_in(text: str) -> set[str]:
    # Any ACL-vocabulary word appearing as a whole identifier, so a planted `acl JSONB` or a multiline
    # `classification` property is caught regardless of the surrounding DDL/JSON formatting.
    return {w for w in ACL_VOCAB if re.search(rf"\b{re.escape(w)}\b", text, re.IGNORECASE)}


def _pgvector_ddl_sources() -> list[str]:
    # The runtime CREATE TABLE plus every migration that touches the pgvector recall relation, so a
    # column added in production DDL (not just the runtime initializer) is still in scope.
    sources = [inspect.getsource(PgVectorSearchIndex._initialize)]
    sources += [
        path.read_text() for path in sorted(_MIGRATIONS.glob("*.py")) if "refaat_chunk_vectors" in path.read_text()
    ]
    return sources


def _opensearch_ddl_source() -> str:
    return inspect.getsource(OpenSearchSearchIndex._ensure_index)


def _columns(source: str) -> set[str]:
    return set(re.findall(r"^\s*(\w+)\s+(?:varchar|vector)\b", source, re.MULTILINE))


def _opensearch_property_keys(source: str) -> set[str]:
    # \s* after the brace so a multiline property block ("name": {\n  "type": ...}) is captured too.
    return set(re.findall(r'"(\w+)":\s*\{\s*"type"', source))


def test_recall_ddl_sources_contain_no_acl_vocabulary() -> None:
    for source in _pgvector_ddl_sources():
        assert _acl_words_in(source) == set(), source
    assert _acl_words_in(_opensearch_ddl_source()) == set()


def test_recall_structures_expose_the_scope_selectors() -> None:
    pgvector_columns: set[str] = set()
    for source in _pgvector_ddl_sources():
        pgvector_columns |= _columns(source)
    assert SCOPE_OK.issubset(pgvector_columns | {"embedding"})

    keys = _opensearch_property_keys(_opensearch_ddl_source())
    assert "chunk_id" in keys
    assert "embedding" in keys


def test_generation_and_vector_schema_tables_have_no_acl_column() -> None:
    for name, table in metadata.tables.items():
        if "generation" in name or "vector" in name:
            assert ACL_VOCAB.isdisjoint(set(table.columns.keys())), name
