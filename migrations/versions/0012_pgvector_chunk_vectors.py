"""Provision the pgvector chunk-vectors table for production, where the runtime does not create it.

The refaat_chunk_vectors relation is pgvector specific and is created at runtime by PgVectorSearchIndex
when schema provisioning is enabled (dev, test, offline). In production the runtime verifies the schema
instead of issuing DDL, so this revision provisions the extension, table, and indexes under the migration
role. It is a no-op offline and on any non-PostgreSQL dialect, and idempotent on PostgreSQL. The embedding
dimension is read from the same environment variable the runtime embedder uses, so both agree.
"""

from __future__ import annotations

import os

from alembic import context, op

revision = "0012_pgvector_chunk_vectors"
down_revision = "0011_fine_tune_job_ref"
branch_labels = None
depends_on = None


def _dimensions() -> int:
    # Resolve the vector dimension the same way the runtime embedder does (config plus any .env), so the
    # provisioned column matches. The runtime also verifies the column dimension at startup, so a residual
    # mismatch fails closed rather than corrupting data. Fall back to the plain environment or the default
    # if the settings cannot be constructed in the migration environment.
    try:
        from refaat.config import Settings

        return int(Settings().embedding_dimensions)
    except Exception:
        try:
            value = int(os.environ.get("REFAAT_EMBEDDING_DIMENSIONS", "384"))
        except ValueError:
            value = 384
        return value if value > 0 else 384


def upgrade() -> None:
    # migration-safety-ack: new pgvector extension/table/indexes with IF NOT EXISTS on an empty relation, idempotent
    if context.is_offline_mode():
        return
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    dimensions = _dimensions()
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS refaat_chunk_vectors (
            chunk_id varchar(96) PRIMARY KEY,
            workspace_id varchar(64) NOT NULL,
            corpus_id varchar(64) NOT NULL,
            document_id varchar(64) NOT NULL,
            embedding vector({dimensions}) NOT NULL
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_refaat_vectors_scope ON refaat_chunk_vectors (workspace_id, corpus_id)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_refaat_vectors_hnsw "
        "ON refaat_chunk_vectors USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    if context.is_offline_mode():
        return
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.execute("DROP TABLE IF EXISTS refaat_chunk_vectors")
