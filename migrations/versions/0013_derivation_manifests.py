"""Create the derivation_manifests table added after 0012: the per-corpus materialization record.

This table was added to the metadata after 0012_pgvector_chunk_vectors was stamped, so a database
already at 0012 does not receive it: alembic does not re-run an applied revision, and production
disables create_all. This revision creates it idempotently (checkfirst), a no-op on a fresh database
or one whose 0012 run already reflected it, so an already migrated production database reaches the
current schema and a derivation manifest has somewhere to persist.
"""

from __future__ import annotations

from alembic import context, op

from refaat.infrastructure.database.schema import derivation_manifests

revision = "0013_derivation_manifests"
down_revision = "0012_pgvector_chunk_vectors"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 0001 emits the full current schema in --sql (offline) mode, so this catch-up revision must
    # no-op offline; otherwise it re-emits CREATE TABLE for a table 0001 already created.
    if context.is_offline_mode():
        return
    bind = op.get_bind()
    derivation_manifests.create(bind, checkfirst=True)
    # Table.create(checkfirst=True) skips indexes when the table already exists, so ensure each index
    # directly in case a partial prior run left the table without one.
    for index in derivation_manifests.indexes:
        index.create(bind, checkfirst=True)


def downgrade() -> None:
    if context.is_offline_mode():
        return
    derivation_manifests.drop(op.get_bind(), checkfirst=True)
