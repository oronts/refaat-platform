"""Create the index_generations table added after 0015: physical index materialization identity.

This table was added to the metadata after 0015_entity_terms was stamped, so a database already at
0015 does not receive it: alembic does not re-run an applied revision, and production disables
create_all. This revision creates it idempotently (checkfirst), a no-op on a fresh database or one
whose 0015 run already reflected it, so an already migrated production database reaches the current
schema and an index generation has somewhere to persist.
"""

from __future__ import annotations

from alembic import context, op

from refaat.infrastructure.database.schema import index_generations

revision = "0016_index_generations"
down_revision = "0015_entity_terms"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 0001 emits the full current schema in --sql (offline) mode, so this catch-up revision must
    # no-op offline; otherwise it re-emits CREATE TABLE for a table 0001 already created.
    if context.is_offline_mode():
        return
    bind = op.get_bind()
    index_generations.create(bind, checkfirst=True)
    # Table.create(checkfirst=True) skips indexes when the table already exists, so ensure each index
    # directly, including the partial-unique active index a UniqueConstraint could not express.
    for index in index_generations.indexes:
        index.create(bind, checkfirst=True)


def downgrade() -> None:
    if context.is_offline_mode():
        return
    index_generations.drop(op.get_bind(), checkfirst=True)
