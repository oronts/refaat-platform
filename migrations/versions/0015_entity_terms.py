"""Create the entity_terms table added after 0014: the term index for the opt-in graph neighborhood read.

Added to the metadata after 0014_hold_permission_history was stamped, so a database already at 0014 does
not receive it: alembic does not re-run an applied revision, and production disables create_all. This
revision creates it idempotently (checkfirst), a no-op on a fresh database or one whose earlier run
already reflected it, so an already migrated production database reaches the current schema and enabling
graph_term_index can populate the index. The table stays empty until the opt-in flag is set.
"""

from __future__ import annotations

from alembic import context, op

from refaat.infrastructure.database.schema import entity_terms

revision = "0015_entity_terms"
down_revision = "0014_hold_permission_history"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 0001 emits the full current schema in --sql (offline) mode, so this catch-up revision must
    # no-op offline; otherwise it re-emits CREATE TABLE for a table 0001 already created.
    if context.is_offline_mode():
        return
    bind = op.get_bind()
    entity_terms.create(bind, checkfirst=True)
    for index in entity_terms.indexes:
        index.create(bind, checkfirst=True)


def downgrade() -> None:
    if context.is_offline_mode():
        return
    entity_terms.drop(op.get_bind(), checkfirst=True)
