"""Create the deferred_erasures table added after 0009.

Added to the metadata after 0009_quota_counters was stamped, so a database already at 0009 does not
receive it (alembic does not re-run an applied revision and production disables create_all). This
revision creates it idempotently, a no-op on a fresh database or one whose 0009 run already reflected
it, so an already migrated production database reaches the current schema.
"""

from __future__ import annotations

from alembic import context, op

from refaat.infrastructure.database.schema import deferred_erasures

revision = "0010_deferred_erasures"
down_revision = "0009_quota_counters"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 0001 emits the full current schema in --sql (offline) mode, so this catch-up revision must
    # no-op offline; otherwise it re-emits CREATE TABLE for a table 0001 already created.
    if context.is_offline_mode():
        return
    bind = op.get_bind()
    deferred_erasures.create(bind, checkfirst=True)
    # Table.create(checkfirst=True) skips indexes when the table already exists, so a partial prior
    # run that left the table without an index would never gain it; ensure each directly.
    for index in deferred_erasures.indexes:
        index.create(bind, checkfirst=True)


def downgrade() -> None:
    if context.is_offline_mode():
        return
    bind = op.get_bind()
    deferred_erasures.drop(bind, checkfirst=True)
