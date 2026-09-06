"""Create the quota_counters and quota_reservations tables added after 0008.

Added to the metadata after 0008_kill_switches was stamped, so a database already at 0008 does not
receive them (alembic does not re-run an applied revision and production disables create_all). This
revision creates them idempotently, a no-op on a fresh database or one whose 0008 run already
reflected them, so an already migrated production database reaches the current schema.
"""

from __future__ import annotations

from alembic import context, op

from refaat.infrastructure.database.schema import quota_counters, quota_reservations

revision = "0009_quota_counters"
down_revision = "0008_kill_switches"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 0001 emits the full current schema in --sql (offline) mode, so this catch-up revision must
    # no-op offline; otherwise it re-emits CREATE TABLE for tables 0001 already created.
    if context.is_offline_mode():
        return
    bind = op.get_bind()
    for table in (quota_counters, quota_reservations):
        table.create(bind, checkfirst=True)
        # Table.create(checkfirst=True) skips indexes when the table already exists, so a partial
        # prior run that left the table without an index would never gain it; ensure each directly.
        for index in table.indexes:
            index.create(bind, checkfirst=True)


def downgrade() -> None:
    if context.is_offline_mode():
        return
    bind = op.get_bind()
    for table in (quota_reservations, quota_counters):
        table.drop(bind, checkfirst=True)
