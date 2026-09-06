"""Create the kill_switches table added after 0007: durable incident kill-switch state.

Added to the metadata after 0007_legal_holds was stamped, so a database already at 0007 does not
receive it (alembic does not re-run an applied revision and production disables create_all). This
revision creates it idempotently, a no-op on a fresh database or one whose 0007 run already
reflected it, so an already migrated production database reaches the current schema.
"""

from __future__ import annotations

from alembic import context, op

from refaat.infrastructure.database.schema import kill_switches

revision = "0008_kill_switches"
down_revision = "0007_legal_holds"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 0001 emits the full current schema in --sql (offline) mode, so this catch-up revision must
    # no-op offline; otherwise it re-emits CREATE TABLE for a table 0001 already created.
    if context.is_offline_mode():
        return
    bind = op.get_bind()
    kill_switches.create(bind, checkfirst=True)
    # Table.create(checkfirst=True) skips indexes when the table already exists, so a partial prior
    # run that left the table without an index would never gain it; ensure each directly.
    for index in kill_switches.indexes:
        index.create(bind, checkfirst=True)


def downgrade() -> None:
    if context.is_offline_mode():
        return
    kill_switches.drop(op.get_bind(), checkfirst=True)
