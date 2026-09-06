"""Create the approval_requests table added after 0004: escalation request anchors.

This table was added to the metadata after 0004_break_glass_columns was stamped, so a database
already at 0004 does not receive it: alembic does not re-run an applied revision, and production
disables create_all. This revision creates it idempotently (checkfirst), a no-op on a fresh
database or one whose 0004 run already reflected it, so an already migrated production database
reaches the current schema and time-based escalation actually has its anchor table.
"""

from __future__ import annotations

from alembic import context, op

from refaat.infrastructure.database.schema import approval_requests

revision = "0005_approval_requests"
down_revision = "0004_break_glass_columns"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 0001 emits the full current schema in --sql (offline) mode, so this catch-up revision must
    # no-op offline; otherwise it re-emits CREATE TABLE for a table 0001 already created.
    if context.is_offline_mode():
        return
    bind = op.get_bind()
    approval_requests.create(bind, checkfirst=True)
    # Table.create(checkfirst=True) skips indexes when the table already exists, so a partial prior
    # run that left the table without its lookup index would never gain it; ensure each index directly.
    for index in approval_requests.indexes:
        index.create(bind, checkfirst=True)


def downgrade() -> None:
    if context.is_offline_mode():
        return
    approval_requests.drop(op.get_bind(), checkfirst=True)
