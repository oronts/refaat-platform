"""Create the hold_permission_history table added after 0013: the held-evidence permission ledger.

Added to the metadata after 0013_derivation_manifests was stamped, so a database already at 0013 does
not receive it: alembic does not re-run an applied revision, and production disables create_all. This
revision creates it idempotently (checkfirst), a no-op on a fresh database or one whose 0013 run already
reflected it, so an already migrated production database reaches the current schema and a held-evidence
permission change has an append-only record.
"""

from __future__ import annotations

from alembic import context, op

from refaat.infrastructure.database.schema import hold_permission_history

revision = "0014_hold_permission_history"
down_revision = "0013_derivation_manifests"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 0001 emits the full current schema in --sql (offline) mode, so this catch-up revision must
    # no-op offline; otherwise it re-emits CREATE TABLE for a table 0001 already created.
    if context.is_offline_mode():
        return
    bind = op.get_bind()
    hold_permission_history.create(bind, checkfirst=True)
    for index in hold_permission_history.indexes:
        index.create(bind, checkfirst=True)


def downgrade() -> None:
    if context.is_offline_mode():
        return
    hold_permission_history.drop(op.get_bind(), checkfirst=True)
