"""Create the legal_holds table added after 0006: durable legal-hold state.

Added to the metadata after 0006_erasure_ledger was stamped, so a database already at 0006 does
not receive it (alembic does not re-run an applied revision and production disables create_all).
This revision creates it idempotently, a no-op on a fresh database or one whose 0006 run already
reflected it, so an already migrated production database reaches the current schema.
"""

from __future__ import annotations

from alembic import context, op

from refaat.infrastructure.database.schema import legal_holds

revision = "0007_legal_holds"
down_revision = "0006_erasure_ledger"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 0001 emits the full current schema in --sql (offline) mode, so this catch-up revision must
    # no-op offline; otherwise it re-emits CREATE TABLE for a table 0001 already created.
    if context.is_offline_mode():
        return
    bind = op.get_bind()
    legal_holds.create(bind, checkfirst=True)
    # Table.create(checkfirst=True) skips indexes when the table already exists, so a partial prior
    # run that left the table without its coverage index would never gain it; ensure each directly.
    for index in legal_holds.indexes:
        index.create(bind, checkfirst=True)


def downgrade() -> None:
    if context.is_offline_mode():
        return
    legal_holds.drop(op.get_bind(), checkfirst=True)
