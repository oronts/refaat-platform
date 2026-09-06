"""Create the governance tables added after 0002: tool_approvals and the audit outbox.

These tables were added to the metadata after 0002_catch_up was stamped, so a database
already at 0002 does not receive them: alembic does not re-run an applied revision, and
production disables create_all. This revision creates them idempotently (checkfirst), a
no-op on a fresh database or one whose 0002 run already reflected them, so an already
migrated production database reaches the current schema and durable approvals and the
durable audit outbox actually have their tables.
"""

from __future__ import annotations

from alembic import context, op

from refaat.infrastructure.database.schema import audit_outbox, audit_outbox_deliveries, tool_approvals

revision = "0003_governance_tables"
down_revision = "0002_catch_up"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 0001 emits the full current schema in --sql (offline) mode, so this catch-up revision must
    # no-op offline; otherwise it re-emits CREATE TABLE for tables 0001 already created.
    if context.is_offline_mode():
        return
    bind = op.get_bind()
    # audit_outbox before audit_outbox_deliveries: the ledger has a foreign key to it.
    for table in (tool_approvals, audit_outbox, audit_outbox_deliveries):
        table.create(bind, checkfirst=True)
        # Table.create(checkfirst=True) skips indexes when the table already exists, so a partial
        # prior run that left a table without its indexes would never gain them; ensure each directly.
        for index in table.indexes:
            index.create(bind, checkfirst=True)


def downgrade() -> None:
    if context.is_offline_mode():
        return
    bind = op.get_bind()
    for table in (audit_outbox_deliveries, audit_outbox, tool_approvals):
        table.drop(bind, checkfirst=True)
