"""Add the break-glass columns to tool_approvals: kind and reason.

These columns were added to the metadata after 0003_governance_tables was stamped, so a
database already at 0003 does not receive them: alembic does not re-run an applied revision,
and production disables create_all. This revision adds each column only when it is absent, so
it is a no-op on a fresh database (already complete from create_all) or one whose 0003 run
predates the columns, and an already migrated production database reaches the current schema.

kind is added with server_default "normal" so existing rows written before it existed read as
normal approvals; without that, the normal approval path (which skips any kind != "normal")
would silently drop every existing approval.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op

revision = "0004_break_glass_columns"
down_revision = "0003_governance_tables"
branch_labels = None
depends_on = None

_TABLE = "tool_approvals"
_COLUMNS = (
    sa.Column("kind", sa.String(16), nullable=False, server_default="normal"),
    sa.Column("reason", sa.Text()),
)


def _present_columns(bind: sa.engine.Connection) -> set[str]:
    return {column["name"] for column in sa.inspect(bind).get_columns(_TABLE)}


def upgrade() -> None:
    if context.is_offline_mode():
        # Offline (--sql) mode cannot inspect the connected database, so the idempotent add is a
        # no-op here: 0001 emits the full current schema (including these columns) in offline mode.
        return
    bind = op.get_bind()
    present = _present_columns(bind)
    missing = [column for column in _COLUMNS if column.name not in present]
    if not missing:
        return
    if bind.dialect.name == "sqlite":
        # SQLite cannot ADD COLUMN with certain clauses in one ALTER, so batch rebuilds the table.
        with op.batch_alter_table(_TABLE) as batch_op:
            for column in missing:
                batch_op.add_column(column._copy())
        return
    for column in missing:
        op.add_column(_TABLE, column._copy())


def downgrade() -> None:
    if context.is_offline_mode():
        return
    bind = op.get_bind()
    present = _present_columns(bind)
    drop = [column.name for column in _COLUMNS if column.name in present]
    if not drop:
        return
    if bind.dialect.name == "sqlite":
        with op.batch_alter_table(_TABLE) as batch_op:
            for name in drop:
                batch_op.drop_column(name)
        return
    for name in drop:
        op.drop_column(_TABLE, name)
