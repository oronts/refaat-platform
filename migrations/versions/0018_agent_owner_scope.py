"""Add agents.owner_scope, added to the metadata after 0017.

A database already at 0017 does not receive the new column (alembic does not re-run an applied revision
and production disables create_all). This revision adds it idempotently, a no-op on a fresh database or
one whose 0017 run already reflected it, so an already migrated production database reaches the current
schema. The column is nullable with no server default, so existing agent rows stay NULL and their
mailboxes remain workspace-readable until an agent is recreated under an active policy.
"""

from __future__ import annotations

from alembic import context, op
from sqlalchemy import Column, String, inspect

revision = "0018_agent_owner_scope"
down_revision = "0017_document_derivation_digest"
branch_labels = None
depends_on = None


def _has_column(inspector: object, table: str, column: str) -> bool:
    return any(col["name"] == column for col in inspector.get_columns(table))  # type: ignore[attr-defined]


def upgrade() -> None:
    # 0001 emits the full current schema in --sql (offline) mode, so this catch-up revision must no-op
    # offline; otherwise it re-adds a column 0001 already created.
    if context.is_offline_mode():
        return
    inspector = inspect(op.get_bind())
    if not _has_column(inspector, "agents", "owner_scope"):
        op.add_column("agents", Column("owner_scope", String(64), nullable=True))


def downgrade() -> None:
    if context.is_offline_mode():
        return
    inspector = inspect(op.get_bind())
    if _has_column(inspector, "agents", "owner_scope"):
        op.drop_column("agents", "owner_scope")
