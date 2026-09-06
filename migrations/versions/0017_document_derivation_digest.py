"""Add documents.derivation_digest, added to the metadata after 0016.

A database already at 0016 does not receive the new column (alembic does not re-run an applied revision
and production disables create_all). This revision adds it idempotently, a no-op on a fresh database or
one whose 0016 run already reflected it, so an already migrated production database reaches the current
schema. The column is nullable with no server default, so existing rows stay NULL until a document is
next rewritten under the destroy lock.
"""

from __future__ import annotations

from alembic import context, op
from sqlalchemy import Column, String, inspect

revision = "0017_document_derivation_digest"
down_revision = "0016_index_generations"
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
    if not _has_column(inspector, "documents", "derivation_digest"):
        op.add_column("documents", Column("derivation_digest", String(64), nullable=True))


def downgrade() -> None:
    if context.is_offline_mode():
        return
    inspector = inspect(op.get_bind())
    if _has_column(inspector, "documents", "derivation_digest"):
        op.drop_column("documents", "derivation_digest")
