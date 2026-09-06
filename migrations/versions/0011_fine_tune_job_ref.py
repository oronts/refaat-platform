"""Add fine_tune_runs.provider_job_ref, added to the metadata after 0010.

A database already at 0010 does not receive the new column (alembic does not re-run an applied revision and
production disables create_all). This revision adds it idempotently, a no-op on a fresh database or one whose
0010 run already reflected it, so an already migrated production database reaches the current schema.
"""

from __future__ import annotations

from alembic import context, op
from sqlalchemy import Column, String, inspect

from refaat.infrastructure.database.schema import fine_tune_runs

revision = "0011_fine_tune_job_ref"
down_revision = "0010_deferred_erasures"
branch_labels = None
depends_on = None


def _has_column(inspector: object, table: str, column: str) -> bool:
    return any(col["name"] == column for col in inspector.get_columns(table))  # type: ignore[attr-defined]


def upgrade() -> None:
    # 0001 emits the full current schema in --sql (offline) mode, so this catch-up revision must no-op
    # offline; otherwise it re-adds a column 0001 already created.
    if context.is_offline_mode():
        return
    bind = op.get_bind()
    # fine_tune_runs was added to schema.py after 0002 was stamped, so a DB stamped mid-history may
    # have no such table; create it (with the column) before the add_column path for an existing one.
    fine_tune_runs.create(bind, checkfirst=True)
    for index in fine_tune_runs.indexes:
        index.create(bind, checkfirst=True)
    inspector = inspect(bind)
    if not _has_column(inspector, "fine_tune_runs", "provider_job_ref"):
        op.add_column("fine_tune_runs", Column("provider_job_ref", String(200), nullable=True))


def downgrade() -> None:
    if context.is_offline_mode():
        return
    bind = op.get_bind()
    inspector = inspect(bind)
    if _has_column(inspector, "fine_tune_runs", "provider_job_ref"):
        op.drop_column("fine_tune_runs", "provider_job_ref")
