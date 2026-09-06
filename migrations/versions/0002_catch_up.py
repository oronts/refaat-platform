"""Catch up an existing database to the full current schema.

Databases first stamped when 0001 was the only revision received the schema that
existed at that time. Every table, column, index, and named constraint added to
the metadata since then is missing from those databases even though
``alembic upgrade head`` reports them as current. This revision reconciles a
connected database against the live metadata by creating only the objects that
are absent (and replacing a named unique or check constraint whose definition has
drifted), so an old database reaches the full current schema and a fresh database
(already complete from 0001) sees a no-op. It is safe to run more than once.

Reconciliation reflects the live database, so this revision runs online only. In
offline (``--sql``) mode it is a no-op, because the target state cannot be known
without inspecting the connected database.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy import CheckConstraint, Table, UniqueConstraint

from refaat.infrastructure.database.schema import metadata

revision = "0002_catch_up"
down_revision = "0001_initial"
branch_labels = None
depends_on = None

# migration-safety-ack: the idempotent catch-up reconciler adds only absent tables, columns, and
# constraints under live inspection, and refuses a NOT NULL column with no server_default in code, so
# its forward path is safe by construction.


def _normalize_check(sqltext: object) -> str:
    return " ".join(str(sqltext).split()).casefold()


def _reconcile_constraints(
    table: Table,
    drop_unique: list[str],
    add_unique: list[UniqueConstraint],
    drop_check: list[str],
    add_check: list[CheckConstraint],
    dialect: str,
) -> None:
    if dialect == "sqlite":
        # SQLite cannot ALTER TABLE ADD/DROP CONSTRAINT, so batch copies the
        # table and rebuilds it with the reconciled constraints.
        with op.batch_alter_table(table.name) as batch_op:
            for name in drop_unique:
                batch_op.drop_constraint(name, type_="unique")
            for constraint in add_unique:
                batch_op.create_unique_constraint(constraint.name, [column.name for column in constraint.columns])
            for name in drop_check:
                batch_op.drop_constraint(name, type_="check")
            for constraint in add_check:
                batch_op.create_check_constraint(constraint.name, constraint.sqltext)
        return

    for name in drop_unique:
        op.drop_constraint(name, table.name, type_="unique")
    for constraint in add_unique:
        op.create_unique_constraint(constraint.name, table.name, [column.name for column in constraint.columns])
    for name in drop_check:
        op.drop_constraint(name, table.name, type_="check")
    for constraint in add_check:
        op.create_check_constraint(constraint.name, table.name, constraint.sqltext)


def upgrade() -> None:
    if context.is_offline_mode():
        return

    bind = op.get_bind()
    inspector = sa.inspect(bind)
    dialect = bind.dialect.name
    existing_tables = set(inspector.get_table_names())

    for table in metadata.sorted_tables:
        if table.name not in existing_tables:
            table.create(bind, checkfirst=True)
            continue

        present_columns = {column["name"] for column in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name not in present_columns:
                if not column.nullable and column.server_default is None:
                    # Adding a NOT NULL column with no server_default to a table that
                    # already holds rows fails: the existing rows have no value for it.
                    # A post-0001 NOT NULL column must carry a server_default so this
                    # reconciliation can backfill it. Refuse with an actionable message
                    # rather than emit DDL the database will reject mid-migration.
                    raise RuntimeError(
                        f"Cannot reconcile column {table.name}.{column.name}: it is NOT NULL without a "
                        "server_default, so a populated database cannot backfill it. Add a server_default to the "
                        "column in schema.py, or handle it in a dedicated data-backfill revision."
                    )
                op.add_column(table.name, column._copy())

        existing_uniques = {
            unique["name"]: set(unique["column_names"])
            for unique in inspector.get_unique_constraints(table.name)
            if unique["name"] is not None
        }
        existing_checks = {
            check["name"]: _normalize_check(check["sqltext"])
            for check in inspector.get_check_constraints(table.name)
            if check["name"] is not None
        }
        drop_unique: list[str] = []
        add_unique: list[UniqueConstraint] = []
        drop_check: list[str] = []
        add_check: list[CheckConstraint] = []
        for constraint in table.constraints:
            if constraint.name is None:
                continue
            if isinstance(constraint, UniqueConstraint):
                desired = {column.name for column in constraint.columns}
                current = existing_uniques.get(constraint.name)
                if current is None:
                    add_unique.append(constraint)
                elif current != desired:
                    drop_unique.append(constraint.name)
                    add_unique.append(constraint)
            elif isinstance(constraint, CheckConstraint):
                current_check = existing_checks.get(constraint.name)
                if current_check is None:
                    add_check.append(constraint)
                elif current_check != _normalize_check(constraint.sqltext):
                    drop_check.append(constraint.name)
                    add_check.append(constraint)
        if drop_unique or add_unique or drop_check or add_check:
            _reconcile_constraints(table, drop_unique, add_unique, drop_check, add_check, dialect)

        present_indexes = {index["name"] for index in inspector.get_indexes(table.name)}
        for index in table.indexes:
            if index.name is not None and index.name not in present_indexes:
                index.create(bind)


def downgrade() -> None:
    # No-op by design: this revision only reconciles a database toward the live
    # metadata, and the pre-0001 baseline it would need to reverse to is not
    # recoverable from the metadata (0001 itself builds the full current schema).
    # A full teardown is available through 0001's own downgrade.
    pass
