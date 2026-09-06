from __future__ import annotations

import io
import logging
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config

from refaat.config import get_settings
from refaat.infrastructure.database.schema import metadata
from refaat.observability import get_logger

REPO_ROOT = Path(__file__).resolve().parents[2]


def _alembic_config(db_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    monkeypatch.setenv("REFAAT_DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("REFAAT_AUTO_CREATE_SCHEMA", "false")
    get_settings.cache_clear()
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    return config


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> object:
    yield
    get_settings.cache_clear()


def _table_columns(engine: sa.Engine, table: str) -> set[str]:
    inspector = sa.inspect(engine)
    return {column["name"] for column in inspector.get_columns(table)}


def _table_indexes(engine: sa.Engine, table: str) -> set[str]:
    inspector = sa.inspect(engine)
    return {index["name"] for index in inspector.get_indexes(table) if index["name"]}


def test_catch_up_repairs_a_database_stamped_before_later_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "old.db"
    engine = sa.create_engine(f"sqlite:///{db_path}")

    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE outbox_deliveries")
        conn.exec_driver_sql("DROP INDEX uq_jobs_active_dedup")
        conn.exec_driver_sql("ALTER TABLE jobs DROP COLUMN dedup_key")

    assert "outbox_deliveries" not in set(sa.inspect(engine).get_table_names())
    assert "dedup_key" not in _table_columns(engine, "jobs")
    assert "uq_jobs_active_dedup" not in _table_indexes(engine, "jobs")

    config = _alembic_config(db_path, monkeypatch)
    command.stamp(config, "0001_initial")
    command.upgrade(config, "head")

    inspector = sa.inspect(engine)
    assert "outbox_deliveries" in set(inspector.get_table_names())
    assert _table_columns(engine, "outbox_deliveries") == {
        column.name for column in metadata.tables["outbox_deliveries"].columns
    }
    assert "dedup_key" in _table_columns(engine, "jobs")
    assert "uq_jobs_active_dedup" in _table_indexes(engine, "jobs")

    engine.dispose()


def test_catch_up_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "old.db"
    engine = sa.create_engine(f"sqlite:///{db_path}")

    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE outbox_deliveries")

    config = _alembic_config(db_path, monkeypatch)
    command.stamp(config, "0001_initial")
    command.upgrade(config, "head")

    command.downgrade(config, "0001_initial")
    command.upgrade(config, "head")

    assert "outbox_deliveries" in set(sa.inspect(engine).get_table_names())
    engine.dispose()


def test_0003_creates_governance_tables_for_a_database_stamped_at_0002(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "at_0002.db"
    engine = sa.create_engine(f"sqlite:///{db_path}")
    governance = {"tool_approvals", "audit_outbox", "audit_outbox_deliveries"}

    # Simulate a production database migrated to 0002 before these tables were added to the metadata.
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE audit_outbox_deliveries")
        conn.exec_driver_sql("DROP TABLE audit_outbox")
        conn.exec_driver_sql("DROP TABLE tool_approvals")
    assert governance.isdisjoint(set(sa.inspect(engine).get_table_names()))

    config = _alembic_config(db_path, monkeypatch)
    command.stamp(config, "0002_catch_up")
    command.upgrade(config, "head")

    assert governance <= set(sa.inspect(engine).get_table_names())
    assert "ix_tool_approvals_lookup" in _table_indexes(engine, "tool_approvals")
    engine.dispose()


def test_0004_adds_break_glass_columns_for_a_database_stamped_at_0003(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "at_0003.db"
    engine = sa.create_engine(f"sqlite:///{db_path}")

    # Simulate a production database migrated to 0003 before the break-glass columns were added.
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql("ALTER TABLE tool_approvals DROP COLUMN kind")
        conn.exec_driver_sql("ALTER TABLE tool_approvals DROP COLUMN reason")
    columns = _table_columns(engine, "tool_approvals")
    assert "kind" not in columns
    assert "reason" not in columns

    config = _alembic_config(db_path, monkeypatch)
    command.stamp(config, "0003_governance_tables")
    command.upgrade(config, "head")

    columns = _table_columns(engine, "tool_approvals")
    assert "kind" in columns
    assert "reason" in columns
    engine.dispose()


def test_0005_creates_approval_requests_for_a_database_stamped_at_0004(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "at_0004.db"
    engine = sa.create_engine(f"sqlite:///{db_path}")

    # Simulate a production database migrated to 0004 before approval_requests was added.
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE approval_requests")
    assert "approval_requests" not in set(sa.inspect(engine).get_table_names())

    config = _alembic_config(db_path, monkeypatch)
    command.stamp(config, "0004_break_glass_columns")
    command.upgrade(config, "head")

    assert "approval_requests" in set(sa.inspect(engine).get_table_names())
    assert "ix_approval_requests_lookup" in _table_indexes(engine, "approval_requests")
    engine.dispose()


def test_0005_adds_the_lookup_index_when_a_partial_run_left_the_table_without_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "partial.db"
    engine = sa.create_engine(f"sqlite:///{db_path}")

    # A crash between the non-transactional CREATE TABLE and CREATE INDEX leaves the table present
    # without its index; Table.create(checkfirst=True) would then skip it, so the index-ensure must add it.
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP INDEX ix_approval_requests_lookup")
    assert "ix_approval_requests_lookup" not in _table_indexes(engine, "approval_requests")

    config = _alembic_config(db_path, monkeypatch)
    command.stamp(config, "0004_break_glass_columns")
    command.upgrade(config, "head")

    assert "ix_approval_requests_lookup" in _table_indexes(engine, "approval_requests")
    engine.dispose()


def _unique_constraints(engine: sa.Engine, table: str) -> dict[str, set[str]]:
    inspector = sa.inspect(engine)
    return {
        constraint["name"]: set(constraint["column_names"])
        for constraint in inspector.get_unique_constraints(table)
        if constraint["name"]
    }


def _metadata_unique_columns(table: str, name: str) -> set[str]:
    for constraint in metadata.tables[table].constraints:
        if isinstance(constraint, sa.UniqueConstraint) and constraint.name == name:
            return {column.name for column in constraint.columns}
    raise AssertionError(f"no unique constraint {name} on {table}")


def test_catch_up_replaces_a_drifted_unique_constraint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "old.db"
    engine = sa.create_engine(f"sqlite:///{db_path}")

    with engine.begin() as conn:
        conn.exec_driver_sql(
            """
            CREATE TABLE model_call_cache (
                id VARCHAR(64) NOT NULL PRIMARY KEY,
                kind VARCHAR(16) NOT NULL,
                model_id VARCHAR(200) NOT NULL,
                cache_key VARCHAR(64) NOT NULL,
                payload JSON NOT NULL,
                expires_at DATETIME NOT NULL,
                created_at DATETIME NOT NULL,
                CONSTRAINT uq_model_call_cache_key UNIQUE (kind, model_id, cache_key)
            )
            """
        )

    assert _unique_constraints(engine, "model_call_cache")["uq_model_call_cache_key"] == {
        "kind",
        "model_id",
        "cache_key",
    }

    config = _alembic_config(db_path, monkeypatch)
    command.stamp(config, "0001_initial")
    command.upgrade(config, "head")

    assert "workspace_id" in _table_columns(engine, "model_call_cache")
    assert _unique_constraints(engine, "model_call_cache")["uq_model_call_cache_key"] == _metadata_unique_columns(
        "model_call_cache", "uq_model_call_cache_key"
    )

    engine.dispose()


def _check_constraints(engine: sa.Engine, table: str) -> dict[str, str]:
    inspector = sa.inspect(engine)
    return {
        constraint["name"]: " ".join(str(constraint["sqltext"]).split()).casefold()
        for constraint in inspector.get_check_constraints(table)
        if constraint["name"]
    }


def _metadata_check(table: str, name: str) -> str:
    for constraint in metadata.tables[table].constraints:
        if isinstance(constraint, sa.CheckConstraint) and constraint.name == name:
            return " ".join(str(constraint.sqltext).split()).casefold()
    raise AssertionError(f"no check constraint {name} on {table}")


def test_catch_up_replaces_a_drifted_check_constraint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "old.db"
    engine = sa.create_engine(f"sqlite:///{db_path}")

    with engine.begin() as conn:
        conn.exec_driver_sql(
            """
            CREATE TABLE session_turns (
                id VARCHAR(64) NOT NULL PRIMARY KEY,
                session_id VARCHAR(64) NOT NULL,
                role VARCHAR(32) NOT NULL,
                content TEXT NOT NULL,
                metadata JSON NOT NULL,
                created_at DATETIME NOT NULL,
                CONSTRAINT ck_session_turn_role CHECK (role in ('user', 'assistant'))
            )
            """
        )

    assert _check_constraints(engine, "session_turns")["ck_session_turn_role"] == "role in ('user', 'assistant')"

    config = _alembic_config(db_path, monkeypatch)
    command.stamp(config, "0001_initial")
    command.upgrade(config, "head")

    assert _check_constraints(engine, "session_turns")["ck_session_turn_role"] == _metadata_check(
        "session_turns", "ck_session_turn_role"
    )

    engine.dispose()


def test_catch_up_refuses_a_not_null_column_without_a_server_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "old.db"
    engine = sa.create_engine(f"sqlite:///{db_path}")

    guard_table = sa.Table(
        "catch_up_guard_probe",
        metadata,
        sa.Column("id", sa.String(64), primary_key=True),
    )
    try:
        metadata.create_all(engine)
        # Introduce the invariant violation only after the database was built, so the
        # column is present in the live metadata but missing from the connected schema.
        guard_table.append_column(sa.Column("mandatory", sa.String(16), nullable=False))

        config = _alembic_config(db_path, monkeypatch)
        command.stamp(config, "0001_initial")
        with pytest.raises(RuntimeError, match="NOT NULL without a server_default"):
            command.upgrade(config, "head")
    finally:
        metadata.remove(guard_table)

    engine.dispose()


def test_offline_upgrade_head_emits_sql_without_inspecting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "offline.db"
    config = _alembic_config(db_path, monkeypatch)
    buffer = io.StringIO()
    config.output_buffer = buffer

    command.upgrade(config, "head", sql=True)

    assert not db_path.exists()
    sql = buffer.getvalue()
    assert "0002_catch_up" in sql
    # 0001 emits the full current schema offline, so every later revision must no-op offline: each
    # table is created exactly once, never re-emitted by a catch-up or governance revision.
    lowered = sql.lower()
    # A trailing space disambiguates audit_outbox from the audit_outbox_deliveries prefix.
    for table in ("tool_approvals", "audit_outbox", "audit_outbox_deliveries", "approval_requests"):
        assert lowered.count(f"create table {table} ") == 1, table


def test_running_migrations_does_not_disable_refaat_loggers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "logger.db"

    # A refaat logger that exists before alembic runs env.py: with the default disable_existing_loggers,
    # env.py's fileConfig would silently disable it, starving refaat's own logging when a migration runs
    # in-process before the app configures logging.
    logger = get_logger("refaat.migration_probe")
    probe = logging.getLogger("refaat.migration_probe")
    probe.disabled = False

    config = _alembic_config(db_path, monkeypatch)
    command.upgrade(config, "head")

    assert probe.disabled is False

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    probe.addHandler(handler)
    probe.setLevel(logging.INFO)
    probe.propagate = False
    try:
        logger.info("after_migration_probe")
        handler.flush()
    finally:
        probe.removeHandler(handler)
    assert "after_migration_probe" in stream.getvalue()


def test_fresh_upgrade_head_yields_the_full_current_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "fresh.db"
    config = _alembic_config(db_path, monkeypatch)
    command.upgrade(config, "head")

    engine = sa.create_engine(f"sqlite:///{db_path}")
    inspector = sa.inspect(engine)

    actual_tables = set(inspector.get_table_names()) - {"alembic_version"}
    assert actual_tables == set(metadata.tables)

    for name, table in metadata.tables.items():
        assert _table_columns(engine, name) == {column.name for column in table.columns}

    engine.dispose()
