from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import migration_safety_check as safety


def _reason(source: str) -> str | None:
    finding = safety._classify("rev", source)
    return None if finding is None else finding.reason


def test_not_null_column_without_server_default_is_flagged() -> None:
    source = (
        "import sqlalchemy as sa\n"
        "from alembic import op\n"
        "def upgrade():\n"
        '    op.add_column("t", sa.Column("x", sa.String(), nullable=False))\n'
    )
    assert _reason(source) == "adds a NOT NULL column with no server_default to an existing table"


def test_nullable_column_add_is_safe() -> None:
    source = (
        "import sqlalchemy as sa\n"
        "from alembic import op\n"
        "def upgrade():\n"
        '    op.add_column("t", sa.Column("x", sa.String()))\n'
    )
    assert _reason(source) is None


def test_not_null_column_with_server_default_is_safe() -> None:
    source = (
        "import sqlalchemy as sa\n"
        "from alembic import op\n"
        "def upgrade():\n"
        '    op.add_column("t", sa.Column("x", sa.String(), nullable=False, server_default="n"))\n'
    )
    assert _reason(source) is None


def test_drop_column_in_upgrade_is_destructive() -> None:
    source = "from alembic import op\ndef upgrade():\n    op.drop_column('t', 'x')\n"
    assert _reason(source) == "destructive or contract operation ['drop_column'] in upgrade()"


def test_alter_column_in_upgrade_is_destructive() -> None:
    source = "from alembic import op\ndef upgrade():\n    op.alter_column('t', 'x', nullable=False)\n"
    assert _reason(source) == "destructive or contract operation ['alter_column'] in upgrade()"


def test_create_all_is_opaque() -> None:
    source = (
        "from alembic import op\nfrom x import metadata\ndef upgrade():\n    metadata.create_all(bind=op.get_bind())\n"
    )
    assert _reason(source) is not None
    assert "opaque" in _reason(source)


def test_raw_sql_execute_is_opaque() -> None:
    source = 'from alembic import op\ndef upgrade():\n    op.execute("ALTER TABLE t DROP COLUMN x")\n'
    assert _reason(source) is not None
    assert "opaque" in _reason(source)


def test_new_table_create_is_safe() -> None:
    source = "def upgrade():\n    mytable.create(bind, checkfirst=True)\n"
    assert _reason(source) is None


def test_add_column_without_a_resolvable_literal_is_flagged() -> None:
    source = "from alembic import op\ndef upgrade():\n    for c in cols:\n        op.add_column('t', c)\n"
    assert _reason(source) == "add_column with no resolvable column literal, so safety cannot be proven"


def test_downgrade_destructive_ops_are_ignored() -> None:
    source = (
        "import sqlalchemy as sa\n"
        "from alembic import op\n"
        "def upgrade():\n"
        "    mytable.create(bind, checkfirst=True)\n"
        "def downgrade():\n"
        "    op.drop_table('t')\n"
    )
    assert _reason(source) is None


def test_ack_marker_is_parsed() -> None:
    assert safety._ack_reason("# migration-safety-ack: because reasons\n") == "because reasons"
    assert safety._ack_reason("# migration-safety-ack:   \n") is None
    assert safety._ack_reason("nothing here\n") is None


def test_empty_ack_reason_does_not_capture_the_next_line() -> None:
    # An empty-reason marker must not steal the following code line as its reason and accept the
    # revision. Only a same-line, non-empty reason is a valid acceptance.
    assert safety._ack_reason("# migration-safety-ack:\ndef upgrade(): pass\n") is None
    assert safety._ack_reason("# migration-safety-ack:\n") is None
    assert safety._ack_reason("x = '# migration-safety-ack: forged'\n") is None


def test_empty_ack_marker_does_not_accept_an_unsafe_migration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(safety, "MIGRATIONS", tmp_path)
    _write_migration(
        tmp_path,
        "0001_bad.py",
        "import sqlalchemy as sa\nfrom alembic import op\n"
        "# migration-safety-ack:\n"
        "def upgrade():\n    op.add_column('t', sa.Column('x', sa.String(), nullable=False))\n",
    )
    assert safety.main() == 1


def _write_migration(directory: Path, name: str, source: str) -> None:
    (directory / name).write_text(source, encoding="utf-8")


def test_check_fails_closed_on_an_unannotated_unsafe_migration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(safety, "MIGRATIONS", tmp_path)
    _write_migration(
        tmp_path,
        "0001_bad.py",
        "import sqlalchemy as sa\nfrom alembic import op\n"
        "def upgrade():\n    op.add_column('t', sa.Column('x', sa.String(), nullable=False))\n",
    )
    assert safety.main() == 1


def test_check_passes_an_annotated_unsafe_migration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(safety, "MIGRATIONS", tmp_path)
    _write_migration(
        tmp_path,
        "0001_accepted.py",
        "import sqlalchemy as sa\nfrom alembic import op\n"
        "# migration-safety-ack: a data backfill revision accepts the not-null add deliberately\n"
        "def upgrade():\n    op.add_column('t', sa.Column('x', sa.String(), nullable=False))\n",
    )
    assert safety.main() == 0


def test_real_migration_chain_passes() -> None:
    assert safety.main() == 0
