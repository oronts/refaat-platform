#!/usr/bin/env python3
"""Static low-downtime safety gate over the alembic migration chain.

Each revision under migrations/versions is analyzed for a forward (upgrade) operation that would
break a running old-code deployment during a rolling upgrade: a NOT NULL column added with no
server_default (existing rows cannot be backfilled), a dropped or altered column, table, or
constraint (a contract step that must follow a prior expand release), or an opaque full-schema
create the analyzer cannot reason about. A revision that is genuinely safe (a new-table create, or
an added column that is nullable or carries a server_default) passes. A revision whose risk is
deliberately accepted carries a marker comment ``# migration-safety-ack: <reason>`` and passes with
the reason recorded, so the gate is an enforced contract with a documented, auditable override.

Only the upgrade path is analyzed; a downgrade is a rollback concern governed separately. Pure
stdlib static analysis, no database, no alembic runtime, deterministic. Fail-closed: any migration
the analyzer cannot prove safe, and any parse error, exits non-zero.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = ROOT / "migrations" / "versions"

# Forward operations that drop or rewrite existing schema. Safe only after a prior expand release,
# so an unannotated occurrence in upgrade() is denied.
DESTRUCTIVE = frozenset({"drop_column", "drop_table", "drop_constraint", "drop_index", "alter_column", "rename_table"})
# Operations whose effect the static analyzer cannot enumerate, so it cannot prove safety: a whole
# schema build or teardown, and a raw SQL statement whose text is not inspected.
OPAQUE = frozenset({"create_all", "drop_all", "execute"})

# The reason is confined to the marker's own line: \s would match a newline, letting an empty-reason
# marker capture the following code line and silently accept the revision, so only [ \t] is allowed
# around it and the reason must start and end with a non-space character.
_ACK = re.compile(r"^[ \t]*#[ \t]*migration-safety-ack:[ \t]*(?P<reason>\S.*\S|\S)[ \t]*$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class Finding:
    revision: str
    reason: str


def _ack_reason(source: str) -> str | None:
    match = _ACK.search(source)
    return match.group("reason") if match else None


def _upgrade(tree: ast.Module) -> ast.FunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "upgrade":
            return node
    return None


def _attr_name(call: ast.Call) -> str | None:
    return call.func.attr if isinstance(call.func, ast.Attribute) else None


def _is_column_call(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr == "Column"
    return isinstance(func, ast.Name) and func.id == "Column"


def _unsafe_column(call: ast.Call) -> bool:
    # A column literal is unsafe to add to a populated table when it is NOT NULL and has no
    # server_default, so existing rows have no value for it. nullable defaults to True, so only an
    # explicit nullable=False makes it NOT NULL.
    nullable_false = False
    has_server_default = False
    for keyword in call.keywords:
        if keyword.arg == "nullable" and isinstance(keyword.value, ast.Constant) and keyword.value.value is False:
            nullable_false = True
        if keyword.arg == "server_default":
            has_server_default = True
    return nullable_false and not has_server_default


def _classify(revision: str, source: str) -> Finding | None:
    tree = ast.parse(source)
    upgrade = _upgrade(tree)
    if upgrade is None:
        return Finding(revision, "no upgrade() function found")

    upgrade_calls = [node for node in ast.walk(upgrade) if isinstance(node, ast.Call)]
    upgrade_attrs = {name for call in upgrade_calls if (name := _attr_name(call)) is not None}

    opaque = upgrade_attrs & OPAQUE
    if opaque:
        return Finding(revision, f"opaque operation {sorted(opaque)} the analyzer cannot verify")

    destructive = upgrade_attrs & DESTRUCTIVE
    if destructive:
        return Finding(revision, f"destructive or contract operation {sorted(destructive)} in upgrade()")

    if "add_column" in upgrade_attrs:
        column_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and _is_column_call(node)]
        if not column_calls:
            return Finding(revision, "add_column with no resolvable column literal, so safety cannot be proven")
        unsafe = [call for call in column_calls if _unsafe_column(call)]
        if unsafe:
            return Finding(revision, "adds a NOT NULL column with no server_default to an existing table")

    return None


def check() -> int:
    if not MIGRATIONS.is_dir():
        print(f"migration_safety: migrations directory not found at {MIGRATIONS}")
        return 1

    findings: list[Finding] = []
    accepted: list[tuple[str, str]] = []
    for path in sorted(MIGRATIONS.glob("*.py")):
        if path.name == "__init__.py":
            continue
        revision = path.stem
        try:
            source = path.read_text(encoding="utf-8")
            finding = _classify(revision, source)
        except (OSError, SyntaxError, ValueError) as error:
            findings.append(Finding(revision, f"could not be analyzed: {error}"))
            continue
        if finding is None:
            continue
        reason = _ack_reason(source)
        if reason is not None:
            accepted.append((revision, reason))
        else:
            findings.append(finding)

    for revision, reason in accepted:
        print(f"migration_safety: {revision} accepted: {reason}")

    if findings:
        print("migration_safety: FAIL")
        for finding in findings:
            print(f"  {finding.revision}: {finding.reason}")
        print("Add a server_default or split the change into an expand and a contract revision, or, if the")
        print("risk is understood, annotate the revision with a '# migration-safety-ack: <reason>' comment.")
        return 1

    print(f"migration_safety: pass ({len(accepted)} accepted)")
    return 0


def main(argv: list[str] | None = None) -> int:
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
