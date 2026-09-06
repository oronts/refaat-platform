from __future__ import annotations

import ast
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = (ROOT / "src", ROOT / "tests", ROOT / "scripts", ROOT / "examples", ROOT / "migrations")
IGNORED_IMPORTS = {"annotations"}


@dataclass(frozen=True, slots=True)
class FunctionBody:
    path: Path
    name: str
    line: int
    digest: str


def python_files() -> list[Path]:
    return sorted(
        path for root in SCAN_ROOTS if root.exists() for path in root.rglob("*.py") if "__pycache__" not in path.parts
    )


def imported_names(tree: ast.AST) -> list[tuple[str, int]]:
    values: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                values.append((alias.asname or alias.name.split(".", 1)[0], node.lineno))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    values.append((alias.asname or alias.name, node.lineno))
    return values


def used_names(tree: ast.AST) -> set[str]:
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def meaningful_body(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.stmt]:
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body.pop(0)
    return body


def function_bodies(path: Path, tree: ast.AST) -> list[FunctionBody]:
    values: list[FunctionBody] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = meaningful_body(node)
        if len(body) < 4 or node.name.startswith("__"):
            continue
        normalized = ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        values.append(FunctionBody(path, node.name, node.lineno, digest))
    return values


def unreachable_lines(tree: ast.AST) -> list[int]:
    found: list[int] = []

    def inspect_block(statements: list[ast.stmt]) -> None:
        terminated = False
        for statement in statements:
            if terminated:
                found.append(statement.lineno)
            if isinstance(statement, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                terminated = True
            for field in ("body", "orelse", "finalbody"):
                child = getattr(statement, field, None)
                if isinstance(child, list):
                    inspect_block(child)
            handlers = getattr(statement, "handlers", None)
            if isinstance(handlers, list):
                for handler in handlers:
                    inspect_block(handler.body)

    if isinstance(tree, ast.Module):
        inspect_block(tree.body)
    return found


def main() -> int:
    problems: list[str] = []
    bodies: list[FunctionBody] = []
    files = python_files()
    for path in files:
        relative = path.relative_to(ROOT)
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(relative))
        except SyntaxError as exc:
            problems.append(f"{relative}:{exc.lineno}: {exc.msg}")
            continue
        if path.name != "__init__.py":
            names = used_names(tree)
            for name, line in imported_names(tree):
                if name not in names and name not in IGNORED_IMPORTS:
                    problems.append(f"{relative}:{line}: imported name {name!r} is unused")
        for line in unreachable_lines(tree):
            problems.append(f"{relative}:{line}: unreachable statement")
        for marker in ("TO" + "DO", "FIX" + "ME", "HA" + "CK"):
            if marker in source:
                problems.append(f"{relative}: unresolved {marker} marker")
        bodies.extend(function_bodies(path, tree))

    by_digest: dict[str, list[FunctionBody]] = {}
    for body in bodies:
        by_digest.setdefault(body.digest, []).append(body)
    for matches in by_digest.values():
        locations = {(match.path, match.line) for match in matches}
        if len(locations) < 2:
            continue
        display = ", ".join(f"{match.path.relative_to(ROOT)}:{match.line} ({match.name})" for match in matches)
        problems.append(f"duplicate non-trivial function body: {display}")

    if problems:
        print("Code audit failed:")
        print("\n".join(f"- {problem}" for problem in problems))
        return 1
    print(f"Code audit passed for {len(files)} Python files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
