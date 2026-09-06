from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOTS = ("src", "tests", "scripts", "migrations", "examples")
TEXT_ROOTS = (".github", "deploy", "docs", "examples", "scripts")
TEXT_SUFFIXES = {".cfg", ".ini", ".json", ".jsonl", ".md", ".sh", ".toml", ".tpl", ".txt", ".yaml", ".yml"}
TEXT_FILES = (
    "README.md",
    "MANIFEST.in",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "NOTICE",
    "pyproject.toml",
    "compose.yaml",
    "Dockerfile",
    "Makefile",
    "mkdocs.yml",
)
MAX_PYTHON_LINE = 120
FORBIDDEN_NAMES = ("ary" + "nth", "prime" + "o")


def tracked_files() -> list[Path]:
    values: list[Path] = []
    for root_name in PYTHON_ROOTS:
        root = ROOT / root_name
        if root.exists():
            values.extend(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    values.extend(ROOT / name for name in TEXT_FILES if (ROOT / name).exists())
    for root_name in TEXT_ROOTS:
        root = ROOT / root_name
        if not root.exists():
            continue
        values.extend(path for path in root.rglob("*") if path.is_file() and path.suffix.casefold() in TEXT_SUFFIXES)
    return sorted(set(values))


def main() -> int:
    problems: list[str] = []
    for path in tracked_files():
        relative = path.relative_to(ROOT)
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            problems.append(f"{relative}: is not valid UTF-8")
            continue
        if content and not content.endswith("\n"):
            problems.append(f"{relative}: missing final newline")
        for line_number, line in enumerate(content.splitlines(), start=1):
            if line.rstrip() != line:
                problems.append(f"{relative}:{line_number}: trailing whitespace")
            if path.suffix == ".py":
                if "\t" in line:
                    problems.append(f"{relative}:{line_number}: tab character in Python source")
                if len(line) > MAX_PYTHON_LINE:
                    problems.append(
                        f"{relative}:{line_number}: line has {len(line)} characters; maximum is {MAX_PYTHON_LINE}"
                    )
        lowered = content.casefold()
        for name in FORBIDDEN_NAMES:
            if name in lowered:
                problems.append(f"{relative}: contains retired project name {name!r}")
    if problems:
        print("Style check failed:")
        print("\n".join(f"- {problem}" for problem in problems))
        return 1
    print(f"Style check passed for {len(tracked_files())} text/source files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
