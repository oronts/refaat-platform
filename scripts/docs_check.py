from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
REQUIRED_README_TEXT = (
    "REFAAT — Retrieval & Evidence Foundation for Adaptive Agentic Tooling",
    "**R**etrieval",
    "**E**vidence",
    "**F**oundation for",
    "**A**daptive",
    "**A**gentic",
    "**T**ooling",
)


def markdown_files() -> list[Path]:
    values = [ROOT / "README.md"]
    values.extend((ROOT / "docs").rglob("*.md"))
    values.extend(ROOT / name for name in ("CONTRIBUTING.md", "SECURITY.md", "CHANGELOG.md"))
    return sorted(path for path in set(values) if path.exists())


def local_target(path: Path, raw_target: str) -> Path | None:
    target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
    if not target or target.startswith(("#", "http://", "https://", "mailto:")):
        return None
    target = target.split("#", 1)[0]
    if not target:
        return None
    return (path.parent / target).resolve()


def nav_paths(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [path for item in value for path in nav_paths(item)]
    if isinstance(value, dict):
        return [path for item in value.values() for path in nav_paths(item)]
    return []


def main() -> int:
    problems: list[str] = []
    for path in markdown_files():
        content = path.read_text(encoding="utf-8")
        for match in LINK_PATTERN.finditer(content):
            target = local_target(path, match.group(1))
            if target is not None and not target.exists():
                line = content.count("\n", 0, match.start()) + 1
                problems.append(f"{path.relative_to(ROOT)}:{line}: missing local link target {match.group(1)!r}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for text in REQUIRED_README_TEXT:
        if text not in readme:
            problems.append(f"README.md: required branding text missing: {text!r}")

    mkdocs_path = ROOT / "mkdocs.yml"
    config = yaml.load(mkdocs_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    for item in nav_paths(config.get("nav", [])):
        target = ROOT / "docs" / item
        if not target.exists():
            problems.append(f"mkdocs.yml: nav target does not exist: {item}")

    if problems:
        print("Documentation check failed:")
        print("\n".join(f"- {problem}" for problem in problems))
        return 1
    print(f"Documentation check passed for {len(markdown_files())} Markdown files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
