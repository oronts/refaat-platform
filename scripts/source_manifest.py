from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SOURCE_MANIFEST.sha256"
EXCLUDED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".quality",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "site",
}
EXCLUDED_FILES = {
    ".coverage",
    "QUALITY_REPORT.md",
    "RELEASE_STATUS.md",
    MANIFEST.name,
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".whl", ".zip"}


def source_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if relative.parts and relative.parts[0] == "data":
            continue
        if any(part in EXCLUDED_DIRECTORIES or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.is_symlink():
            raise RuntimeError(f"Source tree contains a symbolic link: {relative}")
        if not path.is_file() or path.name in EXCLUDED_FILES:
            continue
        if path.suffix.casefold() in EXCLUDED_SUFFIXES:
            continue
        if path.name.startswith("refaat-") and path.suffix in {".gz", ".tar"}:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def current_entries() -> dict[str, str]:
    return {path.relative_to(ROOT).as_posix(): digest(path) for path in source_files()}


def parse_manifest() -> dict[str, str]:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Source manifest does not exist: {MANIFEST}")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(MANIFEST.read_text(encoding="utf-8").splitlines(), start=1):
        checksum, separator, relative = line.partition("  ")
        if not separator or len(checksum) != 64 or not relative:
            raise ValueError(f"Invalid manifest entry at line {line_number}")
        if relative in entries:
            raise ValueError(f"Duplicate manifest entry: {relative}")
        entries[relative] = checksum
    return entries


def write_manifest() -> None:
    entries = current_entries()
    content = "".join(f"{checksum}  {relative}\n" for relative, checksum in entries.items())
    MANIFEST.write_text(content, encoding="utf-8")
    print(f"Wrote {MANIFEST.relative_to(ROOT)} for {len(entries)} source files.")


def check_manifest() -> None:
    expected = parse_manifest()
    actual = current_entries()
    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    changed = sorted(path for path in expected.keys() & actual.keys() if expected[path] != actual[path])
    if missing or unexpected or changed:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        if changed:
            details.append(f"changed={changed}")
        raise RuntimeError("Source manifest mismatch: " + "; ".join(details))
    print(f"Source manifest verified for {len(actual)} files.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Write or verify the Refaat source manifest")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true", help="write SOURCE_MANIFEST.sha256")
    action.add_argument("--check", action="store_true", help="verify SOURCE_MANIFEST.sha256")
    args = parser.parse_args()
    try:
        if args.write:
            write_manifest()
        else:
            check_manifest()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
