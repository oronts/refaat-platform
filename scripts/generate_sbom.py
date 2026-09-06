#!/usr/bin/env python3
"""Generate the committed software bill of materials from the uv lockfile.

The SBOM is a deterministic, sorted projection of uv.lock: every resolved package with its version,
its source registry, and the sha256 hashes of its distributions. It is committed so the dependency
tree is auditable, and drift-checked in the gate so the committed file can never diverge from the
lockfile. ``--check`` regenerates the SBOM in memory and byte-compares it to the committed file;
``--write`` rewrites it. Pure stdlib, offline, deterministic.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCKFILE = ROOT / "uv.lock"
SBOM = ROOT / "docs" / "reference" / "sbom.json"
SCHEMA = "refaat-sbom-1"


def _hashes(package: dict[str, object]) -> list[str]:
    found: set[str] = set()
    sdist = package.get("sdist")
    if isinstance(sdist, dict) and isinstance(sdist.get("hash"), str):
        found.add(sdist["hash"])
    wheels = package.get("wheels")
    if isinstance(wheels, list):
        for wheel in wheels:
            if isinstance(wheel, dict) and isinstance(wheel.get("hash"), str):
                found.add(wheel["hash"])
    return sorted(found)


def _source(package: dict[str, object]) -> str:
    source = package.get("source")
    if isinstance(source, dict):
        for key in ("registry", "git", "url", "path", "editable", "virtual"):
            value = source.get(key)
            if isinstance(value, str):
                return f"{key}:{value}" if key != "registry" else value
    return "unknown"


def build_sbom() -> dict[str, object]:
    with LOCKFILE.open("rb") as handle:
        lock = tomllib.load(handle)
    packages = []
    for package in lock.get("package", []):
        if not isinstance(package, dict):
            continue
        name = package.get("name")
        version = package.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            continue
        packages.append(
            {
                "name": name,
                "version": version,
                "source": _source(package),
                "hashes": _hashes(package),
            }
        )
    packages.sort(key=lambda entry: (entry["name"], entry["version"]))
    return {"schema": SCHEMA, "generated_from": "uv.lock", "package_count": len(packages), "packages": packages}


def _render(sbom: dict[str, object]) -> str:
    return json.dumps(sbom, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the committed SBOM is stale")
    args = parser.parse_args(argv)

    try:
        rendered = _render(build_sbom())
    except (OSError, tomllib.TOMLDecodeError, ValueError) as error:
        print(f"sbom: could not build from uv.lock: {error}")
        return 1

    if args.check:
        if not SBOM.exists():
            print(f"sbom: committed SBOM missing at {SBOM.relative_to(ROOT)}; run scripts/generate_sbom.py")
            return 1
        current = SBOM.read_text(encoding="utf-8")
        if current != rendered:
            print("sbom: committed SBOM is stale; run scripts/generate_sbom.py to regenerate it from uv.lock")
            return 1
        print(f"sbom: current ({build_sbom()['package_count']} packages)")
        return 0

    SBOM.write_text(rendered, encoding="utf-8")
    print(f"sbom: wrote {SBOM.relative_to(ROOT)} ({build_sbom()['package_count']} packages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
