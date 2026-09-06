#!/usr/bin/env python3
"""Generate the committed source provenance attestation.

The provenance binds the release subject (the package name and version) to the exact dependency tree
it was resolved against, by recording the sha256 of the committed software bill of materials. It is a
deterministic projection of committed inputs (pyproject.toml and docs/reference/sbom.json), so it can
be drift-checked in the gate. It deliberately does not record the built wheel or sdist hashes, which
are not deterministic across builds and are added out of band at release time by the signing step,
and it deliberately does not record the source manifest digest, which would be circular since this
file is itself part of the manifest.

``--check`` regenerates in memory and byte-compares to the committed file; ``--write`` rewrites it.
Pure stdlib, offline, deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
SBOM = ROOT / "docs" / "reference" / "sbom.json"
PROVENANCE = ROOT / "docs" / "reference" / "provenance.json"
SCHEMA = "refaat-provenance-1"


def build_provenance() -> dict[str, object]:
    with PYPROJECT.open("rb") as handle:
        project = tomllib.load(handle).get("project", {})
    sbom_digest = hashlib.sha256(SBOM.read_bytes()).hexdigest()
    return {
        "schema": SCHEMA,
        "subject_name": project.get("name"),
        "subject_version": project.get("version"),
        "requires_python": project.get("requires-python"),
        "sbom_sha256": f"sha256:{sbom_digest}",
        "generated_from": ["pyproject.toml", "docs/reference/sbom.json"],
    }


def _render(provenance: dict[str, object]) -> str:
    return json.dumps(provenance, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the committed provenance is stale")
    args = parser.parse_args(argv)

    try:
        rendered = _render(build_provenance())
    except (OSError, tomllib.TOMLDecodeError, ValueError) as error:
        print(f"provenance: could not build: {error}")
        return 1

    if args.check:
        if not PROVENANCE.exists():
            print(f"provenance: committed file missing at {PROVENANCE.relative_to(ROOT)}; run the generator")
            return 1
        if PROVENANCE.read_text(encoding="utf-8") != rendered:
            print(
                "provenance: committed provenance is stale; run scripts/generate_provenance.py after the SBOM changes"
            )
            return 1
        print("provenance: current")
        return 0

    PROVENANCE.write_text(rendered, encoding="utf-8")
    print(f"provenance: wrote {PROVENANCE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
