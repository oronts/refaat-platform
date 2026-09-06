from __future__ import annotations

import json
import tomllib
from pathlib import Path

import yaml

from refaat.version import __version__

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    chart = yaml.safe_load((ROOT / "deploy" / "helm" / "refaat" / "Chart.yaml").read_text())
    openapi = json.loads((ROOT / "docs" / "reference" / "openapi.json").read_text())
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    versions = {
        "package": pyproject["project"]["version"],
        "runtime": __version__,
        "openapi": openapi["info"]["version"],
        "helm_chart": chart["version"],
        "helm_app": str(chart["appVersion"]),
    }
    mismatches = {name: value for name, value in versions.items() if value != __version__}
    if mismatches:
        print(json.dumps({"expected": __version__, "mismatches": mismatches}, indent=2))
        return 1
    if f"Release status: `{__version__}`" not in readme:
        raise RuntimeError("README release status does not match the package version")
    if f"## {__version__} - " not in changelog:
        raise RuntimeError("CHANGELOG has no entry for the package version")
    print(json.dumps({"version": __version__, "surfaces": sorted(versions)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
