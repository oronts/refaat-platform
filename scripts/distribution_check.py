from __future__ import annotations

import json
import re
import tarfile
import zipfile
from pathlib import Path

from refaat.version import __version__

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / ".quality" / "current" / "dist"


def _single(pattern: str) -> Path:
    matches = sorted(DIST.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern} artifact, found {len(matches)}")
    return matches[0]


def _forbidden(names: list[str]) -> list[str]:
    return [
        name
        for name in names
        if "__pycache__" in name
        or name.endswith((".pyc", ".pyo"))
        or "/.quality/" in name
        or "/.venv/" in name
        or "/build/" in name
    ]


def _wheel_members(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    forbidden = _forbidden(names)
    if forbidden:
        raise RuntimeError(f"Wheel contains generated files: {forbidden[:5]}")
    required = {
        "refaat/__init__.py",
        "refaat/version.py",
        "refaat/py.typed",
        "refaat/api/app.py",
        "refaat/application/runtime.py",
    }
    missing = required - set(names)
    if missing:
        raise RuntimeError(f"Wheel is missing files: {sorted(missing)}")
    metadata = [name for name in names if name.endswith(".dist-info/METADATA")]
    if len(metadata) != 1:
        raise RuntimeError("Wheel must contain exactly one METADATA file")
    return set(names)


def _sdist_members(path: Path) -> set[str]:
    with tarfile.open(path, "r:gz") as archive:
        names = archive.getnames()
    forbidden = _forbidden(names)
    if forbidden:
        raise RuntimeError(f"Source distribution contains generated files: {forbidden[:5]}")
    root = f"refaat-{__version__}"
    required = {
        f"{root}/pyproject.toml",
        f"{root}/README.md",
        f"{root}/SOURCE_MANIFEST.sha256",
        f"{root}/CHANGELOG.md",
        f"{root}/LICENSE",
        f"{root}/NOTICE",
        f"{root}/MANIFEST.in",
        f"{root}/Dockerfile",
        f"{root}/compose.yaml",
        f"{root}/mkdocs.yml",
        f"{root}/docs/index.md",
        f"{root}/docs/reference/openapi.json",
        f"{root}/examples/manifest.json",
        f"{root}/examples/01_local_rag_graph_cache.py",
        f"{root}/examples/06_live_api_sdk.py",
        f"{root}/examples/data/company/02_release_policy.md",
        f"{root}/examples/data/evaluation_cases.json",
        f"{root}/tests/e2e/test_api.py",
        f"{root}/tests/integration/test_context_cache_isolation.py",
        f"{root}/migrations/env.py",
        f"{root}/deploy/helm/refaat/Chart.yaml",
        f"{root}/scripts/quality.sh",
        f"{root}/.github/workflows/quality.yml",
    }
    missing = required - set(names)
    if missing:
        raise RuntimeError(f"Source distribution is missing files: {sorted(missing)}")
    return set(names)


def main() -> int:
    wheel = _single(f"refaat-{__version__}-*.whl")
    source = _single(f"refaat-{__version__}.tar.gz")
    wheel_names = _wheel_members(wheel)
    source_names = _sdist_members(source)

    wheel_version = re.search(r"refaat-([^-]+)-", wheel.name)
    if wheel_version is None or wheel_version.group(1) != __version__:
        raise RuntimeError("Wheel filename version does not match refaat.__version__")

    print(
        json.dumps(
            {
                "version": __version__,
                "wheel": wheel.name,
                "wheel_files": len(wheel_names),
                "sdist": source.name,
                "sdist_files": len(source_names),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
