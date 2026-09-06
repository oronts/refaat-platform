from __future__ import annotations

import shutil
from pathlib import Path

from setuptools.build_meta import build_sdist, build_wheel

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / ".quality" / "current" / "dist"
GENERATED = (ROOT / "build", ROOT / "src" / "refaat.egg-info")


def _clean_generated() -> None:
    for path in GENERATED:
        shutil.rmtree(path, ignore_errors=True)


def main() -> int:
    shutil.rmtree(DIST, ignore_errors=True)
    DIST.mkdir(parents=True)
    _clean_generated()
    try:
        wheel = build_wheel(str(DIST), config_settings={})
        source = build_sdist(str(DIST), config_settings={})
        for name in (wheel, source):
            if not (DIST / name).is_file():
                raise RuntimeError(f"Build backend did not create {name}")
            print(name)
    finally:
        _clean_generated()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
