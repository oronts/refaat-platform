#!/usr/bin/env python3
"""Enforce the dependency supply-chain policy.

Two invariants, checked statically and offline from pyproject.toml and the uv lockfile:

- Every declared dependency, in the core set and in each optional extra, carries both a lower and an
  upper version bound. A library must not float unbounded above a major version, so a bare name or a
  ``>=`` with no ``<`` ceiling is denied.
- The lockfile exists and every package resolved from a registry is hash-locked, so a reproducible
  install cannot silently pull an unverified artifact.

Pure stdlib, deterministic. Fail-closed: any policy breach, a missing lockfile, or a parse error
exits non-zero.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
LOCKFILE = ROOT / "uv.lock"


def _is_bounded(spec: str) -> bool:
    # Analyze only the version constraint, before any environment marker, so an operator inside a
    # marker (python_version=='3.13') cannot be read as a version bound. A version operator only ever
    # appears in the constraint, never the package name. == and ~= each fix both a floor and a
    # ceiling; otherwise both >= and < must be present.
    constraint = spec.split(";", 1)[0]
    has_upper = "<" in constraint or "==" in constraint or "~=" in constraint
    has_lower = ">=" in constraint or "==" in constraint or "~=" in constraint
    return has_upper and has_lower


def _declared_dependencies(pyproject: dict[str, object]) -> list[tuple[str, str]]:
    project = pyproject.get("project")
    project = project if isinstance(project, dict) else {}
    result: list[tuple[str, str]] = []
    core = project.get("dependencies")
    if isinstance(core, list):
        result.extend(("core", str(spec)) for spec in core)
    optional = project.get("optional-dependencies")
    if isinstance(optional, dict):
        for group, specs in optional.items():
            if isinstance(specs, list):
                result.extend((f"extra:{group}", str(spec)) for spec in specs)
    return result


def _package_hashes(package: dict[str, object]) -> bool:
    sdist = package.get("sdist")
    if isinstance(sdist, dict) and isinstance(sdist.get("hash"), str):
        return True
    wheels = package.get("wheels")
    if isinstance(wheels, list):
        return any(isinstance(wheel, dict) and isinstance(wheel.get("hash"), str) for wheel in wheels)
    return False


def check() -> int:
    problems: list[str] = []

    try:
        with PYPROJECT.open("rb") as handle:
            pyproject = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        print(f"dependency_policy: could not read pyproject.toml: {error}")
        return 1

    declared = _declared_dependencies(pyproject)
    if not declared:
        problems.append("no dependencies declared in pyproject.toml, so the policy cannot be verified")
    for group, spec in declared:
        if not _is_bounded(spec):
            problems.append(f"unbounded dependency in {group}: '{spec}' needs both a lower and an upper bound")

    if not LOCKFILE.exists():
        problems.append("uv.lock is missing, so the dependency tree is not hash-locked")
    else:
        try:
            with LOCKFILE.open("rb") as handle:
                lock = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as error:
            print(f"dependency_policy: could not read uv.lock: {error}")
            return 1
        for package in lock.get("package", []):
            if not isinstance(package, dict):
                continue
            source = package.get("source")
            if isinstance(source, dict) and isinstance(source.get("registry"), str) and not _package_hashes(package):
                problems.append(f"registry package not hash-locked: {package.get('name')} {package.get('version')}")

    if problems:
        print("dependency_policy: FAIL")
        for problem in problems:
            print(f"  {problem}")
        return 1

    print(f"dependency_policy: pass ({len(declared)} declared dependencies bounded, lockfile hash-locked)")
    return 0


def main(argv: list[str] | None = None) -> int:
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
