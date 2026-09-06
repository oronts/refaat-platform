from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import dependency_policy_check as policy


def test_bounded_specs_are_accepted() -> None:
    assert policy._is_bounded("alembic>=1.14,<2")
    assert policy._is_bounded("uvicorn[standard]>=0.34,<1")
    assert policy._is_bounded("pkg==1.2.3")
    assert policy._is_bounded("pkg~=1.2")


def test_unbounded_specs_are_rejected() -> None:
    assert not policy._is_bounded("pkg")
    assert not policy._is_bounded("pkg>=1")
    assert not policy._is_bounded("pkg>1.0")


def test_environment_marker_operators_are_not_read_as_bounds() -> None:
    # An operator inside an environment marker must not be mistaken for a version bound.
    assert not policy._is_bounded("pkg; python_version=='3.13'")
    assert not policy._is_bounded("pkg; python_version<'3.13'")
    assert policy._is_bounded("pkg>=1,<2; python_version<'3.13'")


def test_package_hashes_detects_sdist_and_wheels() -> None:
    assert policy._package_hashes({"sdist": {"hash": "sha256:x"}})
    assert policy._package_hashes({"wheels": [{"hash": "sha256:y"}]})
    assert not policy._package_hashes({"wheels": [{"no": "hash"}]})
    assert not policy._package_hashes({})


def test_declared_dependencies_covers_core_and_extras() -> None:
    pyproject = {"project": {"dependencies": ["a>=1,<2"], "optional-dependencies": {"x": ["b>=1,<2"]}}}
    declared = policy._declared_dependencies(pyproject)
    assert ("core", "a>=1,<2") in declared
    assert ("extra:x", "b>=1,<2") in declared


def test_real_project_passes_the_policy() -> None:
    assert policy.check() == 0
