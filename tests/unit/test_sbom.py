from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import generate_sbom as sbom


def test_build_sbom_is_sorted_and_counted() -> None:
    built = sbom.build_sbom()
    assert built["schema"] == sbom.SCHEMA
    assert built["generated_from"] == "uv.lock"
    packages = built["packages"]
    assert isinstance(packages, list)
    assert packages
    assert built["package_count"] == len(packages)
    keys = [(entry["name"], entry["version"]) for entry in packages]
    assert keys == sorted(keys)


def test_build_sbom_is_deterministic() -> None:
    assert sbom.build_sbom() == sbom.build_sbom()


def test_every_registry_package_carries_hashes() -> None:
    for entry in sbom.build_sbom()["packages"]:
        if entry["source"].startswith("http"):
            assert entry["hashes"], f"{entry['name']} has no hashes"
            assert all(h.startswith("sha256:") for h in entry["hashes"])


def test_hashes_extracts_sdist_and_wheels() -> None:
    package = {
        "sdist": {"hash": "sha256:aaa"},
        "wheels": [{"hash": "sha256:bbb"}, {"hash": "sha256:aaa"}, {"no": "hash"}],
    }
    assert sbom._hashes(package) == ["sha256:aaa", "sha256:bbb"]


def test_committed_sbom_is_current() -> None:
    assert sbom.main(["--check"]) == 0
