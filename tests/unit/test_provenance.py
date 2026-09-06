from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import generate_provenance as prov


def test_build_provenance_shape() -> None:
    built = prov.build_provenance()
    assert built["schema"] == prov.SCHEMA
    assert built["subject_name"] == "refaat"
    assert isinstance(built["subject_version"], str)
    assert built["subject_version"]
    assert str(built["sbom_sha256"]).startswith("sha256:")


def test_build_provenance_is_deterministic() -> None:
    assert prov.build_provenance() == prov.build_provenance()


def test_sbom_digest_matches_the_committed_sbom() -> None:
    expected = "sha256:" + hashlib.sha256(prov.SBOM.read_bytes()).hexdigest()
    assert prov.build_provenance()["sbom_sha256"] == expected


def test_committed_provenance_is_current() -> None:
    assert prov.main(["--check"]) == 0
