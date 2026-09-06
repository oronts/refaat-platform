from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import threat_coverage_check as coverage


def _model(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "threat-model.md"
    path.write_text(body, encoding="utf-8")
    return path


def _tests(tmp_path: Path, body: str) -> Path:
    directory = tmp_path / "tests"
    directory.mkdir()
    (directory / "test_x.py").write_text(body, encoding="utf-8")
    return directory


def test_defined_and_covered_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(coverage, "THREAT_MODEL", _model(tmp_path, "| TM-01 | a |\n| TM-02 | b |\n"))
    monkeypatch.setattr(coverage, "TESTS", _tests(tmp_path, "# threat: TM-01\n"))
    assert coverage._defined_ids() == {"TM-01", "TM-02"}
    assert coverage._covered_ids() == {"TM-01"}


def test_uncovered_threat_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(coverage, "THREAT_MODEL", _model(tmp_path, "| TM-01 |\n| TM-02 |\n"))
    monkeypatch.setattr(coverage, "TESTS", _tests(tmp_path, "# threat: TM-01\n"))
    assert coverage.check() == 1


def test_orphan_marker_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(coverage, "THREAT_MODEL", _model(tmp_path, "| TM-01 |\n"))
    monkeypatch.setattr(coverage, "TESTS", _tests(tmp_path, "# threat: TM-01\n# threat: TM-99\n"))
    assert coverage.check() == 1


def test_full_coverage_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(coverage, "THREAT_MODEL", _model(tmp_path, "| TM-01 |\n| TM-02 |\n"))
    monkeypatch.setattr(coverage, "TESTS", _tests(tmp_path, "# threat: TM-01\n# threat: TM-02\n"))
    assert coverage.check() == 0


def test_no_threat_ids_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(coverage, "THREAT_MODEL", _model(tmp_path, "no ids here\n"))
    monkeypatch.setattr(coverage, "TESTS", _tests(tmp_path, ""))
    assert coverage.check() == 1


def test_real_threat_model_is_fully_covered() -> None:
    assert coverage.main() == 0
