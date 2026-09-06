from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import vuln_policy_check as vuln

_TODAY = date(2026, 9, 1)


def _allowlist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: object) -> None:
    path = tmp_path / "vuln-allowlist.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(vuln, "ALLOWLIST", path)


def _entry(**over: object) -> dict[str, object]:
    entry = {
        "id": "GHSA-test",
        "package": "somepkg",
        "reason": "not reachable in this library's usage",
        "accepted_on": "2026-08-01",
        "expires": "2026-10-01",
    }
    entry.update(over)
    return entry


def test_empty_allowlist_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _allowlist(tmp_path, monkeypatch, {"schema": vuln.SCHEMA, "accepted": []})
    assert vuln.check(today=_TODAY) == 0


def test_valid_current_acceptance_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _allowlist(tmp_path, monkeypatch, {"schema": vuln.SCHEMA, "accepted": [_entry()]})
    assert vuln.check(today=_TODAY) == 0


def test_lapsed_acceptance_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _allowlist(tmp_path, monkeypatch, {"schema": vuln.SCHEMA, "accepted": [_entry(expires="2026-08-15")]})
    assert vuln.check(today=_TODAY) == 1


def test_missing_field_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _allowlist(tmp_path, monkeypatch, {"schema": vuln.SCHEMA, "accepted": [{"id": "GHSA-x"}]})
    assert vuln.check(today=_TODAY) == 1


def test_non_iso_date_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _allowlist(tmp_path, monkeypatch, {"schema": vuln.SCHEMA, "accepted": [_entry(expires="soon")]})
    assert vuln.check(today=_TODAY) == 1


def test_over_long_window_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _allowlist(tmp_path, monkeypatch, {"schema": vuln.SCHEMA, "accepted": [_entry(expires="2027-08-01")]})
    assert vuln.check(today=_TODAY) == 1


def test_expiry_not_after_acceptance_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _allowlist(tmp_path, monkeypatch, {"schema": vuln.SCHEMA, "accepted": [_entry(expires="2026-07-01")]})
    assert vuln.check(today=_TODAY) == 1


def test_wrong_schema_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _allowlist(tmp_path, monkeypatch, {"schema": "wrong", "accepted": []})
    assert vuln.check(today=_TODAY) == 1


def test_missing_file_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vuln, "ALLOWLIST", tmp_path / "does-not-exist.json")
    assert vuln.check(today=_TODAY) == 1


def test_committed_allowlist_is_consistent() -> None:
    assert vuln.main() == 0
