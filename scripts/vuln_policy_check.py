#!/usr/bin/env python3
"""Vulnerability process gate: the accepted-advisory allowlist stays consistent.

The network audit is the separate pip-audit gate. This gate is the offline, deterministic half: it
verifies the committed advisory allowlist (docs/reference/vuln-allowlist.json) is well formed and
that no accepted advisory has lapsed. Each accepted entry temporarily exempts a known advisory with a
recorded reason and an expiry, so an acceptance cannot be permanent: once it expires the gate fails
until the advisory is fixed or the acceptance is deliberately renewed after re-review.

Fail-closed: a missing or malformed allowlist, a malformed entry, an out-of-order or lapsed expiry,
or an over-long acceptance window exits non-zero.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ROOT / "docs" / "reference" / "vuln-allowlist.json"
SCHEMA = "refaat-vuln-allowlist-1"
MAX_ACCEPTANCE_DAYS = 180
_REQUIRED = ("id", "package", "reason", "accepted_on", "expires")


def _parse_date(value: object) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _entry_problems(index: int, entry: object) -> list[str]:
    if not isinstance(entry, dict):
        return [f"accepted[{index}] is not an object"]
    problems: list[str] = []
    for field in _REQUIRED:
        value = entry.get(field)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"accepted[{index}] field '{field}' is missing or empty")
    return problems


def check(today: date | None = None) -> int:
    today = today or date.today()
    if not ALLOWLIST.is_file():
        display = ALLOWLIST.relative_to(ROOT) if ALLOWLIST.is_relative_to(ROOT) else ALLOWLIST
        print(f"vuln_policy: allowlist missing at {display}")
        return 1
    try:
        data = json.loads(ALLOWLIST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"vuln_policy: could not read the allowlist: {error}")
        return 1

    problems: list[str] = []
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        problems.append(f"allowlist must be an object with schema '{SCHEMA}'")
    accepted = data.get("accepted") if isinstance(data, dict) else None
    if not isinstance(accepted, list):
        problems.append("allowlist 'accepted' must be a list")
        accepted = []

    for index, entry in enumerate(accepted):
        entry_problems = _entry_problems(index, entry)
        problems.extend(entry_problems)
        if entry_problems or not isinstance(entry, dict):
            continue
        accepted_on = _parse_date(entry.get("accepted_on"))
        expires = _parse_date(entry.get("expires"))
        label = f"accepted[{index}] ({entry.get('id')})"
        if accepted_on is None or expires is None:
            problems.append(f"{label} has a non-ISO accepted_on or expires date")
            continue
        if expires <= accepted_on:
            problems.append(f"{label} expires on or before it was accepted")
        if expires < today:
            problems.append(f"{label} acceptance lapsed on {expires}; fix the advisory or renew after re-review")
        if (expires - accepted_on).days > MAX_ACCEPTANCE_DAYS:
            problems.append(f"{label} acceptance window exceeds {MAX_ACCEPTANCE_DAYS} days")

    if problems:
        print("vuln_policy: FAIL")
        for problem in problems:
            print(f"  {problem}")
        return 1

    print(f"vuln_policy: pass ({len(accepted)} accepted advisories, none lapsed)")
    return 0


def main(argv: list[str] | None = None) -> int:
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
