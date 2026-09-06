#!/usr/bin/env python3
"""Network dependency audit that honors the committed advisory allowlist.

Runs pip-audit and ignores exactly the advisories accepted in docs/reference/vuln-allowlist.json, so
that file is the single source of truth for both this gate and the offline vuln_policy gate. The
vuln_policy gate independently enforces that every acceptance carries a reason and an unexpired,
bounded expiry, so an ignored advisory here can never be a permanent, undocumented suppression.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ROOT / "docs" / "reference" / "vuln-allowlist.json"


def _accepted_ids() -> list[str]:
    try:
        data = json.loads(ALLOWLIST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    accepted = data.get("accepted") if isinstance(data, dict) else None
    if not isinstance(accepted, list):
        return []
    ids: list[str] = []
    for entry in accepted:
        if isinstance(entry, dict) and isinstance(entry.get("id"), str) and entry["id"].strip():
            ids.append(entry["id"].strip())
    return ids


def main() -> int:
    command = [sys.executable, "-m", "pip_audit"]
    for advisory in _accepted_ids():
        command += ["--ignore-vuln", advisory]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
