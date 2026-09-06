#!/usr/bin/env python3
"""Traceability gate for the executable threat model.

Every threat named in docs/security/threat-model.md carries a stable id (TM-NN), and every id must be
covered by at least one test that marks itself with ``# threat: TM-NN``. The gate fails closed on an
uncovered threat (a control asserted nowhere) and on an orphan marker (a test claiming a threat id
the model does not define, which catches a typo or a deleted threat). Pure stdlib, offline,
deterministic.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THREAT_MODEL = ROOT / "docs" / "security" / "threat-model.md"
TESTS = ROOT / "tests"

_ID = re.compile(r"\bTM-\d{2,}\b")
_MARKER = re.compile(r"#\s*threat:\s*(TM-\d{2,})")
# The gate's own unit test constructs synthetic marker strings to exercise this scanner, so it is
# excluded to keep those fixtures from being read as real coverage or orphan markers.
_EXCLUDE = frozenset({"test_threat_coverage.py"})


def _defined_ids() -> set[str]:
    return set(_ID.findall(THREAT_MODEL.read_text(encoding="utf-8")))


def _covered_ids() -> set[str]:
    covered: set[str] = set()
    for path in TESTS.rglob("*.py"):
        if path.name in _EXCLUDE:
            continue
        covered.update(_MARKER.findall(path.read_text(encoding="utf-8")))
    return covered


def check() -> int:
    if not THREAT_MODEL.is_file():
        print(f"threat_coverage: threat model not found at {THREAT_MODEL}")
        return 1

    defined = _defined_ids()
    if not defined:
        print("threat_coverage: no threat ids (TM-NN) found in the threat model")
        return 1
    covered = _covered_ids()

    uncovered = sorted(defined - covered)
    orphans = sorted(covered - defined)

    if uncovered or orphans:
        print("threat_coverage: FAIL")
        for threat in uncovered:
            print(f"  {threat}: defined in the threat model but no test marks '# threat: {threat}'")
        for threat in orphans:
            print(f"  {threat}: a test marks it but the threat model does not define it")
        return 1

    print(f"threat_coverage: pass ({len(defined)} threats, each covered by a marked test)")
    return 0


def main(argv: list[str] | None = None) -> int:
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
