from __future__ import annotations

import json
import platform
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
QUALITY = ROOT / ".quality" / "current"
STATUS_FILE = QUALITY / "status.json"


def command_version(*command: str) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    output = (result.stdout or result.stderr).strip().splitlines()
    return output[0] if output else f"exit {result.returncode}"


def environment_summary() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "uv": command_version("uv", "--version"),
        "docker": command_version("docker", "--version"),
        "helm": command_version("helm", "version", "--short"),
    }


def load_status(path: Path = STATUS_FILE) -> dict[str, int]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Quality status must be an object: {path}")
    return {str(name): int(code) for name, code in raw.items()}


def junit_totals(path: Path | None = None) -> dict[str, int] | None:
    target = path or QUALITY / "junit.xml"
    if not target.exists():
        return None
    root = ET.parse(target).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        return None
    return {
        key: sum(int(float(suite.attrib.get(key, "0"))) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def pytest_summary(path: Path | None = None) -> str:
    target = path or QUALITY / "tests_coverage.log"
    if target.exists():
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        pattern = re.compile(r"(?:\d+\s+(?:passed|failed|error|errors|skipped|deselected|xfailed|xpassed))")
        for line in reversed(lines):
            clean = line.strip("= ")
            if pattern.search(clean):
                return clean
    totals = junit_totals()
    if totals is None:
        return "not available"
    return (
        f"{totals['tests']} tests; {totals['failures']} failures; "
        f"{totals['errors']} errors; {totals['skipped']} skipped"
    )


def coverage_summary(path: Path | None = None) -> str:
    target = path or QUALITY / "coverage.json"
    if not target.exists():
        return "not available"
    raw = json.loads(target.read_text(encoding="utf-8"))
    totals = raw.get("totals", {})
    if not isinstance(totals, dict):
        return "not available"
    percent = totals.get("percent_covered_display", totals.get("percent_covered", "unknown"))
    covered = totals.get("covered_lines", "unknown")
    statements = totals.get("num_statements", "unknown")
    branches = totals.get("num_branches", "unknown")
    return f"{percent}% ({covered}/{statements} statements; {branches} branches measured)"


def json_log(name: str) -> dict[str, Any] | None:
    path = QUALITY / f"{name}.log"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def unavailable_reason(name: str) -> str:
    path = QUALITY / f"{name}.log"
    if not path.exists():
        return "reason not recorded"
    return path.read_text(encoding="utf-8", errors="replace").strip() or "reason not recorded"


def log_excerpt(name: str, limit: int = 45) -> str:
    path = QUALITY / f"{name}.log"
    if not path.exists():
        return "Log file missing."
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-limit:])


def release_decision(status: dict[str, int], required: set[str]) -> tuple[str, list[str], list[str]]:
    failed = [name for name, code in status.items() if code not in (0, 125)]
    missing = [name for name in sorted(required) if status.get(name, 125) == 125]
    decision = "PASS" if status and not failed and not missing else "FAIL"
    return decision, failed, missing
