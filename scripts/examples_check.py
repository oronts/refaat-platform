from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = ROOT / "examples"
MANIFEST_PATH = EXAMPLES_ROOT / "manifest.json"


def load_manifest() -> list[dict[str, Any]]:
    value = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("examples/manifest.json must contain an array of objects")
    return value


def run_example(entry: dict[str, Any], *, timeout_seconds: int = 60) -> dict[str, Any]:
    script = EXAMPLES_ROOT / str(entry["script"])
    if not script.is_file():
        raise FileNotFoundError(f"Example script does not exist: {script.relative_to(ROOT)}")
    # A deep example may exercise the whole catalog over many ingests, so it can
    # declare a larger budget than the shared default; a missing or malformed value
    # keeps the default.
    override = entry.get("timeout_seconds")
    if isinstance(override, int) and override > 0:
        timeout_seconds = override
    env = os.environ.copy()
    source_root = str(ROOT / "src")
    env["PYTHONPATH"] = os.pathsep.join(value for value in (source_root, env.get("PYTHONPATH", "")) if value)
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{script.name} failed with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{script.name} did not emit one JSON object: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{script.name} output must be a JSON object")
    missing = [key for key in entry.get("required_keys", []) if key not in payload]
    if missing:
        raise ValueError(f"{script.name} output is missing keys: {', '.join(missing)}")
    return payload


def main() -> int:
    results: list[dict[str, object]] = []
    try:
        for entry in load_manifest():
            payload = run_example(entry)
            results.append(
                {
                    "script": entry["script"],
                    "title": entry["title"],
                    "status": "passed",
                    "output_keys": sorted(payload),
                }
            )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"Example verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"examples": results, "passed": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
