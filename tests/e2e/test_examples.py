from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
DATASET = EXAMPLES / "data" / "company"


def test_example_catalog_is_complete_and_runnable_by_release_gate() -> None:
    manifest_path = EXAMPLES / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(manifest) == 28
    assert len({entry["script"] for entry in manifest}) == len(manifest)

    for entry in manifest:
        assert set(entry) >= {"script", "title", "required_keys"}
        assert entry["title"].strip()
        assert entry["required_keys"]
        script = EXAMPLES / entry["script"]
        assert script.is_file(), f"Missing example script: {script}"
        assert script.read_text(encoding="utf-8").startswith("from __future__ import annotations")

    assert (EXAMPLES / "_shared.py").is_file()
    assert (EXAMPLES / "data" / "evaluation_cases.json").is_file()
    assert len(list(DATASET.glob("*.md"))) == 6


def test_example_execution_is_a_required_quality_gate() -> None:
    quality_script = (ROOT / "scripts" / "quality.sh").read_text(encoding="utf-8")
    checker = (ROOT / "scripts" / "examples_check.py").read_text(encoding="utf-8")

    assert "run_gate examples" in quality_script
    assert '"examples"' in quality_script
    assert "manifest.json" in checker
    assert "subprocess.run" in checker
