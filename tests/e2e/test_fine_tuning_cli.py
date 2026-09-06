from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from refaat.cli import app
from refaat.config import get_settings

pytestmark = pytest.mark.e2e


def _configure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    values = {
        "REFAAT_ENVIRONMENT": "test",
        "REFAAT_DATABASE_URL": f"sqlite:///{tmp_path / 'cli.db'}",
        "REFAAT_AUTO_CREATE_SCHEMA": "true",
        "REFAAT_BLOB_ROOT": str(tmp_path / "blobs"),
        "REFAAT_CONNECTOR_FILE_ROOT": str(tmp_path / "connectors"),
        "REFAAT_SKILL_ROOT": str(tmp_path / "skills"),
        "REFAAT_API_KEY_PEPPER": "cli-test-pepper-with-enough-entropy",
        "REFAAT_LOG_JSON": "false",
        "REFAAT_RATE_LIMIT_REQUESTS": "10000",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()


def _invoke(runner: CliRunner, args: list[str]) -> object:
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout.strip())


def test_fine_tune_cli_flow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure(monkeypatch, tmp_path)
    runner = CliRunner()

    initialized = _invoke(runner, ["init", "--slug", "cli-ft", "--name", "CLI FT"])
    workspace_id = str(initialized["workspace_id"])

    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{"query": "release sign-off", "reference_answer": "A checklist."}]), encoding="utf-8")

    requested = _invoke(runner, ["fine-tune", "request", workspace_id, "refaat-base", str(cases)])
    run_id = str(requested["run_id"])

    # Governance: the run refuses to execute before it is approved.
    unapproved = runner.invoke(app, ["fine-tune", "run", workspace_id, run_id])
    assert unapproved.exit_code != 0

    approved = _invoke(runner, ["fine-tune", "approve", workspace_id, run_id, "--approver", "founder"])
    assert approved["approved"] is True

    _invoke(runner, ["fine-tune", "enqueue", workspace_id, run_id])
    assert _invoke(runner, ["worker", "--once"]) == {"worked": True}

    status = _invoke(runner, ["fine-tune", "status", workspace_id, run_id])
    assert status["status"] == "succeeded"
    assert str(status["result_model_ref"]).startswith("ft:refaat:")
