from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from refaat.cli import app
from refaat.config import get_settings

pytestmark = pytest.mark.e2e


def parse_json(output: str) -> object:
    return json.loads(output.strip())


def configure_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    values = {
        "REFAAT_ENVIRONMENT": "test",
        "REFAAT_DATABASE_URL": f"sqlite:///{tmp_path / 'cli.db'}",
        "REFAAT_AUTO_CREATE_SCHEMA": "true",
        "REFAAT_BLOB_ROOT": str(tmp_path / "blobs"),
        "REFAAT_CONNECTOR_FILE_ROOT": str(tmp_path / "connectors"),
        "REFAAT_SKILL_ROOT": str(tmp_path / "skills"),
        "REFAAT_API_KEY_PEPPER": "cli-test-pepper-with-enough-entropy",
        "REFAAT_LOG_JSON": "false",
        "REFAAT_DEFAULT_CHUNK_TOKENS": "64",
        "REFAAT_CHUNK_OVERLAP_TOKENS": "12",
        "REFAAT_QUERY_TOKEN_BUDGET": "600",
        "REFAAT_RATE_LIMIT_REQUESTS": "10000",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()


def invoke(runner: CliRunner, args: list[str]) -> object:
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    return parse_json(result.stdout)


def test_serve_passes_configured_workers_to_uvicorn(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure_cli(monkeypatch, tmp_path)
    monkeypatch.setenv("REFAAT_API_WORKERS", "3")
    get_settings.cache_clear()
    recorded: dict[str, object] = {}
    fake_uvicorn = SimpleNamespace(run=lambda *args, **kwargs: recorded.update(args=args, kwargs=kwargs))
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)

    result = CliRunner().invoke(app, ["serve"])

    assert result.exit_code == 0, result.output
    assert recorded["args"] == ("refaat.api.app:create_app",)
    assert recorded["kwargs"]["workers"] == 3, "the configured worker count is passed through to uvicorn"


def test_cli_full_local_workflow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure_cli(monkeypatch, tmp_path)
    runner = CliRunner()

    initialized = invoke(runner, ["init", "--slug", "cli-acme", "--name", "CLI Acme"])
    workspace_id = str(initialized["workspace_id"])
    assert str(initialized["api_key"]).startswith("refaat_")

    corpus = invoke(runner, ["corpus", "create", workspace_id, "engineering"])
    corpus_id = str(corpus["id"])
    corpora = invoke(runner, ["corpus", "list", workspace_id])
    assert [item["id"] for item in corpora] == [corpus_id]

    policy = tmp_path / "release.md"
    policy.write_text(
        "Production releases require peer review, rollback evidence, and incident ownership.",
        encoding="utf-8",
    )
    ingested = invoke(runner, ["ingest", workspace_id, corpus_id, str(policy)])
    assert ingested["changed"] is True

    connector_root = tmp_path / "connectors"
    connector_root.mkdir(exist_ok=True)
    (connector_root / "security.txt").write_text(
        "Security changes require least privilege and audit evidence.",
        encoding="utf-8",
    )
    sourced = invoke(
        runner,
        ["ingest-source", workspace_id, corpus_id, "file://security.txt"],
    )
    assert sourced["changed"] is True

    answer = invoke(
        runner,
        ["query", workspace_id, corpus_id, "What does a production release require?"],
    )
    assert answer["citations"]

    recursive = invoke(
        runner,
        [
            "rlm",
            workspace_id,
            corpus_id,
            "Compare release and security controls",
            "--max-depth",
            "1",
            "--max-nodes",
            "4",
            "--branch-factor",
            "2",
            "--leaf-token-budget",
            "128",
        ],
    )
    assert recursive["citations"]
    assert recursive["nodes"]

    background = invoke(
        runner,
        ["rlm", workspace_id, corpus_id, "Summarize all controls", "--background"],
    )
    assert str(background["job_id"]).startswith("job_")
    assert invoke(runner, ["worker", "--once"]) == {"worked": True}

    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps([{"query": "What requires review?", "expected_terms": ["review"]}]),
        encoding="utf-8",
    )
    suite = invoke(runner, ["evaluation", "create", workspace_id, "release", str(cases)])
    evaluation = invoke(
        runner,
        ["evaluation", "run", workspace_id, str(suite["suite_id"]), corpus_id],
    )
    assert evaluation["status"] == "succeeded"

    skills = invoke(runner, ["skill", "list"])
    assert skills[0]["name"] == "corpus_health"
    created = invoke(runner, ["skill", "create", "release_review", "Review release evidence"])
    assert Path(str(created["path"])).is_dir()

    doctor = invoke(runner, ["doctor"])
    assert doctor["status"] == "ok"
    assert {"file", "http", "https"} <= set(doctor["connector_schemes"])
    assert invoke(runner, ["worker", "--once"]) == {"worked": False}


def test_cli_validation_and_process_commands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure_cli(monkeypatch, tmp_path)
    runner = CliRunner()

    missing = runner.invoke(app, ["ingest", "workspace", "corpus", str(tmp_path / "missing")])
    assert missing.exit_code != 0
    assert "existing file" in missing.output

    called: dict[str, object] = {}

    def fake_uvicorn_run(target: str, **kwargs: object) -> None:
        called["target"] = target
        called.update(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_uvicorn_run)
    served = runner.invoke(app, ["serve", "--host", "127.0.0.1", "--port", "9001", "--reload"])
    assert served.exit_code == 0, served.output
    assert called == {
        "target": "refaat.api.app:create_app",
        "host": "127.0.0.1",
        "port": 9001,
        "reload": True,
        "factory": True,
        "workers": None,
    }

    forever: dict[str, bool] = {}

    def fake_run_forever(self: object) -> None:
        forever["called"] = True

    monkeypatch.setattr("refaat.cli.DurableWorker.run_forever", fake_run_forever)
    worker = runner.invoke(app, ["worker"])
    assert worker.exit_code == 0, worker.output
    assert forever == {"called": True}
