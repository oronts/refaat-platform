from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx

from refaat import Refaat, RefaatProject, RuntimeBuilder
from refaat.config import Settings

EXAMPLES_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXAMPLES_ROOT.parent
DOCUMENTS_ROOT = EXAMPLES_ROOT / "data" / "company"
EVALUATION_CASES_PATH = EXAMPLES_ROOT / "data" / "evaluation_cases.json"


def demo_settings(root: Path, *, allow_private_connector_hosts: bool = False) -> Settings:
    return Settings(
        environment="test",
        database_url=f"sqlite+pysqlite:///{root / 'refaat.db'}",
        auto_create_schema=True,
        blob_root=root / "blobs",
        connector_file_root=root / "connectors",
        skill_root=root / "skills",
        api_key_pepper="example-api-key-pepper-with-enough-entropy",
        bootstrap_token="example-bootstrap-token-with-enough-entropy",
        trusted_hosts=["127.0.0.1", "localhost", "testserver"],
        rate_limit_requests=10_000,
        default_chunk_tokens=96,
        chunk_overlap_tokens=16,
        query_token_budget=900,
        query_top_k=10,
        session_compaction_turns=4,
        retained_session_turns=2,
        allow_private_connector_hosts=allow_private_connector_hosts,
        log_json=False,
    )


def build_demo_runtime(root: Path) -> Refaat:
    return RuntimeBuilder(demo_settings(root)).build()


def copy_demo_documents(destination: Path) -> Path:
    shutil.copytree(DOCUMENTS_ROOT, destination)
    return destination


def seed_demo_project(runtime: Refaat, documents: Path | None = None) -> RefaatProject:
    project = runtime.project(
        workspace="northstar",
        workspace_name="Northstar Systems",
        corpus="operations-handbook",
    )
    project.ingest(documents or DOCUMENTS_ROOT, metadata={"example": "northstar"})
    return project


def load_evaluation_cases() -> list[dict[str, Any]]:
    value = json.loads(EVALUATION_CASES_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("Evaluation fixture must contain a JSON array of objects")
    return value


def emit(value: object) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default))


def _json_default(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, (date, datetime, Path)):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as value:
        value.bind(("127.0.0.1", 0))
        return int(value.getsockname()[1])


@contextmanager
def live_api_server(root: Path) -> Iterator[tuple[str, str]]:
    port = free_tcp_port()
    bootstrap_token = "example-bootstrap-token-with-enough-entropy"
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    source_root = str(REPOSITORY_ROOT / "src")
    env["PYTHONPATH"] = os.pathsep.join(value for value in (source_root, env.get("PYTHONPATH", "")) if value)
    env.update(
        {
            "REFAAT_ENVIRONMENT": "test",
            "REFAAT_DATABASE_URL": f"sqlite+pysqlite:///{root / 'api.db'}",
            "REFAAT_AUTO_CREATE_SCHEMA": "true",
            "REFAAT_BLOB_ROOT": str(root / "api-blobs"),
            "REFAAT_CONNECTOR_FILE_ROOT": str(root / "api-connectors"),
            "REFAAT_SKILL_ROOT": str(root / "api-skills"),
            "REFAAT_API_KEY_PEPPER": "example-api-key-pepper-with-enough-entropy",
            "REFAAT_BOOTSTRAP_TOKEN": bootstrap_token,
            "REFAAT_TRUSTED_HOSTS": '["127.0.0.1", "localhost"]',
            "REFAAT_RATE_LIMIT_REQUESTS": "10000",
            "REFAAT_DEFAULT_CHUNK_TOKENS": "96",
            "REFAAT_CHUNK_OVERLAP_TOKENS": "16",
            "REFAAT_QUERY_TOKEN_BUDGET": "900",
            "REFAAT_SESSION_COMPACTION_TURNS": "4",
            "REFAAT_RETAINED_SESSION_TURNS": "2",
            "REFAAT_LOG_JSON": "false",
        }
    )
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "refaat.api.app:create_app",
        "--factory",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    process = subprocess.Popen(
        command,
        cwd=REPOSITORY_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_api(process, f"{base_url}/health/ready")
        yield base_url, bootstrap_token
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _wait_for_api(process: subprocess.Popen[str], url: str) -> None:
    deadline = time.monotonic() + 15
    with httpx.Client(timeout=0.5) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise RuntimeError(f"Refaat API exited before readiness:\n{stdout}\n{stderr}")
            try:
                response = client.get(url)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
    process.kill()
    stdout, stderr = process.communicate()
    raise RuntimeError(f"Refaat API did not become ready:\n{stdout}\n{stderr}")


def bootstrap_workspace(base_url: str, bootstrap_token: str) -> dict[str, str]:
    response = httpx.post(
        f"{base_url}/v1/bootstrap/workspaces",
        headers={"X-Bootstrap-Token": bootstrap_token},
        json={"slug": "northstar-api", "name": "Northstar API Demo"},
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    return {key: str(payload[key]) for key in ("workspace_id", "api_key_id", "api_key")}


def cli_environment(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    source_root = str(REPOSITORY_ROOT / "src")
    env["PYTHONPATH"] = os.pathsep.join(value for value in (source_root, env.get("PYTHONPATH", "")) if value)
    env.update(
        {
            "REFAAT_ENVIRONMENT": "test",
            "REFAAT_DATABASE_URL": f"sqlite+pysqlite:///{root / 'cli.db'}",
            "REFAAT_AUTO_CREATE_SCHEMA": "true",
            "REFAAT_BLOB_ROOT": str(root / "cli-blobs"),
            "REFAAT_CONNECTOR_FILE_ROOT": str(root / "cli-connectors"),
            "REFAAT_SKILL_ROOT": str(root / "cli-skills"),
            "REFAAT_API_KEY_PEPPER": "example-api-key-pepper-with-enough-entropy",
            "REFAAT_BOOTSTRAP_TOKEN": "example-bootstrap-token-with-enough-entropy",
            "REFAAT_TRUSTED_HOSTS": '["127.0.0.1", "localhost"]',
            "REFAAT_RATE_LIMIT_REQUESTS": "10000",
            "REFAAT_LOG_JSON": "false",
        }
    )
    return env


def run_cli(root: Path, *arguments: str) -> object:
    completed = subprocess.run(
        [sys.executable, "-m", "refaat.cli", *arguments],
        cwd=REPOSITORY_ROOT,
        env=cli_environment(root),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Refaat CLI failed ({' '.join(arguments)}):\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Refaat CLI did not return JSON ({' '.join(arguments)}):\n{completed.stdout}") from exc
