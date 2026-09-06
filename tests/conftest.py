from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from refaat.api.app import create_app
from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path / 'refaat.db'}",
        auto_create_schema=True,
        blob_root=tmp_path / "blobs",
        connector_file_root=tmp_path / "connectors",
        skill_root=tmp_path / "skills",
        api_key_pepper="test-pepper-with-enough-entropy",
        bootstrap_token="bootstrap-test-token",
        trusted_hosts=["testserver", "localhost", "127.0.0.1"],
        rate_limit_requests=10_000,
        max_body_bytes=2 * 1024 * 1024,
        default_chunk_tokens=64,
        chunk_overlap_tokens=12,
        query_token_budget=600,
        session_compaction_turns=4,
        retained_session_turns=2,
        log_json=False,
    )


@pytest.fixture
def runtime(settings: Settings) -> Iterator[Refaat]:
    value = RuntimeBuilder(settings).build()
    try:
        yield value
    finally:
        value.close()


@pytest.fixture
def workspace_id(runtime: Refaat) -> str:
    return runtime.create_workspace("acme", "Acme")


@pytest.fixture
def api_key(runtime: Refaat, workspace_id: str) -> str:
    _, raw = runtime.issue_api_key(
        workspace_id,
        "test-admin",
        scopes=("read", "write", "admin"),
    )
    return raw


@pytest.fixture
def corpus_id(runtime: Refaat, workspace_id: str) -> str:
    return runtime.create_corpus(workspace_id, "handbook")


@pytest.fixture
def client(runtime: Refaat, settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings, runtime), raise_server_exceptions=False) as value:
        yield value
