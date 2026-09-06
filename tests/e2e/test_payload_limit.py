from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from refaat.api.app import create_app
from refaat.application.runtime import RuntimeBuilder
from refaat.config import Settings

pytestmark = pytest.mark.e2e


def test_payload_limit_is_enforced(tmp_path: Path) -> None:
    settings = Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path / 'limit.db'}",
        blob_root=tmp_path / "blobs",
        skill_root=tmp_path / "skills",
        api_key_pepper="test-pepper",
        bootstrap_token="token",
        trusted_hosts=["testserver"],
        max_body_bytes=128,
        rate_limit_requests=100,
        log_json=False,
    )
    runtime = RuntimeBuilder(settings).build()
    try:
        with TestClient(create_app(settings, runtime), raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/bootstrap/workspaces",
                headers={"X-Bootstrap-Token": "token"},
                json={"slug": "large", "name": "x" * 300},
            )
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "payload_too_large"
    finally:
        runtime.close()
