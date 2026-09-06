from __future__ import annotations

import io

import pytest

pytestmark = pytest.mark.e2e

_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc``\x00\x00\x00"
    b"\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _bootstrap(client) -> tuple[str, str]:
    response = client.post(
        "/v1/bootstrap/workspaces",
        headers={"X-Bootstrap-Token": "bootstrap-test-token"},
        json={"slug": "api-images", "name": "Api Images"},
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    return payload["workspace_id"], payload["api_key"]


def test_image_upload_route_ingests_citable_chunk(client) -> None:
    _workspace_id, key = _bootstrap(client)
    headers = {"Authorization": f"Bearer {key}"}
    corpus_id = client.post("/v1/corpora", headers=headers, json={"name": "assets"}).json()["id"]

    upload = client.post(
        f"/v1/corpora/{corpus_id}/documents/image",
        headers=headers,
        data={"source_id": "diagram.png", "title": "Diagram", "caption": "A red circle architecture diagram"},
        files={"file": ("diagram.png", io.BytesIO(_PNG), "image/png")},
    )
    assert upload.status_code == 201, upload.text
    assert upload.json()["chunks"] == 1

    answer = client.post(
        f"/v1/corpora/{corpus_id}/query",
        headers=headers,
        json={"query": "red circle architecture diagram"},
    )
    assert answer.status_code == 200, answer.text
    assert answer.json()["citations"]
