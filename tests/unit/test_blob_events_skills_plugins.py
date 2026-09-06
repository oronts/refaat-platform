from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from refaat.domain.models import ContextPacket
from refaat.errors import ConflictError, DependencyUnavailableError, NotFoundError, ValidationError
from refaat.infrastructure.blob import LocalBlobStore
from refaat.infrastructure.events import OutboxPublisher, WebhookEventSink
from refaat.infrastructure.plugins import PluginRegistry
from refaat.skills.registry import SkillCreator, SkillRegistry


def test_local_blob_store_is_atomic_rooted_and_idempotent(tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path / "blobs")
    uri = store.put("workspace/corpus/document", b"evidence", media_type="text/plain")
    with store.open(uri) as handle:
        assert handle.read() == b"evidence"
    with pytest.raises(ValidationError):
        store.open("https://example.com/blob")
    with pytest.raises(ValidationError):
        store.open("file:///etc/passwd")
    store.delete(uri)
    store.delete(uri)
    with pytest.raises(NotFoundError):
        store.open(uri)


def test_s3_blob_open_missing_key_maps_to_not_found() -> None:
    """Parity with LocalBlobStore: an absent object verifies as NotFound, not a raw ClientError.

    Without this, erasure verification (delete then open) sees a ClientError after a correct S3
    delete, falsely records FAILED, and skips the document.erased audit event.
    """
    pytest.importorskip("boto3")
    from botocore.exceptions import ClientError

    from refaat.infrastructure.blob import S3BlobStore

    store = S3BlobStore(
        bucket="refaat-test",
        endpoint_url=None,
        region="us-east-1",
        access_key_id="test-key",
        secret_access_key="test-secret",
    )

    class _MissingKeyClient:
        def get_object(self, **_kwargs: object) -> object:
            raise ClientError(
                {
                    "Error": {"Code": "NoSuchKey", "Message": "The specified key does not exist."},
                    "ResponseMetadata": {"HTTPStatusCode": 404},
                },
                "GetObject",
            )

    class _AccessDeniedClient:
        def get_object(self, **_kwargs: object) -> object:
            raise ClientError(
                {
                    "Error": {"Code": "AccessDenied", "Message": "Access Denied"},
                    "ResponseMetadata": {"HTTPStatusCode": 403},
                },
                "GetObject",
            )

    store.client = _MissingKeyClient()
    with pytest.raises(NotFoundError):
        store.open("s3://refaat-test/workspace/corpus/document")

    # A genuine failure (not a missing key) must still propagate, never be swallowed as absent.
    store.client = _AccessDeniedClient()
    with pytest.raises(ClientError):
        store.open("s3://refaat-test/workspace/corpus/document")


def test_webhook_event_sink_contract_and_failure() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(204)

    success_client = httpx.Client(transport=httpx.MockTransport(handler))
    success_sink = WebhookEventSink(
        "https://events.test/refaat",
        "event-secret",
        allow_private_hosts=True,
        client=success_client,
    )
    success_sink.emit("document.indexed", {"document_id": "doc_1"}, event_id="evt_1")
    success_sink.close()
    assert not success_client.is_closed
    assert captured[0].headers["authorization"] == "Bearer event-secret"
    import json

    body = json.loads(captured[0].content)
    assert body["id"] == "evt_1"
    assert body["topic"] == "document.indexed"
    assert body["payload"] == {"document_id": "doc_1"}

    failure_client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(503, request=request)))
    failure_sink = WebhookEventSink(
        "https://events.test/refaat",
        "event-secret",
        allow_private_hosts=True,
        client=failure_client,
    )
    with pytest.raises(DependencyUnavailableError):
        failure_sink.emit("document.indexed", {})

    success_client.close()
    failure_client.close()


def test_outbox_failure_is_retained_for_retry() -> None:
    events = [
        {"id": "evt", "topic": "topic", "payload": {}, "attempts": 0},
    ]
    failed: list[tuple[str, str, int]] = []

    class Store:
        def claim_outbox(self, worker_id: str, lease_seconds: int, limit: int):
            assert limit == 10
            assert worker_id == "outbox-publisher"
            return events

        def delivered_outbox_sinks(self, event_id: str) -> set[str]:
            return set()

        def record_outbox_delivery(self, event_id: str, sink_name: str) -> None:
            raise AssertionError("a failing sink must not be recorded as delivered")

        def mark_outbox_failed(self, event_id: str, worker_id: str, error: str, delay: int) -> None:
            assert worker_id == "outbox-publisher"
            failed.append((event_id, error, delay))

        def mark_outbox_delivered(self, event_id: str, worker_id: str) -> bool:
            raise AssertionError(event_id)

    class FailingSink:
        def emit(self, topic: str, payload: dict[str, object]) -> None:
            raise RuntimeError(f"cannot deliver {topic}")

    assert OutboxPublisher(Store(), [FailingSink()]).publish(limit=10) == 0
    assert failed[0][0] == "evt"
    assert failed[0][2] == 1


def test_sink_ledger_names_are_stable_across_reordering() -> None:
    class NamedSink:
        def __init__(self, name: str) -> None:
            self.sink_name = name

        def emit(self, topic: str, payload: dict[str, object]) -> None:
            del topic, payload

    class UnnamedSink:
        def emit(self, topic: str, payload: dict[str, object]) -> None:
            del topic, payload

    first, second = NamedSink("alpha"), NamedSink("beta")
    ordered = OutboxPublisher._stable_sink_names((first, second))
    reordered = OutboxPublisher._stable_sink_names((second, first))
    assert ordered == ("alpha", "beta")
    assert reordered == ("beta", "alpha")
    # Each distinct sink keeps its own key regardless of position, so inserting or
    # reordering a configured sink never reassigns a delivery record.
    assert dict(zip(ordered, (first, second), strict=True)) == {"alpha": first, "beta": second}

    dup_a, dup_b = UnnamedSink(), UnnamedSink()
    assert OutboxPublisher._stable_sink_names((dup_a, dup_b)) == ("UnnamedSink", "UnnamedSink#1")


def test_skill_registry_creator_and_plugin_loading(tmp_path: Path, monkeypatch) -> None:
    class CustomSkill:
        name = "custom_skill"
        description = "Custom test skill"

        def run(self, context: ContextPacket, arguments: dict[str, object]):
            return {"query": context.query, "arguments": arguments}

    registry = SkillRegistry()
    registry.register(CustomSkill())
    assert registry.get("custom_skill").description == "Custom test skill"
    with pytest.raises(ConflictError):
        registry.register(CustomSkill())
    with pytest.raises(NotFoundError):
        registry.get("missing")

    creator = SkillCreator(tmp_path / "skills")
    project = creator.create("release_guard", "Checks release evidence")
    assert "refaat>=0.1.0,<1" in (project / "pyproject.toml").read_text()
    assert (project / "tests" / "test_skill.py").is_file()
    with pytest.raises(ConflictError):
        creator.create("release_guard", "Duplicate")
    with pytest.raises(ValidationError):
        creator.create("Bad-Name", "Invalid")

    class EntryPoint:
        name = "loaded"

        @staticmethod
        def load():
            return CustomSkill

    monkeypatch.setattr(
        "refaat.infrastructure.plugins.entry_points",
        lambda group: [EntryPoint()] if group == "refaat.skills" else [],
    )
    discovered = PluginRegistry().discover("refaat.skills")
    assert discovered["loaded"] is CustomSkill
    assert PluginRegistry(set()).discover("refaat.skills") == {}
    assert PluginRegistry({"other"}).discover("refaat.skills") == {}

    class BrokenEntryPoint:
        name = "broken"

        @staticmethod
        def load():
            raise ImportError("broken plugin")

    monkeypatch.setattr("refaat.infrastructure.plugins.entry_points", lambda group: [BrokenEntryPoint()])
    with pytest.raises(Exception, match="Failed to load plugin"):
        PluginRegistry().discover("refaat.skills")
