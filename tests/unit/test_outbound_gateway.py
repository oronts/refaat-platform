"""The governed outbound gateway audits the admission decision before any byte leaves the socket."""

from __future__ import annotations

import itertools

import pytest

from refaat.application.outbound import CapabilityOutbound
from refaat.clock import SystemClock
from refaat.domain.audit import AuditEvent
from refaat.domain.egress import EgressPurpose
from refaat.errors import AuditUnavailableError, EgressDeniedError, ValidationError
from refaat.infrastructure.egress_client import PreparedRequest
from refaat.infrastructure.outbound import OutboundResponse

_URL = "https://api.example.com/v1/secret-path?token=PLANT-abc"


class _RecordingSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        self.events.append(event)


class _FailingSink:
    def record(self, event: AuditEvent) -> None:
        raise RuntimeError("sink down")


class _StubClient:
    def __init__(self, log: list[str], *, prepare_error: Exception | None = None) -> None:
        self._log = log
        self._prepare_error = prepare_error
        self.sent = False

    def prepare(self, url: str) -> PreparedRequest:
        del url
        self._log.append("prepare")
        if self._prepare_error is not None:
            raise self._prepare_error
        return PreparedRequest(url="https://93.184.216.34/v1", headers={}, extensions={}, host="api.example.com")

    def send(
        self, prepared: PreparedRequest, *, method: str = "GET", headers: object = None, content: object = None
    ) -> OutboundResponse:
        del prepared, method, headers, content
        self._log.append("send")
        self.sent = True
        return OutboundResponse(status_code=200, headers={}, content=b"ok", host="api.example.com")


def _view(client: object, *, sink: object | None, fail_closed: bool = False) -> CapabilityOutbound:
    return CapabilityOutbound(
        client,  # type: ignore[arg-type]
        workspace_id="acme",
        capability_name="reach",
        principal=None,
        clock=SystemClock(),
        audit_sink=sink,  # type: ignore[arg-type]
        audit_fail_closed=fail_closed,
        sequence=itertools.count(),
        run_nonce="preg_test",
    )


def test_admission_is_audited_before_the_send() -> None:
    log: list[str] = []
    sink = _RecordingSink()
    view = _view(_StubClient(log), sink=sink)
    response = view.get(_URL)
    assert response.status_code == 200
    assert log == ["prepare", "send"]
    assert [event.event_type for event in sink.events] == ["plugin.egress_admitted"]
    # The admitted event was recorded before the send call, so a fail-closed sink can gate the egress.
    assert sink.events[0].capability_name == "reach"


def test_the_event_is_redacted_no_url_or_secret() -> None:
    sink = _RecordingSink()
    _view(_StubClient([]), sink=sink).get(_URL)
    serialized = str(sink.events[0].to_json())
    assert "secret-path" not in serialized
    assert "PLANT-abc" not in serialized
    assert "/v1/secret-path" not in serialized


def test_an_egress_denial_is_audited_and_no_bytes_are_sent() -> None:
    log: list[str] = []
    sink = _RecordingSink()
    client = _StubClient(
        log, prepare_error=EgressDeniedError("network_destination_denied", "api.example.com", EgressPurpose.PLUGIN)
    )
    with pytest.raises(EgressDeniedError):
        _view(client, sink=sink).get(_URL)
    assert log == ["prepare"]
    assert client.sent is False
    assert [event.event_type for event in sink.events] == ["plugin.egress_denied"]


def test_an_ssrf_pin_denial_is_audited() -> None:
    sink = _RecordingSink()
    client = _StubClient([], prepare_error=ValidationError("pinned to a private address"))
    with pytest.raises(ValidationError):
        _view(client, sink=sink).get(_URL)
    assert [event.event_type for event in sink.events] == ["plugin.egress_denied"]


def test_a_malformed_destination_fails_closed_and_is_audited() -> None:
    sink = _RecordingSink()
    client = _StubClient([], prepare_error=ValueError("Invalid IPv6 URL"))
    with pytest.raises(EgressDeniedError):
        _view(client, sink=sink).get(_URL)
    assert [event.event_type for event in sink.events] == ["plugin.egress_denied"]


def test_an_unserved_run_refuses_and_sends_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("refaat.application.services.in_unserved_run", lambda: True)
    log: list[str] = []
    sink = _RecordingSink()
    client = _StubClient(log)
    with pytest.raises(EgressDeniedError):
        _view(client, sink=sink).get(_URL)
    assert log == []
    assert client.sent is False
    assert sink.events == []


def test_a_fail_closed_sink_outage_blocks_the_send() -> None:
    log: list[str] = []
    client = _StubClient(log)
    with pytest.raises(AuditUnavailableError):
        _view(client, sink=_FailingSink(), fail_closed=True).get(_URL)
    # The admitted-audit raised before the send, so no byte left under the outage.
    assert log == ["prepare"]
    assert client.sent is False
