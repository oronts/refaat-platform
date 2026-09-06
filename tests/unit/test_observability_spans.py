from __future__ import annotations

import io
import json
import logging
import sys

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from refaat.observability import (
    REDACTED,
    SAFE_STR_KEYS,
    JsonFormatter,
    get_logger,
    redact_field_value,
    span,
)
from refaat.programs.react import governance_reason_code


def test_program_span_is_noop_without_tracer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("refaat.observability._program_tracer", lambda: None)
    with span("refaat.program.run", program="p"):
        pass


def test_program_span_records_when_tracer_active(monkeypatch: pytest.MonkeyPatch) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr("refaat.observability._program_tracer", lambda: provider.get_tracer("test"))

    with span("refaat.program.evaluate", program="grounded-rag", cases=2):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "refaat.program.evaluate"
    assert spans[0].attributes["program"] == "grounded-rag"
    assert spans[0].attributes["cases"] == 2


def test_span_attribute_values_are_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr("refaat.observability._program_tracer", lambda: provider.get_tracer("test"))

    with span(
        "refaat.tool.step",
        tool="lookup",
        classification=3,
        requires_approval=True,
        secret_arg="CANARY-RAW-1",
        payload={"a": 1},
    ):
        pass

    attrs = exporter.get_finished_spans()[0].attributes
    assert attrs["tool"] == "lookup"  # allowlisted string passes
    assert attrs["classification"] == 3  # int passes
    assert attrs["requires_approval"] is True  # bool passes
    assert attrs["secret_arg"] == REDACTED  # non-allowlisted string redacted
    assert attrs["payload"] == "[redacted:dict]"  # non-scalar collapses to a typed sentinel
    assert all("CANARY-RAW-1" not in str(value) for value in attrs.values())


def _serialize_log(event: str, **fields: object) -> str:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    base = logging.getLogger("test.observability.redaction")
    base.handlers.clear()
    base.addHandler(handler)
    base.setLevel(logging.INFO)
    base.propagate = False
    try:
        get_logger("test.observability.redaction").info(event, **fields)
        handler.flush()
        return stream.getvalue()
    finally:
        base.removeHandler(handler)


def test_log_fields_are_redacted() -> None:
    line = _serialize_log(
        "evt",
        document_id="doc-1",
        secret_value="CANARY-LOG-1",
        payload={"marker": "CANARY-NESTED-1"},
    )
    parsed = json.loads(line)
    assert parsed["document_id"] == "doc-1"  # allowlisted string passes
    assert parsed["secret_value"] == REDACTED  # non-allowlisted string redacted
    assert parsed["payload"] == "[redacted:dict]"  # non-scalar collapses to a typed sentinel
    assert "CANARY-LOG-1" not in line
    assert "CANARY-NESTED-1" not in line


def test_every_currently_emitted_string_field_survives_the_seam() -> None:
    # Each allowlisted key carries its representative string unchanged, proving the seam did not
    # regress any existing span attribute or structured log field.
    for key in SAFE_STR_KEYS:
        value = f"{key}-value"
        assert redact_field_value(key, value) == value


def test_long_allowlisted_string_survives() -> None:
    # An allowlisted key carries an id or name, so its length is not bounded; a length cap would
    # silently redact a legitimate long program name and regress existing observability.
    name = "x" * 300
    assert redact_field_value("program", name) == name


def test_client_controlled_keys_are_redacted() -> None:
    # A raw request path or HTTP method is client-controlled free text, so neither is allowlisted.
    assert redact_field_value("path", "/v1/corpora/SECRET-CORPUS/programs/grounded-rag/run") == REDACTED
    assert redact_field_value("method", "CANARYMETHODSECRET9") == REDACTED


def test_governance_reason_code_strips_model_detail() -> None:
    code = governance_reason_code("argument validation failed: unexpected field: CANARY-FIELD-9x")
    assert code == "argument_validation_failed"
    assert "CANARY-FIELD-9x" not in code


def test_governance_reason_code_maps_known_categories() -> None:
    assert governance_reason_code("authorized") == "authorized"
    assert governance_reason_code("approval_required:already_consumed") == "approval_required"
    assert governance_reason_code("missing scopes: read:corpus") == "missing_scopes"
    assert governance_reason_code("network_destination_denied:evil.example:443") == "network_destination_denied"


def test_governance_reason_code_unknown_maps_to_other() -> None:
    assert governance_reason_code("no grant registered for tool 'evil'") == "other"
    assert governance_reason_code("side effect external exceeds ceiling read_only") == "other"


def _exc_record(message: str) -> logging.LogRecord:
    # The message flows through a variable, so the raised source line reads "raise ValueError(message)"
    # and the secret lives only in the exception's str(), never in a source-code frame line.
    try:
        raise ValueError(message)
    except ValueError:
        exc_info = sys.exc_info()
    return logging.LogRecord("test.exc", logging.ERROR, __file__, 0, "boom", (), exc_info)


def test_exception_field_is_byte_identical_when_redaction_off() -> None:
    record = _exc_record("secret-CANARY")
    payload = json.loads(JsonFormatter().format(record))
    assert payload["exception"] == logging.Formatter().formatException(record.exc_info)
    assert "ValueError: secret-CANARY" in payload["exception"]
    assert "Traceback (most recent call last):" in payload["exception"]


def test_exception_message_redacted_to_type_when_flag_on() -> None:
    record = _exc_record("secret-CANARY")
    payload = json.loads(JsonFormatter(redact_exception_text=True).format(record))
    exc = payload["exception"]
    assert "secret-CANARY" not in exc  # the message never egresses
    assert "Traceback (most recent call last):" in exc  # frames preserved
    assert "_exc_record" in exc  # the raising frame (a code location) is preserved
    assert exc.splitlines()[-1] == "ValueError"  # message redacted to the type name only


def test_exception_notes_are_redacted_when_flag_on() -> None:
    # A PEP 678 note is runtime data, the same category as the message, so it must not egress.
    try:
        err = ValueError("plain")
        err.add_note("NOTE-CANARY")
        raise err
    except ValueError:
        exc_info = sys.exc_info()
    record = logging.LogRecord("test.exc", logging.ERROR, __file__, 0, "boom", (), exc_info)
    exc = json.loads(JsonFormatter(redact_exception_text=True).format(record))["exception"]
    assert "NOTE-CANARY" not in exc
