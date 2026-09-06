from __future__ import annotations

import contextvars
import json
import logging
import sys
import threading
import traceback
from collections.abc import Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from refaat.config import Settings

trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")
workspace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("workspace_id", default="")


_telemetry_lock = threading.Lock()
_telemetry_initialized = False
_instrumented_apps: set[int] = set()
_instrumented_engines: set[int] = set()


REDACTED = "[redacted]"

# Allowlist of structured-log field and span-attribute keys whose string values are safe to
# egress. It lists every string key any span() or structured log emits today, so the seam is
# byte-identical for existing observability and only a new or unexpected string value is redacted.
# Every key here is an id, an enum, a decision code, or a route template, never client-controlled
# free text: a string under any other key, and any non-scalar value, never reaches the sink. This is
# regression prevention: no current field leaks evidence text, a prompt, or a secret. A new safe
# string field is added here deliberately, and only when its value can never carry free text.
SAFE_STR_KEYS: frozenset[str] = frozenset(
    {
        "program",
        "job_id",
        "kind",
        "event_type",
        "topic",
        "event_id",
        "blob_uri",
        "environment",
        "served",
        "document_id",
        "chunk_id",
        "resource_type",
        "route",
        "tool",
        "capability",
        "reason_code",
        "side_effect",
    }
)


def redact_field_value(key: str, value: object) -> object:
    """Return the value when it is safe to egress, otherwise a redaction sentinel.

    A string passes unchanged only under an allowlisted key; any other string becomes ``REDACTED``.
    A bool, int, float, or None passes unchanged. Any non-scalar (dict, list, tuple, bytes, object)
    becomes a type-tagged sentinel so no structured value ever reaches the sink. Allowlisted keys
    carry ids, enums, decision codes, or route templates, so their length is not bounded (bounding it
    would silently redact a legitimate long id such as a program name).
    """
    if isinstance(value, str):
        if key in SAFE_STR_KEYS:
            return value
        return REDACTED
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return f"[redacted:{type(value).__name__}]"


def redact_fields(fields: Mapping[str, object]) -> dict[str, object]:
    return {key: redact_field_value(key, value) for key, value in fields.items()}


def _walk_exception_chain(root: traceback.TracebackException) -> Iterator[traceback.TracebackException]:
    seen: set[int] = set()
    stack = [root]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        if current.__cause__ is not None:
            stack.append(current.__cause__)
        if current.__context__ is not None:
            stack.append(current.__context__)
        for member in getattr(current, "exceptions", None) or ():
            stack.append(member)


def _redacted_exception_text(exc_info: Any) -> str:
    """Format a traceback with the frames preserved but every exception message redacted to its type.

    The traceback frames (file, line, and code context) are code locations, not data, so they stay
    for debuggability; each exception's ``str()`` and any PEP 678 notes across the whole cause and
    context chain are replaced or dropped, so a message or note that embedded evidence text cannot
    egress through the exc_info channel. This is an opt-in operator control, not a guarantee that no
    exception can ever carry data: a frame local rendered by a custom traceback handler is out of scope.
    A hostile exception (an adversarial ``__str__`` or getter that raises) has its message and notes
    omitted like any other, and on any residual failure the whole render falls back to a fixed
    redaction, so it never crashes the log emit or egresses partly-built text.
    """
    try:
        formatted = traceback.TracebackException(*exc_info, max_group_width=16, max_group_depth=8)
        for member in _walk_exception_chain(formatted):
            member._str = ""  # type: ignore[attr-defined]
            if getattr(member, "__notes__", None):
                member.__notes__ = []
        text = "".join(formatted.format())
    except BaseException:
        # A hostile exception must never crash the log emit or egress partly-built text.
        return "[exception text redacted]"
    return text[:-1] if text.endswith("\n") else text


class JsonFormatter(logging.Formatter):
    def __init__(self, *, redact_exception_text: bool = False) -> None:
        super().__init__()
        self._redact_exception_text = redact_exception_text

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        if trace_id := trace_id_var.get():
            payload["trace_id"] = trace_id
        if workspace_id := workspace_id_var.get():
            payload["workspace_id"] = workspace_id
        fields = getattr(record, "fields", None)
        if isinstance(fields, Mapping):
            payload.update(fields)
        if record.exc_info:
            payload["exception"] = (
                _redacted_exception_text(record.exc_info)
                if self._redact_exception_text
                else self.formatException(record.exc_info)
            )
        return json.dumps(payload, default=str, separators=(",", ":"))


class ContextLogger(logging.LoggerAdapter[logging.Logger]):
    def process(
        self,
        msg: object,
        kwargs: MutableMapping[str, Any],
    ) -> tuple[object, MutableMapping[str, Any]]:
        extra = dict(kwargs.pop("extra", {}))
        fields = dict(extra.get("fields", {}))
        fields.update(kwargs.pop("fields", {}))
        logging_keys = {"exc_info", "stack_info", "stacklevel"}
        for key in list(kwargs):
            if key not in logging_keys:
                fields[key] = kwargs.pop(key)
        extra["fields"] = redact_fields(fields)
        kwargs["extra"] = extra
        return msg, kwargs


def get_logger(name: str) -> ContextLogger:
    return ContextLogger(logging.getLogger(name), {})


def configure_logging(settings: Settings) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        JsonFormatter(redact_exception_text=settings.observability_redact_exception_text)
        if settings.log_json
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.log_level.upper())


def current_trace_id() -> str:
    value = trace_id_var.get()
    if value:
        return value
    value = uuid4().hex
    trace_id_var.set(value)
    return value


@contextmanager
def trace_context(
    *,
    trace_id: str | None = None,
    workspace_id: str | None = None,
) -> Iterator[str]:
    trace_token = trace_id_var.set(trace_id or uuid4().hex)
    workspace_token = workspace_id_var.set(workspace_id or "")
    try:
        yield trace_id_var.get()
    finally:
        trace_id_var.reset(trace_token)
        workspace_id_var.reset(workspace_token)


def _program_tracer() -> Any:
    if not _telemetry_initialized:
        return None
    try:
        from opentelemetry import trace
    except ImportError:  # pragma: no cover - optional dependency
        return None
    return trace.get_tracer("refaat.programs")


@contextmanager
def span(name: str, **attributes: object) -> Iterator[None]:
    """Open an OpenTelemetry span when telemetry is active, otherwise a zero-cost no-op."""

    tracer = _program_tracer()
    if tracer is None:
        yield
        return
    with tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            safe = redact_field_value(key, value)
            # Set only a native scalar; a redacted None is skipped because OTel forbids a None
            # attribute, and a non-scalar has already collapsed to a sentinel string.
            if isinstance(safe, (str, int, float, bool)):
                current.set_attribute(key, safe)
        yield


def configure_telemetry(
    settings: Settings,
    app: object | None = None,
    engine: object | None = None,
) -> None:
    if not settings.otel_enabled:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install refaat[otel] to enable OpenTelemetry") from exc

    global _telemetry_initialized
    with _telemetry_lock:
        if not _telemetry_initialized:
            provider = TracerProvider(resource=Resource.create({SERVICE_NAME: settings.otel_service_name}))
            if settings.otel_exporter_endpoint:
                provider.add_span_processor(
                    BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_exporter_endpoint))
                )
            trace.set_tracer_provider(provider)
            HTTPXClientInstrumentor().instrument()
            _telemetry_initialized = True

        if engine is not None and id(engine) not in _instrumented_engines:
            SQLAlchemyInstrumentor().instrument(engine=engine)
            _instrumented_engines.add(id(engine))
        if app is not None and id(app) not in _instrumented_apps:
            FastAPIInstrumentor.instrument_app(app)  # type: ignore[arg-type]
            _instrumented_apps.add(id(app))
