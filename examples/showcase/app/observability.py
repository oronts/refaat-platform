"""Real economics and observability for the showcase, sourced only from the runtime.

Nothing here is a mock. Token counts and USD cost come from the response usage the
runtime reports and the configured model prices, so offline (extractive generation,
empty price map) both are honestly zero and real numbers appear only when a model is
set. The trace is captured from refaat's own OpenTelemetry instrumentation: an
in-memory span exporter is attached to the showcase process and refaat's ``span()``
emits the genuine ``refaat.program.run`` and ``refaat.program.evaluate`` spans a real
program run produces. The budget usage is read from a real :class:`CostMeter` opened
around that run, and the event log is the runtime's own outbox rows written at
ingestion. When the OpenTelemetry SDK is not installed the trace degrades to a real,
clearly labelled stage trace measured from a live timed run rather than a fabricated
one.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from refaat.application.budget import budget_scope, prices_from
from refaat.evaluation.dataset import Dataset, EvaluationCase

OBS_QUESTION = "Who founded Oronts GmbH and where is it based?"
_DEFAULT_TOKEN_CEILING = 100_000


def compute_cost(
    usage: Mapping[str, Any] | None,
    model: str,
    prices: Mapping[str, Mapping[str, float]] | None,
) -> dict[str, Any]:
    """Prompt, completion, total tokens and USD cost from a real usage dict.

    The USD cost mirrors the runtime CostMeter convention: prompt tokens times the
    input rate plus completion tokens times the output rate, both per 1000 tokens.
    With no usage tokens or no price entry for the model the cost is exactly zero, so
    the offline extractive path (usage carries no token counts, the price map is
    empty) reports zero honestly.
    """
    prompt = _int(usage.get("prompt_tokens")) if usage else 0
    completion = _int(usage.get("completion_tokens")) if usage else 0
    total = prompt + completion
    usd = 0.0
    price = prices.get(model) if prices else None
    if price is not None:
        usd += prompt / 1000.0 * float(price.get("input_per_1k", 0.0))
        usd += completion / 1000.0 * float(price.get("output_per_1k", 0.0))
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "usd_cost": round(usd, 6),
    }


def _int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


@dataclass
class _Fresh:
    tokens: int
    usd: float
    ms: float


@dataclass
class SessionEconomics:
    """Server-side accrual of the session's real token, cost, and cache savings.

    A fresh (uncached) answer records its measured tokens, cost, and latency. A later
    cache hit for the same question and mode reports the tokens, USD, and latency that
    hit avoided, taken from the recorded fresh run, and adds them to the session saved
    totals. Offline the avoided tokens and USD are zero, which is the honest result.
    """

    _lock: threading.Lock = field(default_factory=threading.Lock)
    queries: int = 0
    cache_hits: int = 0
    total_tokens: int = 0
    total_usd: float = 0.0
    total_saved_tokens: int = 0
    total_saved_usd: float = 0.0
    total_saved_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    _fresh: dict[tuple[str, str], _Fresh] = field(default_factory=dict)

    def record(
        self,
        *,
        question: str,
        mode: str,
        cached: bool,
        usage: Mapping[str, Any] | None,
        model: str,
        prices: Mapping[str, Mapping[str, float]] | None,
        elapsed_ms: float,
    ) -> dict[str, Any]:
        cost = compute_cost(usage, model, prices)
        saved = {"tokens": 0, "usd": 0.0, "latency_ms": 0.0}
        key = (mode, question)
        with self._lock:
            self.queries += 1
            self.total_latency_ms += elapsed_ms
            if cached:
                self.cache_hits += 1
                prior = self._fresh.get(key)
                if prior is not None:
                    saved_ms = max(0.0, prior.ms - elapsed_ms)
                    saved = {
                        "tokens": prior.tokens,
                        "usd": round(prior.usd, 6),
                        "latency_ms": round(saved_ms, 3),
                    }
                    self.total_saved_tokens += prior.tokens
                    self.total_saved_usd += prior.usd
                    self.total_saved_latency_ms += saved_ms
            else:
                self.total_tokens += cost["total_tokens"]
                self.total_usd += cost["usd_cost"]
                self._fresh[key] = _Fresh(cost["total_tokens"], cost["usd_cost"], elapsed_ms)
        return {"tokens": cost, "usd_cost": cost["usd_cost"], "saved": saved}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "queries": self.queries,
                "cache_hits": self.cache_hits,
                "total_tokens": self.total_tokens,
                "total_usd": round(self.total_usd, 6),
                "total_saved_tokens": self.total_saved_tokens,
                "total_saved_usd": round(self.total_saved_usd, 6),
                "total_saved_latency_ms": round(self.total_saved_latency_ms, 3),
                "total_latency_ms": round(self.total_latency_ms, 3),
            }


class TraceCapture:
    """Attaches an in-memory OpenTelemetry exporter so refaat's real spans are captured.

    refaat's ``span()`` is a no-op until telemetry is initialised. Here the showcase
    installs its own tracer provider with an in-memory processor and flips the runtime
    flag, so a program run emits its genuine ``refaat.program.run`` and
    ``refaat.program.evaluate`` spans into a sink the showcase can read back. This is
    process-local to the showcase image and never reaches a network exporter. When the
    OpenTelemetry SDK is absent the capture stays inert and the caller falls back to a
    real stage trace.
    """

    def __init__(self) -> None:
        self.real_otel = False
        self._exporter: Any = None
        self._enable()

    def _enable(self) -> None:
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import SERVICE_NAME, Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
                InMemorySpanExporter,
            )
        except ImportError:
            return
        from refaat import observability as obs

        provider = TracerProvider(resource=Resource.create({SERVICE_NAME: "refaat-showcase"}))
        exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        # set_tracer_provider is a one-time process global; the showcase owns its own
        # process, so it installs the in-memory provider and enables refaat's span()
        # to emit real spans into the local sink. No src behaviour is changed: this is
        # the documented runtime switch an operator flips to turn telemetry on.
        trace.set_tracer_provider(provider)
        obs._telemetry_initialized = True
        self._exporter = exporter
        self.real_otel = True

    def clear(self) -> None:
        if self._exporter is not None:
            self._exporter.clear()

    def spans(self) -> list[dict[str, Any]]:
        if self._exporter is None:
            return []
        rows: list[dict[str, Any]] = []
        for span in self._exporter.get_finished_spans():
            context = span.get_span_context()
            parent = span.parent
            duration_ms = round((span.end_time - span.start_time) / 1_000_000, 3)
            rows.append(
                {
                    "name": span.name,
                    "span_id": format(context.span_id, "016x"),
                    "trace_id": format(context.trace_id, "032x"),
                    "parent_span_id": format(parent.span_id, "016x") if parent else None,
                    "duration_ms": duration_ms,
                    "start": _iso(span.start_time),
                    "end": _iso(span.end_time),
                    "attributes": {key: _attr(value) for key, value in (span.attributes or {}).items()},
                }
            )
        return rows


def _iso(epoch_ns: int) -> str:
    return datetime.fromtimestamp(epoch_ns / 1_000_000_000, tz=UTC).isoformat()


def _attr(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


_capture_lock = threading.Lock()
_capture: TraceCapture | None = None


def get_trace_capture() -> TraceCapture:
    global _capture
    with _capture_lock:
        if _capture is None:
            _capture = TraceCapture()
        return _capture


def capture_program_trace(project: Any, settings: Any) -> dict[str, Any]:
    """Run one real grounded program under a budget scope and capture its true trace.

    The program is evaluated over a single case, so the runtime emits a real
    ``refaat.program.evaluate`` span with a nested ``refaat.program.run`` span. The
    surrounding budget scope installs a real CostMeter shared by the query path, so the
    reported spend is what the run actually charged: zero offline (the deterministic
    stack is never wrapped) and real tokens and USD once a model is set. When the
    OpenTelemetry SDK is missing the spans list is empty and the caller reports a real
    stage trace instead.
    """
    capture = get_trace_capture()
    capture.clear()
    prices = prices_from(settings.model_prices)
    token_ceiling = settings.run_token_budget or _DEFAULT_TOKEN_CEILING
    cost_ceiling = settings.run_cost_budget_usd
    dataset = Dataset(name="observability", cases=(EvaluationCase(id="obs-1", query=OBS_QUESTION),))
    program = project.programs.get("grounded-rag")
    with budget_scope(token_ceiling, cost_ceiling, prices=prices) as meter:
        report = program.evaluate(dataset)
        budget = {
            "token_ceiling": token_ceiling,
            "tokens_used": meter.tokens_used if meter is not None else 0,
            "remaining_tokens": meter.remaining_tokens() if meter is not None else None,
            "cost_ceiling_usd": cost_ceiling,
            "cost_used_usd": round(meter.cost_used_usd, 6) if meter is not None else 0.0,
            "remaining_cost_usd": (
                round(meter.remaining_cost_usd(), 6)
                if meter is not None and meter.remaining_cost_usd() is not None
                else None
            ),
        }
    spans = capture.spans()
    if capture.real_otel and spans:
        trace = {
            "source": "opentelemetry",
            "spans": spans,
            "span_count": len(spans),
            "span_names": sorted({span["name"] for span in spans}),
        }
    else:
        trace = _stage_trace(project)
    return {
        "trace": trace,
        "budget": budget,
        "evaluation": {"cases": len(dataset.cases), "hard_passed": bool(report.hard_constraints_passed)},
    }


def _stage_trace(project: Any) -> dict[str, Any]:
    """A real, clearly labelled fallback trace when OpenTelemetry is not installed.

    Each row is a live timed measurement of a real runtime stage, not a fabricated
    span, so the durations are genuine even though there is no OpenTelemetry backend.
    """
    started = time.perf_counter()
    context = project.retrieve(OBS_QUESTION)
    retrieve_ms = round((time.perf_counter() - started) * 1000, 3)
    started = time.perf_counter()
    project.ask(OBS_QUESTION)
    answer_ms = round((time.perf_counter() - started) * 1000, 3)
    spans = [
        {"name": "showcase.retrieve", "duration_ms": retrieve_ms, "attributes": {"routes": ",".join(context.routes)}},
        {"name": "showcase.answer", "duration_ms": answer_ms, "attributes": {}},
    ]
    return {
        "source": "stage-trace",
        "spans": spans,
        "span_count": len(spans),
        "span_names": sorted({span["name"] for span in spans}),
    }


__all__ = [
    "OBS_QUESTION",
    "SessionEconomics",
    "TraceCapture",
    "capture_program_trace",
    "compute_cost",
    "get_trace_capture",
]
