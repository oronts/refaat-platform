"""Configuring the circuit breaker leaves the offline stack byte-identical and never enters a shadow run."""

from __future__ import annotations

from typing import Any

import pytest

from refaat.application.runtime import RuntimeBuilder
from refaat.config import Settings
from refaat.domain.models import DocumentInput
from refaat.domain.resilience import CircuitBreakerConfig
from refaat.infrastructure.resilience import InMemoryCircuitBreaker

pytestmark = pytest.mark.integration

_DOC = DocumentInput(
    source_id="ops.txt",
    title="Ops",
    media_type="text/plain",
    content=b"Rollback restores the previously approved harness snapshot before deployment.",
)


def _answer(settings: Settings, **overrides: object) -> tuple[str, tuple[str, ...]]:
    runtime = RuntimeBuilder(settings.model_copy(update=overrides)).build()
    try:
        project = runtime.project(workspace="acme", corpus="cb")
        runtime.ingest(project.workspace_id, project.corpus_id, _DOC)
        answer = project.ask("how does rollback work")
        return answer.answer, tuple(citation.chunk_id for citation in answer.citations)
    finally:
        runtime.close()


def test_offline_is_byte_identical_with_and_without_a_breaker(settings: Settings) -> None:
    # The deterministic ExtractiveGenerator is never breaker-wrapped, so an in_memory breaker changes
    # nothing about the offline answer or its citations.
    baseline = _answer(settings)
    with_breaker = _answer(settings, circuit_breaker_backend="in_memory")
    assert with_breaker == baseline
    assert baseline[1]  # the answer actually grounds


def test_an_injected_breaker_is_omitted_from_a_shadow(settings: Settings) -> None:
    breaker = InMemoryCircuitBreaker(
        CircuitBreakerConfig(failure_threshold=3, reset_timeout_seconds=30.0, half_open_deadline_seconds=60.0)
    )
    builder: Any = RuntimeBuilder(settings).with_circuit_breaker(breaker)
    composition = builder._capture_composition()
    shadow = RuntimeBuilder(settings).reproduce_from(composition)
    assert "circuit_breaker" in {omitted.name for omitted in shadow.omitted}
