from __future__ import annotations

from typing import Any

import pytest

from refaat.application import runtime as runtime_module
from refaat.application.runtime import RuntimeBuilder
from refaat.clock import Clock
from refaat.config import Settings
from refaat.core.text import STOPWORDS_BY_LANGUAGE


class ClosableResource:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_runtime_builder_closes_partial_resources_after_failure(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stores: list[FakeStore] = []

    class FakeStore:
        def __init__(self, database_url: str, clock: Clock, *, lock_max_waiters: int = 64) -> None:
            del database_url, clock, lock_max_waiters
            self.closed = False
            stores.append(self)

        def initialize(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    blob = ClosableResource()
    builder = RuntimeBuilder(settings)
    monkeypatch.setattr(runtime_module, "SQLStore", FakeStore)
    monkeypatch.setattr(builder, "_blob_store", lambda egress_policy=None: blob)

    def fail_embedder(egress_policy: Any = None) -> Any:
        raise RuntimeError("embedder setup failed")

    monkeypatch.setattr(builder, "_embedder", fail_embedder)

    with pytest.raises(RuntimeError, match="embedder setup failed"):
        builder.build()

    assert blob.closed
    assert len(stores) == 1
    assert stores[0].closed


class DualRoleSinkConnector:
    """An object registered as both an event sink and a connector."""

    def __init__(self) -> None:
        self.close_count = 0
        self.scheme = "dual"

    def emit(self, topic: str, payload: Any, *, event_id: str | None = None) -> None:
        del topic, payload, event_id

    def fetch(self, source: str) -> tuple[bytes, str, dict[str, Any]]:
        del source
        return b"", "text/plain", {}

    def close(self) -> None:
        self.close_count += 1


def test_dual_role_sink_connector_is_closed_once_on_build_failure(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dual = DualRoleSinkConnector()
    builder = RuntimeBuilder(settings).with_event_sink(dual).with_connector(dual)

    def fail_reranker() -> Any:
        raise RuntimeError("reranker setup failed")

    monkeypatch.setattr(builder, "_reranker", fail_reranker)

    with pytest.raises(RuntimeError, match="reranker setup failed"):
        builder.build()

    assert dual.close_count == 1


class SchemeClashConnector:
    def __init__(self, scheme: str) -> None:
        self.scheme = scheme
        self.close_count = 0

    def fetch(self, source: str) -> tuple[bytes, str, dict[str, Any]]:
        del source
        return b"", "text/plain", {}

    def close(self) -> None:
        self.close_count += 1


def test_dual_role_sink_connector_is_closed_once_when_connectors_build_fails(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure raised inside _connectors closes a dual-role object exactly once.

    The registration clash raises after the dual-role connector is registered, so
    the outer cleanup must own the registry's close through the shared already-closed
    set rather than an eager internal close that would double-close the sink.
    """
    dual = DualRoleSinkConnector()
    clash = SchemeClashConnector(dual.scheme)
    builder = RuntimeBuilder(settings).with_event_sink(dual).with_connector(dual).with_connector(clash)

    with pytest.raises(ValueError, match="already registered"):
        builder.build()

    assert dual.close_count == 1


def test_effective_stopwords_is_none_when_unconfigured(settings: Settings) -> None:
    builder = RuntimeBuilder(settings)
    assert builder._effective_stopwords() is None
    extractor = builder._graph_extractor()
    assert extractor._stopwords_by_language is None


def test_with_stopwords_merges_onto_the_shipped_defaults(settings: Settings) -> None:
    builder = RuntimeBuilder(settings).with_stopwords({"IT": ["Della", "Sono"], "en": ["Widget"]})
    registry = builder._effective_stopwords()
    assert registry is not None
    assert registry["it"] == frozenset({"della", "sono"})
    assert registry["en"] == frozenset({"widget"})
    assert registry["de"] is STOPWORDS_BY_LANGUAGE["de"]
    assert builder._graph_extractor()._stopwords_by_language == registry


def test_settings_stopwords_build_the_effective_registry(settings: Settings) -> None:
    configured = settings.model_copy(update={"stopwords_by_language": {"it": ["della", "sono"]}})
    builder = RuntimeBuilder(configured)
    registry = builder._effective_stopwords()
    assert registry is not None
    assert registry["it"] == frozenset({"della", "sono"})
    assert registry["en"] is STOPWORDS_BY_LANGUAGE["en"]
