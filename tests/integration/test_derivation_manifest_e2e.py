"""Derivation manifest: a corpus materialization records a content-addressed manifest when enabled."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from refaat.application.derivation import DerivationManifest, compare_manifests
from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.config import Settings
from refaat.domain.models import DocumentInput

pytestmark = pytest.mark.integration

_BODY = b"The production release checklist requires reviewer approval before deployment."
_OTHER = b"Warehouse throughput and carrier routing options are covered by the logistics report."


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {
        "environment": "test",
        "database_url": f"sqlite:///{tmp_path / 'refaat.db'}",
        "auto_create_schema": True,
        "blob_root": tmp_path / "blobs",
        "connector_file_root": tmp_path / "connectors",
        "skill_root": tmp_path / "skills",
        "api_key_pepper": "test-pepper-with-enough-entropy",
        "bootstrap_token": "bootstrap-test-token",
        "log_json": False,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _ingest(runtime: Refaat, ws: str, corpus: str, source_id: str, body: bytes) -> str:
    return runtime.ingest(
        ws,
        corpus,
        DocumentInput(source_id=source_id, title=source_id, media_type="text/plain", content=body, metadata={}),
    ).document_id


def test_manifest_written_and_verifies_when_enabled(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_settings(tmp_path, derivation_manifest_enabled=True)).build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "handbook")
        _ingest(runtime, ws, corpus, "a", _BODY)

        manifests = runtime.store.list_derivation_manifests(ws, corpus)
        assert len(manifests) == 1
        manifest = manifests[0]
        assert manifest["status"] == "ready"
        assert manifest["row_counts"]["documents"] == 1
        assert manifest["row_counts"]["chunks"] >= 1
        assert manifest["content_checksum"]
        assert runtime.ingestion.derivation_service is not None
        assert runtime.ingestion.derivation_service.verify(ws, corpus) is True
        first_checksum = manifest["content_checksum"]

        # A second document under the same recipe refreshes the same manifest row and moves the checksum.
        _ingest(runtime, ws, corpus, "b", _OTHER)
        refreshed = runtime.store.list_derivation_manifests(ws, corpus)
        assert len(refreshed) == 1
        assert refreshed[0]["row_counts"]["documents"] == 2
        assert refreshed[0]["content_checksum"] != first_checksum
        assert refreshed[0]["derivation_key"] == manifest["derivation_key"]
    finally:
        runtime.close()


def test_manifest_refreshes_after_erase(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_settings(tmp_path, derivation_manifest_enabled=True)).build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "handbook")
        doc_a = _ingest(runtime, ws, corpus, "a", _BODY)
        _ingest(runtime, ws, corpus, "b", _OTHER)
        assert runtime.store.list_derivation_manifests(ws, corpus)[0]["row_counts"]["documents"] == 2

        runtime.project(workspace="acme", corpus="handbook").erase(doc_a)
        manifest = runtime.store.list_derivation_manifests(ws, corpus)[0]
        assert manifest["row_counts"]["documents"] == 1
        assert runtime.ingestion.derivation_service is not None
        assert runtime.ingestion.derivation_service.verify(ws, corpus) is True
    finally:
        runtime.close()


def test_a_recipe_change_creates_a_second_manifest_row(tmp_path: Path) -> None:
    first = RuntimeBuilder(
        _settings(tmp_path, derivation_manifest_enabled=True, default_chunk_tokens=32, chunk_overlap_tokens=4)
    ).build()
    try:
        ws = first.create_workspace("acme", "Acme")
        corpus = first.create_corpus(ws, "handbook")
        _ingest(first, ws, corpus, "a", _BODY)
        key_a = first.store.list_derivation_manifests(ws, corpus)[0]["derivation_key"]
    finally:
        first.close()

    second = RuntimeBuilder(
        _settings(tmp_path, derivation_manifest_enabled=True, default_chunk_tokens=512, chunk_overlap_tokens=0)
    ).build()
    try:
        _ingest(second, ws, corpus, "b", _OTHER)
        manifests = second.store.list_derivation_manifests(ws, corpus)
        assert len(manifests) == 2
        assert key_a in {manifest["derivation_key"] for manifest in manifests}
    finally:
        second.close()


def test_no_manifest_when_disabled(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_settings(tmp_path)).build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "handbook")
        _ingest(runtime, ws, corpus, "a", _BODY)
        assert runtime.store.list_derivation_manifests(ws, corpus) == []
        assert runtime.ingestion.derivation_service is None
        # The read-surface accessor is None when the feature is off, so the HTTP/CLI/SDK surface 404s.
        assert runtime.derivation is None
    finally:
        runtime.close()


def test_compare_reports_every_field_including_config_and_status() -> None:
    base = DerivationManifest(
        derivation_key="k",
        corpus_version=1,
        config={"chunk_tokens": 64},
        embedding_model="m",
        embedding_dimensions=8,
        row_counts={"documents": 1},
        content_checksum="c",
        status="ready",
    )
    assert all(compare_manifests(base, base).values())

    # Differ only in config: identical is false and exactly same_config among the same_* fields is false.
    on_config = compare_manifests(base, replace(base, config={"chunk_tokens": 512}))
    assert on_config["identical"] is False
    assert on_config["same_config"] is False
    assert on_config["same_status"] is True
    assert on_config["same_content_checksum"] is True

    # Differ only in status: without a same_status field this would report identical=false with every
    # same_* true, the reporting gap this field completes.
    on_status = compare_manifests(base, replace(base, status="building"))
    assert on_status["identical"] is False
    assert on_status["same_status"] is False
    assert on_status["same_config"] is True


def test_read_surface_lists_gets_verifies_and_compares(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_settings(tmp_path, derivation_manifest_enabled=True)).build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "handbook")
        _ingest(runtime, ws, corpus, "a", _BODY)
        service = runtime.derivation
        assert service is not None

        manifests = service.manifests(ws, corpus)
        assert len(manifests) == 1
        key = manifests[0].derivation_key
        current = service.current(ws, corpus)
        assert current is not None
        assert current.derivation_key == key
        assert service.manifest(ws, corpus, key) == current
        assert service.manifest(ws, corpus, "not-a-real-key") is None
        assert service.verify_detail(ws, corpus) == "match"

        comparison = compare_manifests(current, current)
        assert comparison["identical"] is True
        assert comparison["same_content_checksum"] is True
    finally:
        runtime.close()


def test_verify_detail_reports_no_manifest_before_any_ingest(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_settings(tmp_path, derivation_manifest_enabled=True)).build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "handbook")
        service = runtime.derivation
        assert service is not None
        assert service.manifests(ws, corpus) == []
        assert service.current(ws, corpus) is None
        assert service.verify_detail(ws, corpus) == "no_manifest"
    finally:
        runtime.close()


def test_compare_distinguishes_two_recipes(tmp_path: Path) -> None:
    first = RuntimeBuilder(
        _settings(tmp_path, derivation_manifest_enabled=True, default_chunk_tokens=32, chunk_overlap_tokens=4)
    ).build()
    try:
        ws = first.create_workspace("acme", "Acme")
        corpus = first.create_corpus(ws, "handbook")
        _ingest(first, ws, corpus, "a", _BODY)
        assert first.derivation is not None
        left = first.derivation.current(ws, corpus)
        assert left is not None
    finally:
        first.close()

    second = RuntimeBuilder(
        _settings(tmp_path, derivation_manifest_enabled=True, default_chunk_tokens=512, chunk_overlap_tokens=0)
    ).build()
    try:
        _ingest(second, ws, corpus, "b", _OTHER)
        assert second.derivation is not None
        by_key = {manifest.derivation_key: manifest for manifest in second.derivation.manifests(ws, corpus)}
        assert len(by_key) == 2
        other_key = next(key for key in by_key if key != left.derivation_key)
        comparison = compare_manifests(left, by_key[other_key])
        assert comparison["same_derivation_key"] is False
        assert comparison["identical"] is False
    finally:
        second.close()
