"""The port catalog stays in lockstep with the live typed Protocols."""

from __future__ import annotations

import inspect
import typing
from pathlib import Path

import pytest

import refaat.ports as ports
from refaat.application.runtime import RuntimeBuilder
from refaat.config import Settings
from refaat.devkit.catalog import PORT_CATALOG, TIERS, PortSpec, get_port, port_names, render_markdown


def _live_protocols() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for name in dir(ports):
        obj = getattr(ports, name)
        if not inspect.isclass(obj):
            continue
        if obj is typing.Protocol or not getattr(obj, "_is_protocol", False):
            continue
        if obj.__module__ != "refaat.ports":
            continue
        found[name] = set(getattr(obj, "__protocol_attrs__", set()))
    return found


def test_every_live_protocol_has_a_catalog_row_with_matching_members() -> None:
    live = _live_protocols()
    assert set(live) == set(port_names())
    for name, members in live.items():
        spec = get_port(name)
        assert spec is not None
        assert set(spec.members) == members


def test_no_stale_or_duplicate_catalog_rows() -> None:
    names = [spec.name for spec in PORT_CATALOG]
    assert len(names) == len(set(names))
    assert set(names) == set(_live_protocols())


def test_every_declared_seam_exists_in_the_code() -> None:
    builder_methods = {n for n in dir(RuntimeBuilder) if n.startswith("with_")}
    settings_fields = set(Settings.model_fields)
    groups = {"refaat.skills", "refaat.extractors", "refaat.connectors", "refaat.event_sinks", "refaat.chunkers"}
    for spec in PORT_CATALOG:
        assert spec.tier in TIERS
        if spec.tier == "builder":
            assert spec.seam in builder_methods
        elif spec.tier == "entry_point":
            assert spec.seam in groups
        elif spec.tier == "config":
            assert spec.seam in settings_fields
        else:
            assert spec.seam == ""


def test_portspec_rejects_an_invalid_tier_or_unsorted_members() -> None:
    with pytest.raises(ValueError, match="unknown tier"):
        PortSpec(name="X", members=("a",), tier="buildr", seam="", default="", conformance_base="XContract", notes="")
    with pytest.raises(ValueError, match="unique and sorted"):
        PortSpec(name="Y", members=("b", "a"), tier="none", seam="", default="", conformance_base="YContract", notes="")
    with pytest.raises(ValueError, match="none-tier"):
        PortSpec(name="Z", members=("a",), tier="none", seam="x", default="", conformance_base="ZContract", notes="")


def test_ports_catalog_doc_is_in_sync_with_the_catalog() -> None:
    doc = Path("docs/reference/ports-catalog.md")
    assert doc.read_text(encoding="utf-8") == render_markdown()
