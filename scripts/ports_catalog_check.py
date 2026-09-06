"""Fail closed when the port catalog drifts from the live typed Protocols.

A new port in refaat.ports must get a PORT_CATALOG row, a removed port must lose its row, and each
row's declared members and injection tier must match the code, so the catalog, the docs reference,
and the plugin CLI can never silently lie about a seam. Standard library plus the refaat package
only; no service required.
"""

from __future__ import annotations

import inspect
import sys
import typing
from pathlib import Path


def _protocol_members(obj: type) -> set[str]:
    return set(getattr(obj, "__protocol_attrs__", set()))


def _live_protocols() -> tuple[dict[str, set[str]], list[str]]:
    import refaat.ports as ports

    found: dict[str, set[str]] = {}
    foreign: list[str] = []
    for name in dir(ports):
        obj = getattr(ports, name)
        if not inspect.isclass(obj):
            continue
        if obj is typing.Protocol or not getattr(obj, "_is_protocol", False):
            continue
        if obj.__module__ != "refaat.ports":
            # A port re-exported from another module would bypass the catalog silently, so fail loudly
            # rather than drop it: ports are defined in refaat.ports.
            foreign.append(f"{name} (defined in {obj.__module__})")
            continue
        found[name] = _protocol_members(obj)
    return found, foreign


def _builder_takes_port(method: object, port_name: str) -> bool:
    try:
        hints = typing.get_type_hints(method)  # type: ignore[arg-type]
    except Exception:
        return False
    for key, value in hints.items():
        if key == "return":
            continue
        # Unwrap a union (SearchIndex | SearchIndexFactory) so an injectable seam that also accepts a
        # factory still matches the port name it is declared to take.
        for candidate in typing.get_args(value) or (value,):
            if getattr(candidate, "__name__", None) == port_name:
                return True
    return False


def main() -> int:
    from refaat.application.runtime import RuntimeBuilder
    from refaat.config import Settings
    from refaat.devkit.catalog import PORT_CATALOG, TIERS, render_markdown

    live, foreign = _live_protocols()
    catalog = {spec.name: spec for spec in PORT_CATALOG}
    errors: list[str] = []

    for entry in foreign:
        errors.append(
            f"Protocol {entry} is exposed in refaat.ports but defined elsewhere; define ports in refaat.ports"
        )

    missing = sorted(set(live) - set(catalog))
    for name in missing:
        errors.append(f"port {name!r} has no PORT_CATALOG row (add it with its members, tier, seam, and doc entry)")
    stale = sorted(set(catalog) - set(live))
    for name in stale:
        errors.append(f"PORT_CATALOG row {name!r} names no Protocol in refaat.ports (remove the stale row)")

    for name in sorted(set(live) & set(catalog)):
        declared = set(catalog[name].members)
        actual = live[name]
        if declared != actual:
            errors.append(f"port {name!r} member drift: catalog {sorted(declared)} vs live {sorted(actual)}")

    builder_methods = {n for n in dir(RuntimeBuilder) if n.startswith("with_")}
    settings_fields = set(getattr(Settings, "model_fields", {}))
    entry_point_groups = {
        "refaat.skills",
        "refaat.extractors",
        "refaat.connectors",
        "refaat.event_sinks",
        "refaat.chunkers",
    }
    for spec in PORT_CATALOG:
        if spec.tier not in TIERS:
            errors.append(f"port {spec.name!r} has unknown tier {spec.tier!r} (one of {sorted(TIERS)})")
        elif spec.tier == "builder":
            method = getattr(RuntimeBuilder, spec.seam, None)
            if spec.seam not in builder_methods or method is None:
                errors.append(
                    f"port {spec.name!r} claims builder seam {spec.seam!r} but RuntimeBuilder has no such method"
                )
            elif not _builder_takes_port(method, spec.name):
                errors.append(f"port {spec.name!r} builder seam {spec.seam!r} does not take a {spec.name}")
        elif spec.tier == "entry_point" and spec.seam not in entry_point_groups:
            errors.append(f"port {spec.name!r} claims entry-point group {spec.seam!r} which is not a discovery group")
        elif spec.tier == "config" and spec.seam not in settings_fields:
            errors.append(f"port {spec.name!r} claims config selector {spec.seam!r} which is not a Settings field")
        elif spec.tier == "none" and spec.seam:
            errors.append(f"port {spec.name!r} is tier none but declares a seam {spec.seam!r}")

    doc = Path("docs/reference/ports-catalog.md")
    if not doc.exists() or doc.read_text(encoding="utf-8") != render_markdown():
        errors.append(
            "docs/reference/ports-catalog.md is stale; run python scripts/generate_ports_catalog_doc.py --write"
        )

    if errors:
        print("Port catalog drift:")
        for line in errors:
            print(f"  - {line}")
        return 1
    print(f"Port catalog matches {len(live)} live Protocols.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
