"""Plugin pinning enforced over real temporary distributions.

Each test builds a throwaway distribution (a package plus a ``.dist-info`` with METADATA, a RECORD,
and an entry point) on a temp path and drives ``PluginRegistry.discover`` against a pin. A matching
entry point is admitted; a version, target, hash, unlisted-file, namespace, missing-RECORD,
duplicate-name, sys.path shadow, or pre-imported drift is refused. The audit observer emits a
redacted plugin.loaded and plugin.rejected, pins-total refuses a forgotten pin, the fail-closed
audit guard fails startup without a sink, and an empty pin set leaves discovery byte-identical.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import pytest

from refaat.application.runtime import RuntimeBuilder
from refaat.clock import SystemClock
from refaat.config import Settings
from refaat.domain.audit import AuditEvent
from refaat.domain.plugins import ObservedPlugin, PluginDecision, PluginDescriptor, manifest_digest
from refaat.errors import ConfigurationError
from refaat.infrastructure.plugins import PluginObserver, PluginRegistry

GROUP = "refaat.skills"
SKILL_SRC = (
    "class DemoSkill:\n"
    "    name = 'demo'\n"
    "    description = 'demo skill'\n\n"
    "    def run(self, context, arguments):\n"
    "        return {}\n"
)


class _CapturingSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        self.events.append(event)


class _RecordingObserver:
    def __init__(self) -> None:
        self.calls: list[tuple[ObservedPlugin, PluginDecision]] = []

    def observe(self, observed: ObservedPlugin, decision: PluginDecision) -> None:
        self.calls.append((observed, decision))


def _code_hash(code_files: dict[str, str]) -> str:
    entries = [
        (path, len(content.encode("utf-8")), hashlib.sha256(content.encode("utf-8")).hexdigest())
        for path, content in code_files.items()
    ]
    return manifest_digest(entries)


def _make_dist(
    root: Path,
    dist_name: str,
    version: str,
    code_files: dict[str, str],
    entry_points_text: str,
    *,
    write_record: bool = True,
    extra_disk_files: dict[str, str] | None = None,
) -> str:
    root.mkdir(parents=True, exist_ok=True)
    for rel, content in code_files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    for rel, content in (extra_disk_files or {}).items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    dist_info = root / f"{dist_name}-{version}.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {dist_name}\nVersion: {version}\n", encoding="utf-8"
    )
    (dist_info / "entry_points.txt").write_text(entry_points_text, encoding="utf-8")
    if write_record:
        record_lines = [f"{rel},," for rel in code_files]
        record_lines += [
            f"{dist_name}-{version}.dist-info/METADATA,,",
            f"{dist_name}-{version}.dist-info/entry_points.txt,,",
            f"{dist_name}-{version}.dist-info/RECORD,,",
        ]
        (dist_info / "RECORD").write_text("\n".join(record_lines) + "\n", encoding="utf-8")
    return _code_hash(code_files)


@contextlib.contextmanager
def _on_path(*roots: Path) -> Iterator[None]:
    added = [str(root) for root in roots]
    before = set(sys.modules)
    for entry in reversed(added):
        sys.path.insert(0, entry)
    importlib.invalidate_caches()
    try:
        yield
    finally:
        for entry in added:
            with contextlib.suppress(ValueError):
                sys.path.remove(entry)
        for name in set(sys.modules) - before:
            sys.modules.pop(name, None)
        importlib.invalidate_caches()


def _descriptor(name: str, value: str, dist_name: str, version: str, code_hash: str) -> PluginDescriptor:
    return PluginDescriptor(
        group=GROUP, name=name, value=value, dist_name=dist_name, dist_version=version, code_hash=code_hash
    )


def _observer(sink: _CapturingSink, *, fail_closed: bool = False) -> PluginObserver:
    builder = RuntimeBuilder(Settings(audit_fail_closed=fail_closed))
    builder.with_audit_sink(sink)
    return builder._plugin_audit_observer(SystemClock())


def test_pinned_load_admits_and_audits(tmp_path: Path) -> None:
    root = tmp_path / "site"
    code = {"okpkg/__init__.py": "", "okpkg/tool.py": SKILL_SRC}
    ch = _make_dist(root, "ok-pkg", "1.0.0", code, f"[{GROUP}]\ndemo = okpkg.tool:DemoSkill\n")
    pins = {f"{GROUP}:demo": _descriptor("demo", "okpkg.tool:DemoSkill", "ok_pkg", "1.0.0", ch)}
    sink = _CapturingSink()
    registry = PluginRegistry({"demo"}, pins=pins, observer=_observer(sink))
    with _on_path(root):
        discovered = registry.discover(GROUP)
    assert set(discovered) == {"demo"}
    assert discovered["demo"].name == "demo"
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.event_type == "plugin.loaded"
    assert event.capability_name == f"{GROUP}:demo"
    assert event.action_fingerprint == ch[:16]
    assert event.principal_id == "system"
    assert event.principal_kind == "system"
    assert str(root) not in json.dumps(event.to_json())


def test_version_swap_refuses_and_audits(tmp_path: Path) -> None:
    root = tmp_path / "site"
    code = {"okpkg/__init__.py": "", "okpkg/tool.py": SKILL_SRC}
    ch = _make_dist(root, "ok-pkg", "1.0.0", code, f"[{GROUP}]\ndemo = okpkg.tool:DemoSkill\n")
    pins = {f"{GROUP}:demo": _descriptor("demo", "okpkg.tool:DemoSkill", "ok_pkg", "9.9.9", ch)}
    sink = _CapturingSink()
    registry = PluginRegistry({"demo"}, pins=pins, observer=_observer(sink))
    with _on_path(root), pytest.raises(ConfigurationError, match="version_mismatch"):
        registry.discover(GROUP)
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.event_type == "plugin.rejected"
    assert event.action_fingerprint == "version_mismatch"
    assert str(root) not in json.dumps(event.to_json())


def test_target_redirect_refuses(tmp_path: Path) -> None:
    root = tmp_path / "site"
    code = {"okpkg/__init__.py": "", "okpkg/tool.py": SKILL_SRC}
    ch = _make_dist(root, "ok-pkg", "1.0.0", code, f"[{GROUP}]\ndemo = okpkg.tool:DemoSkill\n")
    pins = {f"{GROUP}:demo": _descriptor("demo", "okpkg.tool:Other", "ok_pkg", "1.0.0", ch)}
    registry = PluginRegistry({"demo"}, pins=pins)
    with _on_path(root), pytest.raises(ConfigurationError, match="value_mismatch"):
        registry.discover(GROUP)


def test_code_hash_mismatch_refuses(tmp_path: Path) -> None:
    root = tmp_path / "site"
    code = {"okpkg/__init__.py": "", "okpkg/tool.py": SKILL_SRC}
    _make_dist(root, "ok-pkg", "1.0.0", code, f"[{GROUP}]\ndemo = okpkg.tool:DemoSkill\n")
    pins = {f"{GROUP}:demo": _descriptor("demo", "okpkg.tool:DemoSkill", "ok_pkg", "1.0.0", "0" * 64)}
    registry = PluginRegistry({"demo"}, pins=pins)
    with _on_path(root), pytest.raises(ConfigurationError, match="code_hash_mismatch"):
        registry.discover(GROUP)


def test_unlisted_importable_file_refuses(tmp_path: Path) -> None:
    root = tmp_path / "site"
    code = {"okpkg/__init__.py": "", "okpkg/tool.py": SKILL_SRC}
    ch = _make_dist(
        root,
        "ok-pkg",
        "1.0.0",
        code,
        f"[{GROUP}]\ndemo = okpkg.tool:DemoSkill\n",
        extra_disk_files={"okpkg/injected.py": "VALUE = 1\n"},
    )
    pins = {f"{GROUP}:demo": _descriptor("demo", "okpkg.tool:DemoSkill", "ok_pkg", "1.0.0", ch)}
    registry = PluginRegistry({"demo"}, pins=pins)
    with _on_path(root), pytest.raises(ConfigurationError, match="unlisted_file_present"):
        registry.discover(GROUP)


def test_unlisted_file_in_subpackage_refuses(tmp_path: Path) -> None:
    root = tmp_path / "site"
    code = {"okpkg/__init__.py": "", "okpkg/tool.py": SKILL_SRC, "okpkg/deep/__init__.py": ""}
    ch = _make_dist(
        root,
        "ok-pkg",
        "1.0.0",
        code,
        f"[{GROUP}]\ndemo = okpkg.tool:DemoSkill\n",
        extra_disk_files={"okpkg/deep/injected.py": "VALUE = 1\n"},
    )
    pins = {f"{GROUP}:demo": _descriptor("demo", "okpkg.tool:DemoSkill", "ok_pkg", "1.0.0", ch)}
    registry = PluginRegistry({"demo"}, pins=pins)
    with _on_path(root), pytest.raises(ConfigurationError, match="unlisted_file_present"):
        registry.discover(GROUP)


def test_namespace_subpackage_from_other_distribution_admits(tmp_path: Path) -> None:
    root = tmp_path / "site"
    code = {"okpkg/__init__.py": "", "okpkg/tool.py": SKILL_SRC}
    ch = _make_dist(
        root,
        "ok-pkg",
        "1.0.0",
        code,
        f"[{GROUP}]\ndemo = okpkg.tool:DemoSkill\n",
        # A namespace subpackage portion (no __init__.py) that a different distribution co-installs
        # into the same tree must not be scanned or flagged against this distribution's RECORD.
        extra_disk_files={"okpkg/ext/plugin_b.py": "VALUE = 1\n"},
    )
    pins = {f"{GROUP}:demo": _descriptor("demo", "okpkg.tool:DemoSkill", "ok_pkg", "1.0.0", ch)}
    registry = PluginRegistry({"demo"}, pins=pins)
    with _on_path(root):
        discovered = registry.discover(GROUP)
    assert set(discovered) == {"demo"}


def test_unrelated_duplicate_name_outside_allowlist_is_ignored(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    ch = _make_dist(
        root_a,
        "ok-pkg",
        "1.0.0",
        {"okpkg/__init__.py": "", "okpkg/tool.py": SKILL_SRC},
        f"[{GROUP}]\ndemo = okpkg.tool:DemoSkill\nlogger = okpkg.tool:DemoSkill\n",
    )
    _make_dist(
        root_b,
        "other-pkg",
        "1.0.0",
        {"otherpkg/__init__.py": "", "otherpkg/tool.py": SKILL_SRC},
        f"[{GROUP}]\nlogger = otherpkg.tool:DemoSkill\n",
    )
    # 'logger' is duplicated across unrelated distributions but is not allowlisted, so it must not
    # fail discovery of the allowlisted, pinned 'demo'.
    pins = {f"{GROUP}:demo": _descriptor("demo", "okpkg.tool:DemoSkill", "ok_pkg", "1.0.0", ch)}
    registry = PluginRegistry({"demo"}, pins=pins)
    with _on_path(root_a, root_b):
        discovered = registry.discover(GROUP)
    assert set(discovered) == {"demo"}


def test_load_failure_does_not_emit_loaded(tmp_path: Path) -> None:
    root = tmp_path / "site"
    code = {"okpkg/__init__.py": "", "okpkg/tool.py": "raise RuntimeError('boom')\n"}
    ch = _make_dist(root, "ok-pkg", "1.0.0", code, f"[{GROUP}]\ndemo = okpkg.tool:DemoSkill\n")
    pins = {f"{GROUP}:demo": _descriptor("demo", "okpkg.tool:DemoSkill", "ok_pkg", "1.0.0", ch)}
    sink = _CapturingSink()
    registry = PluginRegistry({"demo"}, pins=pins, observer=_observer(sink))
    with _on_path(root), pytest.raises(ConfigurationError, match="Failed to load"):
        registry.discover(GROUP)
    assert all(event.event_type != "plugin.loaded" for event in sink.events)


def test_missing_record_refuses(tmp_path: Path) -> None:
    root = tmp_path / "site"
    code = {"recpkg/__init__.py": "", "recpkg/tool.py": SKILL_SRC}
    _make_dist(root, "rec-pkg", "1.0.0", code, f"[{GROUP}]\ndemo = recpkg.tool:DemoSkill\n", write_record=False)
    pins = {f"{GROUP}:demo": _descriptor("demo", "recpkg.tool:DemoSkill", "rec_pkg", "1.0.0", "a" * 64)}
    registry = PluginRegistry({"demo"}, pins=pins)
    with _on_path(root), pytest.raises(ConfigurationError, match="record_missing"):
        registry.discover(GROUP)


def test_namespace_package_refuses(tmp_path: Path) -> None:
    root = tmp_path / "site"
    code = {"nspkg/sub.py": SKILL_SRC}  # no __init__.py, so nspkg is a namespace package
    ch = _make_dist(root, "ns-pkg", "1.0.0", code, f"[{GROUP}]\ndemo = nspkg.sub:DemoSkill\n")
    pins = {f"{GROUP}:demo": _descriptor("demo", "nspkg.sub:DemoSkill", "ns_pkg", "1.0.0", ch)}
    registry = PluginRegistry({"demo"}, pins=pins)
    with _on_path(root), pytest.raises(ConfigurationError, match="origin_unbound"):
        registry.discover(GROUP)


def test_duplicate_entry_point_name_refuses(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    ch = _make_dist(
        root_a,
        "dup-a",
        "1.0.0",
        {"dupa/__init__.py": "", "dupa/tool.py": SKILL_SRC},
        f"[{GROUP}]\ndup = dupa.tool:DemoSkill\n",
    )
    _make_dist(
        root_b,
        "dup-b",
        "1.0.0",
        {"dupb/__init__.py": "", "dupb/tool.py": SKILL_SRC},
        f"[{GROUP}]\ndup = dupb.tool:DemoSkill\n",
    )
    pins = {f"{GROUP}:dup": _descriptor("dup", "dupa.tool:DemoSkill", "dup_a", "1.0.0", ch)}
    registry = PluginRegistry({"dup"}, pins=pins)
    with _on_path(root_a, root_b), pytest.raises(ConfigurationError, match="duplicate entry-point names"):
        registry.discover(GROUP)


def test_sys_path_shadow_refuses(tmp_path: Path) -> None:
    pinned = tmp_path / "pinned"
    shadow = tmp_path / "shadow"
    code = {"shadowpkg/__init__.py": "", "shadowpkg/tool.py": SKILL_SRC}
    ch = _make_dist(pinned, "shadow-pkg", "1.0.0", code, f"[{GROUP}]\ndemo = shadowpkg.tool:DemoSkill\n")
    # A same-named package earlier on sys.path, without a dist-info, so it registers no entry point
    # yet wins import resolution: the origin bind must reject it.
    (shadow / "shadowpkg").mkdir(parents=True)
    (shadow / "shadowpkg" / "__init__.py").write_text("", encoding="utf-8")
    (shadow / "shadowpkg" / "tool.py").write_text("class DemoSkill:\n    name = 'evil'\n", encoding="utf-8")
    pins = {f"{GROUP}:demo": _descriptor("demo", "shadowpkg.tool:DemoSkill", "shadow_pkg", "1.0.0", ch)}
    registry = PluginRegistry({"demo"}, pins=pins)
    with _on_path(shadow, pinned), pytest.raises(ConfigurationError, match="origin_unbound"):
        registry.discover(GROUP)


def test_preimported_module_with_foreign_file_refuses(tmp_path: Path) -> None:
    root = tmp_path / "site"
    code = {"prepkg/__init__.py": "", "prepkg/sub.py": SKILL_SRC}
    ch = _make_dist(root, "pre-pkg", "1.0.0", code, f"[{GROUP}]\ndemo = prepkg.sub:DemoSkill\n")
    pins = {f"{GROUP}:demo": _descriptor("demo", "prepkg.sub:DemoSkill", "pre_pkg", "1.0.0", ch)}
    registry = PluginRegistry({"demo"}, pins=pins)
    foreign = types.ModuleType("prepkg.sub")
    foreign.__file__ = str(tmp_path / "elsewhere" / "sub.py")
    with _on_path(root):
        sys.modules["prepkg.sub"] = foreign
        with pytest.raises(ConfigurationError, match="preimport_mismatch"):
            registry.discover(GROUP)


def test_empty_pins_is_byte_identical(tmp_path: Path) -> None:
    root = tmp_path / "site"
    code = {"okpkg/__init__.py": "", "okpkg/tool.py": SKILL_SRC}
    _make_dist(root, "ok-pkg", "1.0.0", code, f"[{GROUP}]\ndemo = okpkg.tool:DemoSkill\n")
    spy = _RecordingObserver()
    plain = PluginRegistry({"demo"})
    empty_pinned = PluginRegistry({"demo"}, pins=None, observer=spy.observe)
    with _on_path(root):
        result_plain = plain.discover(GROUP)
        result_empty = empty_pinned.discover(GROUP)
    assert set(result_plain) == set(result_empty) == {"demo"}
    assert result_plain["demo"] is result_empty["demo"]
    assert spy.calls == []


def test_pins_total_refuses_a_forgotten_pin(tmp_path: Path) -> None:
    root = tmp_path / "site"
    code = {"twopkg/__init__.py": "", "twopkg/tool.py": SKILL_SRC}
    ch = _make_dist(
        root,
        "two-pkg",
        "1.0.0",
        code,
        f"[{GROUP}]\ndemo = twopkg.tool:DemoSkill\nother = twopkg.tool:DemoSkill\n",
    )
    settings = Settings(
        plugin_allowlist=["demo", "other"],
        plugin_pins={
            f"{GROUP}:demo": {
                "value": "twopkg.tool:DemoSkill",
                "dist_name": "two_pkg",
                "dist_version": "1.0.0",
                "code_hash": ch,
            }
        },
    )
    builder = RuntimeBuilder(settings)
    with _on_path(root), pytest.raises(ConfigurationError, match="no pin"):
        builder._verify_plugin_pins_total(builder._plugin_pins())


def test_fail_closed_audit_without_sink_fails_startup(settings: Settings) -> None:
    updated = settings.model_copy(
        update={
            "audit_fail_closed": True,
            "plugin_pins": {
                f"{GROUP}:demo": {
                    "value": "m.tool:C",
                    "dist_name": "d",
                    "dist_version": "1.0.0",
                    "code_hash": "a" * 64,
                }
            },
        }
    )
    with pytest.raises(ConfigurationError, match="requires an audit sink"):
        RuntimeBuilder(updated).build()
