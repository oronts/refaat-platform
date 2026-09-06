from __future__ import annotations

import contextlib
import hashlib
import importlib.machinery
import importlib.util
import os
import sys
from collections.abc import Callable
from importlib.metadata import EntryPoint, distributions, entry_points
from typing import Any

from refaat.domain.plugins import (
    ObservedPlugin,
    PluginDecision,
    PluginDescriptor,
    decide_plugin,
    manifest_digest,
)
from refaat.errors import ConfigurationError

PluginObserver = Callable[[ObservedPlugin, PluginDecision], None]

PLUGIN_PIN_GROUPS = (
    "refaat.skills",
    "refaat.extractors",
    "refaat.connectors",
    "refaat.event_sinks",
    "refaat.chunkers",
)

_FILE_LOADERS = (importlib.machinery.SourceFileLoader, importlib.machinery.SourcelessFileLoader)


class PluginRegistry:
    def __init__(
        self,
        allowlist: set[str] | None = None,
        *,
        pins: dict[str, PluginDescriptor] | None = None,
        observer: PluginObserver | None = None,
    ) -> None:
        self.allowlist = allowlist
        self._pins = pins or {}
        self._observer = observer

    def discover(self, group: str) -> dict[str, Any]:
        if not self._pins:
            discovered: dict[str, Any] = {}
            for entry_point in entry_points(group=group):
                if self.allowlist is not None and entry_point.name not in self.allowlist:
                    continue
                discovered[entry_point.name] = self._load(entry_point)
            return discovered
        return self._discover_pinned(group)

    def _discover_pinned(self, group: str) -> dict[str, Any]:
        # entry_points(group=...) deduplicates by name and keeps whichever distribution scans first,
        # so a squatter registering an allowlisted name from a second distribution would be resolved
        # silently. Count the name across every distribution and refuse an ambiguous duplicate.
        counts: dict[str, int] = {}
        for distribution in distributions():
            for point in distribution.entry_points:
                if point.group == group and (self.allowlist is None or point.name in self.allowlist):
                    counts[point.name] = counts.get(point.name, 0) + 1
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        if duplicates:
            raise ConfigurationError(f"duplicate entry-point names in group {group!r}: {', '.join(duplicates)}")
        discovered: dict[str, Any] = {}
        for point in entry_points(group=group):
            if self.allowlist is not None and point.name not in self.allowlist:
                continue
            pinned = self._pins.get(f"{group}:{point.name}")
            if pinned is None:
                raise ConfigurationError(f"no plugin pin for {group}:{point.name}")
            observed = _observe(point, pinned)
            decision = decide_plugin(observed, pinned)
            if not decision.allowed:
                if self._observer is not None:
                    self._observer(observed, decision)
                raise ConfigurationError(f"plugin {group}:{point.name} rejected ({decision.reason})")
            loaded = self._load(point)
            if self._observer is not None:
                self._observer(observed, decision)
            discovered[point.name] = loaded
        return discovered

    @staticmethod
    def _load(entry_point: EntryPoint) -> Any:
        try:
            return entry_point.load()
        except Exception as exc:
            raise ConfigurationError(f"Failed to load plugin {entry_point.name!r}") from exc


def _observe(entry_point: EntryPoint, pinned: PluginDescriptor) -> ObservedPlugin:
    value = entry_point.value
    dotted = value.split(":", 1)[0].strip()
    distribution = entry_point.dist
    if distribution is None:
        return ObservedPlugin(
            group=pinned.group,
            name=entry_point.name,
            value=value,
            distribution_present=False,
            dist_name="",
            dist_version="",
            record_present=False,
            source_file_listed=False,
            unlisted_importable_present=False,
            code_hash=None,
            origin_bound=False,
            preimport_bound=False,
        )
    files = distribution.files
    record_present = files is not None
    record_posix = {str(path) for path in files} if files is not None else set()
    dist_file_realpaths = _record_realpaths(distribution, files) if files is not None else set()
    code_hash = _code_hash(files) if files is not None else None
    source_candidates = {f"{dotted.replace('.', '/')}.py", f"{dotted.replace('.', '/')}/__init__.py"}
    source_file_listed = bool(source_candidates & record_posix)
    unlisted = _unlisted_importable_present(distribution, dotted, record_posix, dist_file_realpaths)
    origin_bound = _origin_bound(dotted, dist_file_realpaths)
    preimport_bound = _preimport_bound(dotted, dist_file_realpaths)
    return ObservedPlugin(
        group=pinned.group,
        name=entry_point.name,
        value=value,
        distribution_present=True,
        dist_name=distribution.name or "",
        dist_version=distribution.version or "",
        record_present=record_present,
        source_file_listed=source_file_listed,
        unlisted_importable_present=unlisted,
        code_hash=code_hash,
        origin_bound=origin_bound,
        preimport_bound=preimport_bound,
    )


def _is_dist_info(path: Any) -> bool:
    parts = path.parts
    return bool(parts) and parts[0].endswith((".dist-info", ".egg-info"))


def _record_realpaths(distribution: Any, files: Any) -> set[str]:
    realpaths: set[str] = set()
    for path in files:
        with contextlib.suppress(Exception):
            realpaths.add(os.path.realpath(os.fspath(distribution.locate_file(path))))
    return realpaths


def _code_hash(files: Any) -> str | None:
    entries: list[tuple[str, int, str]] = []
    for path in files:
        if _is_dist_info(path):
            continue
        try:
            data = path.read_binary()
        except OSError:
            return None
        entries.append((str(path), len(data), hashlib.sha256(data).hexdigest()))
    return manifest_digest(entries)


def _unlisted_importable_present(
    distribution: Any,
    dotted: str,
    record_posix: set[str],
    dist_file_realpaths: set[str],
) -> bool:
    # Scan the plugin's own top-level regular package tree for a source file present on disk but
    # absent from RECORD. A regular package directory belongs to a single distribution, so this
    # never sees another distribution's files; a namespace package (no recorded __init__) and a
    # top-level single-file module have no owned tree and are skipped rather than scanning a shared
    # site-packages root, which would falsely flag unrelated distributions.
    top = dotted.split(".", 1)[0]
    init_rel = f"{top}/__init__.py"
    if init_rel not in record_posix:
        return False
    resolved: str | None = None
    with contextlib.suppress(Exception):
        resolved = os.path.realpath(os.fspath(distribution.locate_file(init_rel)))
    if resolved is None or not os.path.isfile(resolved):
        return False
    package_dir = os.path.dirname(resolved)
    for current, subdirs, names in os.walk(package_dir):
        # Descend only into regular subpackages the pinned distribution itself records, so a namespace
        # portion contributed by another distribution into the same tree is never scanned or flagged.
        subdirs[:] = [
            sub for sub in subdirs if os.path.realpath(os.path.join(current, sub, "__init__.py")) in dist_file_realpaths
        ]
        for name in names:
            if not name.endswith(".py"):
                continue
            candidate_path = os.path.realpath(os.path.join(current, name))
            if os.path.isfile(candidate_path) and candidate_path not in dist_file_realpaths:
                return True
    return False


def _origin_bound(dotted: str, dist_file_realpaths: set[str]) -> bool:
    if not dist_file_realpaths:
        return False
    segments = dotted.split(".")
    try:
        top = importlib.util.find_spec(segments[0])
    except (ImportError, ValueError, AttributeError):
        return False
    if top is None:
        return False
    specs = [top]
    parent = top
    for depth in range(1, len(segments)):
        locations = parent.submodule_search_locations
        if locations is None:
            return False
        try:
            child = importlib.machinery.PathFinder.find_spec(".".join(segments[: depth + 1]), list(locations))
        except (ImportError, ValueError, AttributeError):
            return False
        if child is None:
            return False
        specs.append(child)
        parent = child
    for spec in specs:
        if not isinstance(spec.loader, _FILE_LOADERS) or spec.origin is None:
            return False
        if os.path.realpath(spec.origin) not in dist_file_realpaths:
            return False
    return True


def _preimport_bound(dotted: str, dist_file_realpaths: set[str]) -> bool:
    segments = dotted.split(".")
    for depth in range(1, len(segments) + 1):
        module = sys.modules.get(".".join(segments[:depth]))
        if module is None:
            continue
        origin = getattr(module, "__file__", None)
        if origin is None or os.path.realpath(origin) not in dist_file_realpaths:
            return False
    return True


def pin_for_entry_point(group: str, name: str) -> PluginDescriptor:
    """Compute the content-hash pin for an installed entry point without importing its code.

    Reads the providing distribution's shipped source through the same manifest the loader verifies,
    never calling entry_point.load(), and confirms the resulting pin would be admitted by decide_plugin
    so an operator never records a pin the runtime would then reject.
    """
    installations = sum(
        1
        for distribution in distributions()
        for point in distribution.entry_points
        if point.group == group and point.name == name
    )
    if installations == 0:
        raise ConfigurationError(f"no entry point {group}:{name} is installed")
    if installations > 1:
        raise ConfigurationError(
            f"entry point {group}:{name} is provided by more than one distribution; resolve the duplicate first"
        )
    point = next(candidate for candidate in entry_points(group=group) if candidate.name == name)
    distribution = point.dist
    if distribution is None:
        raise ConfigurationError(f"entry point {group}:{name} has no distribution to pin")
    files = distribution.files
    if files is None:
        raise ConfigurationError(f"the distribution for {group}:{name} ships no RECORD, so its code cannot be hashed")
    code_hash = _code_hash(files)
    if code_hash is None:
        raise ConfigurationError(f"the source for {group}:{name} is not listed in its RECORD, so it cannot be pinned")
    candidate = PluginDescriptor(
        group=group,
        name=name,
        value=point.value,
        dist_name=distribution.name or "",
        dist_version=distribution.version or "",
        code_hash=code_hash,
    )
    decision = decide_plugin(_observe(point, candidate), candidate)
    if not decision.allowed:
        raise ConfigurationError(f"the computed pin for {group}:{name} would be rejected ({decision.reason})")
    return candidate
