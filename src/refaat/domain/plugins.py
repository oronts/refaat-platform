"""Plugin identity and code-hash pinning decision core.

Pure, stdlib-only, fail-closed decisions for whether a discovered entry point may load. A
:class:`PluginDescriptor` is the operator's pin: the exact target module, distribution name and
version, and the sha256 of a canonical manifest over the distribution's shipped source. An
:class:`ObservedPlugin` carries the facts an infrastructure resolver gathered import-free about the
entry point actually present on the path. ``decide_plugin`` compares the two and returns a
:class:`PluginDecision`; the digest comparison uses ``hmac.compare_digest`` so an equal-length code
hash cannot be probed byte by byte. This module reads no files and imports no plugin: the file
reads, the ``importlib.metadata`` resolution, and the origin binding live in the infrastructure
resolver. Importing this module changes nothing.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Iterable
from dataclasses import dataclass

_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_NAME_SEPARATORS = re.compile(r"[-_.]+")

REASON_ALLOWED = "pinned_ok"
REASON_VALUE_MISMATCH = "value_mismatch"
REASON_DISTRIBUTION_ABSENT = "distribution_absent"
REASON_DISTRIBUTION_NAME_MISMATCH = "distribution_name_mismatch"
REASON_VERSION_MISMATCH = "version_mismatch"
REASON_RECORD_MISSING = "record_missing"
REASON_SOURCE_UNLISTED = "source_unlisted"
REASON_UNLISTED_FILE_PRESENT = "unlisted_file_present"
REASON_CODE_HASH_MISMATCH = "code_hash_mismatch"
REASON_ORIGIN_UNBOUND = "origin_unbound"
REASON_PREIMPORT_MISMATCH = "preimport_mismatch"


def canonicalize_distribution_name(name: str) -> str:
    """The PEP 503 normalized form of a distribution name, so an underscore, dot, or case
    variant of the same project resolves to one identity for comparison."""
    return _NAME_SEPARATORS.sub("-", name).lower()


def canonical_manifest(entries: Iterable[tuple[str, int, str]]) -> bytes:
    """The one canonical byte form of a distribution's shipped source, for content addressing.

    Each entry is (posix relative path, byte length, sha256 hex of the content). The list is
    sorted and every field is length-delimited, so a path and the content that follows it are
    unambiguously bound: no rename, split, or content edit can produce the same bytes as another
    layout. The digest is a sha256 over this serialization.
    """
    out = bytearray()
    for path, length, digest in sorted(entries):
        path_bytes = path.encode("utf-8")
        digest_bytes = digest.encode("ascii")
        out += str(len(path_bytes)).encode("ascii") + b":" + path_bytes
        out += b"|" + str(length).encode("ascii")
        out += b"|" + str(len(digest_bytes)).encode("ascii") + b":" + digest_bytes
        out += b"\n"
    return bytes(out)


def manifest_digest(entries: Iterable[tuple[str, int, str]]) -> str:
    """The sha256 hex digest of the canonical manifest over the shipped source entries."""
    return hashlib.sha256(canonical_manifest(entries)).hexdigest()


def digests_match(observed: str, pinned: str) -> bool:
    """Constant-time equality of two code hashes, so a mismatch reveals nothing byte by byte."""
    return hmac.compare_digest(observed, pinned)


@dataclass(frozen=True, slots=True)
class PluginDescriptor:
    """The operator's pin for one entry point, keyed by its ``group`` and ``name``.

    ``value`` is the exact ``module:attr`` (or ``module``) target, ``dist_name`` and
    ``dist_version`` are the providing distribution, and ``code_hash`` is the sha256 of the
    canonical manifest over that distribution's shipped source. A load is admitted only when the
    entry point on the path matches every field; any drift forces the operator to re-pin.
    """

    group: str
    name: str
    value: str
    dist_name: str
    dist_version: str
    code_hash: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("plugin pin value must be a non-empty target")
        if not self.dist_name:
            raise ValueError("plugin pin dist_name must be a non-empty distribution name")
        if not self.dist_version:
            raise ValueError("plugin pin dist_version must be a non-empty version")
        if not _HEX_DIGEST.match(self.code_hash):
            raise ValueError("plugin pin code_hash must be a lowercase 64-character sha256 hex digest")


@dataclass(frozen=True, slots=True)
class ObservedPlugin:
    """The import-free facts an infrastructure resolver gathered about an entry point on the path.

    ``code_hash`` is the manifest digest computed from the distribution's RECORD-listed source, or
    None when it could not be computed (RECORD absent, or a listed file unreadable). The booleans
    record whether the RECORD lists the target source file, whether the target's package directory
    holds an importable file the RECORD does not list, whether the full import origin chain resolved
    inside the pinned distribution, and whether every already-imported level's file is inside it.
    """

    group: str
    name: str
    value: str
    distribution_present: bool
    dist_name: str
    dist_version: str
    record_present: bool
    source_file_listed: bool
    unlisted_importable_present: bool
    code_hash: str | None
    origin_bound: bool
    preimport_bound: bool


@dataclass(frozen=True, slots=True)
class PluginDecision:
    allowed: bool
    reason: str


def decide_plugin(observed: ObservedPlugin, pinned: PluginDescriptor) -> PluginDecision:
    """Whether an observed entry point may load against its pin, fail-closed.

    The checks run in a fixed order so the first drift is the reported reason: the import target,
    the providing distribution's name and version, the RECORD's presence and coverage of the target
    source, a code file on disk the RECORD omits, the canonical code-hash match, the import origin
    bound inside the distribution, and every already-imported level bound inside it. A load is
    admitted only when all pass, so no unpinned bytes reach ``entry_point.load``.
    """
    if observed.value != pinned.value:
        return PluginDecision(False, REASON_VALUE_MISMATCH)
    if not observed.distribution_present:
        return PluginDecision(False, REASON_DISTRIBUTION_ABSENT)
    if canonicalize_distribution_name(observed.dist_name) != canonicalize_distribution_name(pinned.dist_name):
        return PluginDecision(False, REASON_DISTRIBUTION_NAME_MISMATCH)
    if observed.dist_version != pinned.dist_version:
        return PluginDecision(False, REASON_VERSION_MISMATCH)
    if not observed.record_present:
        return PluginDecision(False, REASON_RECORD_MISSING)
    if not observed.source_file_listed:
        return PluginDecision(False, REASON_SOURCE_UNLISTED)
    if observed.unlisted_importable_present:
        return PluginDecision(False, REASON_UNLISTED_FILE_PRESENT)
    if observed.code_hash is None or not digests_match(observed.code_hash, pinned.code_hash):
        return PluginDecision(False, REASON_CODE_HASH_MISMATCH)
    if not observed.origin_bound:
        return PluginDecision(False, REASON_ORIGIN_UNBOUND)
    if not observed.preimport_bound:
        return PluginDecision(False, REASON_PREIMPORT_MISMATCH)
    return PluginDecision(True, REASON_ALLOWED)


__all__ = [
    "REASON_ALLOWED",
    "REASON_CODE_HASH_MISMATCH",
    "REASON_DISTRIBUTION_ABSENT",
    "REASON_DISTRIBUTION_NAME_MISMATCH",
    "REASON_ORIGIN_UNBOUND",
    "REASON_PREIMPORT_MISMATCH",
    "REASON_RECORD_MISSING",
    "REASON_SOURCE_UNLISTED",
    "REASON_UNLISTED_FILE_PRESENT",
    "REASON_VALUE_MISMATCH",
    "REASON_VERSION_MISMATCH",
    "ObservedPlugin",
    "PluginDecision",
    "PluginDescriptor",
    "canonical_manifest",
    "canonicalize_distribution_name",
    "decide_plugin",
    "digests_match",
    "manifest_digest",
]
