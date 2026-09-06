"""Value objects describing a search backend's capabilities.

A backend that cannot serve a search mode declares it here and raises for that mode, so the retriever
skips the leg instead of failing every query. A backend that declares nothing (the built-in Local and
pgvector indexes, and any full hybrid such as the reference OpenSearch backend) is treated as a full
in-process hybrid, so the default path is unchanged. Only the two flags the runtime and the conformance
contract actually enforce live here; a backend's fail-closed transport behavior is asserted directly by
its adapter and the conformance suite, not by a flag.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SearchIndexCapabilities:
    supports_semantic: bool = True
    supports_lexical: bool = True
