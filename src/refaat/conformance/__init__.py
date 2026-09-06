"""The plugin conformance kit: check that an adapter satisfies a port's contract.

Structural conformance (every member the port declares is present) is the universal check every
adapter must pass; it is what `refaat plugin verify` runs. Per-port invariant contracts (the citation
contract, fail-closed egress) build on this and are added per port. The kit ships in the package so a
scaffolded plugin's test can import it, and it is never imported by a served request path.
"""

from refaat.conformance.contract import ConformanceViolation, structural_conformance
from refaat.conformance.graph_read import GraphReadConformanceCase, graph_read_conformance
from refaat.conformance.search_index import SearchIndexConformanceCase, search_index_conformance

__all__ = [
    "ConformanceViolation",
    "GraphReadConformanceCase",
    "SearchIndexConformanceCase",
    "graph_read_conformance",
    "search_index_conformance",
    "structural_conformance",
]
