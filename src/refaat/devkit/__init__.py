"""Plugin-authoring devkit: a machine-readable catalog of the runtime's typed ports.

The devkit is runtime-inert. Nothing here is imported by the query, ingest, or RLM paths; it exists
for the plugin author and the ports-catalog drift gate. The catalog is the single source of truth
consumed by the docs reference, the CLI, and the gate, so they cannot drift from the live ports.
"""

from refaat.devkit.catalog import PORT_CATALOG, PortSpec, get_port, port_names

__all__ = ["PORT_CATALOG", "PortSpec", "get_port", "port_names"]
