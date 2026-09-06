"""Anti-bypass census: no infrastructure module constructs a bare httpx client.

Every outbound httpx seam must build its owned client through ``egress_client.build_http_client`` so
a new outbound path cannot silently skip the egress chokepoint. This AST census enumerates every
``httpx.Client(...)`` and ``httpx.AsyncClient(...)`` construction under ``src/refaat/infrastructure``
and asserts the only module that constructs one directly is ``egress_client`` itself (the factory).

Scope: httpx-in-infrastructure only. boto3 (the S3 blob store), OTLP gRPC, and any user-injected
port that does its own I/O are out of scope by construction and are covered by the deploy-layer
NetworkPolicy backstop, documented in docs/deployment-hardening.md.
"""

from __future__ import annotations

import ast
import pathlib

_INFRA_ROOT = pathlib.Path(__file__).resolve().parents[2] / "src" / "refaat" / "infrastructure"
# The only module allowed to construct httpx.Client directly: the guarded factory itself.
_EXEMPT = {"egress_client.py"}


def _constructs_httpx_client(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in ("Client", "AsyncClient"):
            value = func.value
            if isinstance(value, ast.Name) and value.id == "httpx":
                return True
    return False


def test_no_infrastructure_module_builds_a_bare_httpx_client() -> None:
    offenders = []
    for path in sorted(_INFRA_ROOT.rglob("*.py")):
        if path.name in _EXEMPT:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if _constructs_httpx_client(tree):
            offenders.append(str(path.relative_to(_INFRA_ROOT)))
    assert offenders == [], (
        "These infrastructure modules construct httpx.Client directly instead of routing through "
        f"egress_client.build_http_client, bypassing the egress chokepoint: {offenders}"
    )
