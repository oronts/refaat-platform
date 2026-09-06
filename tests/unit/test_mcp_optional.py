from __future__ import annotations

import importlib.util
import json

import pytest

from refaat.mcp import connect_stdio, run, serve_stdio
from refaat.mcp.bridge import MCPToolDescriptor, MCPToolResult

_MCP_PRESENT = importlib.util.find_spec("mcp") is not None


@pytest.mark.skipif(_MCP_PRESENT, reason="exercises the guard only when mcp is absent")
def test_transport_entry_points_raise_actionable_error_without_mcp() -> None:
    for call in (
        lambda: serve_stdio(None),  # type: ignore[arg-type]
        lambda: connect_stdio("server"),
        run,
    ):
        with pytest.raises(RuntimeError) as excinfo:
            call()
        assert "refaat[mcp]" in str(excinfo.value)


def test_transport_guard_raises_when_find_spec_reports_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    import refaat.mcp.transport as transport

    monkeypatch.setattr(transport.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError) as excinfo:
        transport._require_mcp()
    assert "refaat[mcp]" in str(excinfo.value)


def test_mcp_present_round_trips_descriptor_and_result_shapes() -> None:
    mcp_types = pytest.importorskip("mcp.types")

    descriptor = MCPToolDescriptor(name="weather", description="Remote weather tool.")
    tool = mcp_types.Tool(
        name=descriptor.name,
        description=descriptor.description,
        inputSchema=descriptor.input_schema,
    )
    assert tool.name == "weather"
    assert tool.inputSchema == descriptor.input_schema

    result = MCPToolResult(name="weather", authorized=True, reason="authorized", observation={"temp": 21})
    content = mcp_types.TextContent(type="text", text=json.dumps(result.to_json()))
    assert json.loads(content.text)["authorized"] is True
