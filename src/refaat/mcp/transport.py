from __future__ import annotations

import importlib.util
import json
from typing import Any

from refaat.domain.models import JsonObject
from refaat.mcp.bridge import MCPToolBridge
from refaat.mcp.remote import MCPToolInvoker

_MCP_INSTALL_HINT = (
    "Model Context Protocol transport needs the optional 'mcp' package. Install it with: pip install 'refaat[mcp]'."
)


def _require_mcp() -> None:
    """Fail fast with an actionable message when the optional mcp package is absent.

    This is the single guard: every entry point calls it before touching any mcp
    symbol, so the pure core stays importable and testable with mcp uninstalled.
    """
    if importlib.util.find_spec("mcp") is None:
        raise RuntimeError(_MCP_INSTALL_HINT)


def serve_stdio(
    bridge: MCPToolBridge, *, server_name: str = "refaat", context: JsonObject | None = None
) -> None:  # pragma: no cover
    """Serve a bridge's governed tools over an MCP stdio server (blocking).

    For stdio there is one client per process, so the supervisor that launched the process supplies the
    fixed trusted context that binds the caller's identity; identity still never comes from the tool
    arguments. With no context the bridge behaves as a plain one (byte-identical).
    """
    _require_mcp()
    import anyio

    anyio.run(_serve_stdio, bridge, server_name, context)


def connect_stdio(command: str, args: list[str] | None = None) -> MCPToolInvoker:  # pragma: no cover
    """Connect to an external MCP server over stdio and return a pure invoker.

    The return type is the pure :class:`MCPToolInvoker` Protocol, so no mcp type
    leaks across the transport boundary into the client core.
    """
    _require_mcp()
    return _StdioInvoker(command, list(args or []))


def run() -> None:  # pragma: no cover
    """Console-script entry point placeholder that fails loudly without mcp."""
    _require_mcp()
    raise RuntimeError("Configure a bridge and call serve_stdio; there is no default MCP server surface.")


async def _serve_stdio(bridge: MCPToolBridge, server_name: str, context: JsonObject | None) -> None:  # pragma: no cover
    import mcp.types as types
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server

    server: Any = Server(server_name)

    async def _list() -> list[Any]:
        return [
            types.Tool(name=descriptor.name, description=descriptor.description, inputSchema=descriptor.input_schema)
            for descriptor in bridge.list_tools(context=context)
        ]

    async def _call(name: str, arguments: dict[str, Any]) -> list[Any]:
        result = bridge.call_tool(name, arguments, context=context)
        return [types.TextContent(type="text", text=json.dumps(result.to_json()))]

    # Register handlers via explicit calls rather than decorator syntax, so mypy
    # does not reject an Any-typed SDK decorator applied to a typed handler.
    server.list_tools()(_list)
    server.call_tool()(_call)

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


class _StdioInvoker:  # pragma: no cover
    def __init__(self, command: str, args: list[str]) -> None:
        self._command = command
        self._args = args

    def invoke(self, name: str, arguments: JsonObject) -> JsonObject:
        import anyio

        result: JsonObject = anyio.run(self._invoke, name, arguments)
        return result

    async def _invoke(self, name: str, arguments: JsonObject) -> JsonObject:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(command=self._command, args=self._args)
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            response: Any = await session.call_tool(name, arguments)
            texts: list[str] = [str(getattr(block, "text", "")) for block in response.content]
            return {"content": texts}
