from __future__ import annotations

from typing import Protocol

from refaat.domain.enums import SideEffectClass
from refaat.domain.models import ContextPacket, JsonObject, ToolGrant


class MCPToolInvoker(Protocol):
    """Calls one tool on an external MCP server and returns its result.

    This is the pure seam the transport layer implements: the client core never
    imports the ``mcp`` package, it only depends on this Protocol, so an external
    MCP tool is fully testable offline with an in-memory invoker.
    """

    def invoke(self, name: str, arguments: JsonObject) -> JsonObject: ...


class MCPRemoteSkill:
    """A Refaat :class:`Skill` backed by a tool on an external MCP server.

    The retrieved context is not forwarded: an external MCP tool runs entirely on
    the remote server from the passed arguments. The skill therefore delegates the
    name and arguments straight to the injected invoker, and drops into
    :class:`GovernedReActProgram` unchanged, gated by :func:`mcp_tool_grant`.
    """

    def __init__(self, *, name: str, description: str, invoker: MCPToolInvoker) -> None:
        self.name = name
        self.description = description
        self._invoker = invoker

    def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
        del context
        return self._invoker.invoke(self.name, arguments)


def mcp_tool_grant(name: str, *, required_scopes: tuple[str, ...] = ()) -> ToolGrant:
    """Bind an external MCP tool to an EXTERNAL-side-effect grant.

    EXTERNAL is the highest side-effect class, so the shipped governor denies the
    tool under the default READ_ONLY ceiling and admits it only when a policy
    both raises the ceiling to EXTERNAL and grants the required scopes.
    """
    return ToolGrant(skill_name=name, required_scopes=required_scopes, side_effect=SideEffectClass.EXTERNAL)
