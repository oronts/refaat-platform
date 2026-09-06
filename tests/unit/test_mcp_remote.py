from __future__ import annotations

from refaat.domain.enums import SideEffectClass
from refaat.domain.models import ContextPacket, JsonObject, ReActPolicy
from refaat.mcp.remote import MCPRemoteSkill, mcp_tool_grant
from refaat.programs.react import ToolGovernor


class _FakeInvoker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, JsonObject]] = []

    def invoke(self, name: str, arguments: JsonObject) -> JsonObject:
        self.calls.append((name, arguments))
        return {"remote": name, "echo": arguments}


def _context() -> ContextPacket:
    return ContextPacket(
        query="q",
        hits=(),
        graph_facts=(),
        session_summary=None,
        profile={},
        supplemental_harness={},
        token_count=0,
        routes=(),
    )


def test_remote_skill_passes_name_and_arguments_through() -> None:
    invoker = _FakeInvoker()
    skill = MCPRemoteSkill(name="weather", description="Remote weather tool.", invoker=invoker)

    observation = skill.run(_context(), {"city": "Munich"})

    assert observation == {"remote": "weather", "echo": {"city": "Munich"}}
    assert invoker.calls == [("weather", {"city": "Munich"})]


def test_mcp_tool_grant_is_external() -> None:
    grant = mcp_tool_grant("weather", required_scopes=("call:weather",))

    assert grant.skill_name == "weather"
    assert grant.required_scopes == ("call:weather",)
    assert grant.side_effect is SideEffectClass.EXTERNAL


def test_governor_denies_external_under_read_only_and_admits_under_external() -> None:
    grant = mcp_tool_grant("weather", required_scopes=("call:weather",))

    read_only = ToolGovernor(
        ReActPolicy(
            granted_scopes=("call:weather",),
            side_effect_ceiling=SideEffectClass.READ_ONLY,
            grants=(grant,),
        )
    )
    denied, denied_reason = read_only.authorize("weather")
    assert denied is False
    assert "exceeds ceiling" in denied_reason

    external = ToolGovernor(
        ReActPolicy(
            granted_scopes=("call:weather",),
            side_effect_ceiling=SideEffectClass.EXTERNAL,
            grants=(grant,),
        )
    )
    admitted, admitted_reason = external.authorize("weather")
    assert admitted is True
    assert admitted_reason == "authorized"


def test_governor_denies_external_when_scope_missing() -> None:
    grant = mcp_tool_grant("weather", required_scopes=("call:weather",))
    governor = ToolGovernor(
        ReActPolicy(
            granted_scopes=(),
            side_effect_ceiling=SideEffectClass.EXTERNAL,
            grants=(grant,),
        )
    )

    admitted, reason = governor.authorize("weather")

    assert admitted is False
    assert "missing scopes" in reason
