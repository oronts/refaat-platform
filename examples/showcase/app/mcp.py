"""The MCP layer: serve corpus read skills and consume an external MCP tool.

Two governed surfaces, both exercised offline and deterministically:

- Serve: :class:`MCPToolBridge` advertises the corpus read skills the shipped tool
  governor authorizes, with a content-addressed manifest hash, and denies an external
  tool under the read-only ceiling.
- Consume: :class:`MCPRemoteSkill` wraps an external MCP tool behind an EXTERNAL grant
  and drops into :class:`GovernedReActProgram`, admitted only when the policy raises the
  ceiling to EXTERNAL and denied under the default read-only ceiling.

The bridge and remote skill are pure and need no ``mcp`` package. The actual stdio wire
lives in the transport shim, the only code importing ``mcp``; when ``mcp`` is absent the
wire is reported as skipped, exactly like the shipped MCP tests, and the gate never
requires the optional dependency.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from refaat import RefaatProject
from refaat.domain.enums import SideEffectClass
from refaat.domain.models import JsonObject, ReActPolicy, ToolGrant
from refaat.mcp import MCPRemoteSkill, MCPToolBridge, mcp_tool_grant
from refaat.programs import GovernedReActProgram, KeywordToolPlanner

CORPUS_HEALTH_TOOL = "corpus_health"
NOTIFY_TOOL = "notify"
WEATHER_TOOL = "weather"


class StaticInvoker:
    """A deterministic in-memory MCP tool invoker for the offline gate.

    A real external MCP tool would run on a remote server; offline this returns a fixed
    structured result so the governed path is exercised without a network call.
    """

    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, name: str, arguments: JsonObject) -> JsonObject:
        del arguments
        self.calls += 1
        return {"tool": name, "status": "ok"}


def _serve_policy() -> ReActPolicy:
    return ReActPolicy(
        granted_scopes=("read:corpus", "write:crm"),
        side_effect_ceiling=SideEffectClass.READ_ONLY,
        grants=(
            ToolGrant(CORPUS_HEALTH_TOOL, required_scopes=("read:corpus",), side_effect=SideEffectClass.READ_ONLY),
            ToolGrant(NOTIFY_TOOL, required_scopes=("write:crm",), side_effect=SideEffectClass.EXTERNAL),
        ),
    )


def serve_corpus_skills(project: RefaatProject) -> dict[str, Any]:
    """Serve the corpus read skills over MCP and prove the manifest is content-addressed."""
    runtime = project.runtime
    bridge = MCPToolBridge(
        registry=runtime.skills.registry,
        skill_service=runtime.skills,
        policy=_serve_policy(),
        workspace_id=project.workspace_id,
        corpus_id=project.corpus_id,
    )
    advertised = [descriptor.name for descriptor in bridge.list_tools()]
    call = bridge.call_tool(CORPUS_HEALTH_TOOL, {"query": "How does OGuardAI protect sensitive data?"})
    denied = bridge.call_tool(NOTIFY_TOOL, {"query": "escalate"})
    manifest = bridge.manifest_hash()
    return {
        "advertised": advertised,
        "external_denied": denied.authorized is False,
        "call_authorized": call.authorized is True and int(call.observation.get("retrieved_chunks", 0)) >= 1,
        "manifest_hash": manifest,
        "manifest_content_addressed": manifest.startswith("sha256:") and manifest == bridge.manifest_hash(),
    }


def _remote_policy(ceiling: SideEffectClass) -> ReActPolicy:
    return ReActPolicy(
        max_iterations=4,
        granted_scopes=("call:weather",),
        side_effect_ceiling=ceiling,
        grants=(mcp_tool_grant(WEATHER_TOOL, required_scopes=("call:weather",)),),
    )


def _remote_program(policy: ReActPolicy, invoker: StaticInvoker) -> GovernedReActProgram:
    skill = MCPRemoteSkill(name=WEATHER_TOOL, description="External weather MCP tool.", invoker=invoker)
    planner = KeywordToolPlanner({WEATHER_TOOL: ("drift", "incident")})
    return GovernedReActProgram(policy=policy, planner=planner, skills={WEATHER_TOOL: skill})


def consume_external_tool(project: RefaatProject, chunk_ids: set[str]) -> dict[str, Any]:
    """Drop an external MCP tool into GovernedReActProgram under an EXTERNAL grant."""
    question = "During a security incident, how does OGuardAI handle sensitive data drift to external services?"
    external_invoker = StaticInvoker()
    external = _remote_program(_remote_policy(SideEffectClass.EXTERNAL), external_invoker)
    external_result = external.execute(project, inputs={"question": question}, user_id=None, session_id=None)
    external_step = external_result.outputs["react_steps"][0]

    read_only_invoker = StaticInvoker()
    read_only = _remote_program(_remote_policy(SideEffectClass.READ_ONLY), read_only_invoker)
    read_only_result = read_only.execute(project, inputs={"question": question}, user_id=None, session_id=None)
    read_only_step = read_only_result.outputs["react_steps"][0]

    cited = [citation.chunk_id for citation in external_result.citations]
    return {
        "external_authorized": external_step["authorized"] is True and external_invoker.calls == 1,
        "read_only_denied": read_only_step["authorized"] is False and read_only_invoker.calls == 0,
        "grounded": bool(cited) and all(chunk_id in chunk_ids for chunk_id in cited),
    }


def wire_status() -> dict[str, Any]:
    """Report whether the stdio wire is available; skipped when the mcp package is absent."""
    if importlib.util.find_spec("mcp") is None:
        return {"status": "skipped", "reason": "optional mcp package absent"}
    from refaat.mcp import connect_stdio, serve_stdio

    return {"status": "available", "serve": serve_stdio.__name__, "connect": connect_stdio.__name__}


def run_mcp_layer(project: RefaatProject, chunk_ids: set[str]) -> dict[str, Any]:
    return {
        "serve": serve_corpus_skills(project),
        "consume": consume_external_tool(project, chunk_ids),
        "wire": wire_status(),
    }


def mcp_layer_ok(report: dict[str, Any]) -> bool:
    serve = report["serve"]
    consume = report["consume"]
    wire = report["wire"]
    return bool(
        serve["advertised"] == [CORPUS_HEALTH_TOOL]
        and serve["external_denied"]
        and serve["call_authorized"]
        and serve["manifest_content_addressed"]
        and consume["external_authorized"]
        and consume["read_only_denied"]
        and consume["grounded"]
        and wire["status"] in ("available", "skipped")
    )


__all__ = [
    "CORPUS_HEALTH_TOOL",
    "NOTIFY_TOOL",
    "WEATHER_TOOL",
    "StaticInvoker",
    "consume_external_tool",
    "mcp_layer_ok",
    "run_mcp_layer",
    "serve_corpus_skills",
    "wire_status",
]
