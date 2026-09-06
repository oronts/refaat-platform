# Model Context Protocol (MCP)

Refaat speaks the Model Context Protocol in both directions, under the same tool
governance that guards the ReAct loop. The whole feature is an optional extra:
the `mcp` package is never a core dependency, and the pure server and client
cores are fully testable offline with `mcp` uninstalled.

Install the extra to enable the transport:

```bash
pip install "refaat[mcp]"
```

The package splits into a pure, offline-testable core and a thin transport shim.
Only `refaat/mcp/transport.py` imports the `mcp` package, and it does so behind a
single `_require_mcp()` guard. Importing `refaat.mcp` never imports `mcp`, so the
core stays usable and testable without the extra installed.

## Server direction: serve Refaat skills over MCP

`MCPToolBridge` exposes a corpus's governed skills to an external MCP client. It
is built from a `SkillRegistry`, a `SkillService`, and a `ReActPolicy`, and it
holds one `ToolGovernor` built from that policy. Every authorization decision is
delegated to the governor, so the served surface obeys the same rules as the
ReAct loop.

```python
from refaat.domain.enums import SideEffectClass
from refaat.domain.models import ReActPolicy, ToolGrant
from refaat.mcp import MCPToolBridge

policy = ReActPolicy(
    granted_scopes=("read:corpus",),
    side_effect_ceiling=SideEffectClass.READ_ONLY,
    grants=(ToolGrant("corpus_health", required_scopes=("read:corpus",)),),
)
bridge = MCPToolBridge(
    registry=runtime.skills.registry,
    skill_service=runtime.skills,
    policy=policy,
    workspace_id=project.workspace_id,
    corpus_id=project.corpus_id,
)
```

- `list_tools()` advertises only skills the governor admits. A grant whose
  side-effect class exceeds the policy ceiling, or whose scopes are not held, is
  never advertised.
- `call_tool(name, arguments)` re-authorizes at the boundary, defense in depth,
  and only on success runs the skill through `SkillService`, so retrieval and
  citations are untouched. The corpus binding is fixed at construction and the
  query is read from the call arguments.
- A governance denial returns a structured `MCPToolResult` with `authorized`
  set to `False` and the governor's reason. It does not raise, so a client sees
  a well-formed result on every path.

### The served-surface content address

`manifest_hash()` content-addresses the served surface. It folds in the advertised
tools, each tool's input schema, the policy ceiling, and the granted scopes,
using the one shared canonical-JSON plus sha256 helper that also produces every
program artifact id. The hash is stable across runs and changes whenever the
policy changes the advertised surface.

Each advertised tool carries an `input_schema`. Skills accept an arbitrary JSON
object, so the default is a permissive object schema with no required fields. A
skill that publishes its own `input_schema` object surfaces that instead. The
schema is part of the manifest hash, so the served surface has a stable,
testable content address offline.

Serve the bridge over stdio with `serve_stdio(bridge)`. That entry point calls
the `_require_mcp()` guard first, so it fails with an actionable message when the
extra is not installed.

### Per-caller identity

By default the bridge authorizes with the policy's granted scopes and serves the
same surface to every caller. `MCPToolBridge.from_runtime(runtime, policy=...,
workspace_id=..., corpus_id=...)` wires in the runtime's identity seams so the
bridge scopes each call to the calling principal. It resolves a `Principal` from a
trusted server-side context through the runtime's `PrincipalResolver`, never from
the tool arguments, maps that principal to its effectively held scopes through the
`EntitlementScopeResolver`, and passes those to the governor, so a tool the caller
lacks scopes for is denied even when the policy grants it. The resolved principal
also threads into the retrieval request, so evidence is scoped to that caller.
`list_tools` and `call_tool` take an optional `context`, and `serve_stdio(bridge,
context=...)` supplies the fixed trusted context for a one-client stdio process.

Under strict authorization with an access policy installed, an unresolved caller
is denied rather than served the policy-blanket surface: `list_tools` returns
nothing and `call_tool` returns `authorized=False` with the reason
`identity_required`. With no identity seams configured, that is no principal
resolver and no entitlement resolver, the bridge is byte-identical to a plain one
on every call. Once an entitlement resolver is configured and a call carries a
context that resolves a principal, the advertised and callable tools are scoped
to that caller's held scopes. If a policy and strict authorization are active but
no entitlement resolver is configured, a resolved caller is scoped to only the
tools that require no scopes rather than the full policy surface, so a missing
resolver fails closed rather than open. The content address stays
identity-independent regardless, because `manifest_hash` folds only the
policy-level surface.

## Client direction: consume an external MCP tool

`MCPRemoteSkill` implements the Refaat `Skill` protocol by delegating to an
injected `MCPToolInvoker`, the pure seam the transport implements. It drops into
`GovernedReActProgram` and `BoundedInferencePolicy` with no changes.

An external tool must be bound with `mcp_tool_grant`, which produces a `ToolGrant`
with the `EXTERNAL` side-effect class. `EXTERNAL` is the highest class, so the
shipped governor denies the tool under the default `READ_ONLY` ceiling and admits
it only when a policy both raises the ceiling to `EXTERNAL` and grants the
required scopes. The full grant is pinned into the program artifact through the
existing definition-extra hook, so a trajectory that can reach an external tool
is a distinct, reproducible artifact.

```python
from refaat.mcp import MCPRemoteSkill, connect_stdio, mcp_tool_grant

invoker = connect_stdio("weather-mcp-server")
skill = MCPRemoteSkill(name="weather", description="External weather tool.", invoker=invoker)
grant = mcp_tool_grant("weather", required_scopes=("call:weather",))
```

`connect_stdio` returns the pure `MCPToolInvoker` protocol, so no `mcp` type
crosses the transport boundary into the client core. Like `serve_stdio`, it calls
the `_require_mcp()` guard first.

## What stays out of core

The `mcp` extra is out of the core dependencies and out of the server bundle. The
offline gate runs with `mcp` absent: the pure cores are exercised in full, the
transport guard is covered by an actionable-error test, and the round-trip test
against a present `mcp` package is skipped. Because `mcp` is in the mypy overrides
list, the strict type gate stays green with the package uninstalled.
