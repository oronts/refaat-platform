# Agentic tools and function calling

The governed ReAct program (`GovernedReActProgram`) runs a bounded, authorized
think-act-observe loop. One planner seam drives both the deterministic offline
default (`KeywordToolPlanner`) and a model that chooses tools through function
calling (`ModelToolPlanner`), without changing governance, the citation
contract, or the answer-cache identity.

## The two-path planner seam

A planner proposes the next tool as a `ToolProposal(tool, arguments)` or returns
`None` to stop. Two planners satisfy the one `ToolPlanner` seam:

- `KeywordToolPlanner` is model-free and deterministic. It routes a query to the
  first not-yet-used tool whose keyword appears in it and proposes empty
  arguments. It needs no model and no schema validation, so a core-only, offline
  governed run stays reproducible.
- `ModelToolPlanner` wraps a `ToolCallingGenerator` and a tool manifest. It
  advertises only the not-yet-used tools to the model and returns the model's
  proposed tool and arguments.

`ToolCallingGenerator` is an optional sibling Protocol of `Generator`, not a
method on it. The offline `ExtractiveGenerator` cannot honestly choose tools and
does not implement it; only a function-calling provider
(`OpenAICompatibleGenerator`) does.

## Per-iteration order of gates

Each loop iteration applies these gates in order, and any failure records a
denied or failed step with a stable reason and continues (deny-and-continue);
only a fully valid and authorized call executes:

1. Plan: the planner returns a `ToolProposal` or `None`.
2. Validate arguments: applied only to model proposals. A deterministic
   empty-argument keyword proposal is exempt, so the offline default never
   regresses.
3. Authorize: the shipped `ToolGovernor` admits the tool only when a grant is
   registered, the caller holds every required scope, and the grant's
   side-effect class is within the policy ceiling.
4. Approve: when the matched tool's `ToolCapability` sets `requires_approval`, a
   durable, fingerprint-bound human approval must satisfy the action (separation
   of duties, an optional approver role, and four-eyes when `min_approvals` is
   greater than one) and is consumed once; otherwise the step is denied with an
   `approval_required` reason. A capability without `requires_approval` skips
   this gate, so the default governed run is unchanged. Approvals are configured
   through `ReActPolicy.capabilities` and can be granted, listed, or cancelled over
   the admin-scoped HTTP approval surface. See [Approvals](approvals.md) and
   [Tool governance](tool-governance.md).
5. Execute: the runtime calls `skill.run(context, proposal.arguments)`. The
   arguments are recorded on the `ReActStep`.

## The argument-schema contract

A model manifest carries one `ToolSpec(name, description, parameters)` per
granted, registered skill. `parameters` is a small JSON-Schema object validated
by a pure-Python validator (no third-party dependency on the runtime path):
required-field presence, declared JSON types, and, when
`additionalProperties` is false, no unexpected fields.

Keyword-routed tools must expose a no-required-field schema. A skill that takes
no arguments falls back to a permissive empty object schema, so it is never
rejected on the model path and a no-argument proposal always validates.

## The observation-only invariant

The loop is observation-gathering only. The final answer still comes from
`project.ask`, and tool observations are never injected into the answer prompt.
Retrieval, generation, caching, citations, and the answer-cache identity are
therefore unchanged by tool calling. Feeding observations into the answer is
deferred to a later slice that would have to fold the planner and manifest
identity into the answer-cache fingerprint.

## Artifact identity

The governance policy is always pinned into the content-addressed artifact. The
planner kind, model id, and tool manifest are folded in only on the model path,
so every already-shipped keyword governed program keeps a byte-identical
artifact id, while two model programs with different tool schemas are distinct,
reproducible artifacts.
