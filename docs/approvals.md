# Human approvals for governed tools

A governed tool can require a human to approve each run before it executes. When a tool's registered `ToolCapability` sets `requires_approval`, the governed ReAct loop refuses to run that tool until a valid approval, bound to an immutable action fingerprint, satisfies the request. Approvals are durable and single-use: one approval authorizes exactly one run, and a concurrent run that races for the same approval is denied.

Approvals are configured through `ReActPolicy.capabilities` and granted through `project.approvals` or an admin-scoped HTTP approval surface. There is no CLI approval command yet, and no per-approval audit event is emitted yet.

Two properties hold throughout:

- **Byte-identical when unconfigured.** A capability without `requires_approval`, or a policy with no `capabilities` at all, leaves the governed run exactly as it was. A capabilities key is folded into the content-addressed artifact only when a policy declares one, so every existing governed artifact id is unchanged.
- **Fail closed when active.** A required approval that cannot be evaluated denies. An unresolved initiator, a missing clock, a fingerprint mismatch, a self-approval, an expired grant, or an insufficient four-eyes count all record a denied step rather than run the tool.

## Where the gate sits

The approval gate is one step in the governed think-act-observe loop. Each iteration plans a tool, validates model arguments, authorizes the tool through the `ToolGovernor` (a registered grant, the required scopes, and a side-effect class within the policy ceiling), and only then checks approval. In `GovernedReActProgram._execute_tool`, once a tool is authorized and its skill resolved, a matched `ToolCapability` with `requires_approval` must pass `_authorize_approval` before `skill.run` executes. A denial records a `ReActStep` with `authorized = False` and an `approval_required:...` reason, and the loop continues. The same gate runs on both the non-streaming run path and the streaming path, because both call `_execute_tool`.

Governance decisions are model-independent. The `ToolCapability` is set at registration and is authoritative: no model output and no request body can widen it. The functions that decide whether an approval is satisfied take the registered capability and the resolved principal, never model output.

## The approval fingerprint

An approval authorizes one specific action, not a tool in general. `approval_fingerprint` in `refaat.domain.governance` produces an immutable digest over:

- the capability name,
- the exact tool arguments,
- the initiating principal (its id, kind, and tenant),
- the policy id and policy version,
- the action version, which is the program's content-addressed artifact id.

Because the fingerprint binds all of these, a change to any one of them invalidates a prior approval and forces re-approval. Different arguments, a different initiating principal, a bumped policy version, or a changed program each produce a different fingerprint, so an approval granted for one action can never authorize a different one.

The action version is derived by the governed program from its artifact id, so an approval is bound to the exact program version it authorizes. An approval is invalidated by a change to the `ReActPolicy`, the capabilities, or the grants, and also by swapping a granted skill for a different implementation class: a plain or plugin skill folds its class identity (its `module.qualname`) into the artifact, so replacing it with a different class flips the artifact id, which flips the fingerprint, and the action must be approved again against the new version. The policy id and policy version come from the run-scoped `ApprovalContext`, which defaults to `policy_id="react"` and `policy_version="1"`. Both the party that grants an approval and the enforcement path must compute the fingerprint against the same program, so a grant must be made against the artifact id of the exact program the run will use.

There is a residual limitation to state honestly. The class identity is the skill's `module.qualname`, a name and not a cryptographic content hash. It does not detect an in-place code change to the same class (same module and qualname, for example monkeypatching its `run` method), a change to instance configuration that keeps the same class (two instances of one webhook skill class pointed at different destinations share one class identity), or a plugin author who controls a class definition making one class present the same module and qualname as another; each of these leaves the artifact id unchanged and does not force re-approval. A hostile author who can register a class or its configuration already controls what runs under a tool name, so this is a trust boundary on skill registration, not on approvals. For a cryptographic content-addressed guarantee, bind the capability to a `ProgramSkill` (program-as-skill), which folds its content-addressed inner artifact id, or to a versioned skill.

## Granting durable approvals

`project.approvals` returns an `ApprovalService` scoped to the project's workspace. It has three methods, backed by the `tool_approvals` table:

```python
# Grant a durable approval for one action fingerprint.
approval_id = project.approvals.grant(
    fingerprint=fingerprint,
    capability_name="deploy",
    approver_id="bob",
    approver_roles=["release-manager"],
)

# Inspect the currently granted approvals for a fingerprint.
active = project.approvals.list_active(fingerprint)

# Cancel a granted approval by its id (returns True when a granted row was cancelled).
cancelled = project.approvals.cancel(approval_id)
```

`grant` records the approval with status `granted` and stamps `granted_at` from the runtime clock when no explicit time is given. `list_active` returns only the still-granted rows for a fingerprint. `cancel` flips a granted row to `cancelled` and returns whether it changed anything, so cancelling an already-consumed or already-cancelled approval returns `False`.

The governed loop reads durable approvals automatically. On each run it builds a `StoreApprovalBackend` for the workspace, loads the granted approvals for the action's fingerprint, and evaluates them alongside any in-memory approvals.

## The HTTP approval surface

The same grant, list, and cancel operations are exposed over the admin-scoped HTTP surface, so an approver service can record an approval without the Python API:

| Method | Path | Scope |
|---|---|---|
| `POST` | `/v1/approvals` | admin |
| `GET` | `/v1/approvals?fingerprint=...` | admin |
| `DELETE` | `/v1/approvals/{approval_id}` | admin |

The request body carries only the `fingerprint` and the `capability_name`. The approver is the server-resolved identity, never a request-body field, so a caller can only ever approve as itself and cannot forge another principal's approval. A grant therefore requires a resolved identity: with no principal resolver installed the endpoint denies, because an approval cannot be recorded without an authenticated approver. The separation-of-duties and four-eyes checks then hold at consume time, because the approver id is the caller's true identity rather than a value it chose.

## Passing an in-memory approval into a run

For a run that supplies an approval without persisting it, the approval travels through a context variable rather than the artifact, because it authorizes one run and not a program definition. `ApprovalContext` carries the approval records plus the policy id and policy version they were granted against. The action version is not carried here: the governed program derives it from its artifact id, so an in-memory approval is still bound to the exact program version. `approval_scope` binds the context for the duration of a run, and the enforcement path reads it through `active_approval_context`, which returns an empty context when nothing is set.

```python
from refaat.application.approvals import ApprovalContext, approval_scope
from refaat.domain.governance import ApprovalRecord

approval = ApprovalRecord(fingerprint=fingerprint, approver_id="bob")
with approval_scope(ApprovalContext(approvals=(approval,))):
    result = program.execute(
        project,
        inputs={"question": question},
        user_id=None,
        session_id=None,
        principal=alice,
    )
```

With nothing set, the active context is empty, so a tool that requires approval is denied until an approval is supplied. In-memory and durable approvals are combined: the loop evaluates the run-scoped `ApprovalContext.approvals` together with the durable granted rows for the same fingerprint.

The `ApprovalContext` is a trusted, in-process channel: a caller that can enter `approval_scope` is already trusted to authorize a run. No HTTP, SDK, CLI, or worker surface populates it, so in production only durable approvals apply. The untrusted program-run endpoints fail closed on a non-empty ambient context, so a future change that piped request data into it cannot silently downgrade the durable single-use trust boundary.

## Single-use consume semantics

A durable approval authorizes exactly one run. When an approval satisfies the action and durable rows backed it, the loop consumes them before the tool executes. `consume_tool_approvals` performs an atomic status flip from `granted` to `consumed` for that workspace and fingerprint and returns the number of rows it changed. If that count is zero, another run already consumed the approval first, so the current run is denied with an `approval_required:already_consumed` reason and the tool does not run. Consuming happens before `skill.run`, so the single-use guarantee holds even under a concurrent race.

An approval satisfied only by an in-memory `ApprovalContext`, with no durable rows, is scoped to that one run by construction and consumes nothing durable.

## When an approval is satisfied

`approval_satisfied` in `refaat.domain.governance` is the fail-closed rule that decides whether a required approval holds for a fingerprint. It enforces:

- **Separation of duties.** An initiator can never approve its own action. An approval whose approver id matches the initiator is ignored, so the person requesting an action cannot self-authorize it.
- **Exact fingerprint match.** Only approvals whose fingerprint equals the action fingerprint count. A stale or mismatched fingerprint contributes nothing.
- **A required approver role.** When the capability sets `approver_role`, only an approval whose captured `approver_roles` includes that role counts.
- **Four-eyes with `min_approvals`.** The rule counts distinct approvers and requires at least `min_approvals` of them, so a value above one demands that many different people. `min_approvals` must be at least one, enforced when the capability is constructed.
- **Identity canonicalization.** Approver and initiator ids are compared under a PRECIS UsernameCaseMapped canonical form (RFC 8264 and RFC 8265): control, format, unassigned, and default-ignorable characters are dropped, then NFKC, casefold, and NFKC again. A case variant, a whitespace or compatibility variant, or an invisible-character variant of the same id therefore cannot self-approve or pad the four-eyes count. This is defense in depth: the caller must still supply the identity provider's canonical, stable principal id, not a display name.
- **TTL expiry.** When the capability sets `approval_ttl_seconds`, an approval counts only within its window. The run clock and each approval's timestamp must be timezone-aware; an approval with no valid timestamp, a timestamp in the future, or one older than the TTL never counts. The TTL must be positive and at most ten years.

A required approval also fails closed when the initiator or the clock is absent. In `_authorize_approval`, a missing principal or a missing clock denies immediately with `approval_required:missing_initiator`, because a required approval cannot be evaluated without a resolved initiator and a point in time.

## Grant-then-run example

```python
from refaat.domain.authz import Principal
from refaat.domain.enums import SideEffectClass
from refaat.domain.governance import ToolCapability, approval_fingerprint
from refaat.domain.models import ReActPolicy, ToolGrant
from refaat.programs import GovernedReActProgram, KeywordToolPlanner

alice = Principal(id="alice", tenant="acme")

policy = ReActPolicy(
    max_iterations=2,
    granted_scopes=("read:corpus",),
    side_effect_ceiling=SideEffectClass.READ_ONLY,
    grants=(ToolGrant("lookup", required_scopes=("read:corpus",), side_effect=SideEffectClass.READ_ONLY),),
    capabilities=(
        ToolCapability(
            name="lookup",
            required_scopes=frozenset({"read:corpus"}),
            side_effect=SideEffectClass.READ_ONLY,
            requires_approval=True,
            approver_role="reviewer",
        ),
    ),
)

program = GovernedReActProgram(
    policy=policy,
    planner=KeywordToolPlanner({"lookup": ("rollback",)}),
    skills={"lookup": lookup_skill},
)

# The fingerprint is computed against the exact program the run will use: the action version is
# the program's content-addressed artifact id. With no approval_scope set, the policy defaults are
# react / 1.
fingerprint = approval_fingerprint(
    capability_name="lookup",
    arguments={},
    principal=alice,
    policy_id="react",
    policy_version="1",
    action_version=program.artifact().artifact_id,
)

# A different, role-holding approver grants the approval; alice cannot approve her own action.
project.approvals.grant(
    fingerprint=fingerprint,
    capability_name="lookup",
    approver_id="bob",
    approver_roles=["reviewer"],
)

result = program.execute(
    project,
    inputs={"question": "how does rollback work"},
    user_id=None,
    session_id=None,
    principal=alice,
)
```

Without the grant, the `lookup` step records `authorized = False` with an `approval_required` reason and the skill never runs. With it, the tool runs once and the durable approval is consumed, so a second run for the same fingerprint is denied `already_consumed` until a fresh approval is granted.

A `ReActPolicy` also validates its capabilities when constructed: a capability whose name has no matching tool grant is rejected, and two capabilities for the same tool are rejected, so an approval gate can never dangle or duplicate.

## The durable record

Approvals live in the `tool_approvals` table, keyed by workspace and fingerprint:

| Column | Meaning |
|---|---|
| `id` | The approval id (`tappr` prefix), returned by `grant`. |
| `workspace_id` | The owning workspace; cascades on workspace delete. |
| `fingerprint` | The action fingerprint the approval is bound to. |
| `capability_name` | The tool capability the approval was granted for. |
| `approver_id` | The approver's stable principal id. |
| `approver_roles` | The approver's roles captured at grant time (stored sorted). |
| `granted_at` | The grant timestamp, timezone-aware. |
| `status` | `granted`, `consumed`, or `cancelled`. |
| `consumed_at` | Set atomically when the approval authorizes a run. |
| `cancelled_at` | Set when the approval is cancelled. |

A lookup index over `(workspace_id, fingerprint, status)` backs the active-approval read on the run path. Only `granted` rows are active; consuming and cancelling are one-way status transitions, so an approval is never silently reused.

## Limits to know

- **No CLI surface yet.** Approvals are granted through `project.approvals` or the admin-scoped HTTP surface above. There is no CLI approval command yet.
- **No per-approval audit event yet.** Granting, consuming, and cancelling an approval mutate the durable record, but they do not yet emit a dedicated audit event.
- **Admission versus approval.** The governed ReAct admission path uses the `ToolGovernor`, which checks the caller's scopes and the side-effect ceiling, and the `ToolCapability` drives the separate approval gate. `capability_permits` is a defined, fail-closed admission function in `refaat.domain.governance`, but it is not wired into the ReAct admission path.

## Related pages

- [Agentic tools and function calling](agentic-tools.md)
- [Authorization](authorization.md)
- [Enterprise foundation](enterprise-foundation.md)
