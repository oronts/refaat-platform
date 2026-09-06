# Tool governance

A governed ReAct run admits a tool through a chain of deterministic, model-independent decisions. `ToolGovernor` admits the call against the run policy, and a `ToolCapability` is the authoritative, per-tool governance declaration that adds a payload bound, a classification ceiling, and a human-approval gate on top. Every field of a capability is set at registration. No model output and no request body can widen it, because the decision functions take the registered capability and the resolved principal, never model output.

This page documents the capability value object, the `capability_permits` admission function and every fail-closed reason it returns, the approval primitives, how a capability binds to a `ReActPolicy` grant, and how it is pinned into the content-addressed program artifact. For the per-iteration gate order inside the loop, see [Agentic tools and function calling](agentic-tools.md).

## Where governance sits

Two principles carry over from [Authorization](authorization.md): the runtime is byte-identical when a capability is unconfigured, and its wired governance fails closed when a capability is active. `refaat.domain.governance` is a pure decision core: importing it changes nothing. The approval checks (`approval_satisfied`) are applied by the governed ReAct loop and the approval service, alongside `ToolGovernor` scope and side-effect admission. `capability_permits`, the pure admission over scopes, side-effect ceiling, classification, payload, network, namespace, and secret, is applied in the governed ReAct admission loop for the measurable dimensions: the scope set, the side-effect ceiling, the evidence-classification ceiling, and the payload bound. The network, namespace, and secret fields stay declared-only, authoritative and model-immutable, pending a measurable runtime source.

The admission chain for one governed tool call:

| Layer | Backed by | What it decides |
|---|---|---|
| Grant admission | `ToolGovernor.authorize` | The tool has a registered grant, the caller holds every required scope, and the grant's side-effect class is within the policy ceiling. |
| Capability admission | `capability_permits` | A single fail-closed function that checks scopes, the side-effect ceiling, an evidence-classification ceiling, and a payload bound in one place. Applied in the governed ReAct loop for these measurable dimensions; network, namespace, and secret stay declared-only. |
| Capability approval gate | `ToolCapability` plus `approval_satisfied` | When the matched capability sets `requires_approval`, a fingerprint-bound human approval must satisfy the exact action before `skill.run` executes. It runs after admission, so an admission denial never consumes a single-use approval. |

The last row is applied in the shipped `GovernedReActProgram`: after `ToolGovernor` admits the grant and before the approval gate, `capability_permits` gates the matched capability on scopes, the side-effect ceiling, the evidence-classification ceiling, and the payload bound. That ordering is load-bearing: an admission denial returns before the approval gate, so it never consumes a single-use durable approval. The evidence classification is derived deterministically from the retrieved chunks' ACL tags, never from the model. The network, namespace, and secret dimensions are not enforced at runtime; they stay declared-only, because they are skill-internal and not measurable at the gate.

## ToolCapability

`ToolCapability` is a frozen dataclass. Only `name` is required; every other field has a governing default, so a bare `ToolCapability(name="lookup")` is a read-only, public, no-approval declaration.

```python
from refaat.domain.authz import Classification
from refaat.domain.enums import SideEffectClass
from refaat.domain.governance import ToolCapability

capability = ToolCapability(
    name="refund",  # must match a ReActPolicy grant of the same name
    required_scopes=frozenset({"billing:write"}),
    side_effect=SideEffectClass.EXTERNAL,
    allowed_classification=Classification.CONFIDENTIAL,
    max_payload_bytes=8_192,
    requires_approval=True,
    min_approvals=2,  # four-eyes: two distinct approvers
    approver_role="finance-lead",
    approval_ttl_seconds=3_600,  # approvals older than one hour no longer count
)
```

| Field | Default | Exact effect |
|---|---|---|
| `name` | required | The tool identity. It must equal a `ReActPolicy` grant's `skill_name`, and the loop matches a capability to a proposed tool by this name. |
| `required_scopes` | `frozenset()` | Scopes the caller must hold. In `capability_permits`, any scope not in the granted set denies with `missing_scopes`. |
| `side_effect` | `READ_ONLY` | The tool's side-effect class (`READ_ONLY` < `WRITE` < `EXTERNAL` by rank). In `capability_permits`, a rank above the passed ceiling denies with `side_effect_exceeds_ceiling`. |
| `allowed_classification` | `PUBLIC` | The most sensitive evidence classification the tool may run against (`PUBLIC` 0, `INTERNAL` 1, `CONFIDENTIAL` 2, `RESTRICTED` 3). In `capability_permits`, evidence above this ceiling denies with `classification_exceeds_capability`. |
| `max_payload_bytes` | `None` | An optional payload cap. `None` means no cap and no payload check. When set, an unmeasurable payload denies with `payload_size_unknown` and an oversized one with `payload_exceeds_max`. |
| `requires_approval` | `False` | When true, execution is gated behind a human approval bound to an immutable action fingerprint. When false, `approval_satisfied` short-circuits to allowed and the default governed run is unchanged. |
| `min_approvals` | `1` | The number of distinct approvers required. Above one it is a four-eyes control: `approval_satisfied` counts distinct approver ids and requires at least this many. |
| `approver_role` | `None` | When set, only an approval whose recorded `approver_roles` contains this role counts. |
| `approval_ttl_seconds` | `None` | When set, an approval counts only within this many seconds of its `granted_at`. `None` means approvals do not expire. |
| `allowed_network` | `frozenset()` | Declared governance metadata: the logical network destinations the tool may reach. Empty means not declared, never deny-all. Entries are destination names, never secret values. |
| `resource_namespace` | `None` | Declared governance metadata: the single logical namespace the tool is scoped to (a tenant, corpus, or resource bucket). `None` means not declared. |
| `required_secrets` | `frozenset()` | Declared governance metadata: the secret handles or names the tool needs. Empty means not declared. Entries are handles or names, never secret values. |

### Declared governance metadata: network, namespace, and secrets

`allowed_network`, `resource_namespace`, and `required_secrets` are declared governance metadata. They are authoritative and model-immutable like every other capability field, and they fold into the content-addressed program artifact, so a change to any of them forces re-approval. They carry only logical destination names, a namespace label, and secret handles or names, never secret values, so they are safe to fold into an artifact and later an audit event.

They are declared metadata plus pure decision logic, not runtime egress or credential enforcement. The runtime does not open sockets or read secrets on the capability's behalf, so it cannot itself block a network call or withhold a credential. That enforcement is architectural: network egress is controlled by network policy, resource scoping and credential issuance by IAM, and secret delivery by the secret provider. The capability declares the intended envelope so those layers and the `capability_permits` decision core can act on it. The governed ReAct loop calls `capability_permits`, and it passes no destination or namespace at the gate, so those two fields stay declared and pinned today, not enforced in the running loop; the classification ceiling and payload cap are enforced there. The `required_secrets` dimension is now the exception: when a `SecretProvider` is installed, the governed loop supplies the resolvable set into admission and resolves and injects the declared secrets at execution time, fail-closed when the provider is absent or a secret is unresolvable. See [Execution-time secrets](execution-time-secrets.md).

Audit level is a field on `ToolCapability` (`audit_level`), documented with the observability-redaction section below. Timeout is declared in a later slice with its consumer, so it is not a field on `ToolCapability` yet.

### Construction-time validation

`ToolCapability.__post_init__` rejects two malformed declarations at construction, so an invalid governance object never reaches the loop:

- `min_approvals` must be at least 1. A value below one raises `ValueError("min_approvals must be at least 1")`.
- `approval_ttl_seconds`, when not `None`, must be positive and at most ten years (315,360,000 seconds). Outside that bound it raises `ValueError`. The upper bound is a sanity ceiling that also avoids a `timedelta` overflow when the TTL window is computed.
- `allowed_network` holds at most 64 entries, each a non-empty, whitespace-free string of at most 256 characters, and `required_secrets` is bounded the same way. `resource_namespace`, when not `None`, is a non-empty string of at most 512 characters. These per-entry and set-cardinality bounds keep the artifact and any future audit event from bloating and reject a malformed declaration at construction.

## capability_permits and its fail-closed reasons

`capability_permits` is the single, model-independent admission function. It takes the authoritative capability plus the caller's runtime context and returns a `CapabilityDecision(allowed: bool, reason: str)`. It performs no I/O, so the decision is reproducible.

```python
from refaat.domain.governance import capability_permits

decision = capability_permits(
    capability,
    granted_scopes={"billing:write"},
    side_effect_ceiling=SideEffectClass.EXTERNAL,
    evidence_classification=Classification.INTERNAL,
    payload_bytes=512,
)
# CapabilityDecision(allowed=True, reason="permitted")
```

The checks run in order, and the first failure returns fail-closed with a machine-readable reason:

| Reason | When it is returned |
|---|---|
| `missing_scopes:<a,b>` | The capability's `required_scopes` are not all present in `granted_scopes`. The missing scopes are sorted and comma-joined into the reason. |
| `side_effect_exceeds_ceiling` | `capability.side_effect.rank` is above the passed `side_effect_ceiling.rank`. |
| `classification_exceeds_capability` | `evidence_classification` is more sensitive than `capability.allowed_classification`. |
| `payload_size_unknown` | `max_payload_bytes` is set but `payload_bytes` is `None`. A declared cap that cannot be measured denies rather than admitting an unmeasured payload. |
| `payload_exceeds_max` | `max_payload_bytes` is set and `payload_bytes` is greater than it. |
| `network_destination_denied:<a,b>` | `allowed_network` is declared and the caller passed `requested_destinations` containing one or more destinations not in the allowlist. The denied destinations are sorted and comma-joined. |
| `resource_namespace_denied` | `resource_namespace` is declared and the caller passed a `requested_namespace` that does not match it exactly. |
| `required_secret_unavailable:<a,b>` | `required_secrets` is declared and the caller passed `available_secrets` missing one or more of them. The missing secrets are sorted and comma-joined. |
| `permitted` | All checks pass. This is the only allowed decision. |

The payload checks apply only when `max_payload_bytes` is set. With the default `None`, the payload is not inspected and the reason can never be `payload_size_unknown` or `payload_exceeds_max`.

The network, namespace, and secret checks are opt-in and asymmetric to the payload check. Payload size is generically measurable at the gate, so a declared `max_payload_bytes` denies on an unknown payload (`payload_size_unknown`). A destination, a namespace, and a secret set are skill-internal and not generically measurable at the gate, so each is checked only when the caller supplies the concrete value through `requested_destinations`, `requested_namespace`, or `available_secrets`. When the caller passes `None` for one of these, that dimension is declared-only and is not checked. Exact-match is enforced for the namespace; prefix and glob matching is a documented later refinement. The governed ReAct loop calls `capability_permits` on every admission but passes `None` for `requested_destinations`, `requested_namespace`, and `available_secrets`, so the network, namespace, and secret checks stay declared-only there; only the scope, side-effect, classification, and payload checks are live in the loop. They become enforced when a measurable runtime source supplies the concrete destination, namespace, or secret value.

## Approval primitives

When a capability sets `requires_approval`, two functions decide whether a human approval authorizes the exact action.

### approval_fingerprint

`approval_fingerprint` returns a canonical digest that binds the action to its capability, arguments, initiating principal, policy, and action version:

```python
from refaat.domain.governance import approval_fingerprint

fingerprint = approval_fingerprint(
    capability_name="refund",
    arguments={"invoice": "INV-42", "amount_cents": 1990},
    principal=principal,  # id, kind, and tenant enter the digest
    policy_id="react-billing",
    policy_version="3",
    action_version="sha256:...",  # derived from the program artifact
)
```

An approval is valid only for this exact fingerprint. The action version is the governed program's content-addressed artifact id, so an approval is bound to the exact program version it authorizes: an approval is invalidated by a change to the `ReActPolicy`, the capabilities, or the grants, and also by swapping a granted skill for a different implementation class, whose `module.qualname` changes the artifact id. A change to the arguments, the initiating principal, or the policy likewise produces a different fingerprint and invalidates a prior approval. This is what makes an approval non-transferable across a changed request or a changed program. Residual limitation: the class identity is the skill's module and qualname (kept as separate fields so two differently structured names cannot alias), not a cryptographic content hash, so it detects a swap to a different implementation class but not an in-place code change to the same class (for example monkeypatching its `run` method), a change to instance configuration that keeps the same class, or a plugin author who controls a class definition presenting another's module and qualname. That is a trust boundary on skill registration, not on approvals. For a cryptographic content-addressed guarantee bind the capability to a `ProgramSkill` (program-as-skill) or a versioned skill.

### approval_satisfied

`approval_satisfied` decides whether the recorded approvals satisfy a required approval for one fingerprint. It returns `ApprovalDecision(satisfied: bool, reason: str)` and is fail-closed:

- A capability with `requires_approval=False` returns `no_approval_required` immediately.
- An empty fingerprint returns `missing_fingerprint`; an empty initiator id returns `missing_initiator`.
- When `approval_ttl_seconds` is set, `now` must be timezone-aware or the decision is `invalid_now`.
- It counts distinct approvers whose `ApprovalRecord.fingerprint` matches exactly, excluding the initiator (separation of duties: an initiator can never approve its own action), requiring `approver_role` in the record's roles when the capability sets one, and inside the TTL window when the capability sets one. An approval under a TTL capability with no `granted_at` never counts.
- Approver identity is compared under a canonical case-and-invisible-character folding, so a case or compatibility variant of the same id cannot self-approve or pad a four-eyes count.
- When the distinct count reaches `min_approvals` the decision is `approved`; otherwise it is `insufficient_approvals:<count>/<min_approvals>`.

In the loop, `GovernedReActProgram` builds the fingerprint, gathers run-scoped and durable approvals for it, and denies the step with an `approval_required:<reason>` message when `approval_satisfied` is not satisfied; a durable approval is consumed once so it authorizes exactly one run.

### Approver re-resolution

A durable approval stores the approver's roles at grant time. By default the consume decision uses that snapshot, so an approver who later loses the required role would still satisfy an old approval. Installing a `PrincipalDirectory` with `RuntimeBuilder.with_principal_directory` closes that gap: at consume time each durable approval's approver is re-resolved by id, its snapshot roles are replaced with the approver's current roles, and the approval is dropped when the directory returns `None` (an unknown or revoked approver) or raises (an unavailable directory), so the approval no longer counts. This is the execution-time counterpart to the request-time `PrincipalResolver`, and it is fail-closed: a role revoked after grant, a deactivated approver, or a directory outage all mean the approval no longer satisfies. With no directory installed the grant-time snapshot is used exactly as before, byte-identical.

## Binding a capability to a policy grant

A capability does not float free. It attaches to a `ReActPolicy` grant of the same name, and `ReActPolicy.__post_init__` enforces the binding:

- Each `capabilities` entry's `name` must equal the `skill_name` of one `grants` entry. A capability with no matching grant raises `ValueError("capability <name> has no matching tool grant")`.
- No two capabilities may share a name. A duplicate raises `ValueError("duplicate capability for tool <name>")`.

So there is exactly one capability per grant at most, with no duplicates, and no dangling capability that governs a tool the policy never grants. `max_iterations` is validated in the same hook to stay between 1 and 32.

```python
from refaat.domain.models import ReActPolicy, ToolGrant

policy = ReActPolicy(
    max_iterations=4,
    granted_scopes=("billing:write",),
    side_effect_ceiling=SideEffectClass.EXTERNAL,
    grants=(
        ToolGrant(
            skill_name="refund",
            required_scopes=("billing:write",),
            side_effect=SideEffectClass.EXTERNAL,
        ),
    ),
    capabilities=(capability,),  # capability.name == grant.skill_name == "refund"
)
```

Inside the loop, `GovernedReActProgram` indexes the policy's capabilities by name and looks one up by the proposed tool name. Only a capability whose `requires_approval` is true engages the approval gate; a tool with no capability, or one that does not require approval, runs once the grant admission passes.

## Governing the direct skill-run path

A skill can also be invoked directly, outside any governed program, through `POST /v1/skills/{name}/run`, `sdk.run_skill`, `project.run_skill`, and the MCP bridge's `call_tool`. All four funnel through the single `SkillService.run` chokepoint. A `DirectSkillPolicy` installed with `RuntimeBuilder.with_direct_skill_policy` governs that chokepoint, so a capability that governs a skill inside a `GovernedReActProgram` can also govern the same skill run directly.

```python
from refaat.application.runtime import RuntimeBuilder
from refaat.domain.governance import DirectSkillPolicy, ToolCapability
from refaat.domain.authz import Classification
from refaat.domain.enums import SideEffectClass

policy = DirectSkillPolicy(
    granted_scopes=frozenset({"kb.write"}),
    side_effect_ceiling=SideEffectClass.WRITE,
    capabilities=(
        ToolCapability(
            name="reindex",
            required_scopes=frozenset({"kb.write"}),
            side_effect=SideEffectClass.WRITE,
            allowed_classification=Classification.INTERNAL,
            max_payload_bytes=4096,
        ),
    ),
)
runtime = RuntimeBuilder(settings).with_direct_skill_policy(policy).build()
```

`DirectSkillPolicy` mirrors the measurable-dimension half of a `ReActPolicy`: a policy-level `granted_scopes` and `side_effect_ceiling`, plus one `ToolCapability` per governed skill. At the chokepoint, a skill named by a capability is admitted through `capability_permits` against the four measurable dimensions (scopes, the side-effect ceiling, the evidence classification derived deterministically from the retrieved chunks' ACL tags, and the payload bound). A denial raises before the skill runs. The network, namespace, and secret fields stay declared-only, consistent with the governed loop.

A capability whose `requires_approval` is true is refused outright on the direct path, before any evidence is retrieved, because the direct path derives no action fingerprint to bind an approval to. A four-eyes control therefore lives only on the governed program surface, and the direct path can never be the weaker route around it. A durable approval that exists for the governed path does not admit the same skill run directly.

The two declaration homes are independent. A `ToolCapability` declared at the `ReActPolicy` level governs the governed loop; the same skill name declared at the `DirectSkillPolicy` level governs the direct path. The physical paths are disjoint (a governed-loop call runs `skill.run` in the loop, a direct call runs it in `SkillService.run`; no single call traverses both), so each path enforces its own home. To govern a skill everywhere, declare it in both homes and keep the direct declaration at least as strict as the governed one.

A skill with no declared capability, and the whole surface with no `DirectSkillPolicy` installed, run exactly as before.

### Boundary: principal-precise retrieval on the direct surfaces

The `DirectSkillPolicy` governs capability admission. Deriving the evidence classification soundly needs principal-scoped evidence: the Policy Enforcement Point filters before the top-k truncation, so a principal-less retrieval is not a superset of a principal-scoped one after truncation and can derive a lower classification. The direct path therefore fails closed. A capability that declares a real classification ceiling, anything below RESTRICTED, cannot run without a resolved principal while an access policy is active. The in-process `project.run_skill` accepts an optional `principal`, so a trusted caller passes one to feed principal-precise retrieval into the admission. The HTTP-under-policy and MCP direct surfaces retrieve without a principal today, so a ceiling-declaring skill on either is refused rather than admitted on an unsound bound; under an active access policy the HTTP direct route is already denied outright with a 403. Threading a resolved principal into the HTTP and MCP direct surfaces is a separate follow-up.

## Pinned into the content-addressed artifact

The full governance policy is pinned into the program's content-addressed artifact, so a request can never widen what was registered. When `policy.capabilities` is non-empty, the artifact's definition folds a `capabilities` block, one entry per capability, carrying the exact registered fields: `name`, sorted `scopes`, the `side_effect` string, the `classification` as its integer level, `max_payload_bytes`, `requires_approval`, `min_approvals`, `approver_role`, and `approval_ttl_seconds`. The break-glass and escalation groups, and the declared `allowed_network`, `resource_namespace`, and `required_secrets` fields, each fold only when the capability sets them, with the frozenset fields folded as a sorted list so equal sets fold identically. Because the block is added only when capabilities are present, and each optional field only when declared, a governed program with no capabilities keeps a byte-identical artifact id, a capability that declares none of the optional fields folds the exact same entry as before, and two programs whose capabilities differ are distinct, reproducible artifacts. Loosening any capability field changes the artifact id, so a widened capability cannot masquerade as the approved one.

## Observability redaction

Non-audit observability (OpenTelemetry spans and structured logs) is held to the same redacted-by-construction contract the audit event already meets: only ids, counts, enums, and decision codes egress, never a raw prompt, evidence text, chunk text, a tool argument value, a tool output, or a secret. This is regression prevention. No current span attribute or log field emits any of those; the seam locks the property so a future field cannot regress it, backed by a canary test.

The seam lives in `refaat.observability`. `SAFE_STR_KEYS` is an allowlist of the id, enum, decision-code, and route-template keys the codebase emits today. A client-controlled value such as a raw request path or HTTP method is never allowlisted, so it is redacted rather than logged. `redact_field_value(key, value)` passes a bool, int, float, or None unchanged; it passes a string only under an allowlisted key, otherwise substitutes `[redacted]`; and it collapses any non-scalar to a `[redacted:<type>]` sentinel. Allowlisted values are not length-bounded, so a legitimate long id such as a program name is preserved. `ContextLogger.process` runs every structured field through it, and `span()` runs every attribute through it. Because the allowlist lists every current key, existing observability serializes identically; a new string field must be added to `SAFE_STR_KEYS` deliberately, and only when its value can never carry free text.

`ToolCapability.audit_level` (default `STANDARD`) is a per-tool declaration. `STANDARD` is byte-identical on the observability path: the governed loop emits no extra span or log. `ENHANCED` opts a governed step into one `refaat.tool.step` span and one `governed_tool_step` structured log carrying only the tool id, the capability name, the authorized flag, an opaque `reason_code`, the declared side-effect and classification, the approval flag, an argument count, an observation-present bool, and the iteration index. It never reads an argument value, an observation's content, or the step thought. The `reason_code` comes from `governance_reason_code`, which reduces a raw governed-step reason to a fixed developer-authored vocabulary (`authorized`, `approval_required`, `argument_validation_failed`, `missing_scopes`, and the `capability_permits` reasons), mapping anything else to `other`, so no model-chosen substring (an argument field name, a denied host list) ever reaches the egress.

The `LoggingEventSink` logs only the event topic, the event id, and a payload field count. The raw outbox payload dict never reaches the log.

Two boundaries stated honestly. First, the field seam does not sanitize an exception message or a traceback: `logger.exception` and `exc_info` carry whatever an exception string contains. The mitigation is the convention that a structured log's first argument is a static event name (never an interpolated message) plus the canary test. Second, the SQLAlchemy and FastAPI instrumentors produce their own spans outside `span()`. Evidence and ACL values travel to the database as bound parameters, not inlined into the SQL text, so the SQLAlchemy instrumentor must stay at its defaults (statements without literal bind values) for that assurance to hold.

## Honesty and current boundaries

- Tool capabilities are configured through `ReActPolicy.capabilities` and `ToolGrant` on the Python API. Approvals can be granted, listed, and cancelled over the admin-scoped HTTP approval surface or `project.approvals`; there is no CLI approval command yet.
- Redacted `approval.granted`, `approval.consumed`, and `approval.cancelled` audit events cover the full approval lifecycle, on the same contract as the per-query and per-RLM events in [Audit and provenance](audit-and-provenance.md), emitted only when an `AuditSink` is installed. Their `policy_id` and `policy_version` carry the standing default approval context; a custom `ApprovalContext` policy id is not plumbed to the grant and cancel events (only the consume event reflects it, from the run context).
- The direct skill-execution surface (`POST /v1/skills/{name}/run`, `sdk.run_skill`, `project.run_skill`, and the MCP bridge) runs a registered skill outside any program policy. With no `DirectSkillPolicy` installed it applies no capability, side-effect, or approval gating, exactly as before. With a `DirectSkillPolicy` installed, a skill named by a capability is admitted through `capability_permits` at the single `SkillService.run` chokepoint (scopes, side-effect ceiling, evidence classification, payload bound), and a capability that requires approval is refused outright because the direct path derives no action fingerprint. The HTTP route stays admin-scoped and denied outright under an active access policy. Principal-precise ACL retrieval on the HTTP and MCP direct surfaces is a separate principal-threading follow-up; the admission is a conservative bound that never loosens under a missing principal. See [Governing the direct skill-run path](#governing-the-direct-skill-run-path).
- `capability_permits` is applied in the shipped `GovernedReActProgram` admission path: after `ToolGovernor` admits the grant and before the approval gate, it enforces the scope set, the side-effect ceiling, the evidence-classification ceiling, and the payload bound for the matched capability. The evidence classification is derived deterministically from the retrieved chunks' ACL tags, never from the model, and an admission denial returns before the approval gate so it never consumes a single-use durable approval. The network, namespace, and secret fields stay declared-only, not enforced at runtime, because they are skill-internal and not measurable at the gate.
- A model never influences any of these decisions. Every gating value is registered outside the model and every decision function takes the registered capability and the resolved principal, not model output.

## Related pages

- [Agentic tools and function calling](agentic-tools.md)
- [Authorization](authorization.md)
- [Audit and provenance](audit-and-provenance.md)
- [Enterprise foundation](enterprise-foundation.md)
