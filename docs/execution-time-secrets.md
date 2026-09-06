# Execution-time secrets

A governed tool sometimes needs a credential to do its job: an API token to call an external
service, a signing key, a database password. Refaat lets a `ToolCapability` declare which secrets a
tool requires, resolves them at the moment of the side effect through an installed provider, injects
them into the authorized `skill.run`, and scrubs them immediately after. The value is redacted by
construction on every other surface, so it never reaches the model, the recorded steps, the program
artifact, the answer cache, an audit event, or a log line.

The declaration half already lives in `ToolCapability.required_secrets`, documented in
[Tool governance](tool-governance.md). This page covers the execution-time half: the redaction-by-construction
value type, the provider port, how the governed ReAct loop admits, resolves, injects, and scrubs a
secret, and the fail-closed guarantees.

## The two principles carry over

Two properties from [Authorization](authorization.md) hold here too. The runtime is byte-identical
when no provider is installed: no shipped or tested program declares a required secret, so with the
default configuration nothing changes, and every pinned artifact id is unmoved. And the wired path
fails closed when it is active: a tool that declares a required secret which cannot be resolved is
denied, never run unenforced.

## SecretValue: redacted by construction

`refaat.domain.secrets.SecretValue` wraps one credential. It is deliberately not a dataclass, not a
string, and not any JSON scalar, so an accidental log, serialization, or pickle cannot leak it. It
exposes a single `reveal()` accessor plus a `scrub()` that zeroes the value after use.

| Surface | Result |
|---|---|
| `str(value)`, `repr(value)`, f-string `{value}` | masked to `[secret]` (the name shows in `repr`, never the value) |
| `json.dumps(value)`, `canonical_json({...: value})` | raises `TypeError`, so a secret can never enter the artifact, the answer cache, or `react_steps` |
| `redact_field_value(key, value)` (structured logs and OpenTelemetry spans) | collapses to `[redacted:SecretValue]` |
| `pickle.dumps(value)` | raises `TypeError`, so a secret cannot ride the outbox or a durable job |
| `value.reveal()` | the only way to read the value, called at the side effect and never stored |
| `value.scrub()` | zeroes the value; a later `reveal()` returns the empty string |

A `SecretBundle` holds the resolved values for exactly one admitted capability, keyed by the names
that capability declared, so a skill can never reach beyond its own grant. It is not JSON
serializable or picklable either.

## The provider port

`refaat.ports.SecretProvider` is the seam a deployment implements. It has a stable `provider_id`,
a cheap `available(names, scope)` check that returns the resolvable subset without materializing any
value, and a `resolve(name, scope)` that materializes one `SecretValue` or returns `None` when it
cannot. The model never influences a name: names come only from the registered
`ToolCapability.required_secrets`, never from model output or a tool argument.

A `SecretScope` (workspace, capability, side effect, and the resolved principal id, kind, and tenant)
is handed to the provider for each admitted capability. A real vault or KMS provider uses it for
per-tenant isolation and least privilege.

Two reference providers ship in `refaat.infrastructure.secrets`:

- `StaticSecretProvider(mapping)`, an in-memory name to value map for tests and simple deployments.
- `EnvSecretProvider(prefix=...)`, which reads `PREFIX + NAME.upper()` from the process environment
  (default prefix `REFAAT_TOOL_SECRET_`).

Both key only on the secret name and ignore the rest of the scope, so neither enforces per-tenant
isolation: that is a real provider's job. A blank value is treated as unresolvable, so an empty
credential fails closed at the gate rather than injecting an empty string.

## Installing a provider

```python
from refaat.application.runtime import RuntimeBuilder
from refaat.infrastructure.secrets import StaticSecretProvider

runtime = RuntimeBuilder(settings).with_secret_provider(StaticSecretProvider({"deploy-key": "..."})).build()
```

The reference env provider can also be selected from settings without code:
`REFAAT_TOOL_SECRET_PROVIDER=env` builds an `EnvSecretProvider` at the configured
`REFAAT_TOOL_SECRET_ENV_PREFIX`. The default, `none`, installs nothing. An injected provider always
wins over the settings selector. A vault or KMS provider is installed with `with_secret_provider`.

## The governed loop: admit, materialize, inject, scrub

Inside `GovernedReActProgram`, one governed tool call runs this order:

1. `ToolGovernor.authorize` admits the grant (scopes and side-effect ceiling).
2. `capability_permits` admits the capability. When the capability declares a required secret, the
   loop asks the provider which of those names are available and passes that set into the admission,
   so a missing secret denies here with `required_secret_unavailable:<names>`. When no provider is
   installed, the available set is empty, so a declared required secret denies rather than running
   unenforced (fail-closed default).
3. The loop materializes the required secrets, then the approval gate consumes any required durable
   approval, then `skill.run` executes. Materializing before the approval consume is deliberate: an
   unresolvable required secret (rotated or deleted after the availability check) denies the step
   without burning a single-use, four-eyes, or break-glass approval.
4. The resolved `SecretBundle` is passed to the skill as a separate `secrets=` keyword, never merged
   into the tool arguments, and is scrubbed in a `finally` on every path.

A skill opts into injection with a `wants_secrets = True` marker, mirroring the `is_program_skill`
marker, and its `run(context, arguments, *, secrets)` reveals a value at the outbound call. A plain
skill keeps the exact `skill.run(context, arguments)` call, so every existing skill is byte-identical.
A skill whose capability declares no secret receives `secrets=None`, so a skill can only ever see the
names its own capability declared.

## Fail-closed guarantees

- A capability that declares a required secret and has no provider installed is denied, not run with
  no secret.
- A declared secret the provider cannot resolve is denied, whether the availability check excluded it
  or the resolution itself failed after the check.
- A tool not granted a secret cannot obtain one: the loop only ever asks the provider for the admitted
  capability's own `required_secrets`, never for a name from a tool argument or model output.
- The model never chooses a secret: names live only on the registration-time, model-immutable
  `ToolCapability`.
- No single-use approval is burned on a secret-resolution failure, because the materialize runs before
  the approval consume.

## Audit

When an `AuditSink` is installed, resolving a secret emits one redacted `tool.secret_resolved` event
and a resolution failure emits a `tool.secret_denied` event, both on the same contract as the other
enterprise events in [Audit and provenance](audit-and-provenance.md). The event carries the workspace,
the principal, the capability name, and an `action_fingerprint` over the provider id and the sorted
secret names, never a value. Delivery follows the configured audit outage policy: best-effort by
default, and fail-closed under `audit_fail_closed`, in which case a sink outage denies the step before
the approval consume rather than proceeding unaudited.

## Byte-identity

A provider is a runtime input, never folded into a program's content-addressed artifact, exactly like
the principal, the audit sink, and the principal directory. The secret names were already folded into
the artifact when declared on the capability, so this slice adds no new fold: a capability that
declares secrets keeps its id, and one that declares none is untouched. Nothing here enters the
answer-cache identity. With no provider installed the whole governed loop is byte-identical, including
the pinned base and governed-capability artifact ids.

## Boundaries and follow-ups

- The generic injection into `skill.run` ships here. Threading a resolved credential into a real MCP
  remote tool needs a secret-aware invoker seam and is a documented follow-up; the MCP remote grant
  stays `EXTERNAL` and is denied under a `READ_ONLY` ceiling regardless.
- The direct skill-run surface and connectors are not wired for secret injection yet; the same
  `available`, `materialize`, and `wants_secrets` seam extends there in a later slice.
- Rotating a secret value does not invalidate a prior approval. The declared secret set already binds
  the approval through the action version, but value rotation is operational, not a policy change.
  Folding a value fingerprint into the approval identity is an owner fork, not the default.

## Related pages

- [Tool governance](tool-governance.md)
- [Approvals](approvals.md)
- [Audit and provenance](audit-and-provenance.md)
- [Authorization](authorization.md)
