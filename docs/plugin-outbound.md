# Governed plugin outbound HTTP

A governed tool sometimes needs to reach an external system of its own, for example to call a partner
API. Refaat gives it a governed outbound client so the call flows through the same controls the runtime
applies to every other outbound seam, rather than the plugin opening a raw socket.

## Declaring the intent

A skill opts in with a `wants_outbound = True` marker, the same shape as `wants_secrets`. When the
governed ReAct loop runs such a tool, it passes a governed `outbound` handle as a separate keyword to
`skill.run`, never merged into the tool arguments:

```python
class CallPartner:
    name = "call_partner"
    description = "Fetches a partner record."
    wants_outbound = True

    def run(self, context, arguments, *, outbound):
        response = outbound.get("https://api.partner.example.com/records/42")
        return {"status": response.status_code}
```

## What the runtime enforces

The handle is granted only under strict conditions, so outbound reach is never ungoverned:

- **An EXTERNAL capability is required.** Outbound network reach is an EXTERNAL side effect. The tool's
  capability must declare `side_effect=SideEffectClass.EXTERNAL`, which is admitted only within the
  program's side-effect ceiling. A READ_ONLY capability, or a tool with no capability, is denied the
  handle and the step is refused. Because the capability is EXTERNAL, the program is not shadow-safe and
  is never canaried.
- **Every request is guarded.** The client applies the capability's `allowed_network` allow-list first,
  then the deployment egress allowlist, then an SSRF IP-pin that always refuses a private, shared, or
  otherwise non-globally-routable address, then a disabled-redirect send under a response byte cap. A transport error, a timeout, or a
  partial result raises rather than returning partial or scanned data.
- **The admission is audited before the send.** A redacted `plugin.egress_admitted` event is emitted
  before any byte leaves the socket, and a `plugin.egress_denied` event is emitted on a denial. Each
  event carries only the host, a machine reason code, and the capability name, never a URL path, a
  query, a body, or a secret. Under a fail-closed audit policy a sink outage denies the egress.
- **An unserved run sends nothing.** During a shadow or canary comparison run the client refuses every
  request, so an optimization or deployment comparison never reaches an external system.

## The direct path

Running a skill directly (HTTP `run_skill`, the MCP bridge, or in-process `run_skill`) has no governed
outbound client, so a `wants_outbound` skill is refused there and must be invoked as a governed tool
inside a program.

## Honest boundary

The `OutboundClient` is the governed path by convention. It cannot stop arbitrary in-process Python from
opening its own socket, so an untrusted plugin still needs an operating-system or network isolation
boundary; the client governs a cooperative plugin's declared outbound calls, which is what the official
scaffolds use.
