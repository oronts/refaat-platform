# Deployment hardening

Refaat ships open by default: with no egress policy installed the runtime reaches whatever host a
configured provider, connector, or webhook points at. A hardened deployment closes that down with
two layers of defense.

## The two layers

1. The in-library egress policy is the decision and enforcement point. It governs every outbound
   httpx seam through one shared chokepoint, so an un-allowlisted destination is denied before the
   socket opens. It is byte-identical when unconfigured and fail-closed when active.
2. The Kubernetes NetworkPolicy is the deployment enforcement point. It is the backstop for what the
   httpx seam does not see: the S3 client (boto3), the OTLP exporter (gRPC), and any port a caller
   injects that does its own network I/O.

Run both. The in-library policy gives a precise, logical-hostname allowlist with an audit trail; the
NetworkPolicy gives a coarse but complete network boundary the process cannot talk around.

## The in-library egress allowlist

Set the policy selector and the allowlist:

```
REFAAT_EGRESS_POLICY=allowlist
REFAAT_EGRESS_ALLOWLIST=[{"host": "api.openai.com", "port": 443, "schemes": ["https"], "purposes": ["model", "fine_tuning"]}]
```

Each allowlist entry is a mapping:

| Key | Required | Meaning |
| --- | --- | --- |
| `host` | yes | An exact hostname, or a leading-dot suffix such as `.example.com` that matches the apex and every subdomain. Casefolded. |
| `port` | no | An explicit port. Absent means any port. A URL with no port normalizes to the scheme default (443 for https, 80 for http) before this comparison. |
| `schemes` | no | A list drawn from `http` and `https`. Absent means any scheme. |
| `purposes` | no | A list drawn from `model`, `connector`, `webhook`, `fine_tuning`, `blob`, `observability`, `mcp`, `plugin`. Absent means any purpose. |

Matching semantics:

- The leading-dot suffix is boundary-safe: `.example.com` matches `example.com` and `a.example.com`
  but never `evilexample.com`.
- An IDN host must be listed in the same form Python's URL parser returns. Every mismatch fails
  closed (over-deny), so a misconfiguration denies your own provider rather than silently widening
  access.

Startup is fail-closed on a coherence error: if a real model, embedding, or fine-tuning provider is
configured and its `base_url` host is not on an active allowlist, the process refuses to start. This
is intentional, so an operator who enables an allowlist without listing the model host learns at
startup rather than on the first request.

## In-library scope and the backstop

The in-library policy governs the built-in httpx adapters (model generation, embedding, graph,
multimodal, fine-tuning, the HTTP connector, the webhook sink, and a governed plugin's own outbound
HTTP through the OutboundClient) and the S3 endpoint host at construction. It does not see OTLP gRPC or a user-injected port that performs its own I/O. Those are
covered by the NetworkPolicy backstop. The chokepoint census test asserts that no infrastructure
module builds an httpx client outside the guarded factory, so a new outbound path cannot silently
skip the chokepoint.

## SSRF layering with the connector

The HTTP connector checks the egress allowlist against the logical hostname before it resolves and
pins the destination IP. The allowlist and the SSRF IP pin are complementary: the allowlist decides
whether the hostname is permitted at all, and the IP pin still blocks a permitted hostname that
resolves to a private or reserved address. The allowlist check runs first, so a denied source never
reaches the resolver.

## The secure Helm overlay

The default chart keeps `networkPolicy.enabled: false` for a backward-compatible open install. The
`values-secure.yaml` overlay turns on a default-deny NetworkPolicy with an explicit egress allowlist
and sets the in-library policy:

```
helm install refaat ./deploy/helm/refaat -f values.yaml -f values-secure.yaml
```

Replace every placeholder CIDR and port in the overlay with the real values for your cluster before
applying. The placeholders are intentionally narrow so an unedited install fails closed rather than
allowing broad egress.
