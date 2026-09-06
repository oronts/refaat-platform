# Security policy

Report suspected vulnerabilities privately to `security@oronts.com` with a reproducible description, affected version, impact, and suggested mitigation. Do not open public issues containing secrets or exploitation details.

The maintainers will acknowledge a complete report, assess severity, coordinate remediation, and publish an advisory when appropriate. No response-time SLA is promised for this foundation release.

Only the latest released minor line receives security fixes during the pre-1.0 phase.

## Vulnerability process

The build runs two complementary checks. The `dependency_audit` gate audits the dependency tree against known advisories over the network when the audit tool is available. The `vuln_policy` gate is the offline half: it verifies the committed advisory allowlist at `docs/reference/vuln-allowlist.json` stays consistent.

An advisory that cannot be fixed immediately, for example one that is not reachable in this library's usage or awaits an upstream release, may be temporarily accepted by adding an entry to the allowlist with its advisory id, the affected package, a reason, an acceptance date, and an expiry. The acceptance is bounded: the expiry may not exceed 180 days from the acceptance date, and once it lapses the `vuln_policy` gate fails until the advisory is fixed or the acceptance is deliberately renewed after re-review. An acceptance is therefore always a dated, auditable decision, never a permanent suppression.

Target handling, offered as goals and not a contractual SLA for this pre-1.0 foundation: acknowledge a complete report within a few business days, assess severity and reproduce, then coordinate a fix and an advisory for a confirmed issue. A critical issue in a released line is prioritized over other work.
