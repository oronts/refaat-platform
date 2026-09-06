# Security model

## Implemented controls

- workspace-scoped API keys with scrypt hashing, application pepper, constant-time verification, scopes, revocation,
  and last-use timestamps;
- trusted-host and explicit CORS configuration;
- request body limits and local or Redis-backed rate limiting;
- safe error envelopes and request/trace IDs;
- local blob path confinement, atomic writes, and owner-only permissions;
- S3 server-side encryption requests;
- rooted and size-limited file connectors;
- URL credential rejection, blocked redirects, streamed byte limits, and SSRF address checks for HTTP connectors;
- default exclusion of hidden files, dependency directories, virtual environments, build output, and VCS metadata from
  directory ingestion;
- metadata credential-field filtering;
- immutable base harness and allowlisted supplemental updates;
- non-root, read-only reference containers with dropped capabilities and `no-new-privileges`;
- workspace predicates on public persistence paths;
- optional per-principal authorization over every evidence path, byte-identical when unconfigured and fail closed when active;
- committed-secret scanning and a documented vulnerability-reporting process.

## Authorization and access control

When an access policy and a principal are active, Refaat enforces per-principal access control over every evidence path: retrieval including HyPE and RAPTOR and hierarchical expansion, graph facts, RLM recursion, citations, the answer cache, session memory, learned preferences, and every program, durable job, and HTTP surface. It is byte-identical when unconfigured and fail closed when active: an untagged or malformed evidence unit is denied, a cross-tenant request is refused before any pluggable policy runs, the HTTP surface never trusts a client-supplied identity, a durable job that carries no principal fails closed rather than running unfiltered, and a model never decides a permission. Refaat is the enforcement layer, not the identity provider: identity and policy plug in. See [Authorization](authorization.md), [Access and multi-tenancy](access-and-multitenancy.md), [Identity lifecycle](identity-lifecycle.md), and [Audit and provenance](audit-and-provenance.md).

## Deployment responsibilities

Terminate TLS at a trusted ingress or service mesh. Store peppers, API keys, model keys, S3 credentials, Redis secrets,
and bootstrap tokens in a secret manager. Restrict database, object-store, connector, and model-provider networks.
Enable encryption at rest, backups, point-in-time recovery, retention, and restore drills.

Application-level SSRF checks are vulnerable to DNS and routing changes outside the process. Enforce egress allowlists,
DNS controls, metadata-service blocking, and connector-specific destination policy at the network layer. Add malware
scanning, archive limits, content classification, DLP/PII controls, and customer-specific retention where required.

Do not expose generated skill installation, arbitrary connector registration, or unrestricted job kinds to untrusted
users. Treat documents, model output, and plugin packages as untrusted until reviewed. Skills and plugins execute with
the process privileges of the API or worker and are not a sandbox.

## RLM and autonomy boundary

The recursive-context core does not execute arbitrary model-generated Python. Applications that add browser, terminal,
code, or infrastructure tools must isolate them, enforce capabilities and budgets, and require approval for sensitive
operations.

## Threat-model limits

Refaat does not claim protection against a compromised host, malicious installed plugin, hostile model provider,
operator misconfiguration, or arbitrary code execution deliberately granted by an embedding application. See
[`SECURITY.md`](https://github.com/oronts/refaat/blob/main/SECURITY.md) for vulnerability reporting.

See the deployment-specific [Threat model](security/threat-model.md) for assets, trust boundaries, controls, and residual risks.
