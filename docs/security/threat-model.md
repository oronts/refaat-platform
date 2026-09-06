# Threat model

This threat model describes the generic Refaat runtime. Every deployment must add organization, data, connector, model-provider, and infrastructure-specific analysis.

## Protected assets

- original documents and extracted evidence;
- embeddings, graph state, session summaries, preferences, and cached answers;
- workspace API keys and bootstrap token;
- database, Redis, object-store, model-provider, and webhook credentials;
- plugin packages and executable skills;
- audit/outbox records and evaluation evidence.

## Trust boundaries

```text
User/client
  │ API key
  ▼
Ingress ── FastAPI process ── PostgreSQL / Redis / object storage
                │
                ├── model endpoints
                ├── connector destinations
                └── event destinations

Worker process ── same durable state and external dependencies
Plugin/skill code ── executes inside API or worker trust boundary
```

## Main threats and controls

Each threat carries a stable id, and the `threat_coverage` gate requires every id to be covered by a test that marks itself `# threat: TM-NN`, so the implemented baseline for each is exercised in the suite.

| ID | Threat | Implemented baseline | Deployment responsibility |
|---|---|---|---|
| TM-01 | Cross-tenant access | workspace-scoped keys and workspace predicates | separate environments/accounts where risk requires it; authorization review |
| TM-02 | Key theft | one-time raw key, scrypt digest, pepper, scopes, revocation | secret manager, TLS, rotation, client hardening |
| TM-03 | Oversized/malicious input | body and connector byte limits, parser validation | malware scanning, archive/decompression limits, content policy |
| TM-04 | SSRF | URL credential rejection, private/reserved address checks, redirects disabled | egress allowlists, DNS controls, metadata endpoint blocking |
| TM-05 | Compromised provider or data exfiltration egress | in-library default-deny egress allowlist at every outbound httpx seam, fail-closed on a policy-read failure and on a model host missing from the allowlist | NetworkPolicy default-deny backstop for boto3, OTLP gRPC, and user-injected ports; see [Deployment hardening](../deployment-hardening.md) |
| TM-06 | Stale/wrong cached answer | corpus version plus context fingerprint | evaluation and application-level freshness policy |
| TM-07 | Hallucinated citation | cited chunk ID validation | source quality review and domain evaluation |
| TM-08 | Prompt mutation | immutable base hash and allowlisted supplemental snapshots | admin access control and evidence review |
| TM-09 | Job duplication | leases, attempts, persisted status | idempotent handlers and external operations |
| TM-10 | Event duplication | durable outbox and retries | idempotent sinks/receivers |
| TM-11 | Malicious plugin/skill | explicit installation and optional allowlist | supply-chain review and process isolation |
| TM-12 | Model/provider data exposure | replaceable provider adapter | DPA, region, retention, encryption, PII controls |
| TM-13 | Host compromise | non-root/read-only reference containers | node hardening, runtime security, least privilege, incident response |

## Prompt injection

Corpus content and connector output are untrusted. Refaat grounds citations and prevents cited IDs outside the selected evidence, but it does not semantically neutralize every instruction embedded in a document. Applications should separate system policy from evidence, constrain tools, classify sources, and evaluate adversarial documents.

## Plugins and autonomous tools

The core RLM runtime does not execute arbitrary generated Python. A host application may add browser, terminal, code, or infrastructure tools, but those capabilities are outside the default safety boundary. Isolate them, restrict credentials and networks, apply budgets, log actions, and require approval for sensitive operations.

## Privacy

User IDs, session content, goals, preferences, originals, and model prompts may contain personal data. Define purpose, retention, access, deletion, residency, processor, and audit policy before deployment. The generic metadata filter is not a complete PII detector or anonymization layer.

## Residual risks

Refaat does not protect against a compromised host, malicious installed plugin, hostile infrastructure administrator, unsafe custom adapter, model-provider compromise, weak deployment secrets, or unrestricted capabilities deliberately granted by the embedding application.

Policy and configuration objects the operator constructs in process (an access policy, a residency rule and its allowed zones, a model router, a tool grant) are trusted code, not attacker input. The untrusted surfaces are ingested evidence and connector output, request headers and bodies, and configuration read from the environment or a config file, all of which arrive as plain data (JSON, strings) and are validated at the boundary. Refaat does not defend against operator code that hand-crafts a pathological domain object, for example a string subclass overriding equality or hashing, or that subclasses a domain type to remove its own validation; that is trusted code deliberately weakening the runtime, the same category as a malicious installed plugin.

Under an active policy the governed job read obscures every denial to the same not-found a missing id returns, so an authenticated caller cannot use the response status, code, or message to learn whether a job it may not read exists, and the true reason is recorded to the audit sink. A microsecond-scale timing difference remains, since a found row parses its payload before the denial and the database itself distinguishes a hit from a miss. This residual is dominated by the storage layer, is impractical to exploit remotely across network jitter, and is not closed at the application layer.
