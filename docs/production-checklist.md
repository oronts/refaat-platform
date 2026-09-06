# Production checklist

## Identity and tenancy

- use a long random API-key pepper from a secret manager;
- remove or rotate the bootstrap token after provisioning;
- issue least-privilege workspace-scoped keys;
- test workspace isolation through API and storage paths;
- define key rotation and revocation operations.

## Persistence

- use PostgreSQL/pgvector rather than SQLite;
- disable automatic schema creation;
- run Alembic as an explicit release step;
- configure backups, point-in-time recovery, and restore drills;
- configure S3-compatible versioning, encryption, retention, and lifecycle policy;
- define reindex procedures before changing embedding models or dimensions.

## Network and connectors

- terminate TLS at a trusted ingress or service mesh;
- restrict database, Redis, object storage, model, and connector networks;
- enforce egress allowlists and DNS protections;
- keep private connector hosts disabled unless the deployment explicitly requires them;
- add malware scanning, content classification, DLP, and PII controls where required.

## Runtime

- run API and worker with independent replicas and resource limits;
- give each worker a unique ID;
- monitor lease expiry, failed jobs, retry depth, schedule lag, and outbox lag;
- configure Redis for distributed rate limiting;
- pin images and dependencies;
- use read-only filesystems and non-root users;
- configure disruption budgets, probes, and autoscaling from measured load.

## Models and quality

- pin embedding, graph, and generation model versions;
- maintain representative evaluation suites;
- test citation recall, answer terms, empty-evidence behavior, and cache invalidation;
- calibrate any model judge against human labels;
- record provider latency, errors, token use, and data-processing terms;
- re-run evaluations before activating a harness snapshot.

## Release evidence

- run `./scripts/quality.sh`;
- run PostgreSQL/pgvector tests against the target major version;
- build and scan the final container;
- validate Compose or Helm rendering;
- run an ingestion, query, RLM, worker, and rollback smoke test;
- retain `QUALITY_REPORT.md`, JUnit, coverage, migration, package, and deployment logs.
