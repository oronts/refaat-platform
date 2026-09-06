# Operations and observability

Refaat emits structured logs with request IDs and supports OTLP traces for FastAPI, HTTPX, and SQLAlchemy. Configure
`REFAAT_OTEL_ENABLED=true`, a service name, and an exporter endpoint. API and worker should use distinct service names
or resource attributes in production.

## Signals to monitor

- API request rate, latency, status, payload rejection, and rate-limit rejection;
- model, embedding, graph, and connector latency, errors, retries, and token/cost usage;
- exact/semantic cache route, hit rate, identity scope, and corpus-version churn;
- lexical/vector/graph candidates, selected chunks, context tokens, and empty-context rate;
- RLM depth, node count, input tokens, citation fallback, and budget exhaustion;
- ingestion throughput, failure, compensation, unsupported media, and document status;
- pending/running/failed jobs, lease expiry, retry depth, schedule delay, and worker heartbeat;
- agent mailbox backlog and unacknowledged messages;
- outbox backlog and publishing failures;
- PostgreSQL connections, latency, locks, storage, WAL, replication, and pgvector index behavior;
- object-store errors, versioning, retention, and capacity.

## Health endpoints

Use `/health/live` only for process liveness. `/health/ready` verifies the relational store and should gate traffic. Do
not route external monitoring through authenticated business endpoints.

## Backup and recovery

Back up PostgreSQL and object storage under one recovery policy. Document recovery point and recovery time objectives,
run restore drills, and verify that retained originals can rebuild vector indexes. SQLite WAL files must be included in
consistent local backups.

## Operational playbooks

Maintain playbooks for key rotation, model-provider outage, connector abuse, failed migration, index rebuild, stuck
lease recovery, outbox backlog, corpus rollback, harness rollback, and deletion requests. Alerts should link to a
specific remediation path rather than only report raw metrics.

Detailed procedures: [Backup and restore](operations/backup-restore.md), [Upgrades](operations/upgrades.md), and [Operational runbooks](operations/runbooks.md).
