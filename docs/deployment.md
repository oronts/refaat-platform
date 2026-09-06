# Production deployment

## Required production settings

Production startup rejects SQLite, automatic schema creation, wildcard CORS, a default or short API-key pepper, and a
short configured bootstrap token. Run `alembic upgrade head` as a controlled release step before API or worker rollout.

Use PostgreSQL/pgvector for relational, lexical, and vector search; Redis for distributed rate limiting; and S3-compatible
object storage for originals. API, worker, and migration roles use the same image and application services.

## Release order

1. Back up PostgreSQL and object storage and verify restore readiness.
2. Run the migration job once.
3. Deploy API replicas and verify readiness.
4. Deploy uniquely identified workers and inspect leases and outbox lag.
5. Run ingestion, query, RLM, evaluation, worker, and rollback smoke tests.
6. Observe errors, latency, empty-context rate, cache routes, and retrieval quality before increasing traffic.

## Docker image

The multi-stage image builds `refaat[server]`, copies only the environment and application wheel into the runtime stage,
and runs as UID/GID `10001` with a writable `/tmp` supplied by the deployment. API is the default command; worker and
migration use explicit commands.

## Compose

`compose.yaml` provides PostgreSQL/pgvector, password-protected Redis, a pinned SeaweedFS S3 service with an authenticated pre-created bucket, migration, API, worker, and optional OpenTelemetry Collector roles. Required secrets use Compose's `${VARIABLE:?message}` form. The API binds to loopback by default. Data services remain on an internal backend network; API and worker roles also join an explicit egress network for configured model providers and approved connectors.

Compose is a functional development/staging topology, not a complete regulated production platform. Pin images by
digest and add managed secrets, TLS, backup automation, monitoring, capacity limits, and network controls.

## Kubernetes

The chart under `deploy/helm/refaat` includes:

- separate API and worker deployments;
- a pre-install/pre-upgrade Alembic job;
- external Secret and ConfigMap wiring;
- non-root, read-only containers with dropped capabilities;
- startup, readiness, and liveness probes;
- rollout checksums, disruption budgets, topology spread, and resource defaults;
- optional HPA, ingress, and network policy resources.

The chart deliberately does not install PostgreSQL, Redis, object storage, or a model provider. Use separately operated
or managed dependencies and provide their credentials through the referenced Secret.

## Model and index changes

Treat embedding model or dimension changes as data migrations. Create a new corpus/index version, replay retained
originals, evaluate the candidate, and switch traffic only after acceptance. Do not mix incompatible vectors in one
pgvector column.

Before launch, complete [Backup and restore](operations/backup-restore.md), [Upgrades](operations/upgrades.md), [Operational runbooks](operations/runbooks.md), and the [Production checklist](production-checklist.md).
