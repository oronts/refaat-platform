# Backup and restore

A Refaat recovery is complete only when relational state and original document blobs are restored to a mutually consistent point. Search indexes can be rebuilt from durable evidence, but they still affect recovery time.

## Recovery inventory

Back up:

- PostgreSQL data, including pgvector columns and migration version;
- S3-compatible original blobs and object versions;
- externally managed Secrets and ConfigMaps through the platform's secure process;
- evaluation cases, deployment configuration, plugin package versions, and model IDs;
- optional Redis data only when the deployment relies on more than ephemeral rate-limit state.

Do not put raw API keys, model keys, or object-store secrets in ordinary application backups or source archives.

## PostgreSQL

Use managed point-in-time recovery or tested base backups plus WAL archiving. Record:

- recovery point objective;
- recovery time objective;
- retention window;
- encryption and key ownership;
- restore owner and approval path.

A backup is not proven until restored into an isolated environment and validated.

## Object storage

Enable versioning where supported. Align object retention with database recovery so a restored document row does not reference a blob version that no longer exists. Protect deletion and lifecycle configuration with separate permissions.

## SQLite development backups

SQLite is not supported for production. For local state, use the SQLite backup API or stop writers before copying the database. Include WAL state when applicable. Copy the configured local blob directory with the database.

## Restore procedure

1. Freeze writes or isolate the recovery environment.
2. Restore PostgreSQL to the selected point.
3. Restore object storage to a compatible point.
4. Deploy the exact Refaat, plugin, and migration versions expected by the database.
5. Run `alembic current` and schema checks.
6. Verify that sampled document blob URIs exist and checksums match.
7. Start one worker and one API replica in an isolated network.
8. Run readiness, ingestion, query, citation, cache, RLM, job, outbox, evaluation, and deletion smoke tests.
9. Rebuild search state when the restore excludes or changes vector/index data.
10. Re-enable traffic gradually while monitoring failures and empty-context rates.

## Rebuild from originals

Refaat retains original bytes and source metadata, but the current release does not expose a one-command corpus replay utility. Build replay automation around the stored document/blob records or the upstream source of truth, use stable source IDs, and ingest into a new corpus when changing embedding dimensions or extraction semantics.

## Restore drill acceptance

A drill should prove:

- database and blobs agree;
- workspace isolation remains intact;
- API keys and secret rotation work;
- citations point to the restored evidence;
- the worker recovers expired leases without duplicate corruption;
- outbox sinks tolerate duplicate delivery;
- evaluation metrics match the accepted baseline;
- a document can be deleted from relational, search, graph, cache-version, and blob state.
