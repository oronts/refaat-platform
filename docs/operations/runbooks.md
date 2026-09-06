# Operational runbooks

These runbooks define the minimum response shape. Adapt commands, ownership, and escalation to the deployment platform.

## API readiness failure

1. Check `/health/live` and `/health/ready` separately.
2. Inspect PostgreSQL reachability, credentials, connection limits, and migration state.
3. Confirm the API and worker use the expected release and configuration.
4. Keep the instance out of service until readiness succeeds.
5. Do not bypass readiness by routing business traffic to liveness.

## Worker backlog or expired leases

1. Inspect pending/running/failed job counts and oldest age.
2. Confirm each worker has a unique `REFAAT_WORKER_ID`.
3. Check database time, lease duration, process health, and dependency latency.
4. Restart failed workers; expired leases become recoverable.
5. Investigate repeatedly failing kinds before increasing attempts.
6. Unknown kinds are terminal configuration/code errors, not retryable work.

## Outbox backlog

1. Identify the failing sink and first error.
2. Verify destination availability, authentication, TLS, and egress policy.
3. Confirm receiver idempotency before manual replay.
4. Restore delivery and watch attempts/next-attempt timestamps.
5. Add a temporary sink only through reviewed configuration; all sinks must succeed before acknowledgement.

## Model-provider outage

1. Separate embedding, generation, and graph failures.
2. Stop ingestion when embeddings or graph extraction cannot meet consistency requirements.
3. Decide whether the deterministic/extractive adapters are an accepted fallback for this environment.
4. Protect providers with timeouts and capacity limits.
5. Run evaluation smoke tests after recovery before resuming full traffic.

## Connector abuse or SSRF alert

1. Disable the affected source/connector path and preserve logs.
2. Check destination resolution, redirects, credentials, byte volume, and egress records.
3. Rotate exposed credentials and revoke affected API keys.
4. Enforce network-level destination policy; application checks alone are insufficient.
5. Review stored originals and metadata for malicious content.

## Failed migration

1. Stop new-version API and workers.
2. Preserve migration logs and database state.
3. Determine whether the migration is transactional and whether old code remains compatible.
4. Apply a tested fix-forward, tested down migration, or restore procedure.
5. Do not repeatedly rerun a partially destructive migration without understanding its state.

## Retrieval-quality regression

1. Capture corpus version, model IDs, active harness ID, effective policy, routes, and trace IDs.
2. Re-run the accepted evaluation suite.
3. Compare ingestion checksums, chunk counts, graph counts, and selected evidence.
4. Roll back the harness when the regression follows a supplemental change.
5. Revert model/index traffic only through the prepared migration rollback.

## Harness rollback

1. Identify the last accepted snapshot.
2. Activate it through the admin API or project service.
3. Re-run evaluations and representative queries.
4. Confirm new queries no longer reuse cache entries from the replaced context state.

## Key rotation

1. Issue a new workspace key with the minimum required scopes.
2. Update the consumer through the secret manager.
3. Verify successful traffic with the new key.
4. Revoke the old key.
5. Rotate API-key pepper only with a migration/reissuance plan because stored digests depend on it.
