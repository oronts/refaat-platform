# ADR 0004: Cache entries are versioned and identity-scoped

- Status: accepted
- Date: 2026-08-07

## Context

A semantic answer cache can return stale evidence after ingestion or leak personalized context across workspaces, users, or sessions.

## Decision

Exact and semantic cache keys include workspace, corpus, corpus version, user identity when present, session identity when present, query policy, and harness snapshot. Document replacement and deletion advance the corpus version. A cache hit is valid only when all dimensions match and its expiration has not elapsed.

## Consequences

Ingestion invalidates cached answers without global cache scans. Personalized responses are isolated. Higher hit rates cannot be obtained by weakening evidence freshness or tenant boundaries.
