# ADR 0005: Fingerprint the effective query context for answer caches

## Status

Accepted.

## Context

Workspace, corpus version, identity, and normalized query are insufficient cache boundaries. The same question can produce different valid context after retrieval-policy, model, harness, preference, or session-summary changes.

## Decision

Derive the persisted cache scope from a hash of:

- anonymous, user, or session identity;
- immutable base prompt hash;
- embedding and generation model IDs;
- resolved retrieval policy;
- active supplemental harness state;
- current user profile;
- current session summary.

Keep corpus version as a separate lookup dimension. Do not store raw identity or the full context fingerprint payload in the cache key.

## Consequences

A policy override, harness activation/rollback, learned preference, compaction update, or model change cannot reuse an answer created under the previous context. Cache hit rate may decrease after context changes, which is the required correctness trade-off. Regression tests cover policy, harness, and profile isolation.
