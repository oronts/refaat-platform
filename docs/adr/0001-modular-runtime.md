# ADR 0001: One modular runtime, separate process roles

- Status: accepted
- Date: 2026-08-07

## Context

A context platform needs HTTP APIs, durable background work, model and storage adapters, SDKs, and framework integrations. Splitting each capability into an independently deployed service would duplicate contracts, migrations, observability, and local-development setup before independent scaling boundaries are proven.

## Decision

Refaat is one installable Python distribution with domain, application, port, and adapter boundaries. API, worker, migration, and CLI are separate process roles built from the same runtime composition root and immutable image. PostgreSQL, Redis, object storage, model providers, and telemetry exporters remain external infrastructure.

## Consequences

Core behavior is exercised once and reused by every public surface. API and worker can scale independently. A future extraction into a service is allowed only when measurements show an operational boundary, and the existing port becomes the service contract.
