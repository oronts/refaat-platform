# Errors and failure behavior

Refaat distinguishes expected platform errors from unexpected defects.

## Domain error classes

| Class | Code | HTTP status |
|---|---|---:|
| `NotFoundError` | `not_found` | 404 |
| `ConflictError` | `conflict` | 409 |
| `AuthenticationError` | `authentication_failed` | 401 |
| `AuthorizationError` | `forbidden` | 403 |
| `ValidationError` | `validation_error` | 422 |
| `RateLimitError` | `rate_limited` | 429 |
| `ConfigurationError` | `configuration_error` | 500 |
| `CapabilityNotSupportedError` | `capability_not_supported` | 501 |
| `DependencyUnavailableError` | `dependency_unavailable` | 503 |

All inherit from `RefaatError`. `CapabilityNotSupportedError` is raised by a search backend that declares it cannot serve a mode; the retriever skips a declared-unsupported mode, so it surfaces only when a backend raises for a mode it did not declare unsupported.

## HTTP behavior

Expected errors use a safe envelope:

```json
{
  "error": {
    "code": "not_found",
    "message": "Corpus not found",
    "trace_id": "..."
  }
}
```

Pydantic request failures use `request_validation_error` with safe field details. Unexpected exceptions are logged with context and returned as `internal_error` without stack data.

## SDK behavior

`RefaatClient` and `AsyncRefaatClient` raise `RefaatAPIError` with:

- `status_code`;
- stable `code`;
- safe message;
- optional `trace_id`.

```python
from refaat.sdk import RefaatAPIError

try:
    client.query(corpus_id, "...")
except RefaatAPIError as exc:
    logger.warning("refaat_request_failed", code=exc.code, trace_id=exc.trace_id)
```

## Ingestion compensation

Original bytes are written before indexing. If extraction, embedding, graph extraction, index replacement, or relational commit fails, Refaat attempts to restore the previous search state and deletes the newly written blob. Cleanup failures are logged for operator action. A successful replacement removes the old blob after commit.

## Jobs

A job failure records the error and retries until its configured attempt limit. Unknown job kinds fail terminally. Leases allow another worker to recover work after the lease expires.

## Outbox

If any sink fails, the outbox event remains pending and receives an exponential next-attempt delay. Already delivered external side effects may be repeated, so sinks must be idempotent.

## External dependencies

Connector, model, webhook, Redis, S3, and PostgreSQL errors are normalized where practical but still require retries, timeouts, circuit breaking, and alerts appropriate to the deployment. Do not retry validation or authorization errors blindly.
