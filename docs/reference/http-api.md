# HTTP API reference

The FastAPI service exposes a workspace-scoped set of paths under one authentication model. The committed machine-readable contract, and the authoritative list of every path, is available as [`openapi.json`](openapi.json).

## Authentication

Use either header:

```http
Authorization: Bearer refaat_...
```

```http
X-API-Key: refaat_...
```

Keys belong to one workspace and contain `read`, `write`, and/or `admin` scopes. The raw key is returned once. Bootstrap uses `X-Bootstrap-Token` and is available only when `REFAAT_BOOTSTRAP_TOKEN` is configured.

## Health

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health/live` | process liveness |
| `GET` | `/health/ready` | relational-store readiness |

## Bootstrap and access

| Method | Path | Scope |
|---|---|---|
| `POST` | `/v1/bootstrap/workspaces` | bootstrap token |
| `POST` | `/v1/api-keys` | admin |
| `DELETE` | `/v1/api-keys/{key_id}` | admin |

## Corpora and documents

| Method | Path | Scope |
|---|---|---|
| `POST` | `/v1/corpora` | write |
| `GET` | `/v1/corpora` | read |
| `GET` | `/v1/corpora/{corpus_id}` | read |
| `GET` | `/v1/corpora/{corpus_id}/stats` | read |
| `GET` | `/v1/corpora/{corpus_id}/derivations` | read |
| `GET` | `/v1/corpora/{corpus_id}/derivations/verify` | admin |
| `GET` | `/v1/corpora/{corpus_id}/derivations/compare` | read |
| `GET` | `/v1/corpora/{corpus_id}/derivations/{derivation_key}` | read |
| `GET` | `/v1/corpora/{corpus_id}/documents/{document_id}/derivation/verify` | admin |
| `POST` | `/v1/corpora/{corpus_id}/documents/text` | write |
| `POST` | `/v1/corpora/{corpus_id}/documents/upload` | write |
| `POST` | `/v1/corpora/{corpus_id}/documents/source` | write |
| `POST` | `/v1/corpora/{corpus_id}/documents/image` | write |
| `GET` | `/v1/corpora/{corpus_id}/documents` | read |
| `DELETE` | `/v1/corpora/{corpus_id}/documents/{document_id}` | write |

The derivation routes are read-only and present only when `derivation_manifest_enabled` is set; otherwise they return 404. A derivation manifest records the content-addressed recipe, per-table row counts, and an order-independent content checksum over stored chunk identities for a corpus. `verify` is admin-scoped because it recomputes the O(corpus) rollup from committed state and returns `no_manifest`, `match`, or `drift`. `compare` takes two `left` and `right` derivation keys of the same corpus. The checksum covers chunk-derived state and the row counts report each table, so a graph-only divergence is reflected in the counts rather than the chunk checksum.

A direct text request contains `source_id`, `title`, `content`, `media_type`, and optional metadata. Uploads use multipart form data. Source ingestion accepts a registered URI scheme such as `file://`, `http://`, `https://`, or a plugin scheme. Image ingestion posts multipart form data (`file`, `source_id`, and optional `title`, `caption`, `alt_text`, `metadata_json`) and stores one citable descriptor chunk.

## Retrieval, query, feedback, and RLM

| Method | Path | Scope |
|---|---|---|
| `POST` | `/v1/corpora/{corpus_id}/retrieve` | read |
| `POST` | `/v1/corpora/{corpus_id}/query` | read |
| `POST` | `/v1/corpora/{corpus_id}/query/stream` | read |
| `POST` | `/v1/corpora/{corpus_id}/feedback` | write |
| `POST` | `/v1/corpora/{corpus_id}/rlm` | write |
| `POST` | `/v1/corpora/{corpus_id}/rlm/jobs` | write |

Query body:

```json
{
  "query": "What evidence is required before release?",
  "user_id": "alice",
  "session_id": null,
  "top_k": 8,
  "token_budget": 2000,
  "metadata": {"channel": "assistant"}
}
```

The query response includes answer text, citations, routes, cache state, corpus version, trace ID, and usage. Retrieval returns the selected chunk text, scores, reasons, graph facts, routes, and token count without generation.

## Sessions, jobs, schedules, and agents

| Method | Path | Scope |
|---|---|---|
| `POST` | `/v1/sessions` | write |
| `GET` | `/v1/sessions/{session_id}` | read |
| `POST` | `/v1/jobs` | write |
| `GET` | `/v1/jobs/{job_id}` | read |
| `POST` | `/v1/schedules` | write |
| `POST` | `/v1/agents` | write |
| `POST` | `/v1/agents/{agent_id}/heartbeat` | write |
| `POST` | `/v1/agents/{sender_agent_id}/messages` | write |
| `GET` | `/v1/agents/{agent_id}/messages` | read |
| `POST` | `/v1/agent-messages/{message_id}/acknowledge` | write |

Only registered worker job kinds are executable. Unknown kinds fail terminally rather than remaining in the lease queue.

## Evaluations, harness, and skills

| Method | Path | Scope |
|---|---|---|
| `POST` | `/v1/evaluations/suites` | write |
| `POST` | `/v1/evaluations/suites/{suite_id}/runs` | write |
| `GET` | `/v1/evaluations/runs/{run_id}` | read |
| `GET` | `/v1/harness/active` | read |
| `POST` | `/v1/harness/snapshots` | admin |
| `POST` | `/v1/harness/snapshots/{snapshot_id}/activate` | admin |
| `POST` | `/v1/harness/proposals/from-evaluation/{run_id}` | admin |
| `GET` | `/v1/skills` | read |
| `POST` | `/v1/skills/{skill_name}/run` | admin |
| `POST` | `/v1/skills/projects` | admin |

The streaming query endpoint serves server-sent events: answer deltas followed by one authoritative grounded `final` event that reuses the same retrieval, cache, and citation pipeline as the non-streaming query.

## Programs, optimization, and deployment

| Method | Path | Scope |
|---|---|---|
| `GET` | `/v1/corpora/{corpus_id}/programs` | read |
| `POST` | `/v1/corpora/{corpus_id}/programs/{name}/run` | read |
| `POST` | `/v1/corpora/{corpus_id}/programs/{name}/run/stream` | read |
| `POST` | `/v1/corpora/{corpus_id}/programs/{name}/evaluate` | read |
| `POST` | `/v1/programs/{name}/optimizations` | write |
| `GET` | `/v1/optimizations/{run_id}` | read |
| `POST` | `/v1/corpora/{corpus_id}/programs/{name}/promote` | admin |
| `POST` | `/v1/corpora/{corpus_id}/programs/{name}/approve` | admin |
| `GET` | `/v1/programs/{name}/deployments/{environment}` | read |
| `POST` | `/v1/programs/{name}/rollback` | admin |
| `POST` | `/v1/corpora/{corpus_id}/programs/{name}/canary` | admin |
| `DELETE` | `/v1/corpora/{corpus_id}/programs/{name}/canary` | admin |
| `GET` | `/v1/programs/{name}/canary/{environment}` | read |
| `POST` | `/v1/corpora/{corpus_id}/programs/{name}/deployments/{environment}/run` | read |

Program run and evaluation are read-scoped. Durable optimization enqueues a worker run under the write scope and is polled through `GET /v1/optimizations/{run_id}`. Promotion, approval, rollback, and attaching or clearing a canary are admin-scoped; a run through the resolved active artifact plus its optional canary and the shadow-comparison status stay read-scoped. The run and run/stream endpoints keep the same grounding and citation contract as the query endpoints.

## Fine-tuning

| Method | Path | Scope |
|---|---|---|
| `POST` | `/v1/fine-tunes` | admin |
| `POST` | `/v1/fine-tunes/{run_id}/approve` | admin |
| `POST` | `/v1/fine-tunes/{run_id}/enqueue` | admin |
| `GET` | `/v1/fine-tunes/{run_id}` | read |

Governed fine-tuning is approval-gated: a run is requested, approved, then enqueued or run, and a completed run registers an immutable, un-promoted model artifact. See [Governed fine-tuning](../fine-tuning.md).

## Approvals

| Method | Path | Scope |
|---|---|---|
| `POST` | `/v1/approvals` | admin |
| `GET` | `/v1/approvals` | admin |
| `DELETE` | `/v1/approvals/{approval_id}` | admin |

A grant records a durable, single-use human approval for one action fingerprint. The request body carries only `fingerprint` and `capability_name`; the approver is the server-resolved identity, never a body field, so a caller can only approve as itself and the separation-of-duties and four-eyes checks hold at consume time. `GET /v1/approvals` takes a `fingerprint` query parameter and lists the active approvals for it. See [Approvals](../approvals.md) and [Tool governance](../tool-governance.md).

## Data lifecycle and operations

| Method | Path | Scope |
|---|---|---|
| `POST` | `/v1/corpora/{corpus_id}/documents/{document_id}/erase` | admin |
| `POST` | `/v1/corpora/{corpus_id}/legal-holds` | admin |
| `GET` | `/v1/corpora/{corpus_id}/legal-holds` | admin |
| `POST` | `/v1/legal-holds/{hold_id}/lift` | admin |
| `POST` | `/v1/kill-switches` | admin |
| `GET` | `/v1/kill-switches` | admin |
| `POST` | `/v1/kill-switches/{scope}/disengage` | admin |
| `GET` | `/v1/status` | admin |

Erasure removes or re-derives every auto-derived surface a document produced. A legal hold blocks erasure, deletion, and overwrite; under the defer conflict policy a held erasure is recorded and completed when the last covering hold lifts. A kill switch freezes a workspace's AI and data plane while the governance and control plane stays available. The status plane is a read-only admin snapshot of health and counts. All are admin-scoped and audited. See [Right to erasure](../erasure.md), [Legal holds](../legal-holds.md), [Kill switches](../kill-switches.md), and [Admin status plane](../admin-status.md).

## Error envelope

Expected errors return:

```json
{
  "error": {
    "code": "validation_error",
    "message": "Query is empty",
    "trace_id": "..."
  }
}
```

Request-schema errors use `request_validation_error` and include safe Pydantic details. Unexpected errors use `internal_error` and never expose a stack trace in the HTTP response.

## Interactive documentation

Swagger UI is enabled at `/docs` outside production. Production disables it by default. The committed OpenAPI file is generated from the actual application and verified by the quality gate:

```bash
PYTHONPATH=src python scripts/generate_openapi.py --check
```
