# API, SDK, and CLI

Refaat exposes one application runtime through local Python, FastAPI, synchronous and asynchronous HTTP SDKs, a CLI,
and a durable worker. These surfaces share the same services and evidence contracts.

## Authentication

Use `Authorization: Bearer <key>` or `X-API-Key`. API keys are workspace-scoped and grant `read`, `write`, or
`admin` scopes. The raw key is returned once; persistence keeps only a lookup prefix and a scrypt-derived digest
combined with the application pepper.

The bootstrap endpoint exists only when `REFAAT_BOOTSTRAP_TOKEN` is configured. Rotate or remove the token after
initial provisioning.

## HTTP resources

| Area | Endpoints |
|---|---|
| Health | `GET /health/live`, `GET /health/ready` |
| Bootstrap and keys | `/v1/bootstrap/workspaces`, `/v1/api-keys` |
| Corpora | `/v1/corpora`, `/v1/corpora/{id}`, `/stats` |
| Documents | direct text/upload ingestion, source ingestion, listing, and deletion |
| Context | `/retrieve`, `/query`, `/feedback` |
| RLM | `/rlm` for synchronous execution and `/rlm/jobs` for durable execution |
| Sessions | create and read sessions with retained turns |
| Jobs and schedules | create/read jobs and create interval schedules |
| Agents | create, heartbeat, send/receive messages, and acknowledge delivery |
| Evaluations | create suites, run synchronously or as a job, and inspect results |
| Harness | inspect active state, create/activate snapshots, and propose bounded changes |
| Skills | list, run, and generate reviewed skill projects |

Development deployments expose OpenAPI and Swagger UI. Production disables interactive documentation by default.
Every error uses a stable envelope with `code`, safe `message`, and `trace_id`; unexpected exceptions do not expose
stack traces through HTTP.

## Local Python API

```python
from refaat import Refaat, RlmLimits

with Refaat() as runtime:
    project = runtime.project(workspace="example", corpus="handbook")
    project.ingest("./documents")

    answer = project.ask("What is the rollback policy?")
    recursive = project.rlm(
        "Compare rollback, release, and incident controls",
        limits=RlmLimits(max_depth=2, max_nodes=12),
    )
```

The project facade accepts workspace/corpus IDs or stable names and creates missing local resources. Lower-level
services remain available for applications that need explicit IDs and transactions.

## HTTP SDKs

```python
from refaat.sdk import AsyncRefaatClient, RefaatClient

client = RefaatClient(base_url="http://localhost:8000", api_key="refaat_...")
answer = client.query(corpus_id, "What changed?")
result = client.run_rlm(corpus_id, "Compare the complete policy set")

async with AsyncRefaatClient(
    base_url="http://localhost:8000",
    api_key="refaat_...",
) as async_client:
    answer = await async_client.query(corpus_id, "What changed?")
```

The SDKs expose typed answers, citations, retrieval documents, RLM node traces, and usage values.

## CLI

```bash
refaat init --slug acme --name "Acme"
refaat corpus create <workspace-id> handbook
refaat ingest <workspace-id> <corpus-id> ./policy.md
refaat ingest-source <workspace-id> <corpus-id> https://example.com/policy.md
refaat query <workspace-id> <corpus-id> "What is the policy?"
refaat rlm <workspace-id> <corpus-id> "Compare all controls"
refaat evaluation create <workspace-id> release ./cases.json
refaat evaluation run <workspace-id> <suite-id> <corpus-id>
refaat skill list
refaat doctor
refaat worker --once
```

Use `refaat serve` for local API development. Production should invoke `refaat-api` behind a trusted ingress and run
`refaat-worker` as an independently scaled process.


## Detailed references

- [Python API](reference/python-api.md)
- [HTTP API and committed OpenAPI contract](reference/http-api.md)
- [CLI](reference/cli.md)
- [Errors](reference/errors.md)
