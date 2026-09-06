# Python API reference

Refaat exposes a high-level project facade for common application work and lower-level services for explicit orchestration.

## Public package exports

```python
from refaat import (
    IngestResult,
    ProjectRef,
    QueryRequest,
    QueryResponse,
    Refaat,
    RefaatProject,
    RlmLimits,
    RlmNodeTrace,
    RlmResult,
    RuntimeBuilder,
    __version__,
)
```

## `Refaat`

`Refaat()` builds the configured runtime. Use it as a context manager so database connections, HTTP clients, object-store clients, connectors, model adapters, and event sinks are closed deterministically.

```python
with Refaat() as runtime:
    workspace_id = runtime.create_workspace("acme", "Acme")
    corpus_id = runtime.create_corpus(workspace_id, "handbook")
```

Important methods:

| Method | Result |
|---|---|
| `create_workspace(slug, name)` | workspace ID |
| `issue_api_key(workspace_id, name, scopes)` | key ID and one-time raw key |
| `create_corpus(workspace_id, name, settings)` | corpus ID |
| `project(...)` | `RefaatProject` facade |
| `ingest(workspace_id, corpus_id, document)` | `IngestResult` |
| `ingest_source(...)` | `IngestResult` |
| `retrieve(QueryRequest)` | `ContextPacket` |
| `query(QueryRequest)` | `QueryResponse` |
| `run_rlm(...)` | `RlmResult` |
| `enqueue_rlm(...)` | durable job ID |
| `submit_feedback(...)` | `None` |
| `close()` | closes owned resources idempotently |

## `RefaatProject`

Create or reuse resources by stable names:

```python
project = runtime.project(
    workspace="acme",
    workspace_name="Acme GmbH",
    corpus="engineering",
    corpus_settings={"language": "en"},
)
```

Project methods:

| Method | Description |
|---|---|
| `ingest(path, recursive=True, metadata=None)` | Ingest one supported file or a directory |
| `ingest_source(source, ...)` | Ingest through a registered connector |
| `ask(query, ...)` | Retrieve, generate, validate citations, cache, and optionally record a session |
| `retrieve(query, **options)` | Return the selected `ContextPacket` without generation |
| `rlm(query, limits=...)` | Run bounded recursive context processing |
| `enqueue_rlm(query, limits=...)` | Persist an RLM job for a worker |
| `create_session(...)` | Create a durable session and optional goal |
| `list_documents()` | List active corpus documents |
| `stats()` | Return document, chunk, entity, and relation counts |
| `delete_document(document_id)` | Delete evidence and advance corpus version |
| `submit_feedback(...)` | Record explicit feedback and bounded preferences |
| `run_skill(name, query, ...)` | Execute an installed skill over retrieved context |
| `create_evaluation_suite(name, cases)` | Persist a regression suite |
| `run_evaluation(suite_id)` | Execute a suite synchronously |
| `enqueue_evaluation(suite_id)` | Execute a suite through the worker |
| `active_harness()` | Read active supplemental state |
| `create_harness_snapshot(...)` | Create an immutable evidence-backed snapshot |
| `activate_harness_snapshot(snapshot_id)` | Activate or roll back to a snapshot |

## Query request

```python
request = QueryRequest(
    workspace_id=workspace_id,
    corpus_id=corpus_id,
    query="What is required before release?",
    user_id="alice",
    session_id=session_id,
    top_k=8,
    token_budget=2_000,
    metadata={"channel": "assistant"},
)
response = runtime.query(request)
```

`top_k` and `token_budget` are optional request overrides. The resolved policy and effective context state are included in the answer-cache fingerprint.

## Query response

`QueryResponse` contains:

- `answer`;
- structured `citations`;
- selected context `routes`;
- `cached` flag;
- `corpus_version`;
- `trace_id`;
- provider and context `usage`.

A cached result has `cag:exact` or `cag:semantic` as its first route.

## RLM limits

```python
limits = RlmLimits(
    max_depth=2,
    max_nodes=16,
    branch_factor=4,
    leaf_token_budget=1_500,
)
```

Validation bounds are depth `0..6`, nodes `1..128`, branch factor `2..16`, and leaf token budget `64..100000`.

## `RuntimeBuilder`

Use `RuntimeBuilder` when an application supplies custom extension objects:

```python
runtime = (
    RuntimeBuilder(settings)
    .with_connector(connector)
    .with_extractor(extractor)
    .with_skill(skill)
    .with_event_sink(event_sink)
    .build()
)
```

Provider and persistence adapters are selected through `Settings`. Application-owned extension instances are registered after built-ins and plugin discovery, and duplicate registry keys are rejected.

## Lower-level services

The runtime exposes composed services such as `runtime.ingestion`, `runtime.queries`, `runtime.evaluations`, `runtime.harness`, `runtime.agents`, `runtime.rlm`, `runtime.skills`, and `runtime.store`. These are public to advanced integrators but have a larger compatibility surface than the project facade.
