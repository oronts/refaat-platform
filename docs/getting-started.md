# Getting started

## Requirements

- Python 3.12 or 3.13
- `uv` for the documented development workflow
- no model API key for the deterministic local path

## Install the repository

```bash
cp .env.example .env
uv sync --extra dev --extra documents --extra docs
```

The default development configuration uses SQLite, local blobs, deterministic embeddings, lexical retrieval, co-occurrence graph extraction, and grounded extractive generation.

## Run the complete local example

```bash
uv run python examples/01_local_rag_graph_cache.py
```

That example performs real ingestion, lexical and vector retrieval, graph extraction and expansion, answer generation with citations, an exact CAG hit, document replacement, corpus-version advancement, and cache invalidation.

## Use Refaat from Python

```python
from pathlib import Path

from refaat import Refaat, RlmLimits

with Refaat() as runtime:
    project = runtime.project(
        workspace="northstar",
        workspace_name="Northstar Systems",
        corpus="operations-handbook",
    )
    project.ingest(Path("examples/data/company"))

    answer = project.ask(
        "What evidence is required before a production release?",
        user_id="developer-42",
    )
    recursive = project.rlm(
        "Compare release, security, and incident requirements.",
        user_id="developer-42",
        limits=RlmLimits(
            max_depth=2,
            max_nodes=12,
            branch_factor=3,
            leaf_token_budget=256,
        ),
    )

    print(answer.answer)
    print(answer.citations)
    print(recursive.usage)
```

`runtime.project(...)` creates a missing workspace and corpus and reuses existing resources by workspace slug and corpus name. `project.ingest(...)` accepts a supported file or directory and applies the same ingestion service used by FastAPI and the CLI.

## Use the CLI

The shipped CLI example uses the same SQLite database across separate command processes:

```bash
uv run python examples/07_cli_workflow.py
```

A minimal manual flow is:

```bash
uv run refaat init --slug demo --name "Demo"
uv run refaat corpus create <workspace-id> handbook
uv run refaat ingest \
  <workspace-id> \
  <corpus-id> \
  examples/data/company/02_release_policy.md
uv run refaat query \
  <workspace-id> \
  <corpus-id> \
  "What evidence is required before a production release?"
```

## Run the HTTP service

```bash
uv run refaat-api
```

In development, Swagger UI is available at `/docs`. Provision the first workspace with `POST /v1/bootstrap/workspaces`, then use the returned workspace-scoped API key. Remove or rotate the bootstrap token after provisioning.

The real-network example launches Uvicorn on a temporary loopback port and exercises both SDKs:

```bash
uv run python examples/06_live_api_sdk.py
```

## Verify the checkout

```bash
./scripts/quality.sh
```

The release gate compiles all Python sources, checks style and code structure, executes examples, runs unit/integration/E2E tests with branch coverage, verifies migrations and documentation links, checks the committed OpenAPI contract, builds distributions, and performs a clean-wheel smoke test. Optional host-dependent gates remain visibly unavailable when their required tools or services do not exist.

Continue with [Runnable examples](examples.md) or [Architecture](architecture.md).
