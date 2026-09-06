# Testing and release evidence

Refaat uses one checked-in release gate:

```bash
./scripts/quality.sh
```

The gate records every result under `.quality/current`, including machine-readable status, JUnit XML, branch-aware coverage JSON/XML, package artifacts, logs, and `QUALITY_REPORT.md`.

## Required local gates

- runtime dependency imports;
- bytecode compilation for source, tests, migrations, scripts, and examples;
- repository style and stale-name checks;
- AST audit for unused imports, duplicate non-trivial function bodies, unreachable statements, and unfinished markers;
- static Compose, shared-image, Helm-schema, lifecycle, and shell validation;
- committed-secret scan;
- all executable product examples in the catalog;
- unit, integration, failure-injection, worker, FastAPI, sync/async SDK, CLI, RLM, and example-catalog E2E tests;
- branch-aware coverage with an 80% release floor;
- Alembic upgrade against a clean database;
- Markdown links, MkDocs navigation, and required REFAAT branding;
- committed OpenAPI drift check;
- package, runtime, OpenAPI, Helm, README, and changelog version consistency;
- checked source-manifest integrity with missing, changed, and unexpected-file detection;
- wheel and source-distribution build;
- wheel and source-distribution content validation, including the complete docs, examples, tests, deployment files, and release tooling in the source archive;
- clean-wheel functional smoke using ingestion, query, citations, and RLM.

## Optional host-dependent gates

Ruff, mypy, MkDocs strict build, dependency auditing, PostgreSQL/pgvector tests, Docker/Compose, and Helm run when their tools or services are available. An unavailable capability receives exit code `125`; it is never represented as a pass. CI should provide the complete matrix.

## Test layers

- **Unit:** deterministic algorithms, extractors, connectors, security boundaries, events, plugins, configuration, and cleanup.
- **Integration:** real temporary SQL persistence, atomic ingestion compensation, cache-context isolation, feedback, harness rollback, sessions, extensions, agents, jobs, evaluations, project API, and RLM.
- **E2E:** FastAPI middleware/authentication, workspace isolation, uploads/source ingestion, SDKs, CLI, payload limits, worker surfaces, and the shipped example catalog. A separate mandatory example gate executes every shipped journey in the catalog; its live-server workflows use a real Uvicorn subprocess over loopback TCP.
- **PostgreSQL:** database-side lexical and pgvector behavior when `REFAAT_TEST_POSTGRES_URL` is configured.

External provider adapters use controlled protocol transports so request/response contracts can be tested without paid model calls. Those contract tests are not substitutes for target-provider staging tests.

## Useful commands

```bash
uv sync --all-extras --dev
uv run ruff format --check .
uv run ruff check .
uv run mypy src/refaat
uv run pytest -m "not postgres"
PYTHONPATH=src uv run python scripts/examples_check.py
PYTHONPATH=src uv run python scripts/generate_openapi.py --check
uv run mkdocs build --strict
uv run pip-audit
uv build
```

A green report is evidence for the environment where it ran, not a substitute for target load, restore, security, compliance, network, and provider validation.
