# Release process

## 1. Prepare

1. Select the version and update `src/refaat/version.py`, `pyproject.toml`, Helm `Chart.yaml`, and changelog.
2. Regenerate the OpenAPI contract.
3. Update examples, documentation, migration notes, and compatibility notes.
4. Remove generated caches, temporary databases, logs, build output, and old release evidence.

## 2. Verify source

```bash
PYTHONPATH=src python scripts/generate_openapi.py --check
PYTHONPATH=src python scripts/version_check.py
PYTHONPATH=src python scripts/examples_check.py
python scripts/source_manifest.py --write
./scripts/quality.sh
```

Required gates must pass. Host-dependent unavailable gates remain recorded and must be executed in CI or another suitable environment before publication.

## 3. Infrastructure matrix

Before a production-facing release, execute:

- Python 3.12 and 3.13 tests;
- PostgreSQL/pgvector integration tests;
- Docker image build and Compose validation/smoke;
- Helm lint and render;
- strict MkDocs build;
- dependency vulnerability audit;
- target-provider contract tests when adapters changed.

## 4. Build distributions

```bash
uv build
```

Install the wheel into a clean environment and run ingestion, retrieval, query, citations, cache, RLM, and version checks. Source archives must not contain local databases, blobs, secrets, caches, or quality logs.

## 5. Package evidence

Preserve:

- JUnit XML;
- branch-aware coverage reports;
- gate logs and status JSON;
- generated distribution names and hashes;
- exact unavailable checks and reasons;
- source manifest and release checksums. Verify extracted source with `python scripts/source_manifest.py --check`.

## 6. Publish

Tag the exact reviewed commit. Publish immutable artifacts and container digests. Update documentation and release notes. Do not rewrite a published artifact under the same version.

## 7. Post-release

Run installation and deployment smoke tests from the published artifacts, monitor issue reports and security channels, and keep the prior compatible release available for rollback.
