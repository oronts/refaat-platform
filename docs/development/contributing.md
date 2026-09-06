# Contributing

Refaat favors explicit contracts, small modules, deterministic tests, and one implementation of each product semantic.

## Development setup

```bash
uv sync --all-extras --dev
cp .env.example .env
```

Run the complete gate:

```bash
./scripts/quality.sh
```

For a faster edit loop:

```bash
uv run ruff format .
uv run ruff check .
uv run mypy src/refaat
uv run pytest -m "not postgres"
PYTHONPATH=src uv run python scripts/examples_check.py
PYTHONPATH=src uv run python scripts/generate_openapi.py --check
```

## Design rules

- Domain models and ports must not depend on FastAPI or a specific agent framework.
- API, SDK, CLI, worker, and framework adapters call shared application services.
- Evidence IDs, cache boundaries, workspace predicates, and corpus versions are centralized semantics.
- Optional dependencies must fail with actionable installation guidance.
- External side effects need timeouts, bounded payloads, and cleanup.
- Durable handlers and event receivers must be idempotent.
- Do not represent missing tools or skipped infrastructure as passing verification.

## Tests required for changes

| Change | Minimum test |
|---|---|
| Pure algorithm | unit test with boundary cases |
| SQL/persistence behavior | integration test with real temporary database |
| API/SDK/CLI behavior | E2E test through the public surface |
| Connector/model/webhook | controlled protocol transport and error paths |
| Cache/tenancy/security | regression test proving isolation |
| Example | self-asserting script plus manifest/E2E execution |
| Documentation contract | docs link check; OpenAPI regeneration for API changes |
| Migration | clean upgrade and target-backend validation |

## Code quality

The repository checks formatting/linting, strict typing, compilation, unused imports, unreachable statements, duplicate non-trivial function bodies, unfinished markers, secret patterns, coverage, documentation links, OpenAPI drift, packaging, and clean-wheel behavior.

Keep comments for invariants and non-obvious constraints. Prefer clear names and small functions over narration. Remove replaced examples, adapters, migrations, and flags rather than leaving dead alternatives.

## Pull requests

A pull request should state the problem, product behavior, compatibility impact, tests, security/tenancy impact, migration impact, and documentation changes. Large architecture changes should add an ADR.
