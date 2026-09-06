.DEFAULT_GOAL := help

.PHONY: help install format lint type test test-postgres examples openapi version docs build quality api worker compose-up compose-down

help:
	@printf '%s\n' \
	  'install        Install development and production adapter dependencies' \
	  'format         Format Python sources' \
	  'lint           Check formatting and lint rules' \
	  'type           Run mypy on the public package' \
	  'test           Run unit, integration, and HTTP E2E tests with coverage' \
	  'test-postgres  Run the pgvector/PostgreSQL test marker' \
	  'examples       Execute every shipped product journey' \
	  'openapi        Verify the committed OpenAPI contract' \
	  'version        Verify package, API, Helm, README, and changelog versions' \
	  'docs           Build documentation in strict mode' \
	  'build          Build wheel and source distribution' \
	  'quality        Run the complete local release gate' \
	  'api            Start the local API' \
	  'worker         Start the durable worker' \
	  'compose-up     Start the full development stack' \
	  'compose-down   Stop the development stack'

install:
	uv sync --extra dev --extra documents --extra server

format:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff format --check .
	uv run ruff check .

type:
	uv run mypy src/refaat

test:
	uv run pytest -m "not postgres" --cov=refaat --cov-report=term-missing --cov-report=xml --cov-fail-under=80

test-postgres:
	uv run pytest -m postgres -vv

examples:
	uv run python scripts/examples_check.py

openapi:
	uv run python scripts/generate_openapi.py --check

version:
	uv run python scripts/version_check.py

docs:
	uv run mkdocs build --strict

build:
	uv build

quality:
	./scripts/quality.sh

api:
	uv run refaat-api

worker:
	uv run refaat-worker

compose-up:
	docker compose up --build

compose-down:
	docker compose down --remove-orphans
