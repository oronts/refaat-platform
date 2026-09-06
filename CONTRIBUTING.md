# Contributing

Use Python 3.12+, `uv`, and conventional focused commits. Start with an issue for behavior or public-contract changes.

```bash
uv sync --all-extras --dev
pre-commit install
make quality
```

Keep domain behavior framework-independent. New infrastructure must implement a typed port and include contract tests. Do not add dependencies to the core for one provider. Document migrations, configuration, security impact, and rollback for production-facing changes.

Pull requests require tests, type checks, formatting, documentation updates, and no secrets or generated build artifacts.
