#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PYTHON_BIN=${PYTHON_BIN:-python}
export PYTHON_BIN
QUALITY="$ROOT/.quality/current"
STATUS_TSV="$QUALITY/status.tsv"
rm -rf "$QUALITY"
mkdir -p "$QUALITY"
: > "$STATUS_TSV"

record() {
  printf '%s\t%s\n' "$1" "$2" >> "$STATUS_TSV"
}

run_gate() {
  local name=$1
  shift
  echo "==> $name"
  "$@" >"$QUALITY/$name.log" 2>&1
  local code=$?
  record "$name" "$code"
  if [[ $code -eq 0 ]]; then
    echo "    PASS"
  else
    echo "    FAIL ($code)"
  fi
  return 0
}

unavailable() {
  local name=$1
  local reason=$2
  printf '%s\n' "$reason" >"$QUALITY/$name.log"
  record "$name" 125
  echo "==> $name: NOT AVAILABLE"
}

has_module() {
  "$PYTHON_BIN" - "$1" <<'PY' >/dev/null 2>&1
import importlib.util
import sys
raise SystemExit(0 if importlib.util.find_spec(sys.argv[1]) else 1)
PY
}

run_gate dependencies "$PYTHON_BIN" -c \
  "import alembic, fastapi, httpx, jsonschema, orjson, pydantic, sqlalchemy, typer, uvicorn, yaml"
run_gate compile "$PYTHON_BIN" -m compileall -q src tests scripts migrations examples
run_gate style "$PYTHON_BIN" scripts/style_check.py
run_gate code_audit "$PYTHON_BIN" scripts/code_audit.py
run_gate ports_catalog env PYTHONPATH=src "$PYTHON_BIN" scripts/ports_catalog_check.py
run_gate deployment_static "$PYTHON_BIN" scripts/deployment_check.py
run_gate examples env PYTHONPATH=src "$PYTHON_BIN" scripts/examples_check.py
run_gate openapi env PYTHONPATH=src "$PYTHON_BIN" scripts/generate_openapi.py --check
run_gate api_compat "$PYTHON_BIN" scripts/api_compat_check.py --check
run_gate migration_safety "$PYTHON_BIN" scripts/migration_safety_check.py
run_gate sbom "$PYTHON_BIN" scripts/generate_sbom.py --check
run_gate provenance "$PYTHON_BIN" scripts/generate_provenance.py --check
run_gate dependency_policy "$PYTHON_BIN" scripts/dependency_policy_check.py
run_gate threat_coverage "$PYTHON_BIN" scripts/threat_coverage_check.py
run_gate vuln_policy "$PYTHON_BIN" scripts/vuln_policy_check.py
run_gate benchmarks env PYTHONPATH=src "$PYTHON_BIN" scripts/benchmarks_check.py
run_gate version env PYTHONPATH=src "$PYTHON_BIN" scripts/version_check.py
run_gate core_import env PYTHONPATH=src "$PYTHON_BIN" scripts/core_import_check.py
run_gate source_manifest "$PYTHON_BIN" scripts/source_manifest.py --check

if has_module ruff; then
  run_gate ruff_format "$PYTHON_BIN" -m ruff format --check .
  run_gate ruff_lint "$PYTHON_BIN" -m ruff check .
else
  unavailable ruff_format "ruff is not installed on this host"
  unavailable ruff_lint "ruff is not installed on this host"
fi

if has_module mypy; then
  run_gate mypy "$PYTHON_BIN" -m mypy src/refaat
else
  unavailable mypy "mypy is not installed on this host"
fi

run_gate secret_scan "$PYTHON_BIN" scripts/secret_scan.py
run_gate tests_coverage bash -c '
  set -euo pipefail
  export PYTHONPATH=src
  "$PYTHON_BIN" -m coverage erase
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON_BIN" -m coverage run --branch -m pytest \
    -p pytest_asyncio.plugin -m "not postgres" \
    --junitxml=.quality/current/junit.xml
  "$PYTHON_BIN" -m coverage xml -o .quality/current/coverage.xml
  "$PYTHON_BIN" -m coverage json -o .quality/current/coverage.json
  "$PYTHON_BIN" -m coverage report --fail-under=80
'

if [[ -n "${REFAAT_TEST_POSTGRES_URL:-}" ]]; then
  run_gate postgres "$PYTHON_BIN" -m pytest -m postgres -vv
else
  unavailable postgres "REFAAT_TEST_POSTGRES_URL is not configured"
fi

run_gate migration bash -c '
  set -euo pipefail
  database=$(mktemp -t refaat-migration-XXXXXX.db)
  trap "rm -f $database" EXIT
  PYTHONPATH=src \
  REFAAT_DATABASE_URL="sqlite+pysqlite:///$database" \
  REFAAT_AUTO_CREATE_SCHEMA=false \
    "$PYTHON_BIN" -m alembic upgrade head
'
run_gate docs_links "$PYTHON_BIN" scripts/docs_check.py

if has_module mkdocs; then
  run_gate mkdocs "$PYTHON_BIN" -m mkdocs build --strict
else
  unavailable mkdocs "mkdocs is not installed on this host"
fi

run_gate package "$PYTHON_BIN" scripts/build_release.py
run_gate distribution env PYTHONPATH=src "$PYTHON_BIN" scripts/distribution_check.py
run_gate wheel_smoke bash -c '
  set -euo pipefail
  wheel=$(find .quality/current/dist -maxdepth 1 -name "refaat-*.whl" -print -quit)
  test -n "$wheel"
  "$PYTHON_BIN" scripts/wheel_smoke.py "$wheel"
'

if has_module pip_audit; then
  run_gate dependency_audit "$PYTHON_BIN" scripts/dependency_audit.py
else
  unavailable dependency_audit "pip-audit is not installed on this host"
fi

if command -v docker >/dev/null 2>&1; then
  run_gate compose_config env \
    POSTGRES_PASSWORD=quality-postgres-password-000000 \
    REDIS_PASSWORD=quality-redis-password-000000000 \
    OBJECT_STORE_SECRET_ACCESS_KEY=quality-object-store-password-000000 \
    REFAAT_API_KEY_PEPPER=quality-api-key-pepper-000000000000 \
    REFAAT_BOOTSTRAP_TOKEN=quality-bootstrap-token-00000000000 \
    docker compose config --quiet
  run_gate docker_build docker build --tag refaat:quality .
else
  unavailable compose_config "docker is not installed on this host"
  unavailable docker_build "docker is not installed on this host"
fi

if command -v helm >/dev/null 2>&1; then
  run_gate helm_lint helm lint deploy/helm/refaat --strict
  run_gate helm_template helm template refaat deploy/helm/refaat --namespace refaat
else
  unavailable helm_lint "helm is not installed on this host"
  unavailable helm_template "helm is not installed on this host"
fi

"$PYTHON_BIN" - "$STATUS_TSV" "$QUALITY/status.json" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
status = {}
for line in source.read_text(encoding="utf-8").splitlines():
    name, code = line.split("\t", 1)
    status[name] = int(code)
target.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
PY

"$PYTHON_BIN" scripts/render_quality_report.py
PYTHONPATH=src "$PYTHON_BIN" scripts/render_release_status.py

"$PYTHON_BIN" - "$QUALITY/status.json" <<'PY'
import json
import sys
from pathlib import Path

required = {
    "dependencies",
    "compile",
    "style",
    "code_audit",
    "ports_catalog",
    "deployment_static",
    "examples",
    "openapi",
    "api_compat",
    "version",
    "core_import",
    "source_manifest",
    "secret_scan",
    "tests_coverage",
    "migration",
    "migration_safety",
    "sbom",
    "provenance",
    "dependency_policy",
    "threat_coverage",
    "vuln_policy",
    "benchmarks",
    "docs_links",
    "package",
    "distribution",
    "wheel_smoke",
}
status = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
failed = [name for name, code in status.items() if code not in (0, 125)]
missing = [name for name in required if status.get(name, 125) == 125]
raise SystemExit(1 if failed or missing else 0)
PY
