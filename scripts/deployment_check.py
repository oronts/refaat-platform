from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "compose.yaml"
CHART = ROOT / "deploy" / "helm" / "refaat"
REQUIRED_SERVICES = {
    "postgres",
    "redis",
    "object-store",
    "migrate",
    "api",
    "worker",
}
REQUIRED_TEMPLATES = {
    "api-deployment.yaml",
    "worker-deployment.yaml",
    "migration-job.yaml",
    "configmap.yaml",
    "service.yaml",
    "pdb.yaml",
    "hpa.yaml",
    "ingress.yaml",
    "networkpolicy.yaml",
}
WORKLOAD_SERVICES = ("migrate", "api", "worker")


def as_mapping(value: object, label: str, problems: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        problems.append(f"{label} must be a mapping")
        return {}
    return {str(key): item for key, item in value.items()}


def check_compose(problems: list[str]) -> None:
    document = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    root = as_mapping(document, "compose document", problems)
    services = as_mapping(root.get("services"), "compose services", problems)
    missing = sorted(REQUIRED_SERVICES - services.keys())
    if missing:
        problems.append(f"compose is missing services: {', '.join(missing)}")

    for name in WORKLOAD_SERVICES:
        service = as_mapping(services.get(name), f"compose service {name}", problems)
        if service.get("read_only") is not True:
            problems.append(f"compose service {name} must use a read-only root filesystem")
        if "ALL" not in service.get("cap_drop", []):
            problems.append(f"compose service {name} must drop all Linux capabilities")
        if "no-new-privileges:true" not in service.get("security_opt", []):
            problems.append(f"compose service {name} must disable privilege escalation")

    api = as_mapping(services.get("api"), "compose service api", problems)
    if "healthcheck" not in api:
        problems.append("compose API service must define a health check")
    migrate = as_mapping(services.get("migrate"), "compose service migrate", problems)
    if migrate.get("command") != ["alembic", "upgrade", "head"]:
        problems.append("compose migration service must run Alembic upgrade head")
    worker = as_mapping(services.get("worker"), "compose service worker", problems)
    if worker.get("command") != ["refaat-worker"]:
        problems.append("compose worker service must run refaat-worker")

    for name in ("api", "worker"):
        service = as_mapping(services.get(name), f"compose service {name}", problems)
        networks = service.get("networks", [])
        if not {"backend", "egress"}.issubset(set(networks)):
            problems.append(f"compose service {name} must use isolated backend and explicit egress networks")

    object_store = as_mapping(
        services.get("object-store"),
        "compose service object-store",
        problems,
    )
    image = str(object_store.get("image", ""))
    if "seaweedfs" not in image.casefold() or ":4.29" not in image:
        problems.append("compose object store must use the pinned SeaweedFS 4.29 image")
    environment = as_mapping(
        object_store.get("environment"),
        "compose object-store environment",
        problems,
    )
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "S3_BUCKET"):
        if key not in environment:
            problems.append(f"compose object store is missing {key}")

    for name, value in services.items():
        service = as_mapping(value, f"compose service {name}", problems)
        image_value = str(service.get("image", ""))
        if image_value.casefold().endswith(":latest}") or image_value.casefold().endswith(":latest"):
            problems.append(f"compose service {name} uses an unpinned latest image")


def check_dockerfile(problems: list[str]) -> None:
    content = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    required = (
        "AS builder",
        "AS runtime",
        '".[server]"',
        "USER 10001:10001",
        'CMD ["refaat-api"]',
    )
    for value in required:
        if value not in content:
            problems.append(f"Dockerfile is missing {value!r}")
    if "HEALTHCHECK" in content:
        problems.append("shared Docker image must not define an API-only HEALTHCHECK")


def check_helm(problems: list[str]) -> None:
    chart = yaml.safe_load((CHART / "Chart.yaml").read_text(encoding="utf-8"))
    values = yaml.safe_load((CHART / "values.yaml").read_text(encoding="utf-8"))
    schema = json.loads((CHART / "values.schema.json").read_text(encoding="utf-8"))
    if not isinstance(chart, dict) or chart.get("name") != "refaat":
        problems.append("Helm Chart.yaml must define the refaat chart")
    for error in sorted(Draft202012Validator(schema).iter_errors(values), key=str):
        location = ".".join(str(item) for item in error.absolute_path) or "<root>"
        problems.append(f"Helm values schema error at {location}: {error.message}")

    templates = CHART / "templates"
    present = {path.name for path in templates.glob("*.yaml")}
    missing = sorted(REQUIRED_TEMPLATES - present)
    if missing:
        problems.append(f"Helm chart is missing templates: {', '.join(missing)}")

    api = (templates / "api-deployment.yaml").read_text(encoding="utf-8")
    for value in (
        "startupProbe:",
        "readinessProbe:",
        "livenessProbe:",
        ".Values.containerSecurityContext",
        ".Values.podSecurityContext",
    ):
        if value not in api:
            problems.append(f"Helm API deployment is missing {value}")

    worker = (templates / "worker-deployment.yaml").read_text(encoding="utf-8")
    for value in ("refaat-worker", ".Values.containerSecurityContext"):
        if value not in worker and value == "refaat-worker":
            values_worker = as_mapping(
                as_mapping(values, "Helm values", problems).get("worker"),
                "Helm worker values",
                problems,
            )
            if values_worker.get("command") != ["refaat-worker"]:
                problems.append("Helm worker deployment must resolve to refaat-worker")
        elif value not in worker:
            problems.append(f"Helm worker deployment is missing {value}")

    migration = (templates / "migration-job.yaml").read_text(encoding="utf-8")
    if 'include "refaat.envFrom"' in migration:
        problems.append("Helm pre-install migration must not reference the release ConfigMap")
    if 'include "refaat.serviceAccountName"' in migration:
        problems.append("Helm pre-install migration must not reference a release ServiceAccount")
    for value in (
        'include "refaat.secretEnvFrom"',
        "range $key, $value := .Values.config",
        'command: ["alembic", "upgrade", "head"]',
    ):
        if value not in migration:
            problems.append(f"Helm migration job is missing {value!r}")
    migration_values = as_mapping(
        as_mapping(values, "Helm values", problems).get("migration"),
        "Helm migration values",
        problems,
    )
    annotations = as_mapping(
        migration_values.get("annotations"),
        "Helm migration annotations",
        problems,
    )
    if annotations.get("helm.sh/hook") != "pre-install,pre-upgrade":
        problems.append("Helm migration must be a pre-install and pre-upgrade hook")


def check_shell_scripts(problems: list[str]) -> None:
    for path in sorted((ROOT / "scripts").glob("*.sh")):
        result = subprocess.run(
            ["bash", "-n", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            detail = result.stderr.strip() or f"exit {result.returncode}"
            problems.append(f"{path.relative_to(ROOT)} is not valid shell: {detail}")


def main() -> int:
    problems: list[str] = []
    check_compose(problems)
    check_dockerfile(problems)
    check_helm(problems)
    check_shell_scripts(problems)
    if problems:
        print("Deployment check failed:")
        print("\n".join(f"- {problem}" for problem in problems))
        return 1
    print("Deployment check passed for Compose, the shared image, Helm structure, values schema, and shell scripts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
