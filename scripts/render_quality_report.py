from __future__ import annotations

from quality_evidence import (
    QUALITY,
    ROOT,
    coverage_summary,
    environment_summary,
    load_status,
    log_excerpt,
    pytest_summary,
    release_decision,
)

REPORT = ROOT / "QUALITY_REPORT.md"
REQUIRED_GATES = {
    "dependencies",
    "compile",
    "style",
    "code_audit",
    "deployment_static",
    "examples",
    "openapi",
    "api_compat",
    "version",
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


def render() -> str:
    status = load_status()
    decision, failed, missing_required = release_decision(status, REQUIRED_GATES)
    unavailable = [name for name, code in status.items() if code == 125]
    environment = environment_summary()

    lines = [
        "# Refaat quality report",
        "",
        f"**Release decision: {decision}.**",
        "",
        (
            "This report is generated from the checked-in release gate. Exit code `125` means "
            "the host capability was unavailable and was not represented as a pass."
        ),
        "",
        "## Environment",
        "",
        f"- Python: `{environment['python']}`",
        f"- Platform: `{environment['platform']}`",
        f"- uv: `{environment['uv']}`",
        f"- Docker: `{environment['docker']}`",
        f"- Helm: `{environment['helm']}`",
        "",
        "## Gate results",
        "",
        "| Gate | Required locally | Exit code | Result |",
        "|---|---|---:|---|",
    ]
    for name, code in status.items():
        result = "PASS" if code == 0 else ("NOT AVAILABLE" if code == 125 else "FAIL")
        required = "yes" if name in REQUIRED_GATES else "no"
        lines.append(f"| `{name}` | {required} | {code} | {result} |")
    if not status:
        lines.append("| release gate | yes | - | NOT RUN |")

    lines.extend(
        [
            "",
            "## Test evidence",
            "",
            f"- Pytest: {pytest_summary()}.",
            f"- Coverage: {coverage_summary()}.",
            (
                "- Layers: unit, persistence integration, failure injection, worker durability, "
                "FastAPI E2E, sync/async SDK E2E, CLI E2E, RLM recursion, cache isolation, "
                "programmatic extensions, executable examples, and migration smoke."
            ),
            "",
            "## Reproduce",
            "",
            "```bash",
            "./scripts/quality.sh",
            "```",
        ]
    )

    if unavailable:
        lines.extend(["", "## Capabilities unavailable on this host", ""])
        lines.extend(f"- `{name}`" for name in unavailable)

    if missing_required:
        lines.extend(["", "## Missing required gates", ""])
        lines.extend(f"- `{name}`" for name in missing_required)

    if failed:
        lines.extend(["", "## Failure excerpts", ""])
        for name in failed:
            lines.extend([f"### `{name}`", "", "```text", log_excerpt(name), "```", ""])
    else:
        lines.extend(
            [
                "",
                "## Verification boundary",
                "",
                (
                    "All required local gates ran successfully. Optional tools marked unavailable "
                    "remain explicit; checked-in CI jobs are configured to validate "
                    "PostgreSQL/pgvector, Docker, Compose, and Helm on suitable hosts. External "
                    "model providers and customer infrastructure still require deployment-specific tests."
                ),
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    REPORT.write_text(render(), encoding="utf-8")
    print(f"Wrote {REPORT.relative_to(ROOT)} from {QUALITY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
