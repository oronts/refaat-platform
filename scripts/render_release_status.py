from __future__ import annotations

from datetime import date

from quality_evidence import (
    ROOT,
    coverage_summary,
    environment_summary,
    json_log,
    load_status,
    pytest_summary,
    release_decision,
    unavailable_reason,
)

from refaat.version import __version__

REPORT = ROOT / "RELEASE_STATUS.md"
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
    decision, failed, missing = release_decision(status, REQUIRED_GATES)
    unavailable = [name for name, code in status.items() if code == 125]
    environment = environment_summary()
    examples = json_log("examples") or {}
    distribution = json_log("distribution") or {}

    lines = [
        f"# Refaat {__version__} release status",
        "",
        f"**Decision: {decision}.** Generated on {date.today().isoformat()} from recorded gate evidence.",
        "",
        "## Verified locally",
        "",
        f"- Pytest: {pytest_summary()}.",
        f"- Branch-aware coverage: {coverage_summary()}.",
        f"- Executable product examples: {examples.get('passed', 'not available')}.",
        f"- Wheel files: {distribution.get('wheel_files', 'not available')}.",
        f"- Source-distribution files: {distribution.get('sdist_files', 'not available')}.",
        "- OpenAPI, documentation links/navigation, migration, package content, and wheel smoke are required gates.",
        "",
        "## Gate matrix",
        "",
        "| Gate | Required locally | Exit code | Result |",
        "|---|---|---:|---|",
    ]
    for name, code in status.items():
        result = "PASS" if code == 0 else ("NOT AVAILABLE" if code == 125 else "FAIL")
        required = "yes" if name in REQUIRED_GATES else "no"
        lines.append(f"| `{name}` | {required} | {code} | {result} |")

    lines.extend(
        [
            "",
            "## Environment",
            "",
            f"- Python: `{environment['python']}`",
            f"- Platform: `{environment['platform']}`",
            f"- uv: `{environment['uv']}`",
            f"- Docker: `{environment['docker']}`",
            f"- Helm: `{environment['helm']}`",
        ]
    )

    if unavailable:
        lines.extend(["", "## Not available on this host", ""])
        for name in unavailable:
            lines.append(f"- `{name}`: {unavailable_reason(name)}")

    if failed or missing:
        lines.extend(["", "## Blocking issues", ""])
        lines.extend(f"- Failed gate: `{name}`" for name in failed)
        lines.extend(f"- Missing required gate: `{name}`" for name in missing)

    lines.extend(
        [
            "",
            "## Verification boundary",
            "",
            (
                "This status proves the recorded checks on the environment above. It does not certify "
                "customer infrastructure, external model providers, production load, restore objectives, "
                "or regulatory compliance. Host-dependent checks marked unavailable must pass in CI or "
                "the target release environment before production promotion."
            ),
            "",
            "## Reproduce",
            "",
            "```bash",
            "./scripts/quality.sh",
            "```",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    REPORT.write_text(render(), encoding="utf-8")
    print(f"Wrote {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
