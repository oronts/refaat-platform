# Release gate

A release is cut only when the full quality gate is green and the release evidence is current and signed. The gate is `scripts/quality.sh`; the steps below compose it with the version, supply-chain, and signing evidence into the production release gate.

## Gate criteria

A release candidate must satisfy all of the following.

- Every gate in `scripts/quality.sh` passes. The per-gate results are written to `.quality/current/status.json`, where 0 is a pass and 125 is a tool that is unavailable on the host; a required gate may never be 125 or non-zero. The required set spans dependencies, compilation, style, code and deployment audits, examples, OpenAPI drift and API compatibility, version consistency, the source manifest, secret scanning, tests and coverage, migration smoke and migration safety, the SBOM, provenance, the dependency policy, threat coverage, the vulnerability policy, the performance benchmark, documentation links, packaging, distribution, and the wheel smoke, plus ruff, mypy, mkdocs, pip-audit, Docker, Helm, and PostgreSQL where their tools and services are available.
- The version agrees across every surface the `version` gate checks: `pyproject.toml`, `src/refaat/version.py`, the committed OpenAPI document, the Helm chart version and appVersion, the README release line, and a matching CHANGELOG heading.
- The API compatibility baseline is current, or a deliberate breaking change has been accepted by re-pinning the baseline with a version bump, per [API compatibility](../reference/api-compatibility.md).
- The bill of materials and the provenance are current against the lockfile, per [Supply chain](../reference/supply-chain.md).

## Cutting a release

1. Land the change and run the full gate: `PYTHON_BIN=.venv/bin/python bash scripts/quality.sh`, then confirm `.quality/current/status.json` has no non-zero and no unexpected 125.
2. For a version change, bump the version across every surface the `version` gate checks, add a CHANGELOG heading, and regenerate the derived artifacts: the OpenAPI document, the SBOM, the provenance, and the source manifest, in that order.
3. Re-run the full gate on the bumped tree until green.
4. Build the distributions: `python -m build`.
5. Sign the distributions and the supply-chain evidence out of band with `scripts/sign_release.sh` on a trusted machine, per [Supply chain](../reference/supply-chain.md). No signing key ever enters CI.
6. Publish the distributions and their signatures, and record the release version in the version-control system the maintainer uses.

## Independent review and penetration test

Two release-gate activities are owner or external and are not part of the offline gate.

- An independent adversarial review of the change set, beyond the per-change review, before a production release. The security-critical governance work is built and reviewed adversarially per change, but a fresh independent pass on the assembled release is the standing discipline.
- A penetration test against a representative deployment, covering the authorization, tenancy, egress, and audit boundaries, before the first production release and after a material change to those boundaries. The threat model in [Threat model](../security/threat-model.md) is the scope input, and each threat there is already traced to a test by the `threat_coverage` gate; the penetration test validates the deployment, which the offline gate cannot reach.

Neither is a substitute for the other: the offline gate proves the code, the independent review proves the reasoning, and the penetration test proves the deployment.
