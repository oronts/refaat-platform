# Supply chain and SBOM

The build commits a software bill of materials and enforces a dependency policy, so the dependency tree is auditable and cannot drift or float unverified.

## Bill of materials

`docs/reference/sbom.json` is a deterministic projection of the uv lockfile: every resolved package with its version, its source registry, and the sha256 hashes of its distributions, sorted by name and version. The `sbom` gate regenerates it in memory from `uv.lock` and byte-compares it to the committed file, so the committed bill of materials can never diverge from the lockfile.

Regenerate it whenever `uv.lock` changes:

```
python scripts/generate_sbom.py
```

Then regenerate the source manifest so the committed file is re-hashed.

## Dependency policy

The `dependency_policy` gate enforces two invariants, statically and offline:

- Every declared dependency, in the core set and in each optional extra, carries both a lower and an upper version bound. A library must not float unbounded above a major version, so a bare name or a `>=` with no `<` ceiling is denied. An `==` pin or a `~=` compatible-release constraint counts as bounded.
- The lockfile exists and every package resolved from a registry is hash-locked, so a reproducible install cannot silently pull an unverified artifact.

Both gates are pure standard-library static checks with no network, and both are fail-closed: a stale bill of materials, an unbounded dependency, a missing lockfile, or an unverified package fails the build.

## Provenance

`docs/reference/provenance.json` binds the release subject (the package name and version) to the exact dependency tree it was resolved against, by recording the sha256 of the committed bill of materials, along with the required Python version. It is a deterministic projection of committed inputs, so the `provenance` gate drift-checks it the same way the SBOM is checked. It deliberately does not record the built wheel or sdist hashes, which are not deterministic across builds and are added at release time by the signing step, nor the source manifest digest, which would be circular. Regenerate it when the SBOM changes:

```
python scripts/generate_provenance.py
```

## Signing

Signing is produced out of band at release time by a release owner on a trusted machine, never in CI, so no signing key enters the build. `scripts/sign_release.sh` signs the built distributions in `dist` plus the committed bill of materials and provenance, writing a detached signature next to each file. The signer is configurable through `REFAAT_SIGN_CMD` (gpg by default, or a keyless signer such as cosign). The committed provenance already binds the version to its dependency tree; signing binds those and the built artifacts to the owner's key.
