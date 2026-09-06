# API compatibility contract

The HTTP surface is a committed contract. A frozen baseline of the OpenAPI document is stored next to the live schema, and a gate classifies every change to the surface as either compatible or breaking. Breaking changes fail the gate until the baseline is deliberately re-pinned in a release commit. Compatible evolution passes without a re-pin, so the contract does not block additive growth.

## The two artifacts

- [`openapi.json`](openapi.json) tracks the current surface. The `openapi` gate proves it is byte-identical to the live `create_app().openapi()`, and the `version` gate pins its `info.version` to the package version.
- `openapi.baseline.json` is the frozen reference. The `api_compat` gate diffs the current surface against it.

Because the `openapi` gate already guarantees `openapi.json` equals the running application, the compatibility gate compares two committed JSON files with the standard library only. It imports no runtime module and builds no application, so it cannot perturb the runtime and is immune to library-version drift on the host. Both gates are required, so a corrupted `openapi.json` fails the `openapi` gate independently. A hand edit of the baseline changes its checksum and trips the `source_manifest` gate.

## What counts as breaking

A change is breaking when it can break an existing client:

- an operation removed (a path or a method dropped)
- a response status removed, or its media type removed or repointed to a different schema
- a new required request parameter, or an existing parameter made required
- a request parameter removed
- a request parameter whose type changed, or an enum value removed from it
- a request body made required, or its media type removed or repointed
- a schema removed from `components.schemas`
- a property removed from a schema
- a property added to a schema's `required` list
- a property or parameter whose JSON type, or whose schema reference (`$ref`) target, changed
- an enum value removed from a property, a parameter, or an enum schema
- an operation's effective security requirement changed, including the document-level requirement every operation inherits
- a security scheme in `components.securitySchemes` removed, or its definition changed (the mechanism, the api-key name or location, the bearer format)

## What counts as compatible

A change is compatible when existing clients keep working:

- a new operation, path, or response status
- a new optional parameter
- a new response or schema property
- a new enum value
- a documentation-only change (descriptions, summaries, examples, and titles are ignored)

## Fail-closed behavior

The gate exits non-zero on any breaking change and on any error: a missing or malformed baseline, a baseline whose version is ahead of the current surface, or an identical surface whose baseline version no longer matches. It never reports a pass on an error. It is pure and deterministic, so it produces the same result on every run offline.

## Re-pinning after an accepted breaking change

Landing a breaking change is one ordered act, run at release time. The order matters: re-pinning the baseline before regenerating the schema would capture a stale version and misclassify the next run.

1. Change the surface.
2. Bump the package version in `src/refaat/version.py` and the other version surfaces the `version` gate enforces.
3. Regenerate the schema: `PYTHONPATH=src python scripts/generate_openapi.py`.
4. Re-pin the baseline: `python scripts/api_compat_check.py --write`.
5. Regenerate the source manifest: `python scripts/source_manifest.py --write`.

After the re-pin the baseline equals the current surface, so the diff is empty and the gate is green. The `version` gate independently proves the version moved. A stale baseline can only surface more differences, never fewer, so the fail-closed guarantee holds.

## Scope

The contract covers the HTTP and OpenAPI surface. URL-path versioning, deprecation headers, and the Python SDK surface are out of scope for this gate and are governed separately.

The gate classifies on the OpenAPI type of each field. It does not distinguish a request schema from a response schema, so it treats a property loosened from required to optional as compatible and a request-side property or enum removal conservatively as breaking. Narrowing that keeps the JSON type but tightens a `format`, `pattern`, or numeric bound is not classified. These are conservative choices: the gate never passes a change the rules above mark as breaking, and a case it does not model surfaces through the exact-drift `openapi` gate and the release-time re-pin.
