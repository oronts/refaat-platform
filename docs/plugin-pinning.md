# Plugin pinning

Refaat discovers optional extensions (skills, extractors, connectors, event sinks, chunking strategies) through Python entry points. The name allowlist (`REFAAT_PLUGIN_ALLOWLIST`) decides which entry-point names may load, but a name is not an identity: any installed distribution can publish an entry point under an allowlisted name, and the resolver would import whatever module that name points at. Plugin pinning closes that gap. It binds each allowlisted entry point to an exact target module, providing distribution, version, and a hash of the shipped source, so a discovered entry point loads only when it matches the pin.

Pinning is opt-in. With no pins configured, discovery is the name-only allowlist and is byte-identical to before: no distribution metadata is read, no file is hashed, and nothing is reordered. When any pin is set, pinning is enforced for every allowlisted entry point.

## What a pin contains

Pins live in configuration, keyed `group:name`, following the nested shape of `REFAAT_MODEL_PRICES`. Each pin records four fields:

| Field | Meaning |
|---|---|
| `value` | The exact `module:attribute` (or `module`) the entry point must resolve to. |
| `dist_name` | The distribution that must provide the entry point, compared under PEP 503 name normalization. |
| `dist_version` | The exact version of that distribution. |
| `code_hash` | The sha256 of a canonical manifest over the distribution's shipped source. |

```python
from refaat.application.runtime import RuntimeBuilder

runtime = (
    RuntimeBuilder()
    .with_plugin_pins(
        {
            "refaat.skills:release_review": {
                "value": "acme_release.skill:ReleaseReviewSkill",
                "dist_name": "acme-release",
                "dist_version": "2.3.1",
                "code_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            }
        }
    )
    .build()
)
```

The same pins can come from `REFAAT_PLUGIN_PINS` in the environment. A builder call overrides the settings value.

## The code hash

The `code_hash` is a sha256 over a canonical manifest, not over a single file. The manifest is the sorted list of `(posix relative path, byte length, sha256 of content)` for every file the distribution's RECORD lists, excluding the `.dist-info` directory. Every field is length-delimited, so a path and the content that follows it are bound unambiguously: no rename, split, or content edit produces the same manifest as another layout. To compute the pin for an installed entry point, run `refaat plugin pin <group>:<name>` (for example `refaat plugin pin refaat.skills:my_skill`). It reads the distribution's RECORD-listed source with the same function the runtime uses, without importing the plugin, and prints the exact `REFAAT_PLUGIN_PINS` fragment; it refuses to print a pin the runtime would then reject.

The comparison is constant-time (`hmac.compare_digest`), so a mismatch reveals nothing byte by byte.

## What is checked, and in what order

At the single load chokepoint, before any entry point in a group is imported, the resolver refuses a group in which a name appears on more than one distribution, because that duplicate is ambiguous under a pin and the standard resolver would silently keep whichever distribution scans first. Then, for each allowlisted entry point, the following run in order and the first drift is the reported reason. All of them are import-free; only the final step imports.

1. The entry point's `value` equals the pinned target.
2. The providing distribution is present, its PEP 503 normalized name equals the pinned name, and its version equals the pinned version.
3. The RECORD is present and lists the target module's source file. An importable `.py` that exists in the target's package directory but is absent from the RECORD is refused, so a stale or incomplete RECORD fails closed.
4. The computed `code_hash` equals the pinned hash.
5. The module the entry point will import resolves, without executing it, to a file inside the pinned distribution at every level of the dotted path. A module of the same name shadowing earlier on the path, a namespace package, or a non-file loader is refused.
6. Every level already present in `sys.modules` resolves to a file inside the pinned distribution, so a preloaded module cannot return cached, unvalidated code from the load.
7. Only when all pass does the entry point load.

When any pin is set, discovery also refuses at startup if an allowlisted entry point across the plugin groups has no pin, so a forgotten pin cannot fall back to the name-only path.

## Audit

When an audit sink is installed, a redacted `plugin.loaded` event is recorded on a pinned load and a `plugin.rejected` event on a refusal. Each event carries the `group:name`, a bounded reason code or the code-hash prefix, and the standard timestamps only. It never carries a file path, file bytes, or free text. Under `REFAAT_AUDIT_FAIL_CLOSED`, plugin pins require an installed sink, checked before discovery, so a pinned load or refusal is never unaudited.

## The honest guarantee

The guarantee plugin pinning provides is narrow and worth stating precisely: a pinned entry point never loads unpinned bytes. It is not the claim that no installed package can execute in this process. Several things are deliberately out of scope:

- Transitive imports across distributions. A pinned module can import a second distribution the pin does not cover. Pin the closure with a lockfile, and pin the deployment with a container image digest.
- Interpreter startup code and custom import hooks. `.pth` files and `sitecustomize` run before the runtime starts, and a custom `sys.meta_path` finder installed that way can transform a module's bytes at import. Origin binding resolves each module through the path-based file finder and requires a file origin inside the pinned distribution, which rejects a substitute file, but a finder that rewrites bytes in memory sits outside this seam because installing it already ran code at startup. Run from an install whose startup surface you control.
- Time-of-check to time-of-use. The hash is read from disk, then the module is imported; a concurrent mutation of `site-packages`, or a poisoned `__pycache__`, or an unrecorded source file written into a package tree between those steps, is not covered. All of these need write access to the install. Closing the read-to-import window fully would mean reading every byte once and executing from that copy, which for dotted packages and C extensions means reimplementing the importer, so it is out of scope. Run from an immutable or read-only install.

Plugin pinning is defense in depth over environment integrity. It complements, and does not replace, the image-digest pinning the shipped deployment already uses and the review, scanning, and sandboxing described in [Plugins and executable skills](plugins-and-skills.md).
