# Plugins and executable skills

## Typed extension points

Embedders, generators, graph extractors, document extractors, blob stores, search indexes, connectors, event sinks, chunking strategies, and
skills are typed protocols. `RuntimeBuilder` supplies defaults and accepts explicit overrides. Installed distributions
can register reviewed extensions through Python entry points.

Entry-point groups are:

- `refaat.skills`;
- `refaat.extractors`;
- `refaat.connectors`;
- `refaat.event_sinks`;
- `refaat.chunkers`.

For the full flow across any of these, see [Authoring a plugin](guides/authoring-a-plugin.md).

`REFAAT_PLUGIN_ALLOWLIST` contains allowed entry-point names. The runtime passes an explicit allowlist, so an empty
list disables automatic third-party loading. Applications may still register reviewed local objects explicitly.

## Skill creator

```bash
refaat skill create release-review "Review grounded release evidence"
```

The creator writes an importable project with `pyproject.toml`, a Refaat skill entry point, a typed implementation
skeleton, a metadata test, and a README. It does not edit or auto-install code into the Refaat package.

## Skill execution

A skill receives the canonical `ContextPacket` used by generation plus explicit JSON-compatible arguments. It can inspect
retrieved chunks and graph facts without reimplementing retrieval. The built-in `corpus_health` skill reports route
coverage and score composition.

## Trust boundary

Plugins and skills are normal Python packages and execute with process privileges. Review, pin, scan, test, and allowlist
them. Run untrusted or customer-supplied executable extensions in a separate sandboxed service with constrained network,
filesystem, identity, and resource access.


## Event sinks and direct registration

Applications can register connectors, extractors, skills, and event sinks directly through `RuntimeBuilder`. Separately packaged sinks use the `refaat.event_sinks` entry-point group. See [Custom extensions](guides/custom-extensions.md) and [Events](reference/events.md).

## Custom chunking strategies

The document chunker is a typed `ChunkingStrategy` port. The three built-in strategies (`flat`, `parent_child`, `sentence_window`) are the default and are byte-identical to the pre-port chunker. A domain splitter (legal clauses, tables, code, transcripts) plugs in exactly like a model adapter:

- inject an instance with `RuntimeBuilder.with_chunking_strategy(strategy)`, or
- ship it under the `refaat.chunkers` entry-point group and select it by name with `REFAAT_CHUNKING_STRATEGY`, gated by the same allowlist and code-hash pin as every other plugin group.

A strategy implements `strategy_id` and `chunk(text, *, metadata, settings)`, returning chunks with contiguous ordinals from zero, non-empty text, a positive `token_count`, and a populated `heading`. The pipeline mints every chunk id and checksum, so a strategy can never forge an id. Contextual embedding, HyPE, and RAPTOR compose on any conforming strategy without change. A configured strategy that cannot be resolved fails startup rather than silently falling back.
