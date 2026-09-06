# Authoring a plugin

This is the single reference for extending Refaat. Every replaceable part of the runtime is a typed Protocol port declared in `refaat.ports`. You supply an implementation and register it in one of two ways: directly on `RuntimeBuilder` for application-owned code, or through a Python entry point for a separately packaged plugin. Model vendors and agent frameworks are adapters over these ports, so the dependency flows into the runtime rather than the other way around.

The path is the same for every extension:

1. implement a typed port;
2. register it (builder injection point or entry point);
3. allowlist the entry point (packaged plugins only);
4. pin it (recommended in production);
5. run the conformance check before you trust it.

## The extension model

Three surfaces make the runtime pluggable:

- The typed Protocol ports in `refaat.ports`. They cover embedders, generators, graph extractors, rerankers, cross-encoder scorers, query transforms, chunking strategies, document extractors, connectors, blob stores, search indexes, event sinks, skills, judge models, summarizers, image captioners, fine-tuning providers, secret providers, and code sandboxes, among others. The required members for the common ports are tabulated in [Custom connectors, extractors, skills, and event sinks](custom-extensions.md).
- Over thirty `with_*` injection points on `RuntimeBuilder`. Each one accepts an explicit instance and overrides the shipped default for that port.
- Five entry-point plugin groups for separately packaged distributions: `refaat.skills`, `refaat.extractors`, `refaat.connectors`, `refaat.event_sinks`, and `refaat.chunkers`.

A builder injection point and an entry point resolve to the same port. Use the builder when your code lives in the application, and an entry point when the extension ships as its own reviewed distribution.

## Step 1: implement a port

Pick the port for the behavior you are adding and implement its members. A chunking strategy is representative:

```python
from collections.abc import Sequence

from refaat.config import Settings
from refaat.core.chunking import TextChunk
from refaat.core.text import estimate_tokens, normalize_text
from refaat.domain.models import JsonObject


class FixedWindowStrategy:
    strategy_id = "fixed-window"

    def __init__(self, window_chars: int = 800) -> None:
        self._window = window_chars

    def chunk(self, text: str, *, metadata: JsonObject, settings: Settings) -> Sequence[TextChunk]:
        body = normalize_text(text)
        pieces = [body[i : i + self._window] for i in range(0, len(body), self._window)] or [body]
        return [
            TextChunk(ordinal=i, text=piece, token_count=estimate_tokens(piece))
            for i, piece in enumerate(pieces)
            if piece.strip()
        ]
```

Each port defines an output contract the pipeline relies on. For chunking, ordinals are contiguous from zero, text is non-empty, `token_count` is positive, and the pipeline mints every chunk id and checksum so a strategy can never forge an id. The full contract for the chunking port, including hierarchical chunks and how contextual embedding, HyPE, and RAPTOR compose on any conforming strategy, is in [Custom chunking strategies](custom-chunking.md).

## Step 2: register it

### Application-owned: the builder

Inject the instance through the matching `with_*` method:

```python
from refaat.application.runtime import RuntimeBuilder

runtime = (
    RuntimeBuilder(settings)
    .with_connector(MyConnector())
    .with_extractor(MyExtractor())
    .with_skill(MySkill())
    .with_event_sink(MyEventSink())
    .with_chunking_strategy(FixedWindowStrategy())
    .build()
)
```

A builder-injected extension takes precedence over any environment selection. Registries reject duplicate connector schemes, media types, and skill names, and `RuntimeBuilder` closes already-created resources when composition fails.

### Packaged: an entry point

A plugin distribution publishes its extensions under the group that matches the port:

```toml
[project.entry-points."refaat.connectors"]
knowledge = "acme_refaat.connectors:KnowledgeConnector"

[project.entry-points."refaat.extractors"]
runbook = "acme_refaat.extractors:RunbookExtractor"

[project.entry-points."refaat.skills"]
release_guard = "acme_refaat.skills:ReleaseGuardSkill"

[project.entry-points."refaat.event_sinks"]
audit = "acme_refaat.events:AuditSink"

[project.entry-points."refaat.chunkers"]
legal_clause = "acme_refaat.chunking:LegalClauseStrategy"
```

A chunking strategy shipped this way is selected by name with `REFAAT_CHUNKING_STRATEGY`. The other groups are discovered and registered automatically once allowlisted.

## Step 3: allowlist the entry point

Discovery is closed by default. `REFAAT_PLUGIN_ALLOWLIST` holds the approved entry-point names, and an empty allowlist disables all automatic third-party loading. Only names on the allowlist are ever imported:

```bash
export REFAAT_CHUNKING_STRATEGY=legal_clause
export REFAAT_PLUGIN_ALLOWLIST=legal_clause,knowledge,runbook,release_guard,audit
```

Applications may still register reviewed local objects through the builder regardless of the allowlist. The allowlist governs entry-point discovery, not direct injection.

## Step 4: pin it

An allowlisted name is not an identity: any installed distribution can publish an entry point under that name. Plugin pinning binds each allowlisted entry point to an exact target module, distribution, version, and a hash of the shipped source, so a discovered entry point loads only when it matches the pin:

```python
runtime = (
    RuntimeBuilder(settings)
    .with_plugin_pins(
        {
            "refaat.chunkers:legal_clause": {
                "value": "acme_refaat.chunking:LegalClauseStrategy",
                "dist_name": "acme-refaat",
                "dist_version": "2.3.1",
                "code_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            }
        }
    )
    .build()
)
```

The same pins can come from `REFAAT_PLUGIN_PINS`. Pinning is opt-in and byte-identical when no pins are set. When any pin is set, every allowlisted entry point in a discovered group must be pinned or the runtime refuses to load it. The full pin format, the ordered checks, and the honest scope of the guarantee are in [Plugin pinning](../plugin-pinning.md).

## Scaffold it

`refaat plugin new <Port> <name>` writes an installable project for you: a typed implementation stub with the port's members, the correct entry-point block (or a builder injection snippet), a conformance test, and a README with the allowlist and pin steps. Run `refaat plugin ports` for the port names and `refaat plugin describe <Port>` for a port's contract.

```bash
refaat plugin new ChunkingStrategy acme_chunker --out ./plugins
```

## Step 5: run the conformance check

Structural conformance (every member the port declares is present) is the universal check. Run it on your adapter:

```bash
refaat plugin verify acme_chunker:AcmeChunker --port ChunkingStrategy
```

Beyond structure, assert the invariants its port depends on. For a chunking strategy that means contiguous ordinals, non-empty text, positive token counts, valid parent spans for hierarchical chunks, and determinism across two runs:

```python
def assert_conforms(strategy, settings, fixtures) -> None:
    assert strategy.strategy_id
    for text in fixtures:
        chunks = list(strategy.chunk(text, metadata={}, settings=settings))
        assert chunks
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))
        for chunk in chunks:
            assert chunk.text.strip()
            assert chunk.token_count > 0
        again = list(strategy.chunk(text, metadata={}, settings=settings))
        assert [c.text for c in chunks] == [c.text for c in again]
```

A misconfigured extension that cannot be resolved fails startup rather than silently falling back, so a broken selection never quietly changes how a corpus is indexed or served.

## Trust boundary

Plugins and skills are ordinary Python packages. They execute with the API or worker process privileges and are not a sandbox for arbitrary model-generated code. Review dependencies, network behavior, filesystem access, and secrets before installation, and run untrusted or customer-supplied extensions in a separate sandboxed service with constrained network, filesystem, identity, and resource access.

## See also

- [Custom connectors, extractors, skills, and event sinks](custom-extensions.md) for the port member tables and durable event delivery.
- [Custom chunking strategies](custom-chunking.md) for the chunking port contract and composition.
- [Plugins and executable skills](../plugins-and-skills.md) for the skill creator and the trust boundary.
- [Plugin pinning](../plugin-pinning.md) for the pin format and the ordered load checks.
- [Configuration reference](../configuration.md) for `REFAAT_PLUGIN_ALLOWLIST`, `REFAAT_PLUGIN_PINS`, and `REFAAT_CHUNKING_STRATEGY`.
