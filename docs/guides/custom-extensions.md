# Custom connectors, extractors, skills, and event sinks

Refaat supports direct programmatic registration for application-owned components and Python entry points for separately packaged plugins.

## Typed extension ports

| Extension | Required members | Purpose |
|---|---|---|
| `Connector` | `scheme`, `fetch(source)` | Load bytes, media type, and source metadata |
| `DocumentExtractor` | `media_types`, `extract(content, metadata=...)` | Convert bytes into text and metadata |
| `Skill` | `name`, `description`, `run(context, arguments)` | Execute reviewed Python behavior over a context packet |
| `EventSink` | `emit(topic, payload)` | Deliver durable outbox events |
| `Embedder` | `model_id`, `dimensions`, `embed(texts)` | Produce vectors |
| `Generator` | `model_id`, `generate(request)` | Produce a grounded result with cited chunk IDs |
| `GraphExtractor` | `extract(...)` | Produce entities and relations |
| `SearchIndex` | lexical/vector replace, delete, and query methods | Replace retrieval storage |
| `ChunkingStrategy` | `strategy_id`, `chunk(text, *, metadata, settings)` | Split a document into ordered chunks |

## Programmatic registration

```python
runtime = (
    RuntimeBuilder(settings)
    .with_connector(MyConnector())
    .with_extractor(MyExtractor())
    .with_skill(MySkill())
    .with_event_sink(MyEventSink())
    .with_chunking_strategy(MyChunkingStrategy())
    .build()
)
```

Registries reject duplicate connector schemes, media types, and skill names. `RuntimeBuilder` closes already-created resources when composition fails.

## Entry-point groups

A plugin package can publish:

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

`REFAAT_PLUGIN_ALLOWLIST` can restrict discovery to approved entry-point names. An empty allowlist disables automatic third-party discovery. Production operators should install only reviewed packages and list each approved entry-point name explicitly.

## Durable event delivery

Ingestion writes an outbox event in the same relational operation as document state. The worker sends each pending event to every configured sink. An event is marked delivered only after all sinks succeed. Failures are recorded with exponential retry delay.

Event sinks must be idempotent because delivery is at least once. Use the outbox event ID or a destination-specific idempotency key when the receiver supports it.

## Security boundary

Plugins and skills are trusted application code. They execute with API or worker process privileges and are not a sandbox for arbitrary model-generated Python. Review dependencies, network behavior, filesystem access, and secrets before installation.

Run `examples/05_programmatic_extensions.py` for a complete working implementation. For the end-to-end plugin flow (implement a port, register it, allowlist, pin, and run the conformance check), see [Authoring a plugin](authoring-a-plugin.md). For a full walkthrough of the chunking port, including a domain splitter, a plugin package, and the conformance contract, see [Custom chunking strategies](custom-chunking.md) and `examples/29_custom_chunking_strategy.py`.
