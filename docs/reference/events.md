# Events and outbox delivery

Refaat writes durable outbox records for state changes that need external delivery. Event sinks are replaceable through `RuntimeBuilder.with_event_sink(...)` or `refaat.event_sinks` entry points.

## Built-in topics

| Topic | Produced by | Payload |
|---|---|---|
| `document.indexed` | successful document replacement or creation | `workspace_id`, `document_id`, `corpus_id`, `corpus_version` |
| `program.shadow_compared` | shadow or canary comparison of a deployment run | `workspace_id`, `program`, `environment`, `shadow_run_id`, `served`, `candidate_artifact_id` |

The event is created as part of the relational document update. Original-blob and search-index work is compensated when ingestion fails before commit.

## Delivery semantics

The worker's `OutboxPublisher`:

1. reads pending events;
2. invokes every configured sink;
3. marks the event delivered only when every sink succeeds;
4. records a failed attempt and an exponential delay when any sink raises.

Delivery is at least once. A sink can receive a duplicate after a crash between external delivery and the database acknowledgement. Destination handlers must therefore be idempotent.

## Custom event sink

```python
from refaat.domain.models import JsonObject


class AuditSink:
    def emit(self, topic: str, payload: JsonObject) -> None:
        audit_client.write(topic=topic, payload=payload)


runtime = RuntimeBuilder(settings).with_event_sink(AuditSink()).build()
```

A sink may implement `close()`; the runtime closes registered sink instances when it owns them.

## Webhook sink

`WebhookEventSink` posts:

```json
{
  "topic": "document.indexed",
  "payload": {
    "document_id": "doc_...",
    "corpus_id": "corpus_...",
    "corpus_version": 7
  }
}
```

It uses a bearer secret and validates the destination URL during construction. Production still requires network egress restrictions, TLS validation policy, receiver authentication, rotation, and replay protection.

## Adding topics

New domain events should be written in the same transaction as the state they represent, use stable topic names, contain workspace-safe identifiers, and avoid secrets or full document content. Add unit tests for retry behavior and integration tests that prove emission from the producing workflow.
