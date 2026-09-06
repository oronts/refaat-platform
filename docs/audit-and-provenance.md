# Audit and provenance

Refaat emits a redacted audit event for every evidence-serving path when authorization is active and an audit sink is installed. The event carries ids, counts, the policy, and timestamps, never a raw query, evidence text, or a secret. A deployment routes events to its own append-only store, SIEM, or signed ledger. For the enforcement model, see [Authorization](authorization.md).

## The event contract

`AuditEvent` is redacted by construction:

| Field | Meaning |
|---|---|
| `event_type` | one redacted event per governed action: `query.authorized`, `rlm.authorized`, `approval.granted`, `approval.consumed`, `approval.cancelled`, `hold.placed`, `hold.lifted`, `hold.enforced`, `document.erased`, `erasure.deferred`, `killswitch.engaged`, `killswitch.disengaged`, `quota.denied`, `degradation.decided`, `identity.refused`, `identity.reresolved`, and `output.denied` |
| `schema_version` | Contract version, currently `2` |
| `occurred_at` | Event timestamp from the runtime clock |
| `workspace_id`, `corpus_id`, `corpus_version` | The scope served |
| `principal_id`, `principal_kind`, `tenant` | Who was served |
| `policy_id`, `policy_version` | Which policy decided |
| `cited_document_ids`, `citation_count` | What was cited, by id and count |
| `abstained`, `cached` | Whether the answer abstained or came from cache |
| `trace_id` | Correlation id for tracing |
| `capability_name` | The governed tool capability, on `approval.consumed`, else null |
| `action_fingerprint` | One-way digest of the approved action, on `approval.consumed`, else null |
| `dedup_id` | Stable digest for at-most-once handling downstream |

The event never contains the query text, the answer text, evidence text, or any ACL internals. It is safe to ship to an external system.

On `approval.consumed` the `dedup_id` folds the consumed single-use approval ids, so two consumptions of the same action by the same principal are distinct records and only a redelivery of one consumption collides.

## The sink port

```python
class AuditSink(Protocol):
    def record(self, event: AuditEvent) -> None: ...
```

Install it on the builder:

```python
runtime = (
    RuntimeBuilder(settings)
    .with_access_policy(CanonicalAccessPolicy())
    .with_principal_resolver(HeaderPrincipalResolver())
    .with_audit_sink(MyAuditSink())
    .build()
)
```

## When events are emitted

An event is recorded once per served query on the query and streaming paths, once per authorized RLM run, and once per approval lifecycle action (`approval.granted`, `approval.consumed`, `approval.cancelled`), but only when an audit sink is installed and, for the evidence paths, a policy and principal are active. With no sink, or with no access policy, nothing is emitted and the runtime is byte-identical; a principal-less request under an active policy is denied by strict authorization before an event would be emitted.

Emission is best-effort by default: a sink failure is logged and swallowed so a read query still serves. Setting `audit_durable` turns on durable at-least-once delivery: each emitted event is persisted to an `audit_outbox` table and a worker delivers it to the configured sink with a bounded retry backoff and a per-sink ledger, so a transient delivery failure is retried rather than lost and a redelivery never doubles a sink that already received the event. With `audit_durable` off, delivery is the direct best-effort call and the runtime is byte-identical. When more than one delivery sink is configured, each must expose a distinct `sink_name` so the ledger mapping is stable across a restart.

Setting `audit_fail_closed` turns the default best-effort emission into a fail-closed one for the synchronous pre-return paths: when the sink (or, under `audit_durable`, the outbox insert) fails, the query, RLM run, or approval grant, consume, or cancel raises `AuditUnavailableError` (HTTP 503) rather than proceeding unaudited. Under an active policy the streaming query path buffers when `audit_fail_closed` is on, so its terminal audit fires before any answer byte and a sink outage returns a 503 with no answer delta rather than serving unaudited. Enabling `audit_fail_closed` together with `audit_durable` gives the strict never-lost-before-enqueue guarantee: the operation cannot complete unless the event is durably enqueued, and the worker then delivers it at least once. With `audit_fail_closed` off, the runtime is byte-identical.

Because fail-closed audit is meaningless with nowhere to send events, a runtime that sets `audit_fail_closed` but installs no sink refuses to build (a configuration error), regardless of whether an access policy or plugin pins are present, rather than silently running any governed operation (a query, an RLM run, an approval, or a legal-hold write) unaudited. On the approval lifecycle the policy holds end to end: a grant whose audit cannot be recorded cancels its own just-created approval so no usable but unaudited approval survives, and a cancel is always audited (a system attribution stands in when no canceller identity resolved) so an admin-key cancel with no resolver still fails closed.

## Provenance of the decision

The audit event names the deciding `policy_id` and `policy_version`, and the `AccessDecision` inside the runtime carries a machine-readable reason. Together with `corpus_version`, which advances on any content or ACL change, this gives a reconstructable record of which policy version served which corpus version to which principal, without exposing the evidence itself.

## Related pages

- [Authorization](authorization.md)
- [Identity lifecycle](identity-lifecycle.md)
- [Observability](operations.md)
