# Legal holds

A legal hold is durable, first-class state that refuses every destructive path for a document while litigation or a regulatory duty requires the evidence to be preserved. It gates all three destructive operations, not just erasure: complete deletion, the ordinary ingestion delete, and a content overwrite of an existing document.

Enforcement of an existing hold is always on. It is state-driven, with no feature flag and no startup probe: every destructive path consults the `legal_holds` table on every attempt, so any runtime refuses to destroy a held document regardless of how the runtime was built. A runtime that never wired the management surface still enforces a hold another runtime placed, and a hold placed after a runtime started is enforced by that already-running runtime. When no hold row covers a document the destructive paths are the exact pre-change behavior, so the feature is byte-identical when no hold exists.

## Enabling the management surface

Placing, listing, and lifting holds is a separate management surface, opt-in at build time:

```python
runtime = RuntimeBuilder(settings).with_legal_hold_registry().build()
```

This enables the place, list, and lift write surface (the `project.legal_holds` accessor, the place and lift HTTP routes, and the CLI). It does not turn enforcement on or off: enforcement is unconditional. A runtime built without it still enforces every existing hold, it just cannot place or lift one.

When the management surface is not wired, the write path fails closed:

- `place` and `lift` raise `LegalHoldError` (HTTP 409), and the `project.legal_holds` accessor raises `ConfigurationError`. A hold row is therefore only ever written by a runtime that deliberately opted into managing holds.

## Placing, listing, and lifting

```python
project = runtime.project(workspace="acme", corpus="policies")
hold_id = project.legal_holds.place(
    corpus_id=corpus_id, document_id=doc_id, placed_by="counsel@acme.com", reason="matter 2026-014"
)
active = project.legal_holds.list_active(corpus_id)
project.legal_holds.lift(hold_id, lifted_by="counsel@acme.com")
```

`place` with no `document_id` records a corpus-wide hold. HTTP and CLI surfaces mirror these calls:

| Verb | HTTP | CLI |
|---|---|---|
| Place | `POST /v1/corpora/{corpus_id}/legal-holds` | `refaat legal-hold place <ws> <corpus> <placed_by>` |
| List | `GET /v1/corpora/{corpus_id}/legal-holds` | `refaat legal-hold list <ws>` |
| Lift | `POST /v1/legal-holds/{hold_id}/lift` | `refaat legal-hold lift <ws> <hold_id> <lifted_by>` |

On the HTTP surface every route is admin-scoped, and `placed_by` and `lifted_by` are the authenticated admin identity, never a request-body field, so a client cannot spoof the actor. The place and lift routes and CLI return HTTP 409 when the management surface is not wired.

## Coverage

Coverage is decided entirely in SQL. A document is held when an active row (`lifted_at IS NULL`) exists for its workspace and corpus whose `document_id` is NULL (a corpus-wide hold) or equals the document id. Liveness keys on `lifted_at`, not the human-readable `status` label, so a partially written row still reads as held and fails toward retention. Two overlapping holds both cover a document; lifting one still leaves it held until the last is lifted.

A brand-new document is never blocked, even under a corpus-wide hold, because ingesting a new document is not spoliation of held evidence. A same-checksum re-tag (an ACL or metadata change with identical bytes) is evidence-preserving and is allowed under a hold; only a genuine content overwrite of an existing, held document is refused.

## The two-shape refusal contract

The three destructive paths refuse in two deliberate shapes, driven by operation shape:

- Erase keeps its shipped complete-deletion contract: a held document returns an `ErasureReceipt` with status `HELD` and HTTP 200, and nothing is erased. The `HELD` row is written to the append-only `erasure_ledger` as the durable refusal trace.
- Delete and a content overwrite are void operations that refuse by raising `LegalHoldError`, which surfaces as HTTP 409.

This divergence is intentional. A uniform 409 would require translating the `HELD` receipt into an exception at the erase route, which would break the already-shipped, tested complete-deletion contract.

## Erasure conflict policy: refuse or defer

By default an erasure request that meets an active legal hold is refused: it returns a `HELD` receipt, erases nothing, and does not retain the request, so the data subject must re-submit after the hold is lifted. Setting `legal_hold_on_conflict` to `defer` changes only what happens on that conflict. A held erase is recorded as a durable pending request in `deferred_erasures` and returns a `DEFERRED` receipt, still erasing nothing. When the last hold covering the document is lifted, the lift runs the recorded request and the document is erased. One pending request is kept per document, so repeating a held erase does not duplicate it.

Execution re-checks the hold under the same corpus destroy lock the direct erase uses, so a document still covered by another hold is never erased and its request stays pending until that hold also lifts. Lifting a hold triggers execution best-effort: a processing error leaves the request pending and never changes the reported lift result. Because that lift-time execution is best-effort, a `deferred_erasure.sweep` worker job is the at-least-once backstop: it re-drives every pending request whose covering holds have all lifted, so a request stranded by a crash or a transient failure at lift time is still completed. Schedule it periodically alongside the other maintenance jobs. The deferral emits a redacted `erasure.deferred` audit event and records a `DEFERRED` `erasure_ledger` row, following the same contract as the other lifecycle events. The default `refuse` policy is byte-identical: no `deferred_erasures` rows are written, no sweep has anything to do, and lifting a hold is unchanged.

## Input bounds

Every id that can reach an audit event is bounded to an id shape before it is used, so client free text never enters a redacted audit trail. `place` validates the corpus id and the optional document id, and `erase` validates its requested document id before any hold check or audit emission. A free-text or control-character id is rejected with a `ValidationError`. This matters most under a corpus-wide hold, where any supplied id reads as held: without the bound, an unvalidated erase id would land in the `hold.enforced` event's cited ids. The human-readable hold reason is bounded for length but stays on the row only and never enters an audit event.

## Concurrency and the destroy lock

Complete deletion spans several transactions, so a naive check-then-destroy has a place-during-destroy race. The erase holds a corpus-level destroy lock across the whole multi-transaction operation and rechecks the hold after acquiring it, so a hold placed before acquisition is seen and a hold attempted during the destroy blocks until the erase finishes. The delete and overwrite chokepoints check the hold inside the same transaction as the write, under the same corpus lock, so they are atomic. Placing a hold takes the same corpus lock, so it serializes against an in-flight destroy.

On PostgreSQL the corpus lock is a session-level advisory lock on a connection drawn from a dedicated `NullPool` engine to the same database, not from the shared store connection pool. A same-corpus waiter parks on a `NullPool` connection, never on a shared-pool slot, so a place, ingest, or delete concurrent with an in-flight erase can never drain the shared pool or starve the erase's inner transactions. Those transient lock connections are bounded by a semaphore (`REFAAT_LEGAL_HOLD_LOCK_MAX_WAITERS`, default 64), so a large same-corpus waiter burst cannot exhaust the server's `max_connections` before parking on the advisory lock.

On SQLite the corpus lock is a process-wide reentrant lock keyed by the database and the corpus, so two `SQLStore` instances over one SQLite database in a single process serialize their destroyers. SQLite has no cross-process advisory lock, so a multi-process SQLite deployment cannot serialize destroyers across processes and is unsupported for enforceable holds. Use PostgreSQL for any multi-process deployment; single-process SQLite is fully enforceable.

## The refusal audit contract

Every refusal writes a durable, redacted trace. Erase records the `HELD` `erasure_ledger` row; delete and overwrite emit a `hold.enforced` audit event when an audit sink is installed. The event carries only ids, the fixed capability name (`document.delete`, `ingest.overwrite`, or `document.erase`), the policy, and timestamps. The human-readable hold reason lives on the `legal_holds` row only and never enters an audit event.

Placing a hold inserts the protective row first and emits `hold.placed` best-effort, so the row lands even under a degraded sink. Lifting a hold emits `hold.lifted` before removing the row: under `audit_fail_closed` a sink outage raises and freezes the hold in place rather than lifting it unaudited. All lifecycle events follow the same redacted contract, carrying only ids, the capability, and timestamps.

## Operations sizing (PostgreSQL)

On PostgreSQL every destructive path takes the corpus destroy lock, whether or not this runtime wired the management surface, because enforcement is state-driven. The transient `NullPool` lock connections it opens are additional to the store connection pool (up to 15 by default) and the pgvector pool. Budget PostgreSQL `max_connections` for the expected concurrent-destroy count on top of those two pools; the lock-connection burst is capped by `REFAAT_LEGAL_HOLD_LOCK_MAX_WAITERS`. The store connection pool itself needs no increase, because the lock no longer draws from it.
