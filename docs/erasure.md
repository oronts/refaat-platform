# Right to erasure

Complete deletion physically removes or re-derives every auto-derived evidence surface a document produced, so no residue of its content survives in a queryable or model-reachable auto-derived surface. With the default configuration (the model-call cache disabled, local blob storage) this is exhaustive for those surfaces, including a document's superseded revisions in the summary-tree surface and in every session turn a current build wrote with the stable cited document id. The bounded items, all noted under Limitations, are the superseded content-addressed blob (reclaimed by the deferred blob.gc sweep rather than in this operation, see the deferred superseded-blob reclamation limitation for the concurrent-overwrite-race rationale and the tracked per-document-lineage enhancement), the opt-in model-call cache embedding orphans, versioned object storage, and the upgrade-time legacy session turn that predates the cited document id field; the operator-review durable records listed there are out of scope by design. It is a separate operation from the ordinary [ingestion delete](ingestion.md); erasure is additive and never changes that path, and the ordinary delete is otherwise behavior-preserving except that it refuses a document under a legal hold (see [legal holds](legal-holds.md)).

Physical deletion is the safe default here because the store is plaintext at rest. Cryptographic erasure through per-document keys is a separate at-rest-encryption concern and is out of scope for this operation.

## The guarantee

For one document, erasure removes or re-derives all of these auto-derived surfaces:

| Surface | What happens |
|---|---|
| Chunks | Deleted (the document row cascades to its chunks) |
| Vectors | Deleted from the search index |
| Graph entities and relations | Deleted (cascade) |
| Hypothetical questions | Deleted (cascade) |
| RAPTOR summary tree | Rebuilt over the surviving chunks; when RAPTOR has since been disabled and no builder is available, the whole corpus tree is purged instead, so no summary of the document remains either way |
| Answer cache | The corpus's cached answers are physically deleted |
| Generation model-call cache | Purged workspace-wide (complete, accepts a cold cache) |
| Embedding model-call cache | Opt-in and off by default. When enabled, the rows for the exact current input the ingester embedded are deleted: each chunk's embedded text (under contextual retrieval the contextualized title, heading, and body, otherwise the raw chunk text), plus the current hypothetical questions and current summary texts, keyed under the current embedder identity. The superseded-derivation and identity-drift orphans (see Limitations) are the bounded exceptions |
| Session summaries | Every session summary in the corpus is cleared (no per-document summary linkage exists) |
| Session turns | Every turn that cites the document is deleted, matched by the stable cited document id so a turn from a superseded revision is caught even after a re-ingest rewrote the cited chunk ids |
| Learned preferences | The rows keyed to the document are deleted |
| Blob | The current stored bytes are deleted and their removal is verified. A superseded content-addressed blob that a prior replacement left is reclaimed by the deferred blob.gc sweep, not by this operation (see the deferred superseded-blob reclamation limitation, which carries the concurrent-overwrite-race rationale and the tracked per-document-lineage enhancement) |
| Outbox | Events referencing the document id are deleted |

The operation returns an `ErasureReceipt` that carries only ids, per-surface counts, a status, and a timestamp. It never carries a chunk text, a hypothetical question, a summary, an answer, or any other content, so it is safe to persist and to derive an audit event from. Each request is recorded in an append-only `erasure_ledger` row, and a redacted `document.erased` audit event is emitted when an audit sink is installed.

Frame the completeness claim honestly: it is scoped to the auto-derived evidence surfaces above. The surfaces listed under Limitations are not auto-erased.

## Using it

Python:

```python
receipt = runtime.project(workspace="acme", corpus="policies").erase(document_id)
assert receipt.status is ErasureStatus.ERASED
```

CLI:

```bash
refaat document erase <workspace_id> <corpus_id> <document_id>
```

HTTP (admin scope):

```
POST /v1/corpora/{corpus_id}/documents/{document_id}/erase
```

SDK:

```python
receipt = client.erase_document(corpus_id, document_id)
```

## Semantics

Erasure is idempotent and fail-closed:

- Idempotent: a re-run over an already-erased document converges on a clean `ERASED` receipt because every delete is a no-op. An absent document is treated as success.
- Fail-closed on the blob: the raw bytes are deleted and their removal is verified. Unlike the ordinary delete, which logs and swallows a blob-cleanup failure, erasure surfaces it, records a `FAILED` ledger row, and raises so the operator learns the erasure did not fully complete.
- Legal hold: a document that an active hold covers is refused with a `HELD` receipt (HTTP 200) and nothing is erased. Enforcement is always on and state-driven: every erase consults the `legal_holds` table, so a hold is honored whether or not this runtime wired the management surface, and erasure is permitted only when no hold covers the document. The erase holds a corpus-level destroy lock across its multiple transactions and rechecks the hold after acquiring it, so a hold that commits before the erase takes the lock is seen and refuses, while a placement attempted after the erase owns the lock blocks until the erase finishes. The requested document id is validated to an id shape before the hold check, so free text cannot reach the `HELD` receipt or an audit event. Full detail, including the deliberate two-shape refusal contract (erase returns a `HELD` receipt with HTTP 200, while delete and a content overwrite raise `LegalHoldError` with HTTP 409), is in [legal holds](legal-holds.md).
- Authoritative record: the durable `erasure_ledger` row is the authoritative, redacted record of every erasure (`ERASED`, `FAILED`, or `HELD`). The external audit event is best-effort for a completed erasure, because the physical deletion is irreversible and already recorded, so an audit sink outage never leaves the erasure unrecorded (the ledger holds it) and never raises after a completed erasure.

## Out of scope and limitations

1. Operator-curated and durable-record surfaces have no automatic document linkage and are flagged for operator review, not auto-erased: `harness_snapshots.evidence`, `optimization_runs.dataset_cases`, `evaluation_suites.cases`, `fine_tune_runs.provenance`, `evaluation_runs.results`, `jobs.result`, `program_shadow_runs.comparison`, `agents.state`, `agent_messages.payload`, and `program_artifacts.document`. The last of these holds bootstrap-derived demonstrations distilled from grounded answers, and a program artifact is immutable and content-addressed, so erasing one in place would break the registry and every pinned id; retire the affected artifact through the registry instead. Review and remove these yourself when a document's content may have been copied into them.
2. Cryptographic erasure through per-document keys is a separate future at-rest-encryption slice. This operation is a physical delete.
3. Concurrency window: an evidence job (RLM, evaluation, or optimization) that read the document before the erasure ran may briefly re-materialize its answer into a durable result. Quiesce evidence jobs for the corpus before erasing. A fencing follow-up is planned.
4. Versioned object storage: on an S3 bucket with object versioning enabled, a delete places a delete marker and retains the prior non-current versions, which the removal check cannot see. Erasure deletes and verifies the current object, but retained versions need a bucket lifecycle rule that expires non-current versions, or version-aware deletion configured at the store. This is a deployment configuration concern, not something the application can enforce from a single delete.
5. Superseded blob reclamation is deferred, not synchronous. When a document is replaced, the prior content-addressed blob is left for the deferred blob.gc sweep, which reclaims a blob no live document references after a grace window and re-checks that reference immediately before deleting. Erasure deletes and verifies the current blob synchronously and fail-closed, but a superseded blob from an earlier revision is reclaimed by that sweep, so the worker must be running for it to be removed. Erasure does not reclaim it inline because a per-erasure global sweep with no grace window can delete a concurrent ingest's not-yet-committed blob. A synchronous, per-document superseded-blob reclaim would need a superseded-blob lineage record written at replace time so exactly those blobs can be deleted fail-closed without a global scan; that is a tracked enhancement, not built.
6. Legal hold refuses erasure whenever an active hold covers the document, so a held document is never partially erased.
7. Superseded-derivation embedding orphan, only when the opt-in model-call cache is enabled. The embedding cache is a global, content-addressed, shared store keyed by the embedder identity and the embedded text, with no per-document lineage. Erasure purges the rows for the document's current chunk, hypothetical, and summary texts, but a text derived from the document in an earlier RAPTOR tree or before a `replace_document`, and since superseded, leaves an inert cache row keyed by a text that no longer exists anywhere. Such a row is never read again (nothing embeds that vanished text), so it is not queryable or model-reachable, but it is a lossy vector derived from the content. The same applies when the embedder backend identity changed between ingest and erasure: the key folds that identity, so erasure recomputing keys under the current identity leaves the rows written under the prior identity, even for a current text. Because the cache is global and lineage-free, targeted deletion cannot reach a key whose source text or embedder identity is gone; remove these with the cache TTL or a periodic embedding-cache sweep. With the model-call cache disabled (the default), no embedding cache exists and these exceptions do not arise.
8. Legacy-format session turn from a build predating the `cited_document_ids` field. A turn a pre-upgrade build persisted carrying only `citations` (the volatile chunk ids, no stable cited document id), that cites a revision a later re-ingest has since superseded, cannot be auto-matched at erasure: its recorded chunk ids no longer resolve and it has no stable cited document id, so the superseded chunk ids are gone and there is nothing to map the turn back to the document. This is a bounded upgrade-time limitation only, a clean install writes `cited_document_ids` on every turn, so it never arises going forward. Flag such turns for operator review.
