# Ingestion and multimodal extraction

## Pipeline

1. Validate workspace/corpus scope and content size.
2. Compute a content checksum and return immediately when source and checksum are already ready.
3. Persist the original through the blob-store port.
4. Extract canonical text and safe metadata.
5. Chunk under a configurable target and overlap.
6. Embed every chunk through one embedding port.
7. Extract a canonical entity/relation graph.
8. Replace document, chunks, entities, and relations transactionally; increment corpus version.
9. Update the vector index and emit a durable outbox event.

Source IDs are unique inside a corpus. Re-ingesting the same source replaces it; unchanged content is idempotent.

## Built-in extractors

Core installations support plain text, Markdown, JSON, CSV, XML/YAML text, HTML, PDF, and DOCX. Optional local media adapters expose Tesseract image OCR and faster-whisper audio/video transcription. Every extractor implements the same protocol and can be loaded as a plugin.

OCR quality, layout understanding, table extraction, and transcription accuracy depend on the selected adapter and source quality. Store originals so better extraction can be replayed.

## How to ingest each format group

Every format flows through the same call. Pass the raw bytes and the correct media type on a `DocumentInput`, and the registry routes to the matching extractor. The examples below use the in-process API; the HTTP `POST /v1/corpora/{id}/documents` endpoint, the CLI, and the SDK take the same content and media type.

Office documents (DOCX, PDF). DOCX needs `refaat[documents]` (python-docx), PDF needs `refaat[documents]` (pypdf). Both extract paragraph text and DOCX table rows into citable chunks.

```python
project.runtime.ingest(
    project.workspace_id,
    project.corpus_id,
    DocumentInput(
        source_id="runbook.docx",
        title="Runbook",
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        content=docx_bytes,
    ),
)
```

Web documents (HTML). HTML needs `refaat[documents]` (beautifulsoup4). Script, style, noscript, and template elements are stripped, the title is captured to metadata, and the visible text becomes the chunk body. Use `media_type="text/html"`.

Data documents (JSON, CSV, XML, YAML, TOML). These need no optional dependency. JSON is flattened to path-value lines, CSV rows become `key: value` pairs per row, and XML, YAML, and TOML are ingested as structured text. Use the matching media type, for example `text/csv` or `application/json`.

Code and config (Python, JavaScript, TypeScript, Go, Rust, Java, PHP, Ruby, shell, SQL, GraphQL, CSS, plain text, Markdown). These flow through the text extractor with no optional dependency. When you ingest a directory with `project.ingest(path)`, the file extension maps to the media type automatically, so a source tree is ingested as-is.

Media (images, audio, video). Image OCR needs the Tesseract adapter and audio and video transcription needs the faster-whisper adapter, both in `refaat[documents]` and its media extras. An image can also be ingested as one citable descriptor chunk through `project.ingest_image(...)`; see [Multimodal ingestion](guides/multimodal.md).

A missing optional dependency raises a clear, actionable installation error rather than failing silently, so the offline core stays dependency-light and each format group is opt-in by install extra.

## Contextual retrieval

Optional deterministic contextual retrieval improves recall on short chunks by prepending the document title and the
nearest section heading to each chunk before embedding, while the stored and cited chunk text stays raw and unchanged.
The chunker keeps heading lines inline in the chunk body and additionally records the nearest heading in chunk metadata;
a module function `contextualize(title, heading, text)` builds the embedding-time prefix and never repeats a heading that
is already present inline. Because the prefix affects only the embedding input, chunk ids and checksums stay byte-stable:
they are derived from the document id, the raw-text checksum, and the ordinal, never from the embedding. Every chunk
records a `heading` and a `context_strategy` of `flat` or `contextual` in its metadata.

Contextual retrieval is off by default and opt-in through `REFAAT_CONTEXTUAL_RETRIEVAL`. It changes the stored
embeddings, so toggling it is not a live switch: it requires a reindex. Re-ingesting unchanged content is idempotent and
will not re-embed, so force a reindex (change or re-add the source) after flipping the setting. On re-ingest the corpus
version advances, which invalidates the answer cache.

## Hierarchical chunking (parent-child and sentence-window)

`REFAAT_CHUNKING_STRATEGY` selects how a document is split. The default `flat` is the layout above and is byte-identical
to earlier releases: flat chunks carry no `granularity` metadata key. The `parent_child` and `sentence_window` strategies
add a small-child layer for precise matching while keeping a broader parent section for context. Both store parents and
children as ordinary chunk rows with globally sequential ordinals; the linkage lives entirely in chunk metadata
(`granularity` of `parent` or `child`, plus a child's `parent_id`, `sent_start`, and `sent_end`), so there is no schema
change. A child's text is a verbatim contiguous slice of its parent text, and its `sent_start`/`sent_end` are character
offsets into that parent.

The strategy is an ingest-time decision that must match retrieval, so it is a `Settings` knob, not a per-request or
harness override. Changing it requires a reindex; on re-ingest the corpus version advances and the answer cache is
invalidated. See [Adaptive context](adaptive-context.md) for how retrieval expands children back to parents or to a
bounded sentence window at query time.

## Metadata safety

Credential-shaped metadata keys are dropped before persistence. The redaction walks nested dicts and lists and removes any key whose casefolded name contains a sensitive substring (`password`, `passwd`, `secret`, `token`, `api_key`, `apikey`, `access_key`, `access_token`, `refresh_token`, `client_secret`, `private_key`, `credential`, `authorization`, `session_token`, `aws_secret_access_key`), guarded by a benign allowlist (`token_count`, `author`, `authors`, plus the structural keys the pipeline stamps). Sensitive-free metadata is returned unchanged, so stored bytes for existing corpora are identical. This is a guardrail, not a data-loss-prevention system. Use OGuardAI or another policy layer before ingestion when PII or secrets require detection and transformation.

## Write ordering and vector consistency

Ingestion commits the relational rows (chunks, entities, relations, the document row, and the `document.indexed` outbox event) in one transaction first, then writes the vector index. The relational store is the source of truth, so a citation can never resolve against a vector for a chunk that is not durably stored. On the offline SQLite path the local index reads directly from the store, so there is no separate vector write.

On PostgreSQL the vector table is eventually consistent: if the process crashes after the relational commit but before the inline vector write, the durable `document.indexed` event is replayed through a reconcile sink that rebuilds the document's vectors from the committed chunk rows. Reconciliation is idempotent, so a redelivered event is harmless.

## Blob garbage collection

Re-ingesting a source writes a new content-addressed blob and updates the document row to point at it, leaving the previous blob behind. That superseded blob is not deleted inline: an inline delete would race a concurrent revert to the old content, which recreates and re-references the very same content-addressed key, and it would strand a live blob if the vector write then failed. Because keys are content-addressed, an orphaned blob is a benign storage leak, while deleting a still-referenced blob is data loss.

A deferred `blob.gc` worker job reclaims orphans. It lists every blob in the store, keeps any blob still referenced by a current document row, keeps any blob newer than `REFAAT_BLOB_GC_GRACE_SECONDS` (ingestion writes the blob before it commits the row, so a blob mid-ingest or just reverted to carries a recent modification time), and re-checks liveness against the database immediately before each delete. The grace window is the primary guard; the re-check narrows but does not fully close a concurrent-revert window, because the filesystem delete is not fenced against a commit that references the key in the same instant. Content-addressed keys make that residual benign: the worst case is a superseded blob that re-ingesting the source re-materializes, never a silent corruption. Deletion is idempotent, so a repeated sweep is harmless. Schedule the job with `create_schedule` alongside the other maintenance jobs; it is global rather than per-workspace because the blob store is shared.

## Outbox delivery contract

The `document.indexed` event and any other outbox event fan out to every configured sink at least once. A per-sink delivery ledger (`outbox_deliveries`, keyed on event id and sink name) records each sink success durably, so a retry after a restart or a lost lease never re-invokes a sink that already delivered. Only the current lease owner finalizes an event as delivered.

At-least-once delivery means a sink can still be invoked more than once for the same event: if a sink completes an external side effect and the worker is reclaimed before the ledger row commits, the reclaiming worker re-invokes that sink. This residual double-delivery is inherent to at-least-once and cannot be removed without a distributed transaction across the sink and the ledger. Sinks must therefore be idempotent, keyed on the event id passed to `emit`.

## Connectors

File and HTTP connector contracts are included. HTTP fetching blocks loopback, private, link-local, multicast, and reserved addresses by default and disables redirects to reduce SSRF risk. Production connectors should add tenant allowlists, credential vault integration, checkpointing, source-specific rate limits, and idempotency tokens.
