# Data model and persistence

Refaat stores durable application state in a relational database, original bytes in a blob store, and retrieval data in the selected search adapter.

## Tenant and evidence hierarchy

```text
Workspace
└── Corpus (monotonic version)
    ├── Document (stable source ID and checksum)
    │   ├── Chunk (canonical evidence ID and embedding)
    │   ├── Entity
    │   └── Relation
    ├── Cache entry
    ├── Evaluation suite/run
    └── Corpus-scoped harness snapshot
```

Every public lookup includes a workspace boundary. Corpus versions advance when evidence changes.

## Main records

| Record | Important fields |
|---|---|
| Workspace | ID, slug, name, creation time |
| Corpus | ID, workspace ID, name, version, settings |
| Document | source ID, title, media type, checksum, blob URI, metadata, status |
| Chunk | ordinal, text, token count, checksum, metadata, embedding, embedding model |
| Entity | normalized name, kind, owning document, metadata |
| Relation | source entity, target entity, kind, weight, owning document |
| API key | workspace, prefix, digest, scopes, revocation, last use |
| Cache entry | corpus version, context scope, query hash/vector, response, expiry |
| Session | corpus, user, goal, summary, metadata |
| Session turn | role, content, metadata, ordinal/time |
| Preference | scope, key, value, confidence, observations |
| Job | kind, payload, status, lease, attempts, result/error |
| Schedule | job kind/payload, interval, next run, enabled state |
| Agent | kind, goal, state, retained flag, parent, heartbeat |
| Agent message | sender, recipient, kind, payload, correlation, status |
| Outbox event | topic, payload, attempts, next attempt, delivery time |
| Evaluation suite/run | cases, status, metrics, results |
| Harness snapshot | parent, base-prompt hash, supplemental state, evidence, active flag |

## Canonical evidence identity

A chunk ID is the citation boundary. Generated answers refer to chunk IDs in model output, and the runtime maps validated IDs into structured citations with document IDs and quotes.

Document IDs remain stable when the same `source_id` is replaced. Chunk IDs change with content/chunk boundaries, which prevents old citations from silently referring to new text.

## Cache context scope

The persisted cache schema keeps a compact `user_scope` string. Refaat derives that value from:

- anonymous, user, or session identity;
- immutable base policy hash;
- embedding and generation model IDs;
- effective retrieval policy;
- active supplemental harness;
- current user profile;
- current session summary.

This allows stronger cache isolation without exposing raw identity or context JSON in the key.

## SQLite and PostgreSQL

SQLite is the local/test backend and can create its schema automatically. Production startup rejects SQLite and automatic schema creation. PostgreSQL uses migrations and pgvector-backed semantic search plus database-side lexical search.

Do not copy a SQLite file while active writes are occurring unless the backup method includes its WAL consistently. See [Backup and restore](../operations/backup-restore.md).
