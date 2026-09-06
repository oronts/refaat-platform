from __future__ import annotations

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    text,
)

metadata = MetaData()

workspaces = Table(
    "workspaces",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("slug", String(96), nullable=False, unique=True),
    Column("name", String(200), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

api_keys = Table(
    "api_keys",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("name", String(120), nullable=False),
    Column("prefix", String(24), nullable=False, unique=True),
    Column("digest", Text, nullable=False),
    Column("scopes", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("last_used_at", DateTime(timezone=True)),
    Column("revoked_at", DateTime(timezone=True)),
    Index("ix_api_keys_workspace", "workspace_id"),
)

corpora = Table(
    "corpora",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("name", String(200), nullable=False),
    Column("version", Integer, nullable=False, default=0),
    Column("settings", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("workspace_id", "name", name="uq_corpora_workspace_name"),
    Index("ix_corpora_workspace", "workspace_id"),
)

documents = Table(
    "documents",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("corpus_id", ForeignKey("corpora.id", ondelete="CASCADE"), nullable=False),
    Column("source_id", String(512), nullable=False),
    Column("title", String(512), nullable=False),
    Column("media_type", String(160), nullable=False),
    Column("checksum", String(64), nullable=False),
    # An order-independent XOR fold over this document's (chunk_id, checksum) set, maintained only
    # under the destroy lock in replace_document, so an operator can localize which document drifted.
    # Null and byte-identical when derivation manifests are off; never enters retrieval, cache, or the
    # corpus content checksum.
    Column("derivation_digest", String(64)),
    Column("blob_uri", Text, nullable=False),
    Column("status", String(32), nullable=False),
    Column("error", Text),
    Column("metadata", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("corpus_id", "source_id", name="uq_documents_corpus_source"),
    Index("ix_documents_workspace_corpus", "workspace_id", "corpus_id"),
)

chunks = Table(
    "chunks",
    metadata,
    Column("id", String(96), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("corpus_id", ForeignKey("corpora.id", ondelete="CASCADE"), nullable=False),
    Column("document_id", ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
    Column("ordinal", Integer, nullable=False),
    Column("text", Text, nullable=False),
    Column("token_count", Integer, nullable=False),
    Column("checksum", String(64), nullable=False),
    Column("metadata", JSON, nullable=False),
    Column("embedding", JSON, nullable=False),
    Column("embedding_model", String(200), nullable=False),
    # Multimodal: the blob key of the stored image bytes and the image media type for
    # an image-descriptor chunk. Both are NULL for an ordinary text chunk, so a
    # text-only corpus is byte-identical and these columns never enter the answer
    # cache identity. A cited image chunk resolves through this table like any chunk;
    # the raw image is never a citation.
    Column("image_ref", Text),
    Column("media_type", String(160)),
    UniqueConstraint("document_id", "ordinal", name="uq_chunks_document_ordinal"),
    Index("ix_chunks_workspace_corpus", "workspace_id", "corpus_id"),
    Index("ix_chunks_document", "document_id"),
)

chunk_hypotheticals = Table(
    "chunk_hypotheticals",
    metadata,
    Column("id", String(96), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("corpus_id", ForeignKey("corpora.id", ondelete="CASCADE"), nullable=False),
    Column("document_id", ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
    Column("chunk_id", ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False),
    Column("ordinal", Integer, nullable=False),
    Column("question", Text, nullable=False),
    Column("embedding", JSON, nullable=False),
    Column("embedding_model", String(200), nullable=False),
    UniqueConstraint("chunk_id", "ordinal", name="uq_chunk_hypotheticals_chunk_ordinal"),
    Index("ix_chunk_hypotheticals_scope", "workspace_id", "corpus_id"),
    Index("ix_chunk_hypotheticals_chunk", "chunk_id"),
)

raptor_nodes = Table(
    "raptor_nodes",
    metadata,
    Column("id", String(96), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("corpus_id", ForeignKey("corpora.id", ondelete="CASCADE"), nullable=False),
    Column("level", Integer, nullable=False),
    Column("parent_node_id", String(96)),
    Column("summary", Text, nullable=False),
    Column("embedding", JSON, nullable=False),
    Column("embedding_model", String(200), nullable=False),
    Column("child_node_ids", JSON, nullable=False),
    Column("leaf_chunk_ids", JSON, nullable=False),
    Index("ix_raptor_nodes_scope", "workspace_id", "corpus_id"),
    Index("ix_raptor_nodes_parent", "parent_node_id"),
)

entities = Table(
    "entities",
    metadata,
    Column("id", String(96), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("corpus_id", ForeignKey("corpora.id", ondelete="CASCADE"), nullable=False),
    Column("document_id", ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
    Column("name", String(300), nullable=False),
    Column("normalized_name", String(300), nullable=False),
    Column("kind", String(100), nullable=False),
    Column("metadata", JSON, nullable=False),
    Index("ix_entities_corpus_name", "corpus_id", "normalized_name"),
    Index("ix_entities_document", "document_id"),
)

relations = Table(
    "relations",
    metadata,
    Column("id", String(128), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("corpus_id", ForeignKey("corpora.id", ondelete="CASCADE"), nullable=False),
    Column("document_id", ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
    Column("source_entity_id", ForeignKey("entities.id", ondelete="CASCADE"), nullable=False),
    Column("target_entity_id", ForeignKey("entities.id", ondelete="CASCADE"), nullable=False),
    Column("kind", String(100), nullable=False),
    Column("weight", Float, nullable=False),
    Column("metadata", JSON, nullable=False),
    Index("ix_relations_corpus", "corpus_id"),
    Index("ix_relations_source", "source_entity_id"),
    Index("ix_relations_target", "target_entity_id"),
)

entity_terms = Table(
    "entity_terms",
    metadata,
    Column("id", String(96), primary_key=True),
    Column("entity_id", ForeignKey("entities.id", ondelete="CASCADE"), nullable=False),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("corpus_id", ForeignKey("corpora.id", ondelete="CASCADE"), nullable=False),
    Column("document_id", ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
    Column("term", String(300), nullable=False),
    Index("ix_entity_terms_lookup", "corpus_id", "term"),
    Index("ix_entity_terms_document", "document_id"),
    Index("ix_entity_terms_entity", "entity_id"),
)

query_cache = Table(
    "query_cache",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("corpus_id", ForeignKey("corpora.id", ondelete="CASCADE"), nullable=False),
    Column("corpus_version", Integer, nullable=False),
    Column("user_scope", String(200), nullable=False),
    Column("query", Text, nullable=False),
    Column("query_hash", String(64), nullable=False),
    Column("query_embedding", JSON, nullable=False),
    Column("response", JSON, nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint(
        "workspace_id",
        "corpus_id",
        "corpus_version",
        "user_scope",
        "query_hash",
        name="uq_query_cache_scope",
    ),
    Index("ix_query_cache_lookup", "workspace_id", "corpus_id", "corpus_version", "user_scope"),
)

model_call_cache = Table(
    "model_call_cache",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("kind", String(16), nullable=False),
    Column("workspace_id", String(64), nullable=False, server_default=""),
    Column("model_id", String(200), nullable=False),
    Column("cache_key", String(64), nullable=False),
    Column("payload", JSON, nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("kind", "workspace_id", "model_id", "cache_key", name="uq_model_call_cache_key"),
    Index("ix_model_call_cache_lookup", "kind", "workspace_id", "model_id", "cache_key"),
    Index("ix_model_call_cache_expiry", "expires_at"),
)

sessions = Table(
    "sessions",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("corpus_id", ForeignKey("corpora.id", ondelete="CASCADE"), nullable=False),
    Column("user_id", String(200)),
    Column("goal", Text),
    Column("summary", Text),
    Column("metadata", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Index("ix_sessions_workspace", "workspace_id"),
)

session_turns = Table(
    "session_turns",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("session_id", ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
    Column("role", String(32), nullable=False),
    Column("content", Text, nullable=False),
    Column("metadata", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    CheckConstraint("role in ('user', 'assistant', 'system', 'tool')", name="ck_session_turn_role"),
    Index("ix_session_turns_session_created", "session_id", "created_at"),
)

preferences = Table(
    "preferences",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("corpus_id", ForeignKey("corpora.id", ondelete="CASCADE"), nullable=False),
    Column("user_scope", String(200), nullable=False),
    Column("key", String(200), nullable=False),
    Column("value", JSON, nullable=False),
    Column("confidence", Float, nullable=False),
    Column("observations", Integer, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("workspace_id", "corpus_id", "user_scope", "key", name="uq_preferences_scope_key"),
    Index("ix_preferences_scope", "workspace_id", "corpus_id", "user_scope"),
)

jobs = Table(
    "jobs",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("kind", String(160), nullable=False),
    Column("payload", JSON, nullable=False),
    Column("status", String(32), nullable=False),
    Column("attempts", Integer, nullable=False),
    Column("max_attempts", Integer, nullable=False),
    Column("run_after", DateTime(timezone=True), nullable=False),
    Column("lease_owner", String(160)),
    Column("leased_until", DateTime(timezone=True)),
    Column("heartbeat_at", DateTime(timezone=True)),
    Column("last_error", Text),
    Column("result", JSON),
    # Optional dedup handle tying a job to the durable entity it drives (an
    # optimization run id). Null for ordinary jobs, which are unaffected.
    Column("dedup_key", String(96)),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Index("ix_jobs_claim", "status", "run_after", "leased_until"),
    Index("ix_jobs_workspace", "workspace_id"),
)

# At most one non-terminal job per dedup_key: a partial unique index scoped to
# PENDING and RUNNING so the legitimate enqueue and the orphan reaper converge on a
# single job for a run (the loser's insert hits the constraint and re-reads the
# winner). Jobs with a null dedup_key are outside the index and never contend.
Index(
    "uq_jobs_active_dedup",
    jobs.c.dedup_key,
    unique=True,
    sqlite_where=text("dedup_key IS NOT NULL AND status IN ('pending', 'running')"),
    postgresql_where=text("dedup_key IS NOT NULL AND status IN ('pending', 'running')"),
)

schedules = Table(
    "schedules",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("name", String(200), nullable=False),
    Column("job_kind", String(160), nullable=False),
    Column("payload", JSON, nullable=False),
    Column("interval_seconds", Integer, nullable=False),
    Column("next_run_at", DateTime(timezone=True), nullable=False),
    Column("enabled", Boolean, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("workspace_id", "name", name="uq_schedules_workspace_name"),
    Index("ix_schedules_due", "enabled", "next_run_at"),
)

agents = Table(
    "agents",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("name", String(200), nullable=False),
    Column("kind", String(120), nullable=False),
    Column("goal", Text),
    Column("state", JSON, nullable=False),
    Column("retained", Boolean, nullable=False),
    Column("parent_agent_id", ForeignKey("agents.id", ondelete="SET NULL")),
    Column("owner_scope", String(64)),
    Column("last_heartbeat_at", DateTime(timezone=True)),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("workspace_id", "name", name="uq_agents_workspace_name"),
    Index("ix_agents_workspace", "workspace_id"),
)

agent_messages = Table(
    "agent_messages",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("sender_agent_id", ForeignKey("agents.id", ondelete="CASCADE"), nullable=False),
    Column("recipient_agent_id", ForeignKey("agents.id", ondelete="CASCADE"), nullable=False),
    Column("kind", String(120), nullable=False),
    Column("payload", JSON, nullable=False),
    Column("correlation_id", String(128)),
    Column("status", String(32), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("acknowledged_at", DateTime(timezone=True)),
    Index("ix_agent_messages_mailbox", "recipient_agent_id", "status", "created_at"),
)

harness_snapshots = Table(
    "harness_snapshots",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("corpus_id", ForeignKey("corpora.id", ondelete="CASCADE")),
    Column("parent_id", ForeignKey("harness_snapshots.id", ondelete="SET NULL")),
    Column("base_prompt_hash", String(64), nullable=False),
    Column("supplemental", JSON, nullable=False),
    Column("evidence", JSON, nullable=False),
    Column("active", Boolean, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Index("ix_harness_active", "workspace_id", "corpus_id", "active"),
)

evaluation_suites = Table(
    "evaluation_suites",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("name", String(200), nullable=False),
    Column("cases", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("workspace_id", "name", name="uq_eval_suites_workspace_name"),
)

evaluation_runs = Table(
    "evaluation_runs",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("suite_id", ForeignKey("evaluation_suites.id", ondelete="CASCADE"), nullable=False),
    Column("corpus_id", ForeignKey("corpora.id", ondelete="CASCADE"), nullable=False),
    Column("status", String(32), nullable=False),
    Column("metrics", JSON, nullable=False),
    Column("results", JSON, nullable=False),
    Column("error", Text),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("finished_at", DateTime(timezone=True)),
    Index("ix_eval_runs_workspace", "workspace_id"),
)

outbox = Table(
    "outbox",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("topic", String(200), nullable=False),
    Column("payload", JSON, nullable=False),
    Column("attempts", Integer, nullable=False),
    Column("available_at", DateTime(timezone=True), nullable=False),
    Column("delivered_at", DateTime(timezone=True)),
    Column("last_error", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("claimed_by", String(160)),
    Column("claimed_until", DateTime(timezone=True)),
    Index("ix_outbox_pending", "delivered_at", "available_at"),
)

outbox_deliveries = Table(
    "outbox_deliveries",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("event_id", ForeignKey("outbox.id", ondelete="CASCADE"), nullable=False),
    Column("sink_name", String(200), nullable=False),
    Column("delivered_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("event_id", "sink_name", name="uq_outbox_deliveries_event_sink"),
    Index("ix_outbox_deliveries_event", "event_id"),
)

# Durable at-least-once delivery of governance audit events. Empty and unused unless
# durable audit is enabled; the payload holds a serialized AuditEvent that the worker
# reconstructs and delivers to the real configured sink with retries and a per-sink ledger.
audit_outbox = Table(
    "audit_outbox",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("event_type", String(200), nullable=False),
    Column("payload", JSON, nullable=False),
    Column("attempts", Integer, nullable=False),
    Column("available_at", DateTime(timezone=True), nullable=False),
    Column("delivered_at", DateTime(timezone=True)),
    Column("last_error", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("claimed_by", String(160)),
    Column("claimed_until", DateTime(timezone=True)),
    Index("ix_audit_outbox_pending", "delivered_at", "available_at"),
)

audit_outbox_deliveries = Table(
    "audit_outbox_deliveries",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("event_id", ForeignKey("audit_outbox.id", ondelete="CASCADE"), nullable=False),
    Column("sink_name", String(200), nullable=False),
    Column("delivered_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("event_id", "sink_name", name="uq_audit_outbox_deliveries_event_sink"),
    Index("ix_audit_outbox_deliveries_event", "event_id"),
)

program_artifacts = Table(
    "program_artifacts",
    metadata,
    Column("row_id", String(64), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("artifact_id", String(96), nullable=False),
    Column("program", String(128), nullable=False),
    Column("definition_hash", String(96), nullable=False),
    Column("document", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("workspace_id", "artifact_id", name="uq_program_artifacts"),
    Index("ix_program_artifacts_program", "workspace_id", "program"),
)

program_deployments = Table(
    "program_deployments",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("program", String(128), nullable=False),
    Column("environment", String(64), nullable=False),
    Column("artifact_id", String(96), nullable=False),
    Column("action", String(32), nullable=False),
    Column("sequence", Integer, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Index("ix_program_deployments_alias", "workspace_id", "program", "environment"),
)

program_approvals = Table(
    "program_approvals",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("program", String(128), nullable=False),
    Column("environment", String(64), nullable=False),
    Column("artifact_id", String(96), nullable=False),
    Column("approver", String(200), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Index("ix_program_approvals", "workspace_id", "program", "environment", "artifact_id"),
)

program_canaries = Table(
    "program_canaries",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("program", String(128), nullable=False),
    Column("environment", String(64), nullable=False),
    Column("candidate_artifact_id", String(96)),
    Column("percent", Integer, nullable=False),
    Column("compare", Boolean, nullable=False),
    Column("action", String(32), nullable=False),
    Column("sequence", Integer, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Index("ix_program_canaries_alias", "workspace_id", "program", "environment"),
    UniqueConstraint("workspace_id", "program", "environment", "sequence", name="uq_program_canaries_sequence"),
)

program_shadow_runs = Table(
    "program_shadow_runs",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("program", String(128), nullable=False),
    Column("environment", String(64), nullable=False),
    Column("active_artifact_id", String(96), nullable=False),
    Column("candidate_artifact_id", String(96)),
    Column("served", String(32), nullable=False),
    Column("bucket", Integer, nullable=False),
    Column("query_hash", String(96), nullable=False),
    Column("comparison", JSON),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Index("ix_program_shadow_runs_alias", "workspace_id", "program", "environment"),
)

optimization_runs = Table(
    "optimization_runs",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("program", String(128), nullable=False),
    Column("corpus_id", ForeignKey("corpora.id", ondelete="CASCADE"), nullable=False),
    Column("dataset_name", String(200), nullable=False),
    Column("dataset_version", String(96), nullable=False),
    Column("dataset_cases", JSON, nullable=False),
    Column("search_space", JSON, nullable=False),
    Column("bundle_id", String(96), nullable=False),
    Column("program_definition_hash", String(96)),
    Column("program_immutable_policy_hash", String(96)),
    Column("budget", Integer),
    Column("candidate_total", Integer, nullable=False),
    Column("plan_hash", String(96), nullable=False),
    Column("status", String(32), nullable=False),
    Column("result", JSON, nullable=False),
    Column("best_artifact_id", String(96)),
    Column("error", Text),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("finished_at", DateTime(timezone=True)),
    Index("ix_optimization_runs_workspace", "workspace_id"),
)

# At most one non-terminal run per (workspace, program, plan): a partial unique
# index scoped to PENDING and RUNNING so two concurrent idempotent enqueues of the
# same materialized plan converge on one run (the loser's insert hits the
# constraint and re-reads the winner), while a fresh enqueue after a terminal
# SUCCEEDED or FAILED run is still allowed.
Index(
    "uq_optimization_runs_active_plan",
    optimization_runs.c.workspace_id,
    optimization_runs.c.program,
    optimization_runs.c.plan_hash,
    unique=True,
    sqlite_where=text("status IN ('pending', 'running')"),
    postgresql_where=text("status IN ('pending', 'running')"),
)

fine_tune_runs = Table(
    "fine_tune_runs",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("dataset_version", String(96), nullable=False),
    Column("base_model", String(200), nullable=False),
    Column("hyperparameters", JSON, nullable=False),
    Column("status", String(32), nullable=False),
    Column("approved", Boolean, nullable=False),
    Column("approved_by", String(200)),
    Column("requester", String(200), nullable=False),
    Column("result_model_ref", String(200)),
    Column("provider_job_ref", String(200)),
    Column("provenance", JSON, nullable=False),
    # Content address of (workspace, dataset_version, base_model, hyperparameters),
    # so a duplicate active request for identical inputs is rejected by the partial
    # unique index below.
    Column("content_key", String(96), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Index("ix_fine_tune_runs_workspace", "workspace_id"),
)

# Durable single-use human approvals bound to an action fingerprint, consumed on the execution they authorize.
# kind partitions a normal approval from a break-glass one so neither path borrows the other's records;
# its server_default "normal" is load-bearing, so a row written before this column existed reads as a
# normal approval and the normal path (which skips any kind != "normal") keeps honoring it. reason holds
# the break-glass justification on the row only and never enters an audit event (the audit contract is
# redacted by construction).
tool_approvals = Table(
    "tool_approvals",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("fingerprint", String(64), nullable=False),
    Column("capability_name", String(200), nullable=False),
    Column("approver_id", String(200), nullable=False),
    Column("approver_roles", JSON, nullable=False),
    Column("granted_at", DateTime(timezone=True), nullable=False),
    Column("status", String(16), nullable=False),
    Column("kind", String(16), nullable=False, server_default="normal"),
    Column("reason", Text),
    Column("consumed_at", DateTime(timezone=True)),
    Column("cancelled_at", DateTime(timezone=True)),
    Index("ix_tool_approvals_lookup", "workspace_id", "fingerprint", "status"),
)

# Open approval requests for an action fingerprint, whose earliest created_at anchors time-based
# escalation: after a bounded delay the governed loop widens who may approve. status "open" scopes
# the anchor to requests still awaiting an approver.
approval_requests = Table(
    "approval_requests",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("fingerprint", String(64), nullable=False),
    Column("capability_name", String(200), nullable=False),
    Column("requested_by", String(200), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("status", String(16), nullable=False, server_default="open"),
    Index("ix_approval_requests_lookup", "workspace_id", "fingerprint", "status"),
)

# At most one non-terminal run per identical training request: a partial unique
# index scoped to PENDING and RUNNING so two concurrent requests for the same
# (workspace, data version, base model, hyperparameters) converge on one run, while
# a fresh request after a terminal SUCCEEDED or FAILED run is still allowed.
Index(
    "uq_fine_tune_runs_active_content",
    fine_tune_runs.c.content_key,
    unique=True,
    sqlite_where=text("status IN ('pending', 'running')"),
    postgresql_where=text("status IN ('pending', 'running')"),
)

# Append-only record of every right-to-erasure request and its outcome. Carries only ids,
# a status, per-surface counts (a JSON object), and a timestamp, never any erased content, so
# the ledger itself can never re-introduce a residue of the document it recorded the erasure of.
erasure_ledger = Table(
    "erasure_ledger",
    metadata,
    Column("request_id", String(64), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("corpus_id", String(64), nullable=False),
    Column("document_id", String(64), nullable=False),
    Column("status", String(16), nullable=False),
    Column("counts", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Index("ix_erasure_ledger_document", "workspace_id", "corpus_id", "document_id"),
)

derivation_manifests = Table(
    "derivation_manifests",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("corpus_id", ForeignKey("corpora.id", ondelete="CASCADE"), nullable=False),
    Column("derivation_key", String(96), nullable=False),
    Column("corpus_version", Integer, nullable=False),
    Column("config", JSON, nullable=False),
    Column("embedding_model", String(200), nullable=False),
    Column("embedding_dimensions", Integer, nullable=False),
    Column("row_counts", JSON, nullable=False),
    Column("content_checksum", String(64), nullable=False),
    Column("status", String(16), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("workspace_id", "corpus_id", "derivation_key", name="uq_derivation_manifest_key"),
    Index("ix_derivation_manifest_status", "workspace_id", "corpus_id", "status"),
)

# One physical build of a corpus index, distinct from the recipe-keyed derivation_manifests row (two
# builds of one recipe collapse there); the id is an opaque surrogate so a rebuild is a distinct row.
# Recall selectors only, never ACL vocabulary: the query-time PEP over chunk.metadata stays the authority.
index_generations = Table(
    "index_generations",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("corpus_id", ForeignKey("corpora.id", ondelete="CASCADE"), nullable=False),
    Column("derivation_key", String(96), nullable=False),
    Column("corpus_version", Integer, nullable=False),
    Column("content_checksum", String(64), nullable=False),
    Column("row_counts", JSON, nullable=False),
    Column("config", JSON, nullable=False),
    Column("embedding_model", String(200), nullable=False),
    Column("embedding_dimensions", Integer, nullable=False),
    Column("physical_corpus_id", String(64)),
    Column("status", String(16), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("activated_at", DateTime(timezone=True)),
    Column("retired_at", DateTime(timezone=True)),
    Index("ix_index_generations_status", "workspace_id", "corpus_id", "status"),
)

# At most one serving generation per corpus, structurally: a partial unique index scoped to status active
# (a standalone Index, not a UniqueConstraint, is the only form that can carry the partial predicate).
Index(
    "uq_index_generations_active",
    index_generations.c.workspace_id,
    index_generations.c.corpus_id,
    unique=True,
    sqlite_where=text("status = 'active'"),
    postgresql_where=text("status = 'active'"),
)

# Durable legal-hold state. An active row (lifted_at IS NULL) refuses every destructive path for
# the covered document, or corpus-wide when document_id IS NULL. The human reason lives on the row
# ONLY; it is never copied into an audit event. Coverage is decided in SQL by is_document_held, the
# single authority, so liveness keys on lifted_at (not the status label).
legal_holds = Table(
    "legal_holds",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("corpus_id", String(64), nullable=False),
    Column("document_id", String(64)),
    Column("status", String(16), nullable=False, server_default="active"),
    Column("reason", Text),
    Column("placed_by", String(200), nullable=False),
    Column("placed_at", DateTime(timezone=True), nullable=False),
    Column("lifted_by", String(200)),
    Column("lifted_at", DateTime(timezone=True)),
    Index("ix_legal_holds_coverage", "workspace_id", "corpus_id", "document_id", "status"),
)

hold_permission_history = Table(
    "hold_permission_history",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("corpus_id", String(64), nullable=False),
    Column("document_id", String(64), nullable=False),
    Column("covering_hold_ids", JSON, nullable=False),
    Column("old_fingerprint", String(64), nullable=False),
    Column("new_fingerprint", String(64), nullable=False),
    Column("old_acl", JSON),
    Column("new_acl", JSON),
    Column("content_checksum", String(64), nullable=False),
    Column("actor", String(200), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Index("ix_hold_permission_history_document", "workspace_id", "corpus_id", "document_id"),
)

kill_switches = Table(
    "kill_switches",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("workspace_id", String(64), nullable=False),
    Column("scope", String(32), nullable=False),
    Column("reason", Text),
    Column("engaged_by", String(200), nullable=False),
    Column("engaged_at", DateTime(timezone=True), nullable=False),
    Column("disengaged_by", String(200)),
    Column("disengaged_at", DateTime(timezone=True)),
    Index(
        "uq_kill_switches_active",
        "workspace_id",
        "scope",
        unique=True,
        sqlite_where=text("disengaged_at IS NULL"),
        postgresql_where=text("disengaged_at IS NULL"),
    ),
    Index("ix_kill_switches_active", "disengaged_at"),
)

quota_counters = Table(
    "quota_counters",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("workspace_id", String(64), nullable=False, server_default=""),
    Column("scope", String(32), nullable=False),
    Column("scope_key", String(256), nullable=False),
    Column("dimension", String(32), nullable=False),
    Column("window_start", BigInteger, nullable=False),
    Column("window_seconds", Integer, nullable=False),
    Column("spent", BigInteger, nullable=False, server_default="0"),
    Column("reserved", BigInteger, nullable=False, server_default="0"),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("scope", "scope_key", "dimension", "window_start", name="uq_quota_counters_window"),
    CheckConstraint("reserved >= 0", name="ck_quota_counters_reserved_nonneg"),
    CheckConstraint("spent >= 0", name="ck_quota_counters_spent_nonneg"),
    Index("ix_quota_counters_sweep", "window_start"),
)

quota_reservations = Table(
    "quota_reservations",
    metadata,
    Column("reservation_id", String(96), primary_key=True),
    Column("counter_id", String(64), nullable=False),
    Column("estimate", BigInteger, nullable=False),
    Column("state", String(16), nullable=False, server_default="open"),
    Column("reserved_at", DateTime(timezone=True), nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    Index("ix_quota_reservations_lease", "state", "expires_at"),
    Index("ix_quota_reservations_counter", "counter_id"),
)

deferred_erasures = Table(
    "deferred_erasures",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("workspace_id", ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
    Column("corpus_id", String(64), nullable=False),
    Column("document_id", String(64), nullable=False),
    Column("status", String(16), nullable=False, server_default="pending"),
    Column("requested_by", String(200), nullable=False),
    Column("requested_at", DateTime(timezone=True), nullable=False),
    Column("executed_request_id", String(64)),
    Column("executed_at", DateTime(timezone=True)),
    # One pending deferred erasure per document, so a re-request while held converges on the existing
    # row rather than duplicating; an executed row does not block a later fresh request.
    Index(
        "ux_deferred_erasures_pending",
        "workspace_id",
        "corpus_id",
        "document_id",
        unique=True,
        sqlite_where=text("status = 'pending'"),
        postgresql_where=text("status = 'pending'"),
    ),
    Index("ix_deferred_erasures_corpus", "workspace_id", "corpus_id", "status"),
)

optimization_candidates = Table(
    "optimization_candidates",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("run_id", ForeignKey("optimization_runs.id", ondelete="CASCADE"), nullable=False),
    Column("ordinal", Integer, nullable=False),
    Column("tunables", JSON, nullable=False),
    Column("artifact_id", String(96), nullable=False),
    # Version-independent identity of the checkpointed variant, so a re-register
    # after a benign library version bump (which shifts the full artifact_id via
    # dependency_versions) is tolerated while a real definition or base-prompt
    # drift is still refused.
    Column("definition_hash", String(96)),
    Column("immutable_policy_hash", String(96)),
    Column("hard_constraints_passed", Boolean, nullable=False),
    Column("score", Float, nullable=False),
    Column("aggregate", JSON, nullable=False),
    Column("mean_tokens", Float, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("run_id", "ordinal", name="uq_optimization_candidates_ordinal"),
    Index("ix_optimization_candidates_run", "run_id"),
)
