from __future__ import annotations

import math
import os
import socket
import uuid
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_worker_id() -> str:
    """A per-process-unique lease owner.

    Two worker processes must never share a lease owner: the lease guards key on
    ``lease_owner == worker_id`` and ``claim_job`` reclaims an expired RUNNING
    job, so a shared owner lets one replica reset or complete a job another is
    actively running. Hostname plus PID plus a random suffix stays unique across
    replicas even inside one host; an explicit ``REFAAT_WORKER_ID`` overrides it.
    """
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _priced(record: Mapping[str, float] | None, *, require_output: bool) -> bool:
    """True when a price record can actually charge a metered call.

    A missing or all-zero record fails closed: the input rate must be positive,
    and the output rate too for a generator (an embedder produces no completion
    tokens, so a zero output rate is legitimate there).
    """
    if not record:
        return False
    if float(record.get("input_per_1k", 0.0)) <= 0.0:
        return False
    return not (require_output and float(record.get("output_per_1k", 0.0)) <= 0.0)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="REFAAT_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    environment: Literal["development", "test", "production"] = "development"
    database_url: str = "sqlite:///./data/refaat.db"
    auto_create_schema: bool = True
    # Bounds the concurrent Postgres advisory-lock connections a same-corpus destroy waiter burst can
    # open, so a large burst cannot exhaust the server's connection limit. Applies only on the
    # PostgreSQL destroy path; SQLite and non-destructive paths never open a lock connection.
    legal_hold_lock_max_waiters: int = 64
    # "defer" records an erasure that meets a hold and runs it when the last covering hold lifts;
    # "refuse" (default) erases nothing and is byte-identical.
    legal_hold_on_conflict: Literal["refuse", "defer"] = "refuse"

    blob_backend: Literal["local", "s3"] = "local"
    blob_root: Path = Path("./data/blobs")
    blob_gc_grace_seconds: int = 86_400
    s3_bucket: str | None = None
    s3_endpoint_url: str | None = None
    s3_region: str = "eu-central-1"
    s3_access_key_id: SecretStr | None = None
    s3_secret_access_key: SecretStr | None = None

    connector_file_root: Path = Path("./data/connectors")
    connector_max_bytes: int = 20 * 1024 * 1024
    allow_private_connector_hosts: bool = False

    bootstrap_token: SecretStr | None = None
    api_key_pepper: SecretStr = Field(default=SecretStr("development-only-change-me"))
    cors_origins: list[str] = Field(default_factory=list)
    trusted_hosts: list[str] = Field(default_factory=lambda: ["localhost", "127.0.0.1", "testserver"])
    max_body_bytes: int = 20 * 1024 * 1024
    rate_limit_requests: int = 120
    rate_limit_window_seconds: int = 60
    rate_limit_trusted_proxies: list[str] = Field(default_factory=list)
    rate_limit_max_buckets: int = 10_000
    api_workers: int = 1
    redis_url: str | None = None
    redis_timeout_seconds: float = 1.0
    answer_cache_backend: Literal["sql", "redis"] = "sql"
    model_call_cache_backend: Literal["sql", "redis"] = "sql"

    embedding_provider: Literal["deterministic", "openai_compatible"] = "deterministic"
    embedding_model: str = "refaat-hash-v1"
    embedding_dimensions: int = 384
    generation_provider: Literal["extractive", "openai_compatible"] = "extractive"
    generation_model: str = "extractive-v1"
    graph_provider: Literal["cooccurrence", "openai_compatible"] = "cooccurrence"
    reranker_provider: Literal["none", "lexical", "cross_encoder"] = "none"
    judge_provider: Literal["none", "generator"] = "none"
    query_transform_provider: Literal["sub_query", "hyde", "query_rewrite", "multi_query", "step_back"] = "sub_query"
    graph_max_entities: int = 80
    graph_min_token_length: int = 5
    # Fallback language for the lexical stopwords and the co-occurrence graph
    # extractor when a document carries no language metadata tag. The default "en"
    # selects the exact English stopword set, so an untagged corpus is byte-identical.
    default_language: str = "en"
    # Per-project override or extension of the built-in graph-extractor stopword
    # registry. None (the default) means unconfigured: the co-occurrence graph uses
    # the shipped per-language sets and is byte-identical. When set, each entry
    # overrides an existing language ("en", "de") or adds a new one ("it", "pt");
    # the words are matched against casefolded tokens by the resolver.
    stopwords_by_language: dict[str, list[str]] | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: SecretStr | None = None
    embedding_base_url: str | None = None
    embedding_api_key: SecretStr | None = None
    embedding_send_dimensions: bool = True
    generation_base_url: str | None = None
    generation_api_key: SecretStr | None = None
    fine_tuning_provider: Literal["deterministic", "openai"] = "deterministic"
    fine_tuning_base_url: str | None = None
    fine_tuning_api_key: SecretStr | None = None
    # Delay before a not-yet-terminal fine-tune run is re-polled, so a long provider job self-progresses.
    fine_tune_poll_seconds: int = 60
    # Multimodal generation, off by default. With both at "none" an image is ingested
    # with an honest deterministic descriptor and answered by the extractive path, and
    # a text-only corpus is byte-identical to before. "openai_vision" activates a real,
    # lazily imported vision model at ingest (captioner) and at query (generation);
    # the offline gate never constructs either.
    image_captioner: Literal["none", "openai_vision"] = "none"
    multimodal_generation: Literal["none", "openai_vision"] = "none"
    multimodal_model: str = "gpt-4o-mini"
    multimodal_base_url: str | None = None
    multimodal_api_key: SecretStr | None = None
    image_max_bytes: int = 8 * 1024 * 1024
    request_timeout_seconds: float = 60.0
    model_max_retries: int = 2
    model_retry_base_delay_seconds: float = 0.25

    default_chunk_tokens: int = 420
    chunk_overlap_tokens: int = 64
    chunking_strategy: str = "flat"
    sentence_window_radius: int = 1
    query_token_budget: int = 4_000
    query_top_k: int = 12
    semantic_weight: float = 0.55
    lexical_weight: float = 0.30
    graph_weight: float = 0.15
    fusion_strategy: Literal["rrf", "linear"] = "rrf"
    rrf_k: int = 60
    autocut_enabled: bool = False
    autocut_threshold: float = 0.5
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    query_expansion: bool = True
    max_query_variants: int = 4
    compression_enabled: bool = False
    compression_min_salience: float = 0.0
    hype_enabled: bool = False
    raptor_enabled: bool = False
    # Opt-in term-indexed graph neighborhood read. Default off is byte-identical (the full list_graph
    # scan); on, the query and RLM paths fetch only the query-relevant subgraph. A query longer than
    # the char cap falls back to the full scan so the substring candidate set can never blow up.
    graph_term_index_enabled: bool = False
    graph_term_index_max_query_chars: int = 256
    derivation_manifest_enabled: bool = False
    # Gates the physical index-generation surface; when off, no rows are read or folded and the answer
    # cache stays keyed on corpora.version alone, byte-identical to before the feature existed.
    index_generations_enabled: bool = False
    router_long_query_tokens: int = 24
    router_multi_clause_threshold: int = 2
    router_entity_threshold: int = 2
    contextual_retrieval: bool = False
    abstain_below_score: float = 0.0
    citation_policy: Literal["optional", "required_when_evidence", "required"] = "optional"
    # Soft down-rank of low-trust evidence in retrieval ranking, off by default. 0.0 applies no
    # penalty and the ranking is byte-identical. When positive, each candidate is penalized by
    # weight * (AUTHORITATIVE - source_trust) / AUTHORITATIVE before the final top_k, so untrusted
    # or injected evidence cannot dominate trusted evidence. It is a soft re-rank, never a hard
    # exclusion: a low-trust chunk is still retrievable and citable, only ranked lower.
    source_trust_weight: float = 0.0
    # Under an active policy, over-fetch limit * factor (capped) so authorized results are not starved.
    acl_overfetch_factor: int = 4
    acl_overfetch_hard_cap: int = 2_000
    cache_ttl_seconds: int = 3_600
    semantic_cache_threshold: float = 0.96
    model_call_cache_enabled: bool = False
    # Output governance, off by default. An empty patterns list installs no governor, so the
    # query and RLM answer paths are byte-identical. A non-empty list auto-installs the
    # reference PatternDenylistGovernor over the final answer: "refuse" withholds a matching
    # answer, "redact" replaces each match with a placeholder. It catches only a configured
    # literal, never reconstructed PII, paraphrase, an unconfigured secret, or injection.
    output_denylist_patterns: list[str] = Field(default_factory=list)
    output_governance_mode: Literal["refuse", "redact"] = "refuse"
    # Generation-evidence egress redaction, off by default. An empty patterns list installs no
    # transform, so the generation path is byte-identical. A non-empty list auto-installs the
    # reference PatternRedactor, which replaces each configured span in the evidence a generation
    # model would see (retrieved chunk bodies, graph facts, session summary by default) with the
    # placeholder before the request reaches a real generator. It covers the generation model's
    # evidence input only; embeddings, fine-tuning, ingestion-time HyPE/RAPTOR/captioner, and
    # evaluation-time judging are separate provider egresses this control does not cover. See
    # docs/egress-redaction.md.
    egress_redaction_patterns: list[str] = Field(default_factory=list)
    egress_redaction_placeholder: str = "[REDACTED]"
    egress_redaction_fields: list[str] = Field(default_factory=lambda: ["evidence", "graph_facts", "session_summary"])
    # Data-residency model routing, off by default. None or an empty mapping installs no
    # residency router, so the generation path is byte-identical. When set, each entry maps a
    # classification name (public/internal/confidential/restricted) to a rule with allowed_regions,
    # a max_retention term (zero/transient/limited/retained), and optional allowed_purposes. The
    # active provider tags below are matched against the rule for the derived evidence
    # classification. Region and purpose admission is exact-match on opaque operator strings, and
    # the tags are trusted operator declarations, not verified facts about the provider. A policy
    # with no rule for a classification refuses that classification (including untagged PUBLIC).
    residency_policy: dict[str, dict[str, object]] | None = None
    residency_provider_region: str = ""
    residency_provider_retention: Literal["zero", "transient", "limited", "retained"] | None = None
    residency_provider_purpose: str = ""
    # Degradation on a generation provider outage, off by default. "none" installs no policy, so a
    # provider outage propagates and the query and RLM answer paths are byte-identical (the outage
    # surfaces as a 503). "refuse_all" installs the fail-closed reference policy (every outage
    # refuses). "safe_defaults" serves a grounded, cited fallback when one survives and abstains
    # otherwise, never serving an ungrounded answer. It governs the query and RLM answer seams only;
    # the citation-miss extractive fallback and the internal degrade-to-deterministic helpers are
    # out of scope. See docs/degradation.md.
    degradation_policy: Literal["none", "refuse_all", "safe_defaults"] = "none"
    # Code execution in the RLM is off by default: "deny" installs a sandbox that
    # refuses every execution, so the shipped runtime runs no code. "safe_expression"
    # is the bounded, provably-safe governance evaluator; "subprocess" is the opt-in
    # isolated reference adapter. Even with a non-deny sandbox, code runs only when a
    # caller passes an admitting CodePolicy on an in-process RLM call.
    code_sandbox: Literal["deny", "safe_expression", "subprocess"] = "deny"
    code_sandbox_max_nodes: int = 256
    code_sandbox_max_output_bytes: int = 65_536
    code_sandbox_max_wall_seconds: float = 2.0
    # Bound the corpus an RLM run loads into memory before recursion. None (default) is unbounded and
    # byte-identical; a positive value fails the run closed when the corpus exceeds it.
    rlm_max_input_chunks: int | None = None
    run_token_budget: int | None = None
    run_cost_budget_usd: float | None = None
    # Per-tool execution deadline (seconds). None (default) is byte-identical: the governed loop runs a
    # tool inline; a positive value runs it under a worker-thread deadline with the three-outcome contract.
    tool_timeout_seconds: float | None = None
    tool_timeout_max_workers: int = 8
    # Per-provider circuit breaker. 'none' (default) is byte-identical: no breaker is installed and the
    # deterministic offline stack is never wrapped. 'in_memory' fast-fails a wedged provider endpoint.
    circuit_breaker_backend: Literal["none", "in_memory"] = "none"
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_reset_timeout_seconds: float = 30.0
    circuit_breaker_half_open_deadline_seconds: float = 60.0
    circuit_breaker_count_429: bool = False
    model_prices: dict[str, dict[str, float]] = Field(default_factory=dict)
    session_compaction_turns: int = 20
    retained_session_turns: int = 8

    worker_id: str = Field(default_factory=_default_worker_id)
    worker_poll_seconds: float = 1.0
    job_lease_seconds: int = 60
    job_max_attempts: int = 5
    shadow_run_retention_days: int = 30
    # Serve-first async shadow: defer the canary comparison to a durable job so it never delays the served
    # response. Off by default (inline compare, byte-identical); only fires with a canary and no policy active.
    defer_shadow: bool = False
    shadow_defer_timeout_seconds: int = 300
    outbox_retention_days: int = 7
    # Durable at-least-once audit delivery. Off by default: audit events are delivered directly
    # to the installed sink (byte-identical). When on with a sink installed, an emission persists
    # to the audit_outbox and a worker delivers it with retries, so a transient failure loses nothing.
    audit_durable: bool = False
    # Fail-closed audit for the synchronous pre-return paths. Off by default: a sink failure is
    # logged and swallowed (best-effort) so the request still serves. When on, a sink failure on
    # a query, RLM, or approval grant/consume/cancel raises AuditUnavailableError so the operation
    # refuses rather than proceeding unaudited. Under an active policy the streaming query path
    # buffers when this is on, so its terminal audit fires before any answer byte and a sink outage
    # denies rather than serving unaudited.
    audit_fail_closed: bool = False
    # Require a principal on an in-process evidence call when an access policy is active. On by
    # default: such a call is denied rather than served unfiltered, so an omitted principal cannot
    # silently widen access; the HTTP boundary already fails closed through the identity dependency.
    # Set to false to restore the trusted-in-process behavior where a principal-less in-process caller
    # runs unfiltered. The no-policy path is unaffected either way.
    strict_authorization: bool = True
    # Execution-time secret provider for governed tools. Off by default ("none"): no provider is
    # installed, so a governed step whose capability declares a required secret fails closed rather
    # than running unenforced, and a program that declares none is byte-identical. "env" installs the
    # reference provider that reads REFAAT_TOOL_SECRET_<NAME> from the process environment; a real
    # vault or KMS provider is installed with with_secret_provider.
    tool_secret_provider: Literal["none", "env"] = "none"
    tool_secret_env_prefix: str = "REFAAT_TOOL_SECRET_"

    # Incident kill switches. Off by default ("none"): no registry is installed, no table is read, and
    # every guard short-circuits, so the runtime is byte-identical. "database" installs the store-backed
    # registry read at request time behind a short-TTL cache, so engaging a switch reaches every worker
    # within one TTL with no restart.
    kill_switch_backend: Literal["none", "database"] = "none"
    kill_switch_cache_ttl_seconds: float = 2.0
    # Serve last-known-good for this window after a switch-registry read fails past the TTL. Default 0.0
    # is strict fail-closed: any read failure denies every guarded operation. A positive value is an
    # explicit operator decision to relax the fail-closed default and ride out a brief store outage.
    kill_switch_stale_deny_seconds: float = 0.0

    # Persisted per-scope quotas. Off by default ("none"): no policy is installed, no counter table
    # is read, and every guard short-circuits, so the runtime is byte-identical. "database" installs
    # the store-backed policy over an immutable rule snapshot loaded once at build time. Each rule is
    # {scope, dimension, limit, window_seconds}; scope is workspace|principal|workspace_principal|
    # corpus, dimension is generation_tokens|cost_usd|requests, limit and window are positive ints.
    quota_backend: Literal["none", "database"] = "none"
    quota_rules: list[dict[str, Any]] = Field(default_factory=list)
    # The reservation lease. A crash between reserve and reconcile leaves an open lease the reaper
    # charges after this many seconds, so it must exceed the longest run wall-clock; too short charges
    # a live long run its full ceiling, too long delays capacity reclaim after a crash.
    quota_reservation_lease_seconds: float = 900.0

    # Secure-by-default egress governance. Off by default ("none"): no policy is installed, every
    # outbound httpx seam's chokepoint is a no-op, and the runtime is byte-identical. "allowlist"
    # installs the default-deny AllowlistEgressPolicy over egress_allowlist. Each allowlist entry is
    # {host, port?, schemes?, purposes?}; host is an exact hostname or a leading-dot suffix
    # (".example.com"); port is optional (any when absent); schemes and purposes are optional lists.
    # A configured model, embedding, or fine-tuning base_url host must appear on an active allowlist
    # or startup fails closed, so an operator cannot silently lock the model out.
    egress_policy: Literal["none", "allowlist"] = "none"
    egress_allowlist: list[dict[str, Any]] = Field(default_factory=list)

    log_level: str = "INFO"
    log_json: bool = True
    # Redact the exception message in a JSON log's traceback. Off by default: the full traceback,
    # including each exception's message, is logged (byte-identical). When on, the traceback frames
    # (code locations) are preserved but every exception message is replaced with the exception type
    # name, so a message that embedded evidence text cannot egress through the exc_info channel.
    observability_redact_exception_text: bool = False
    otel_enabled: bool = False
    otel_service_name: str = "refaat"
    otel_exporter_endpoint: str | None = None

    plugin_allowlist: list[str] = Field(default_factory=list)
    # Opt-in identity and code-hash pins for discovered entry points, keyed "group:name" to a
    # record of {value, dist_name, dist_version, code_hash}. Empty by default: discovery stays the
    # name-only allowlist and is byte-identical. When set, every allowlisted entry point must be
    # pinned and loads only when it matches its pin. See docs/plugin-pinning.md.
    plugin_pins: dict[str, dict[str, str]] = Field(default_factory=dict)
    skill_root: Path = Path("./data/skills")

    def validate_for_startup(self) -> None:
        self._validate_numbers()
        self._validate_quota()
        self._validate_egress()
        if self.blob_backend == "s3" and not self.s3_bucket:
            raise ValueError("REFAAT_S3_BUCKET is required when the S3 blob backend is selected")
        if (self.answer_cache_backend == "redis" or self.model_call_cache_backend == "redis") and not self.redis_url:
            raise ValueError("REFAAT_REDIS_URL is required when a Redis cache backend is selected")
        if self.model_call_cache_backend == "redis" and not self.model_call_cache_enabled:
            # A never-written Redis model cache adds only a fail-closed erasure risk; require the pair.
            raise ValueError("REFAAT_MODEL_CALL_CACHE_BACKEND='redis' requires REFAAT_MODEL_CALL_CACHE_ENABLED=true")
        if not math.isfinite(self.redis_timeout_seconds) or self.redis_timeout_seconds <= 0.0:
            raise ValueError("REFAAT_REDIS_TIMEOUT_SECONDS must be a positive finite number")
        if self.graph_term_index_max_query_chars <= 0:
            raise ValueError("REFAAT_GRAPH_TERM_INDEX_MAX_QUERY_CHARS must be positive")
        if self.environment != "production":
            return
        pepper = self.api_key_pepper.get_secret_value()
        if pepper == "development-only-change-me" or len(pepper) < 24:
            raise ValueError("REFAAT_API_KEY_PEPPER must contain at least 24 non-default characters")
        if self.bootstrap_token and len(self.bootstrap_token.get_secret_value()) < 24:
            raise ValueError("REFAAT_BOOTSTRAP_TOKEN must contain at least 24 characters")
        if self.database_url.startswith("sqlite"):
            raise ValueError("SQLite is not supported for production deployments")
        if self.auto_create_schema:
            raise ValueError("REFAAT_AUTO_CREATE_SCHEMA must be false in production; run migrations explicitly")
        if "*" in self.cors_origins:
            raise ValueError("Wildcard CORS is not supported in production")
        if (
            (self.image_captioner == "openai_vision" or self.multimodal_generation == "openai_vision")
            and self.multimodal_api_key is None
            and self.openai_api_key is None
        ):
            # A real vision model must carry a key. Enforced only in production, so the
            # off-by-default "none" path and the offline gate never need a key.
            raise ValueError("A multimodal vision provider requires REFAAT_MULTIMODAL_API_KEY or REFAAT_OPENAI_API_KEY")
        if self.fine_tuning_provider == "openai" and self.fine_tuning_api_key is None and self.openai_api_key is None:
            # The deterministic default needs no external resource; a real provider must
            # carry a key. Enforced only in production, so the default and test paths are
            # unaffected and the offline gate never needs a key.
            raise ValueError(
                "REFAAT_FINE_TUNING_PROVIDER='openai' requires REFAAT_FINE_TUNING_API_KEY or REFAAT_OPENAI_API_KEY"
            )
        if self.api_workers > 1 and not self.redis_url:
            # The in-memory rate limiter is per process, so behind N API workers one
            # address would get about N times its allowance. A multi-process production
            # deployment must use the shared Redis backend.
            raise ValueError(
                "REFAAT_API_WORKERS greater than 1 requires REFAAT_REDIS_URL: the in-memory rate limiter is "
                "per process and cannot enforce a shared limit across workers"
            )

    def _validate_quota(self) -> None:
        # Nothing configured: the default is byte-identical, so skip all quota validation.
        if self.quota_backend == "none" and not self.quota_rules:
            return
        from refaat.domain.quota import QuotaDimension, quota_rules_from_config

        rules = quota_rules_from_config(self.quota_rules)
        if self.quota_reservation_lease_seconds <= 0:
            raise ValueError("REFAAT_QUOTA_RESERVATION_LEASE_SECONDS must be positive")
        has_token = any(rule.dimension is QuotaDimension.GENERATION_TOKENS for rule in rules)
        has_cost = any(rule.dimension is QuotaDimension.COST_USD for rule in rules)
        # A token or cost reservation is a ceiling; you cannot reserve a ceiling you do not have.
        if has_token and not (self.run_token_budget is not None and self.run_token_budget > 0):
            raise ValueError("A generation_tokens quota requires a positive REFAAT_RUN_TOKEN_BUDGET ceiling")
        if not has_cost:
            return
        if not (self.run_cost_budget_usd is not None and self.run_cost_budget_usd > 0):
            raise ValueError("A cost_usd quota requires a positive REFAAT_RUN_COST_BUDGET_USD ceiling")
        # Fail closed: an unpriced model would meter zero cost and never trip a cost quota.
        if self.generation_provider == "openai_compatible" and not _priced(
            self.model_prices.get(self.generation_model), require_output=True
        ):
            raise ValueError(
                "A cost_usd quota requires a REFAAT_MODEL_PRICES entry with positive input and output rates for the "
                f"configured generation model {self.generation_model!r}"
            )
        if self.embedding_provider == "openai_compatible" and not _priced(
            self.model_prices.get(self.embedding_model), require_output=False
        ):
            raise ValueError(
                "A cost_usd quota requires a REFAAT_MODEL_PRICES entry with a positive input rate for the configured "
                f"embedding model {self.embedding_model!r}"
            )

    def _validate_egress(self) -> None:
        # Nothing configured: the default is byte-identical, so skip all egress validation.
        if self.egress_policy == "none" and not self.egress_allowlist:
            return
        from refaat.domain.egress import (
            AllowlistEgressPolicy,
            EgressDestination,
            EgressPurpose,
            allowlist_from_config,
        )

        rules = allowlist_from_config(self.egress_allowlist)
        if self.egress_policy != "allowlist":
            raise ValueError("REFAAT_EGRESS_ALLOWLIST requires REFAAT_EGRESS_POLICY='allowlist'")
        policy = AllowlistEgressPolicy(rules)
        checks: list[tuple[str, str, EgressPurpose]] = []
        if self.embedding_provider == "openai_compatible":
            checks.append(
                ("REFAAT_EMBEDDING_BASE_URL", self.embedding_base_url or self.openai_base_url, EgressPurpose.MODEL)
            )
        if self.generation_provider == "openai_compatible" or self.graph_provider == "openai_compatible":
            checks.append(
                ("REFAAT_GENERATION_BASE_URL", self.generation_base_url or self.openai_base_url, EgressPurpose.MODEL)
            )
        if self.multimodal_generation == "openai_vision" or self.image_captioner == "openai_vision":
            checks.append(
                (
                    "REFAAT_MULTIMODAL_BASE_URL",
                    self.multimodal_base_url or self.generation_base_url or self.openai_base_url,
                    EgressPurpose.MODEL,
                )
            )
        if self.fine_tuning_provider == "openai":
            checks.append(
                (
                    "REFAAT_FINE_TUNING_BASE_URL",
                    self.fine_tuning_base_url or self.openai_base_url,
                    EgressPurpose.FINE_TUNING,
                )
            )
        for name, url, purpose in checks:
            decision = policy.decide(EgressDestination.from_url(url, purpose))
            if not decision.allowed:
                # Fail closed at startup: an operator enabling an allowlist without listing the model
                # host would otherwise deny its own model on the first request.
                raise ValueError(
                    f"REFAAT_EGRESS_POLICY='allowlist' does not permit the configured {name} host "
                    f"{decision.host!r}; add it to REFAAT_EGRESS_ALLOWLIST"
                )

    def _validate_numbers(self) -> None:
        positive = {
            "REFAAT_EMBEDDING_DIMENSIONS": self.embedding_dimensions,
            "REFAAT_GRAPH_MAX_ENTITIES": self.graph_max_entities,
            "REFAAT_GRAPH_MIN_TOKEN_LENGTH": self.graph_min_token_length,
            "REFAAT_DEFAULT_CHUNK_TOKENS": self.default_chunk_tokens,
            "REFAAT_QUERY_TOKEN_BUDGET": self.query_token_budget,
            "REFAAT_QUERY_TOP_K": self.query_top_k,
            "REFAAT_RRF_K": self.rrf_k,
            "REFAAT_CACHE_TTL_SECONDS": self.cache_ttl_seconds,
            "REFAAT_SESSION_COMPACTION_TURNS": self.session_compaction_turns,
            "REFAAT_RETAINED_SESSION_TURNS": self.retained_session_turns,
            "REFAAT_MAX_BODY_BYTES": self.max_body_bytes,
            "REFAAT_BLOB_GC_GRACE_SECONDS": self.blob_gc_grace_seconds,
            "REFAAT_CONNECTOR_MAX_BYTES": self.connector_max_bytes,
            "REFAAT_IMAGE_MAX_BYTES": self.image_max_bytes,
            "REFAAT_RATE_LIMIT_REQUESTS": self.rate_limit_requests,
            "REFAAT_RATE_LIMIT_WINDOW_SECONDS": self.rate_limit_window_seconds,
            "REFAAT_RATE_LIMIT_MAX_BUCKETS": self.rate_limit_max_buckets,
            "REFAAT_API_WORKERS": self.api_workers,
            "REFAAT_JOB_LEASE_SECONDS": self.job_lease_seconds,
            "REFAAT_JOB_MAX_ATTEMPTS": self.job_max_attempts,
            "REFAAT_SHADOW_RUN_RETENTION_DAYS": self.shadow_run_retention_days,
            "REFAAT_SHADOW_DEFER_TIMEOUT_SECONDS": self.shadow_defer_timeout_seconds,
            "REFAAT_OUTBOX_RETENTION_DAYS": self.outbox_retention_days,
            "REFAAT_ROUTER_LONG_QUERY_TOKENS": self.router_long_query_tokens,
            "REFAAT_ROUTER_MULTI_CLAUSE_THRESHOLD": self.router_multi_clause_threshold,
            "REFAAT_ROUTER_ENTITY_THRESHOLD": self.router_entity_threshold,
            "REFAAT_CODE_SANDBOX_MAX_NODES": self.code_sandbox_max_nodes,
            "REFAAT_CODE_SANDBOX_MAX_OUTPUT_BYTES": self.code_sandbox_max_output_bytes,
            "REFAAT_LEGAL_HOLD_LOCK_MAX_WAITERS": self.legal_hold_lock_max_waiters,
            "REFAAT_ACL_OVERFETCH_FACTOR": self.acl_overfetch_factor,
            "REFAAT_ACL_OVERFETCH_HARD_CAP": self.acl_overfetch_hard_cap,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Positive values required for: {', '.join(sorted(invalid))}")
        if self.chunk_overlap_tokens < 0 or self.chunk_overlap_tokens >= self.default_chunk_tokens:
            raise ValueError("REFAAT_CHUNK_OVERLAP_TOKENS must be smaller than the chunk size")
        if self.retained_session_turns >= self.session_compaction_turns:
            raise ValueError("REFAAT_RETAINED_SESSION_TURNS must be smaller than the compaction threshold")
        if not 0.0 <= self.semantic_cache_threshold <= 1.0:
            raise ValueError("REFAAT_SEMANTIC_CACHE_THRESHOLD must be between 0 and 1")
        if not 0.0 <= self.autocut_threshold <= 1.0:
            raise ValueError("REFAAT_AUTOCUT_THRESHOLD must be between 0 and 1")
        if not 0.0 <= self.abstain_below_score <= 1.0:
            raise ValueError("REFAAT_ABSTAIN_BELOW_SCORE must be between 0 and 1")
        if self.source_trust_weight < 0.0:
            raise ValueError("REFAAT_SOURCE_TRUST_WEIGHT must be non-negative")
        if self.kill_switch_cache_ttl_seconds < 0.0:
            raise ValueError("REFAAT_KILL_SWITCH_CACHE_TTL_SECONDS must be non-negative")
        if self.kill_switch_stale_deny_seconds < 0.0:
            raise ValueError("REFAAT_KILL_SWITCH_STALE_DENY_SECONDS must be non-negative")
        if not 0.0 <= self.bm25_k1 <= 5.0:
            raise ValueError("REFAAT_BM25_K1 must be between 0 and 5")
        if not 0.0 <= self.bm25_b <= 1.0:
            raise ValueError("REFAAT_BM25_B must be between 0 and 1")
        if not 1 <= self.max_query_variants <= 8:
            raise ValueError("REFAAT_MAX_QUERY_VARIANTS must be between 1 and 8")
        if not 0.0 <= self.compression_min_salience <= 1.0:
            raise ValueError("REFAAT_COMPRESSION_MIN_SALIENCE must be between 0 and 1")
        if not 0 <= self.sentence_window_radius <= 8:
            raise ValueError("REFAAT_SENTENCE_WINDOW_RADIUS must be between 0 and 8")
        weights = (self.semantic_weight, self.lexical_weight, self.graph_weight)
        if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
            raise ValueError("Retrieval weights must be non-negative and have a positive sum")
        if self.request_timeout_seconds <= 0 or self.worker_poll_seconds <= 0:
            raise ValueError("Timeout and polling intervals must be positive")
        if self.fine_tune_poll_seconds < 1:
            raise ValueError("REFAAT_FINE_TUNE_POLL_SECONDS must be at least 1 second")
        if not 0.0 < self.code_sandbox_max_wall_seconds <= 60.0:
            raise ValueError("REFAAT_CODE_SANDBOX_MAX_WALL_SECONDS must be between 0 and 60")
        if self.model_max_retries < 0 or self.model_retry_base_delay_seconds < 0:
            raise ValueError("Model retry settings must be non-negative")
        if not 2 <= len(self.default_language) <= 16:
            raise ValueError("REFAAT_DEFAULT_LANGUAGE must be a short language code such as 'en' or 'de'")
        if self.stopwords_by_language is not None:
            for code, words in self.stopwords_by_language.items():
                if not 2 <= len(code) <= 16:
                    raise ValueError(
                        "REFAAT_STOPWORDS_BY_LANGUAGE keys must be short language codes such as 'en' or 'it'"
                    )
                if not words or any(not word or not word.strip() for word in words):
                    raise ValueError("REFAAT_STOPWORDS_BY_LANGUAGE values must be non-empty lists of non-empty strings")
        if self.run_token_budget is not None and self.run_token_budget <= 0:
            raise ValueError("REFAAT_RUN_TOKEN_BUDGET must be positive when set")
        if self.rlm_max_input_chunks is not None and self.rlm_max_input_chunks <= 0:
            raise ValueError("REFAAT_RLM_MAX_INPUT_CHUNKS must be positive when set")
        if self.tool_timeout_seconds is not None and not (
            math.isfinite(self.tool_timeout_seconds) and self.tool_timeout_seconds > 0
        ):
            raise ValueError("REFAAT_TOOL_TIMEOUT_SECONDS must be a positive finite number when set")
        if self.tool_timeout_max_workers < 1:
            raise ValueError("REFAAT_TOOL_TIMEOUT_MAX_WORKERS must be at least 1")
        if self.circuit_breaker_failure_threshold < 1:
            raise ValueError("REFAAT_CIRCUIT_BREAKER_FAILURE_THRESHOLD must be at least 1")
        for _name, _value in (
            ("REFAAT_CIRCUIT_BREAKER_RESET_TIMEOUT_SECONDS", self.circuit_breaker_reset_timeout_seconds),
            ("REFAAT_CIRCUIT_BREAKER_HALF_OPEN_DEADLINE_SECONDS", self.circuit_breaker_half_open_deadline_seconds),
        ):
            if not (math.isfinite(_value) and _value > 0):
                raise ValueError(f"{_name} must be a positive finite number")
        if self.run_cost_budget_usd is not None:
            if self.run_cost_budget_usd <= 0:
                raise ValueError("REFAAT_RUN_COST_BUDGET_USD must be positive when set")
            # Fail closed: a cost budget over a real model that carries no usable price
            # would never trip (unpriced == zero cost), silently defeating the ceiling.
            # An empty record ({}) passes a bare membership check yet both rates
            # default to zero, so require a positive input rate for both metered roles
            # and a positive output rate for the generator (an embedder legitimately
            # has a zero output rate).
            if self.generation_provider == "openai_compatible" and not _priced(
                self.model_prices.get(self.generation_model), require_output=True
            ):
                raise ValueError(
                    "REFAAT_RUN_COST_BUDGET_USD requires a REFAAT_MODEL_PRICES entry with positive input and output "
                    f"rates for the configured generation model {self.generation_model!r}; an unpriced or all-zero "
                    "record makes the cost budget ineffective"
                )
            if self.embedding_provider == "openai_compatible" and not _priced(
                self.model_prices.get(self.embedding_model), require_output=False
            ):
                raise ValueError(
                    "REFAAT_RUN_COST_BUDGET_USD requires a REFAAT_MODEL_PRICES entry with a positive input rate for "
                    f"the configured embedding model {self.embedding_model!r}; an unpriced or all-zero record makes "
                    "the cost budget ineffective"
                )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_for_startup()
    return settings
