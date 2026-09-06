"""Persisted per-scope quotas end to end plus byte-identity, concurrency, and fail-closed behavior.

With the backend off nothing is installed, no counter table is read or written, the pinned artifact
id is unchanged, and a query runs normally. With a policy active a request counts against its quota
after the kill-switch freeze and before generation, an exhausted quota denies with 429 and audits,
two concurrent reservations against a near-exhausted counter can never both pass, a leaked
reservation the reaper reclaims never permanently blocks the tenant, and a cross-tenant counter is
isolated. The reserve-then-reconcile releases the over-reserved ceiling down to the measured spend.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select

from refaat.application.budget import CostMeter, quota_run
from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.config import Settings
from refaat.domain.audit import AUDIT_SCHEMA_VERSION, AuditEvent
from refaat.domain.models import DocumentInput
from refaat.domain.quota import (
    QuotaContext,
    QuotaDimension,
    QuotaKey,
    QuotaScope,
    quota_rules_from_config,
)
from refaat.errors import QuotaExceededError
from refaat.infrastructure.database.schema import quota_counters
from refaat.infrastructure.database.store import SQLStore
from refaat.infrastructure.quota import DatabaseQuotaPolicy
from refaat.programs.program import GroundedRagProgram

pytestmark = pytest.mark.integration

PINNED_GROUNDED_RAG_ID = "sha256:b76d73f8d956bcd38ccc9d4c24372b07b387647a189f3a9e300cded186febbb4"
_BODY = b"The checkout production release requires an approved rollback plan before deploy."


class _RecordingSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        self.events.append(event)


class _Clock:
    def __init__(self) -> None:
        self._now = datetime(2026, 1, 1, tzinfo=UTC)

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)


class _ChargingGenerator:
    """Wraps the default extractive generator and charges the active run meter, so an offline test
    can drive a token or cost quota to exhaustion deterministically."""

    def __init__(self, inner: object, *, tokens: int, model_id: str) -> None:
        self._inner = inner
        self._tokens = tokens
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate(self, request: object) -> object:
        from refaat.application.budget import active_meter as _active_meter

        meter = _active_meter()
        if meter is not None:
            meter.charge({"prompt_tokens": self._tokens, "completion_tokens": 0}, model_id=self._model_id)
        return self._inner.generate(request)  # type: ignore[attr-defined]

    def stream(self, request: object) -> object:
        return self._inner.stream(request)  # type: ignore[attr-defined]


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {
        "environment": "test",
        "database_url": f"sqlite:///{tmp_path / 'refaat.db'}",
        "auto_create_schema": True,
        "blob_root": tmp_path / "blobs",
        "connector_file_root": tmp_path / "connectors",
        "skill_root": tmp_path / "skills",
        "api_key_pepper": "test-pepper-with-enough-entropy",
        "bootstrap_token": "bootstrap-test-token",
        "default_chunk_tokens": 48,
        "chunk_overlap_tokens": 8,
        "log_json": False,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _seed(runtime: Refaat, *, slug: str = "acme") -> tuple[str, str]:
    workspace = runtime.create_workspace(slug, slug.title())
    corpus = runtime.create_corpus(workspace, "handbook")
    document = DocumentInput(source_id=f"{slug}-doc", title="doc", media_type="text/plain", content=_BODY)
    runtime.ingest(workspace, corpus, document)
    return workspace, corpus


def _counter_rows(store: SQLStore) -> int:
    with store.engine.connect() as conn:
        return int(conn.execute(select(func.count()).select_from(quota_counters)).scalar_one())


def test_pinned_grounded_rag_artifact_id_is_unchanged() -> None:
    assert GroundedRagProgram().artifact().artifact_id == PINNED_GROUNDED_RAG_ID


def test_backend_off_installs_nothing_writes_no_rows_and_serves_normally(tmp_path: Path) -> None:
    runtime = RuntimeBuilder(_settings(tmp_path)).build()
    try:
        assert runtime.quota_policy is None
        assert runtime.store.quota_management_enabled is False
        _seed(runtime)
        answer = runtime.project(workspace="acme", corpus="handbook").ask("rollback plan")
        assert answer.citations
        assert _counter_rows(runtime.store) == 0
    finally:
        runtime.close()


def test_requests_quota_admits_then_refuses_with_429_and_audits(tmp_path: Path) -> None:
    sink = _RecordingSink()
    rules = [{"scope": "workspace", "dimension": "requests", "limit": 2, "window_seconds": 3600}]
    settings = _settings(tmp_path, quota_backend="database", quota_rules=rules)
    runtime = RuntimeBuilder(settings).with_audit_sink(sink).build()
    try:
        assert runtime.quota_policy is not None
        assert runtime.store.quota_management_enabled is True
        _seed(runtime)
        project = runtime.project(workspace="acme", corpus="handbook")
        assert project.ask("rollback plan").citations
        assert project.ask("rollback plan").citations
        with pytest.raises(QuotaExceededError) as excinfo:
            project.ask("rollback plan")
        assert excinfo.value.status_code == 429
        denied = [event for event in sink.events if event.event_type == "quota.denied"]
        assert denied, "a quota denial must emit a quota.denied audit event"
        event = denied[-1]
        assert event.capability_name == QuotaDimension.REQUESTS.value
        assert event.schema_version == AUDIT_SCHEMA_VERSION == "2"
        # Redaction: only ids, the dimension, and digests, never query or evidence text.
        assert "rollback" not in (event.action_fingerprint or "")
        assert "rollback" not in event.dedup_id
    finally:
        runtime.close()


def test_token_quota_admits_then_refuses_via_the_query_path(tmp_path: Path) -> None:
    from refaat.infrastructure.generation import ExtractiveGenerator

    rules = [{"scope": "workspace", "dimension": "generation_tokens", "limit": 150, "window_seconds": 3600}]
    settings = _settings(tmp_path, quota_backend="database", quota_rules=rules, run_token_budget=100)
    generator = _ChargingGenerator(ExtractiveGenerator(), tokens=100, model_id="det-gen")
    runtime = RuntimeBuilder(settings).with_generator(generator).build()
    try:
        _seed(runtime)
        project = runtime.project(workspace="acme", corpus="handbook")
        # Distinct queries so both miss the answer cache and reach generation.
        # First run reserves the 100 ceiling (fits 150), spends 100, reconciles to spent=100.
        assert project.ask("rollback plan").citations
        # Second run would reserve another 100 ceiling on top of spent 100, exceeding 150: 429.
        with pytest.raises(QuotaExceededError):
            project.ask("production release deploy")
    finally:
        runtime.close()


def test_cost_quota_admits_then_refuses_via_the_query_path(tmp_path: Path) -> None:
    from refaat.infrastructure.generation import ExtractiveGenerator

    rules = [{"scope": "workspace", "dimension": "cost_usd", "limit": 1500, "window_seconds": 3600}]
    prices = {"det-gen": {"input_per_1k": 10.0, "output_per_1k": 10.0}}
    settings = _settings(
        tmp_path,
        quota_backend="database",
        quota_rules=rules,
        run_token_budget=1000,
        run_cost_budget_usd=0.001,
        model_prices=prices,
    )
    # Each run reserves the 0.001 USD ceiling (1000 micro-USD) and commits at least it, so the second
    # run's reservation (1000 + 1000) exceeds the 1500 micro-USD limit and is refused.
    generator = _ChargingGenerator(ExtractiveGenerator(), tokens=100, model_id="det-gen")
    runtime = RuntimeBuilder(settings).with_generator(generator).build()
    try:
        _seed(runtime)
        project = runtime.project(workspace="acme", corpus="handbook")
        assert project.ask("rollback plan").citations
        with pytest.raises(QuotaExceededError):
            project.ask("production release deploy")
    finally:
        runtime.close()


def test_concurrent_reservations_never_both_pass_a_near_exhausted_counter(tmp_path: Path) -> None:
    for trial in range(5):
        store = SQLStore(f"sqlite:///{tmp_path / f'quota-{trial}.db'}", _Clock())
        store.initialize()
        try:
            key = QuotaKey(
                scope=QuotaScope.WORKSPACE,
                scope_key="ws1",
                dimension=QuotaDimension.GENERATION_TOKENS,
                window_start=0,
                window_seconds=60,
                limit=100,
                workspace_id="ws1",
            )
            results: list[bool] = []
            lock = threading.Lock()
            barrier = threading.Barrier(2)

            def worker(
                reservation_id: str,
                *,
                store: SQLStore = store,
                key: QuotaKey = key,
                results: list[bool] = results,
                lock: threading.Lock = lock,
                barrier: threading.Barrier = barrier,
            ) -> None:
                barrier.wait()
                admitted = store.quota_reserve(key=key, reservation_id=reservation_id, estimate=60, ttl_seconds=900)
                with lock:
                    results.append(admitted)

            threads = [
                threading.Thread(target=worker, args=("a",)),
                threading.Thread(target=worker, args=("b",)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            # 60 + 60 > 100, so exactly one of the two concurrent reservations admits.
            assert sum(1 for admitted in results if admitted) == 1, results
        finally:
            store.close()


def test_reserve_then_reconcile_releases_the_over_reserved_ceiling(tmp_path: Path) -> None:
    clock = _Clock()
    store = SQLStore(f"sqlite:///{tmp_path / 'quota.db'}", clock)
    store.initialize()
    store.enable_quota_management()
    try:
        rules = quota_rules_from_config(
            [{"scope": "workspace", "dimension": "generation_tokens", "limit": 300, "window_seconds": 3600}]
        )
        policy = DatabaseQuotaPolicy(store, clock, rules=rules, lease_seconds=900)
        ctx = QuotaContext(workspace_id="ws1", corpus_validated=True)
        # Reserve the 200 ceiling, actually spend 40, reconcile down.
        meter = CostMeter(token_ceiling=200)
        with quota_run(policy, ctx, meter=meter, token_ceiling=200, cost_ceiling_usd=None, idem="run-1"):
            meter.tokens_used = 40
        key = QuotaKey(
            scope=QuotaScope.WORKSPACE,
            scope_key="ws1",
            dimension=QuotaDimension.GENERATION_TOKENS,
            window_start=int(clock.now().timestamp()) - (int(clock.now().timestamp()) % 3600),
            window_seconds=3600,
            limit=300,
            workspace_id="ws1",
        )
        # spent is 40 (not 200), so 260 headroom remains for a fresh 200-ceiling run.
        assert store.quota_reserve(key=key, reservation_id="probe", estimate=260, ttl_seconds=900) is True
    finally:
        store.close()


def test_a_leaked_reservation_is_reaped_so_the_tenant_is_not_blocked(tmp_path: Path) -> None:
    clock = _Clock()
    store = SQLStore(f"sqlite:///{tmp_path / 'quota.db'}", clock)
    store.initialize()
    store.enable_quota_management()
    try:
        rules = quota_rules_from_config(
            [{"scope": "workspace", "dimension": "generation_tokens", "limit": 100, "window_seconds": 3600}]
        )
        policy = DatabaseQuotaPolicy(store, clock, rules=rules, lease_seconds=1)
        ctx = QuotaContext(workspace_id="ws1", corpus_validated=True)
        # Simulate a crash: reserve the full ceiling, then never reconcile (no commit or release).
        reservation = policy.reserve(ctx, dimension=QuotaDimension.GENERATION_TOKENS, estimate=100, idem="crashed")
        assert reservation.active
        window = int(clock.now().timestamp()) - (int(clock.now().timestamp()) % 3600)
        key = QuotaKey(
            scope=QuotaScope.WORKSPACE,
            scope_key="ws1",
            dimension=QuotaDimension.GENERATION_TOKENS,
            window_start=window,
            window_seconds=3600,
            limit=100,
            workspace_id="ws1",
        )
        # While the lease is open the window is fully reserved: a new run is blocked.
        assert store.quota_reserve(key=key, reservation_id="blocked", estimate=1, ttl_seconds=900) is False
        clock.advance(5)
        assert store.quota_reap() == 1
        # After the reaper the window is not permanently blocked: the ceiling landed in spent and a
        # fresh run within the remaining headroom is admitted only once the window rolls, but the
        # tenant regains capacity in the next window rather than being locked out forever.
        next_window = window + 3600
        next_key = QuotaKey(
            scope=QuotaScope.WORKSPACE,
            scope_key="ws1",
            dimension=QuotaDimension.GENERATION_TOKENS,
            window_start=next_window,
            window_seconds=3600,
            limit=100,
            workspace_id="ws1",
        )
        assert store.quota_reserve(key=next_key, reservation_id="recovered", estimate=100, ttl_seconds=900) is True
    finally:
        store.close()


def test_cross_tenant_isolation_via_the_query_path(tmp_path: Path) -> None:
    rules = [{"scope": "workspace", "dimension": "requests", "limit": 1, "window_seconds": 3600}]
    settings = _settings(tmp_path, quota_backend="database", quota_rules=rules)
    runtime = RuntimeBuilder(settings).build()
    try:
        _seed(runtime, slug="acme")
        _seed(runtime, slug="pfizer")
        acme_project = runtime.project(workspace="acme", corpus="handbook")
        pfizer_project = runtime.project(workspace="pfizer", corpus="handbook")
        assert acme_project.ask("rollback plan").citations
        with pytest.raises(QuotaExceededError):
            acme_project.ask("rollback plan")
        # A different workspace has its own counter and is unaffected by acme's exhaustion.
        assert pfizer_project.ask("rollback plan").citations
    finally:
        runtime.close()


def test_unpriced_model_under_a_cost_quota_fails_startup(tmp_path: Path) -> None:
    rules = [{"scope": "workspace", "dimension": "cost_usd", "limit": 1000, "window_seconds": 3600}]
    with pytest.raises(ValueError, match="MODEL_PRICES"):
        _settings(
            tmp_path,
            quota_backend="database",
            quota_rules=rules,
            run_cost_budget_usd=0.01,
            generation_provider="openai_compatible",
        ).validate_for_startup()


def test_token_quota_without_a_budget_ceiling_fails_startup(tmp_path: Path) -> None:
    rules = [{"scope": "workspace", "dimension": "generation_tokens", "limit": 1000, "window_seconds": 3600}]
    with pytest.raises(ValueError, match="RUN_TOKEN_BUDGET"):
        _settings(tmp_path, quota_backend="database", quota_rules=rules).validate_for_startup()


def test_a_store_outage_denies_admission_fail_closed(tmp_path: Path) -> None:
    clock = _Clock()
    store = SQLStore(f"sqlite:///{tmp_path / 'quota.db'}", clock)
    store.initialize()
    try:
        rules = quota_rules_from_config(
            [{"scope": "workspace", "dimension": "requests", "limit": 10, "window_seconds": 3600}]
        )
        policy = DatabaseQuotaPolicy(store, clock, rules=rules, lease_seconds=900)

        def _boom(**_kwargs: object) -> bool:
            raise RuntimeError("counter store is down")

        store.quota_reserve = _boom  # type: ignore[method-assign]
        decision = policy.admit(QuotaContext(workspace_id="ws1"), dimension=QuotaDimension.REQUESTS, idem="x")
        assert decision.admitted is False
        assert decision.code == "unavailable"
    finally:
        store.close()


def test_multi_key_admit_is_all_or_nothing(tmp_path: Path) -> None:
    # A denied per-principal cap must not still charge the shared workspace counter, so a principal
    # past its own limit cannot burn the workspace budget through denied requests.
    clock = _Clock()
    store = SQLStore(f"sqlite:///{tmp_path / 'quota.db'}", clock)
    store.initialize()
    try:
        rules = quota_rules_from_config(
            [
                {"scope": "workspace", "dimension": "requests", "limit": 100, "window_seconds": 3600},
                {"scope": "principal", "dimension": "requests", "limit": 3, "window_seconds": 3600},
            ]
        )
        policy = DatabaseQuotaPolicy(store, clock, rules=rules, lease_seconds=900)
        ctx = QuotaContext(workspace_id="ws1", principal_id="p1")
        admitted = sum(policy.admit(ctx, dimension=QuotaDimension.REQUESTS, idem=f"r{i}").admitted for i in range(10))
        assert admitted == 3
        with store.engine.connect() as conn:
            workspace_spent = conn.execute(
                select(quota_counters.c.spent).where(quota_counters.c.scope == QuotaScope.WORKSPACE.value)
            ).scalar_one()
        assert workspace_spent == 3, "denied per-principal admissions must not charge the workspace counter"
    finally:
        store.close()


def test_ingestion_counts_against_a_requests_quota_and_denies_when_exhausted(tmp_path: Path) -> None:
    rules = [{"scope": "workspace", "dimension": "ingest_requests", "limit": 2, "window_seconds": 3600}]
    settings = _settings(tmp_path, quota_backend="database", quota_rules=rules)
    runtime = RuntimeBuilder(settings).build()
    try:
        workspace = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(workspace, "handbook")

        def _doc(source_id: str) -> DocumentInput:
            return DocumentInput(source_id=source_id, title="doc", media_type="text/plain", content=_BODY)

        runtime.ingest(workspace, corpus, _doc("a"))
        runtime.ingest(workspace, corpus, _doc("b"))
        with pytest.raises(QuotaExceededError) as excinfo:
            runtime.ingest(workspace, corpus, _doc("c"))
        assert excinfo.value.status_code == 429
    finally:
        runtime.close()


def test_ingestion_quota_uses_corpus_scope_and_skips_a_principal_rule(tmp_path: Path) -> None:
    # A corpus-scoped rule governs ingestion (it validates the corpus first), while a principal-scoped
    # rule skips it rather than denying, because ingestion carries no request principal.
    rules = [
        {"scope": "principal", "dimension": "ingest_requests", "limit": 1, "window_seconds": 3600},
        {"scope": "corpus", "dimension": "ingest_requests", "limit": 2, "window_seconds": 3600},
    ]
    settings = _settings(tmp_path, quota_backend="database", quota_rules=rules)
    runtime = RuntimeBuilder(settings).build()
    try:
        workspace = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(workspace, "handbook")

        def _doc(source_id: str) -> DocumentInput:
            return DocumentInput(source_id=source_id, title="doc", media_type="text/plain", content=_BODY)

        # The principal rule (limit 1) would deny the second ingest if it applied; it skips instead, so
        # the corpus rule (limit 2) is the only governor and admits exactly two ingests.
        runtime.ingest(workspace, corpus, _doc("a"))
        runtime.ingest(workspace, corpus, _doc("b"))
        with pytest.raises(QuotaExceededError):
            runtime.ingest(workspace, corpus, _doc("c"))
    finally:
        runtime.close()


def test_ingestion_quota_denial_emits_a_redacted_audit_event(tmp_path: Path) -> None:
    sink = _RecordingSink()
    rules = [{"scope": "workspace", "dimension": "ingest_requests", "limit": 1, "window_seconds": 3600}]
    settings = _settings(tmp_path, quota_backend="database", quota_rules=rules)
    runtime = RuntimeBuilder(settings).with_audit_sink(sink).build()
    try:
        workspace = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(workspace, "handbook")

        def _doc(source_id: str) -> DocumentInput:
            return DocumentInput(source_id=source_id, title="doc", media_type="text/plain", content=_BODY)

        runtime.ingest(workspace, corpus, _doc("a"))
        with pytest.raises(QuotaExceededError):
            runtime.ingest(workspace, corpus, _doc("b"))
        denied = [event for event in sink.events if event.event_type == "quota.denied"]
        assert denied, "an ingestion quota denial must emit a quota.denied audit event"
        event = denied[-1]
        assert event.capability_name == QuotaDimension.INGEST_REQUESTS.value
        # Redaction: only ids, the dimension, and digests, never document content.
        assert "checkout" not in (event.action_fingerprint or "")
    finally:
        runtime.close()


def test_shadow_reingest_is_not_throttled_by_an_ingest_requests_quota(tmp_path: Path) -> None:
    # An optimization shadow re-ingests already-authorized documents into a throwaway runtime; it must
    # not consume the tenant ingest_requests quota, or an optimization run would fail once the corpus
    # exceeds the limit. The base carries a workspace ingest_requests rule of limit 1, and the shadow
    # re-ingests three documents; with the shadow quota-isolated, all three ingest.
    from refaat.optimization.shadow import shadow_index

    rules = [{"scope": "workspace", "dimension": "ingest_requests", "limit": 1, "window_seconds": 3600}]
    settings = _settings(tmp_path, quota_backend="database", quota_rules=rules)
    runtime = RuntimeBuilder(settings).build()
    try:
        workspace = runtime.create_workspace("acme", "Acme")
        runtime.create_corpus(workspace, "handbook")
        project = runtime.project(workspace="acme", corpus="handbook")
        docs = [DocumentInput(source_id=f"d{i}", title="d", media_type="text/plain", content=_BODY) for i in range(3)]
        with shadow_index(project, docs, {}) as (shadow_project, _shadow_composition):
            assert len(shadow_project.list_documents()) == 3
    finally:
        runtime.close()


def test_program_run_over_http_counts_against_a_program_requests_quota(tmp_path: Path) -> None:
    # A served program run is charged at the HTTP program-run route, so an operator can cap the rate of
    # program executions independently of the query and token budgets. A denial returns 429.
    from fastapi.testclient import TestClient

    from refaat.api.app import create_app

    rules = [{"scope": "workspace", "dimension": "program_requests", "limit": 2, "window_seconds": 3600}]
    settings = _settings(
        tmp_path,
        quota_backend="database",
        quota_rules=rules,
        trusted_hosts=["testserver", "localhost", "127.0.0.1"],
        rate_limit_requests=10_000,
    )
    runtime = RuntimeBuilder(settings).build()
    ws = runtime.create_workspace("acme", "Acme")
    corpus = runtime.create_corpus(ws, "handbook")
    runtime.ingest(ws, corpus, DocumentInput(source_id="a", title="a", media_type="text/plain", content=_BODY))
    _, api_key = runtime.issue_api_key(ws, "admin", scopes=("read", "write", "admin"))
    try:
        with TestClient(create_app(runtime.settings, runtime), raise_server_exceptions=False) as client:
            headers = {"x-api-key": api_key}
            run_path = f"/v1/corpora/{corpus}/programs/grounded-rag/run"
            assert client.post(run_path, headers=headers, json={"question": "rollback plan"}).status_code == 200
            # An evaluate is a served program-execution surface, so it charges too and cannot bypass the cap.
            evaluate = client.post(
                f"/v1/corpora/{corpus}/programs/grounded-rag/evaluate",
                headers=headers,
                json={"cases": [{"query": "rollback plan", "expected_terms": ["rollback"]}]},
            )
            assert evaluate.status_code == 200, evaluate.text
            assert client.post(run_path, headers=headers, json={"question": "rollback plan"}).status_code == 429
    finally:
        runtime.close()


def test_in_process_program_run_and_evaluate_do_not_charge_program_requests(tmp_path: Path) -> None:
    # BoundProgram.run itself never charges, only the served HTTP surface does. Every internal reuse path
    # (optimization, deployment shadow and canary, nested program-skill) and the in-process evaluate all
    # go through BoundProgram.run, so this invariant is what keeps them from throttling the served quota.
    from refaat.evaluation import Dataset

    rules = [{"scope": "workspace", "dimension": "program_requests", "limit": 1, "window_seconds": 3600}]
    settings = _settings(tmp_path, quota_backend="database", quota_rules=rules)
    runtime = RuntimeBuilder(settings).build()
    try:
        _seed(runtime)
        project = runtime.project(workspace="acme", corpus="handbook")
        program = project.programs.get("grounded-rag")
        for _ in range(3):
            assert program.run(question="rollback plan").answer is not None
        dataset = Dataset.from_records(
            "eval", [{"query": "rollback plan", "expected_terms": ["rollback"]} for _ in range(3)]
        )
        report = program.evaluate(dataset)
        assert report.case_count == 3
    finally:
        runtime.close()


def test_skill_run_over_http_counts_against_a_skill_requests_quota(tmp_path: Path) -> None:
    # A served skill run is charged at the HTTP skill-run route against its own dimension, so a program
    # exposed as a skill cannot bypass the program cap and a skill-run rate can be capped on its own.
    from fastapi.testclient import TestClient

    from refaat.api.app import create_app

    rules = [{"scope": "workspace", "dimension": "skill_requests", "limit": 1, "window_seconds": 3600}]
    settings = _settings(
        tmp_path,
        quota_backend="database",
        quota_rules=rules,
        trusted_hosts=["testserver", "localhost", "127.0.0.1"],
        rate_limit_requests=10_000,
    )
    runtime = RuntimeBuilder(settings).build()
    ws = runtime.create_workspace("acme", "Acme")
    corpus = runtime.create_corpus(ws, "handbook")
    runtime.ingest(ws, corpus, DocumentInput(source_id="a", title="a", media_type="text/plain", content=_BODY))
    _, api_key = runtime.issue_api_key(ws, "admin", scopes=("read", "write", "admin"))
    try:
        with TestClient(create_app(runtime.settings, runtime), raise_server_exceptions=False) as client:
            headers = {"x-api-key": api_key}
            path = "/v1/skills/corpus_health/run"
            payload = {"corpus_id": corpus, "query": "health"}
            assert client.post(path, headers=headers, json=payload).status_code == 200
            assert client.post(path, headers=headers, json=payload).status_code == 429
    finally:
        runtime.close()


def test_in_process_skill_run_charges_skill_requests(tmp_path: Path) -> None:
    # The skill charge lives at the SkillService funnel, so it covers the in-process and MCP surfaces too,
    # not only the HTTP route. This is the mirror of the program case: SkillService.run is not reused by
    # internal machinery (a governed-ReAct tool calls the skill's own run), so charging there is safe.
    rules = [{"scope": "workspace", "dimension": "skill_requests", "limit": 1, "window_seconds": 3600}]
    settings = _settings(tmp_path, quota_backend="database", quota_rules=rules)
    runtime = RuntimeBuilder(settings).build()
    try:
        _seed(runtime)
        project = runtime.project(workspace="acme", corpus="handbook")
        assert project.run_skill("corpus_health", "health") is not None
        with pytest.raises(QuotaExceededError):
            project.run_skill("corpus_health", "health")
    finally:
        runtime.close()


def test_governed_react_skill_tool_does_not_charge_skill_requests(tmp_path: Path) -> None:
    # A governed-ReAct tool reaches the skill's own run through the ToolGovernor, never SkillService.run,
    # so it must not consume skill_requests: three program runs invoke the tool under a limit-1 rule.
    from refaat.domain.enums import SideEffectClass
    from refaat.domain.models import ContextPacket, JsonObject, ReActPolicy, ToolGrant
    from refaat.programs import GovernedReActProgram, KeywordToolPlanner

    class _Lookup:
        name = "lookup"
        description = "read-only lookup over the retrieved context"

        def __init__(self) -> None:
            self.calls = 0

        def run(self, context: ContextPacket, arguments: JsonObject) -> JsonObject:
            del arguments
            self.calls += 1
            return {"hits": len(context.hits)}

    rules = [{"scope": "workspace", "dimension": "skill_requests", "limit": 1, "window_seconds": 3600}]
    settings = _settings(tmp_path, quota_backend="database", quota_rules=rules)
    runtime = RuntimeBuilder(settings).build()
    try:
        _seed(runtime)
        project = runtime.project(workspace="acme", corpus="handbook")
        skill = _Lookup()
        policy = ReActPolicy(
            max_iterations=2,
            granted_scopes=("read:corpus",),
            side_effect_ceiling=SideEffectClass.READ_ONLY,
            grants=(ToolGrant("lookup", required_scopes=("read:corpus",), side_effect=SideEffectClass.READ_ONLY),),
        )
        program = project.programs.register(
            GovernedReActProgram(
                policy=policy, planner=KeywordToolPlanner({"lookup": ("rollback",)}), skills={"lookup": skill}
            )
        )
        for _ in range(3):
            program.run(question="how does rollback work")
        assert skill.calls == 3
    finally:
        runtime.close()


def test_principal_scoped_skill_rule_is_enforced_when_identity_is_threaded(tmp_path: Path) -> None:
    # A principal-scoped skill_requests rule is honored only when the resolved identity is threaded into
    # the request (the HTTP route now passes it, as in-process run_skill already did). Without a
    # principal the rule resolves to missing_scope and fails closed.
    from refaat.domain.authz import Principal

    rules = [{"scope": "principal", "dimension": "skill_requests", "limit": 1, "window_seconds": 3600}]
    settings = _settings(tmp_path, quota_backend="database", quota_rules=rules)
    runtime = RuntimeBuilder(settings).build()
    try:
        _seed(runtime)
        project = runtime.project(workspace="acme", corpus="handbook")
        alice = Principal(id="alice", tenant="acme")
        assert project.run_skill("corpus_health", "health", principal=alice) is not None
        with pytest.raises(QuotaExceededError):
            project.run_skill("corpus_health", "health", principal=alice)
    finally:
        runtime.close()


def test_unknown_skill_name_does_not_charge_skill_requests(tmp_path: Path) -> None:
    # The skill is resolved before the charge, so an unknown name raises NotFoundError without burning a
    # skill_requests unit; a valid run then still succeeds under a limit of one.
    from refaat.errors import NotFoundError

    rules = [{"scope": "workspace", "dimension": "skill_requests", "limit": 1, "window_seconds": 3600}]
    settings = _settings(tmp_path, quota_backend="database", quota_rules=rules)
    runtime = RuntimeBuilder(settings).build()
    try:
        _seed(runtime)
        project = runtime.project(workspace="acme", corpus="handbook")
        with pytest.raises(NotFoundError):
            project.run_skill("does_not_exist", "health")
        assert project.run_skill("corpus_health", "health") is not None
    finally:
        runtime.close()


def test_mcp_skill_run_counts_against_a_skill_requests_quota(tmp_path: Path) -> None:
    # The MCP bridge funnels through SkillService.run, so a served MCP skill run is metered too, not just
    # the HTTP route: the second call under a limit of one is denied.
    from refaat.domain.enums import SideEffectClass
    from refaat.domain.models import ReActPolicy, ToolGrant
    from refaat.mcp.bridge import MCPToolBridge

    rules = [{"scope": "workspace", "dimension": "skill_requests", "limit": 1, "window_seconds": 3600}]
    settings = _settings(tmp_path, quota_backend="database", quota_rules=rules)
    runtime = RuntimeBuilder(settings).build()
    try:
        workspace, corpus = _seed(runtime)
        policy = ReActPolicy(
            granted_scopes=("read:corpus",),
            side_effect_ceiling=SideEffectClass.READ_ONLY,
            grants=(
                ToolGrant("corpus_health", required_scopes=("read:corpus",), side_effect=SideEffectClass.READ_ONLY),
            ),
        )
        bridge = MCPToolBridge(
            registry=runtime.skills.registry,
            skill_service=runtime.skills,
            policy=policy,
            workspace_id=workspace,
            corpus_id=corpus,
        )
        assert bridge.call_tool("corpus_health", {"query": "health"}).authorized is True
        with pytest.raises(QuotaExceededError):
            bridge.call_tool("corpus_health", {"query": "health"})
    finally:
        runtime.close()


def test_optimization_enqueue_over_http_counts_against_optimization_requests(tmp_path: Path) -> None:
    # An optimization run spawns many candidate evaluations, so the enqueue is charged on its own
    # dimension at the served HTTP route: the second enqueue under a limit of one is denied.
    from fastapi.testclient import TestClient

    from refaat.api.app import create_app

    rules = [{"scope": "workspace", "dimension": "optimization_requests", "limit": 1, "window_seconds": 3600}]
    settings = _settings(
        tmp_path,
        quota_backend="database",
        quota_rules=rules,
        trusted_hosts=["testserver", "localhost", "127.0.0.1"],
        rate_limit_requests=10_000,
    )
    runtime = RuntimeBuilder(settings).build()
    ws = runtime.create_workspace("acme", "Acme")
    corpus = runtime.create_corpus(ws, "handbook")
    runtime.ingest(ws, corpus, DocumentInput(source_id="a", title="a", media_type="text/plain", content=_BODY))
    _, api_key = runtime.issue_api_key(ws, "admin", scopes=("read", "write", "admin"))
    try:
        with TestClient(create_app(runtime.settings, runtime), raise_server_exceptions=False) as client:
            headers = {"x-api-key": api_key}
            payload = {
                "corpus_id": corpus,
                "cases": [{"query": "rollback", "expected_terms": ["rollback"]}],
                "budget": 1,
            }
            first = client.post("/v1/programs/grounded-rag/optimizations", headers=headers, json=payload)
            assert first.status_code == 202, first.text
            second = client.post("/v1/programs/grounded-rag/optimizations", headers=headers, json=payload)
            assert second.status_code == 429
    finally:
        runtime.close()


def test_fine_tune_request_over_http_counts_against_fine_tune_requests(tmp_path: Path) -> None:
    # Fine-tune carries no corpus, so a workspace-scoped rule governs it: the second request is denied.
    from fastapi.testclient import TestClient

    from refaat.api.app import create_app

    rules = [{"scope": "workspace", "dimension": "fine_tune_requests", "limit": 1, "window_seconds": 3600}]
    settings = _settings(
        tmp_path,
        quota_backend="database",
        quota_rules=rules,
        trusted_hosts=["testserver", "localhost", "127.0.0.1"],
        rate_limit_requests=10_000,
    )
    runtime = RuntimeBuilder(settings).build()
    ws = runtime.create_workspace("acme", "Acme")
    _, api_key = runtime.issue_api_key(ws, "admin", scopes=("read", "write", "admin"))
    try:
        with TestClient(create_app(runtime.settings, runtime), raise_server_exceptions=False) as client:
            headers = {"x-api-key": api_key}
            payload = {"cases": [{"query": "q", "reference_answer": "a"}], "base_model": "refaat-base"}
            first = client.post("/v1/fine-tunes", headers=headers, json=payload)
            assert first.status_code == 202, first.text
            second = client.post("/v1/fine-tunes", headers=headers, json=payload)
            assert second.status_code == 429
    finally:
        runtime.close()


def test_corpus_scoped_rule_denies_a_corpus_less_fine_tune_fail_closed(tmp_path: Path) -> None:
    # Fine-tune carries no corpus, so a corpus-scoped fine_tune_requests rule resolves to a missing scope
    # and denies fail closed rather than minting a row; the workspace-scoped test above proves the govern
    # path, so this locks the corpus-optional deny branch.
    from fastapi.testclient import TestClient

    from refaat.api.app import create_app

    rules = [{"scope": "corpus", "dimension": "fine_tune_requests", "limit": 5, "window_seconds": 3600}]
    settings = _settings(
        tmp_path,
        quota_backend="database",
        quota_rules=rules,
        trusted_hosts=["testserver", "localhost", "127.0.0.1"],
        rate_limit_requests=10_000,
    )
    runtime = RuntimeBuilder(settings).build()
    ws = runtime.create_workspace("acme", "Acme")
    _, api_key = runtime.issue_api_key(ws, "admin", scopes=("read", "write", "admin"))
    try:
        with TestClient(create_app(runtime.settings, runtime), raise_server_exceptions=False) as client:
            headers = {"x-api-key": api_key}
            payload = {"cases": [{"query": "q", "reference_answer": "a"}], "base_model": "refaat-base"}
            first = client.post("/v1/fine-tunes", headers=headers, json=payload)
            assert first.status_code == 429
    finally:
        runtime.close()
