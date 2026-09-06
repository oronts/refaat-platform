from __future__ import annotations

import threading
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlparse

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError

from refaat.api.app import create_app
from refaat.api.middleware import InMemoryRateLimiter, client_ip
from refaat.application.blob_gc import collect_unreferenced_blobs
from refaat.application.runtime import Refaat, RuntimeBuilder
from refaat.application.services import IngestionService
from refaat.application.worker import DurableWorker
from refaat.config import Settings
from refaat.domain.enums import EvaluationStatus
from refaat.domain.models import (
    CacheEntry,
    DocumentInput,
    GenerationRequest,
    GenerationResult,
    QueryRequest,
    QueryResponse,
)
from refaat.errors import ConflictError, NotFoundError
from refaat.infrastructure.database.schema import schedules
from refaat.infrastructure.events import OutboxPublisher
from refaat.infrastructure.plugins import PluginRegistry
from refaat.skills.registry import SkillRegistry

pytestmark = pytest.mark.integration


def _policy_document(content: bytes = b"Release approval requires two reviewers.") -> DocumentInput:
    return DocumentInput(source_id="policy.txt", title="Policy", media_type="text/plain", content=content)


def _doc(source_id: str, content: bytes) -> DocumentInput:
    return DocumentInput(source_id=source_id, title=source_id, media_type="text/plain", content=content)


class _HallucinatingGenerator:
    model_id = "hallucinating-test"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return GenerationResult(answer="Release approval needs two reviewers.", cited_chunk_ids=("chunk_missing",))


def test_query_survives_hallucinated_citation(runtime: Refaat, workspace_id: str, corpus_id: str) -> None:
    runtime.ingest(
        workspace_id,
        corpus_id,
        DocumentInput(
            source_id="policy.txt",
            title="Policy",
            media_type="text/plain",
            content=b"Release approval requires two reviewers before deployment to production.",
        ),
    )
    runtime.queries.generator = _HallucinatingGenerator()

    response = runtime.query(
        QueryRequest(workspace_id=workspace_id, corpus_id=corpus_id, query="release approval reviewers")
    )

    assert response.answer
    assert response.citations, "an invented citation must fall back to grounded evidence, not fail the query"


def test_worker_survives_lost_lease_on_completion(runtime: Refaat, workspace_id: str, monkeypatch) -> None:
    worker = DurableWorker(runtime)
    worker.register("test.noop", lambda workspace_id, payload: {"ok": True})
    runtime.store.enqueue_job(workspace_id, "test.noop", {})

    def _lease_lost(*args: object, **kwargs: object) -> None:
        raise ConflictError("lease reclaimed by another worker")

    monkeypatch.setattr(runtime.store, "complete_job", _lease_lost)

    assert worker.run_once() is True


def test_compaction_only_removes_named_turns(runtime: Refaat, workspace_id: str, corpus_id: str) -> None:
    session_id = runtime.store.create_session(workspace_id, corpus_id, user_id=None, goal=None, metadata_value={})
    for index in range(5):
        runtime.store.append_session_turn(workspace_id, session_id, "user", f"turn {index}", {})
    turns = runtime.store.list_session_turns(workspace_id, session_id)
    removed = [str(turns[0]["id"]), str(turns[1]["id"])]

    runtime.store.compact_session(workspace_id, session_id, summary="prior context", remove_turn_ids=removed)

    remaining = {str(turn["id"]) for turn in runtime.store.list_session_turns(workspace_id, session_id)}
    assert remaining == {str(turns[2]["id"]), str(turns[3]["id"]), str(turns[4]["id"])}


def test_sqlite_busy_timeout_is_configured(runtime: Refaat) -> None:
    with runtime.store.engine.connect() as conn:
        timeout = conn.exec_driver_sql("PRAGMA busy_timeout").scalar()
    assert int(timeout) == 5000


def test_skill_registry_defaults_to_deny_all_plugins(monkeypatch) -> None:
    class _InjectedSkill:
        name = "injected_skill"
        description = "must not auto-load without an allowlist"

        def run(self, context: object, arguments: object) -> dict[str, object]:
            return {}

    class _EntryPoint:
        name = "injected_skill"

        @staticmethod
        def load() -> type[_InjectedSkill]:
            return _InjectedSkill

    monkeypatch.setattr(
        "refaat.infrastructure.plugins.entry_points",
        lambda group: [_EntryPoint()] if group == "refaat.skills" else [],
    )

    with pytest.raises(NotFoundError):
        SkillRegistry().get("injected_skill")
    assert SkillRegistry(PluginRegistry({"injected_skill"})).get("injected_skill").name == "injected_skill"


def test_outbox_claim_lease_and_delivery_are_idempotent(runtime: Refaat, workspace_id: str, corpus_id: str) -> None:
    runtime.ingest(workspace_id, corpus_id, _policy_document())
    pending = runtime.store.pending_outbox(50)
    assert len(pending) == 1
    event_id = str(pending[0]["id"])

    claimed = runtime.store.claim_outbox("w1", 60, 50)
    assert [str(event["id"]) for event in claimed] == [event_id]
    assert runtime.store.claim_outbox("w2", 60, 50) == [], "a leased event must not be re-claimable"

    assert runtime.store.mark_outbox_delivered(event_id, "w1") is True
    assert runtime.store.mark_outbox_delivered(event_id, "w1") is False, "delivery must be idempotent"
    assert runtime.store.claim_outbox("w3", 60, 50) == [], "a delivered event must not be re-claimable"


def test_outbox_finalize_requires_claim_ownership(runtime: Refaat, workspace_id: str, corpus_id: str) -> None:
    runtime.ingest(workspace_id, corpus_id, _policy_document())
    event_id = str(runtime.store.pending_outbox(50)[0]["id"])

    claimed = runtime.store.claim_outbox("owner", 60, 50)
    assert [str(event["id"]) for event in claimed] == [event_id]

    assert runtime.store.mark_outbox_delivered(event_id, "intruder") is False, "a non-owner cannot finalize the event"
    runtime.store.mark_outbox_failed(event_id, "intruder", "boom", 60)
    assert runtime.store.claim_outbox("intruder", 60, 50) == [], "a non-owner mark_failed must not release the lease"

    assert runtime.store.mark_outbox_delivered(event_id, "owner") is True, "the lease owner finalizes the event"


def _force_outbox_available(runtime: Refaat) -> None:
    from refaat.infrastructure.database.schema import outbox

    with runtime.store.engine.begin() as conn:
        conn.execute(
            update(outbox)
            .where(outbox.c.delivered_at.is_(None))
            .values(available_at=runtime.store.clock.now(), claimed_by=None, claimed_until=None)
        )


def test_outbox_fanout_does_not_reinvoke_a_succeeded_sink_on_retry(
    runtime: Refaat, workspace_id: str, corpus_id: str
) -> None:
    runtime.ingest(workspace_id, corpus_id, _policy_document())
    first_calls: list[str | None] = []
    state = {"fail": True}

    class FirstSink:
        def emit(self, topic: str, payload: dict[str, object], *, event_id: str | None = None) -> None:
            first_calls.append(event_id)

    class SecondSink:
        def emit(self, topic: str, payload: dict[str, object], *, event_id: str | None = None) -> None:
            if state["fail"]:
                raise RuntimeError("second sink is down")

    publisher = OutboxPublisher(runtime.store, [FirstSink(), SecondSink()], worker_id="w1")
    assert publisher.publish(50) == 0, "a failing sink leaves the event undelivered"
    assert len(first_calls) == 1, "the first sink succeeded once"

    state["fail"] = False
    _force_outbox_available(runtime)
    assert publisher.publish(50) == 1, "the retry delivers once the second sink recovers"
    assert len(first_calls) == 1, "the already-succeeded first sink is not re-invoked on retry"


def test_outbox_delivery_ledger_survives_a_fresh_publisher(runtime: Refaat, workspace_id: str, corpus_id: str) -> None:
    runtime.ingest(workspace_id, corpus_id, _policy_document())
    first_calls: list[str | None] = []
    state = {"fail": True}

    class FirstSink:
        def emit(self, topic: str, payload: dict[str, object], *, event_id: str | None = None) -> None:
            first_calls.append(event_id)

    class SecondSink:
        def emit(self, topic: str, payload: dict[str, object], *, event_id: str | None = None) -> None:
            if state["fail"]:
                raise RuntimeError("second sink is down")

    event_id = str(runtime.store.pending_outbox(50)[0]["id"])
    assert OutboxPublisher(runtime.store, [FirstSink(), SecondSink()], worker_id="w1").publish(50) == 0
    assert len(first_calls) == 1, "the first sink succeeded once"
    assert runtime.store.delivered_outbox_sinks(event_id) == {"FirstSink"}, "the success is recorded durably"

    state["fail"] = False
    _force_outbox_available(runtime)
    # A brand-new publisher instance models a worker restart: the in-memory tracking
    # is gone, but the durable ledger still skips the already-succeeded first sink and
    # invokes only the not-yet-delivered second sink.
    fresh = OutboxPublisher(runtime.store, [FirstSink(), SecondSink()], worker_id="w2")
    assert fresh.publish(50) == 1, "a fresh publisher delivers once the second sink recovers"
    assert len(first_calls) == 1, "the ledger prevents re-invoking the succeeded sink after a restart"


def test_purge_delivered_outbox_bounds_the_ledger(runtime: Refaat, workspace_id: str, corpus_id: str) -> None:
    from refaat.infrastructure.database.schema import outbox, outbox_deliveries

    runtime.ingest(workspace_id, corpus_id, _policy_document())
    event_id = str(runtime.store.pending_outbox(50)[0]["id"])
    runtime.store.record_outbox_delivery(event_id, "logging")
    assert runtime.store.delivered_outbox_sinks(event_id) == {"logging"}

    now = runtime.store.clock.now()
    with runtime.store.engine.begin() as conn:
        conn.execute(update(outbox).where(outbox.c.id == event_id).values(delivered_at=now - timedelta(days=30)))

    assert runtime.store.purge_delivered_outbox(7) == 1
    with runtime.store.engine.connect() as conn:
        assert conn.execute(outbox.select().where(outbox.c.id == event_id)).first() is None
        remaining = conn.execute(outbox_deliveries.select().where(outbox_deliveries.c.event_id == event_id)).first()
    assert remaining is None, "the ledger rows cascade away with the purged event"


def test_purge_delivered_outbox_keeps_undelivered_and_recent_events(
    runtime: Refaat, workspace_id: str, corpus_id: str
) -> None:
    runtime.ingest(workspace_id, corpus_id, _policy_document())
    # The single ingest event is still undelivered, so retention never drops it.
    assert runtime.store.purge_delivered_outbox(0) == 0
    assert len(runtime.store.pending_outbox(50)) == 1


def test_outbox_publisher_calls_positional_only_event_id_sink_without_keyword(
    runtime: Refaat, workspace_id: str, corpus_id: str
) -> None:
    runtime.ingest(workspace_id, corpus_id, _policy_document())
    received: list[tuple[str, str | None]] = []

    class PositionalOnlySink:
        def emit(self, topic: str, payload: dict[str, object], event_id: str | None = None, /) -> None:
            received.append((topic, event_id))

    delivered = OutboxPublisher(runtime.store, [PositionalOnlySink()], worker_id="w1").publish(50)

    assert delivered == 1
    assert received == [("document.indexed", None)], "a positional-only emit is called with two args, no keyword"


def test_outbox_publisher_signature_gates_event_id(runtime: Refaat, workspace_id: str, corpus_id: str) -> None:
    runtime.ingest(workspace_id, corpus_id, _policy_document())
    event_id = str(runtime.store.pending_outbox(50)[0]["id"])
    legacy: list[tuple[str, object]] = []
    modern: list[tuple[str, str | None]] = []

    class LegacySink:
        def emit(self, topic: str, payload: dict[str, object]) -> None:
            legacy.append((topic, payload))

    class ModernSink:
        def emit(self, topic: str, payload: dict[str, object], *, event_id: str | None = None) -> None:
            modern.append((topic, event_id))

    delivered = OutboxPublisher(runtime.store, [LegacySink(), ModernSink()]).publish(50)

    assert delivered == 1
    assert len(legacy) == 1, "a legacy emit(topic, payload) sink still receives the event"
    assert legacy[0][0] == "document.indexed"
    assert modern == [("document.indexed", event_id)], "an opt-in sink receives the outbox id"


def test_outbox_claim_is_at_most_once_under_concurrency(runtime: Refaat, workspace_id: str, corpus_id: str) -> None:
    runtime.ingest(workspace_id, corpus_id, _policy_document())
    barrier = threading.Barrier(2)
    claims: list[list[str]] = []
    lock = threading.Lock()

    def worker(worker_id: str) -> None:
        barrier.wait()
        events = runtime.store.claim_outbox(worker_id, 60, 50)
        with lock:
            claims.append([str(event["id"]) for event in events])

    threads = [threading.Thread(target=worker, args=(f"w{index}",)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    claimed_ids = [event_id for batch in claims for event_id in batch]
    assert len(claimed_ids) == len(set(claimed_ids)) == 1, "exactly one worker may claim the event"


def _force_due(runtime: Refaat, schedule_id: str) -> None:
    with runtime.store.engine.begin() as conn:
        conn.execute(
            update(schedules).where(schedules.c.id == schedule_id).values(next_run_at=runtime.store.clock.now())
        )


def test_schedule_dispatch_is_idempotent_and_atomic(runtime: Refaat, workspace_id: str) -> None:
    schedule_id = runtime.store.create_schedule(workspace_id, "nightly", "cache.purge", {}, interval_seconds=3600)
    _force_due(runtime, schedule_id)

    assert runtime.store.dispatch_due_schedules() == 1
    assert runtime.store.dispatch_due_schedules() == 0, "an advanced schedule must not re-dispatch"


def test_schedule_dispatch_claims_once_under_concurrency(runtime: Refaat, workspace_id: str) -> None:
    schedule_id = runtime.store.create_schedule(workspace_id, "nightly", "cache.purge", {}, interval_seconds=3600)
    _force_due(runtime, schedule_id)
    barrier = threading.Barrier(2)
    dispatched: list[int] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        count = runtime.store.dispatch_due_schedules()
        with lock:
            dispatched.append(count)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sum(dispatched) == 1, "the guarded UPDATE claim enqueues the due schedule exactly once"


def test_replace_document_maps_first_write_race_to_conflict(runtime: Refaat, workspace_id: str, corpus_id: str) -> None:
    def call() -> object:
        try:
            return runtime.store.replace_document(
                workspace_id=workspace_id,
                corpus_id=corpus_id,
                document_id=f"doc_{threading.get_ident()}",
                source_id="race.txt",
                title="Race",
                media_type="text/plain",
                checksum="deadbeef",
                blob_uri="blob://race",
                document_metadata={},
                chunk_rows=[],
                entity_rows=[],
                relation_rows=[],
            )
        except BaseException as exc:
            return exc

    barrier = threading.Barrier(2)
    outcomes: list[object] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        result = call()
        with lock:
            outcomes.append(result)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not any(isinstance(outcome, IntegrityError) for outcome in outcomes), "a raw IntegrityError must not bubble"
    conflicts = [outcome for outcome in outcomes if isinstance(outcome, ConflictError)]
    successes = [outcome for outcome in outcomes if isinstance(outcome, int)]
    assert len(successes) >= 1
    assert len(conflicts) == 2 - len(successes)


def test_ingest_retries_once_on_first_write_conflict(
    runtime: Refaat, workspace_id: str, corpus_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_replace = runtime.store.replace_document
    calls = {"n": 0}

    def flaky_replace(**kwargs: object) -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConflictError("Document source already exists")
        return real_replace(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(runtime.store, "replace_document", flaky_replace)

    result = runtime.ingest(workspace_id, corpus_id, _policy_document(b"Retry this once."))

    assert result.changed is True
    assert calls["n"] == 2, "the first ConflictError triggers exactly one bounded retry"
    rows = runtime.store.list_documents(workspace_id, corpus_id)
    assert len([doc for doc in rows if doc["source_id"] == "policy.txt"]) == 1

    live_row = runtime.store.get_document_by_source(workspace_id, corpus_id, "policy.txt")
    assert live_row is not None
    live_path = Path(urlparse(str(live_row["blob_uri"])).path)
    blob_files = [path for path in runtime.settings.blob_root.rglob("*") if path.is_file()]
    assert len(blob_files) == 2, "the losing attempt's orphan is left for the deferred sweep"

    reclaimed = collect_unreferenced_blobs(
        blob_store=runtime.ingestion.blob_store,
        store=runtime.store,
        now=runtime.store.clock.now() + timedelta(hours=1),
        grace_seconds=0,
    )
    assert reclaimed == 1, "the deferred sweep reclaims exactly the losing attempt's orphan"
    assert live_path.exists(), "the committed document's blob is untouched"
    survivors = [path for path in runtime.settings.blob_root.rglob("*") if path.is_file()]
    assert survivors == [live_path]


def test_concurrent_first_write_yields_one_document(runtime: Refaat, workspace_id: str, corpus_id: str) -> None:
    barrier = threading.Barrier(2)
    results: list[object] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        try:
            outcome: object = runtime.ingest(workspace_id, corpus_id, _policy_document(b"Shared brand new source."))
        except BaseException as exc:
            outcome = exc
        with lock:
            results.append(outcome)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert all(not isinstance(result, BaseException) for result in results), "both concurrent ingests succeed"
    rows = [doc for doc in runtime.store.list_documents(workspace_id, corpus_id) if doc["source_id"] == "policy.txt"]
    assert len(rows) == 1, "the unique (corpus, source) constraint leaves exactly one document row"


def test_safe_metadata_is_recursive_and_round_trips_equal() -> None:
    redact = IngestionService._safe_metadata
    nested = {
        "api": {"client_secret": "x", "note": "ok"},
        "access_token": "y",
        "token_count": 5,
        "author": "jane",
        "items": [{"password": "p", "keep": 1}, {"session_token": "s", "value": 2}],
    }

    cleaned = redact(nested)

    assert cleaned == {
        "api": {"note": "ok"},
        "token_count": 5,
        "author": "jane",
        "items": [{"keep": 1}, {"value": 2}],
    }
    benign = {"title": "Report", "author": "jane", "nested": {"pages": 3, "tags": ["a", "b"]}, "token_count": 9}
    assert redact(benign) == benign, "sensitive-free metadata must round-trip equal"


def test_safe_metadata_uses_token_boundaries_and_recurses_into_tuples() -> None:
    redact = IngestionService._safe_metadata
    value = {
        "x-api-key": "sekret",
        "creds": ("user", {"password": "p", "role": "admin"}),
        "tokenizer": "bert",
        "secretary": "jane",
        "credentialing_status": "active",
        "token_count": 12,
        "author": "kim",
    }

    cleaned = redact(value)

    assert "x-api-key" not in cleaned, "the hyphenated api key spelling is stripped"
    assert cleaned["creds"] == ["user", {"role": "admin"}], "a password nested in a tuple is stripped"
    assert cleaned["tokenizer"] == "bert", "a benign key that merely contains 'token' survives"
    assert cleaned["secretary"] == "jane", "a benign key that merely contains 'secret' survives"
    assert cleaned["credentialing_status"] == "active", "a benign key that merely contains 'credential' survives"
    assert cleaned["token_count"] == 12
    assert cleaned["author"] == "kim"


def test_rate_limiter_keys_on_client_ip_not_api_key(runtime: Refaat, settings: Settings) -> None:
    low = settings.model_copy(update={"rate_limit_requests": 2})
    with TestClient(create_app(low, runtime), raise_server_exceptions=False) as api:
        statuses = [
            api.get("/v1/nonexistent", headers={"x-api-key": f"forged-{index}"}).status_code for index in range(3)
        ]
    assert statuses[-1] == 429, "distinct forged api keys from one IP share a bucket, closing the bypass"


def test_in_memory_rate_limiter_reuses_capacity_once_windows_expire(monkeypatch: pytest.MonkeyPatch) -> None:
    from refaat.api import middleware

    now = {"t": 1000.0}
    monkeypatch.setattr(middleware.time, "monotonic", lambda: now["t"])
    limiter = InMemoryRateLimiter(requests=1, window_seconds=10, max_buckets=5)
    for index in range(5):
        assert limiter.consume(f"ip-{index}") is True, "the first max_buckets active addresses are admitted"
    assert limiter.consume("ip-overflow") is False, "a new address is refused while every bucket is active"

    now["t"] = 1100.0
    assert limiter.consume("fresh") is True, "once the active windows expire the swept capacity is reused"
    assert len(limiter.values) <= limiter.max_buckets, "the active-bucket map stays bounded"


def test_in_memory_rate_limiter_does_not_evict_active_buckets(monkeypatch: pytest.MonkeyPatch) -> None:
    from refaat.api import middleware

    now = {"t": 2000.0}
    monkeypatch.setattr(middleware.time, "monotonic", lambda: now["t"])
    limiter = InMemoryRateLimiter(requests=1, window_seconds=60, max_buckets=5)
    for index in range(5):
        now["t"] += 0.001
        assert limiter.consume(f"ip-{index}") is True

    # A limited address must stay limited even while an attacker cycles fresh addresses
    # to try to force its active bucket out of the map.
    assert limiter.consume("ip-0") is False, "the first address is already at its limit"
    for index in range(5, 100):
        now["t"] += 0.001
        assert limiter.consume(f"ip-{index}") is False, "no new address evicts an active bucket to make room"
    assert limiter.consume("ip-0") is False, "the active bucket was never evicted, so its count still holds"
    assert len(limiter.values) <= limiter.max_buckets, "the map is hard-bounded without evicting active buckets"


def test_client_ip_prefers_socket_peer_and_untrusted_forwarded_entry() -> None:
    direct = {"client": ("203.0.113.9", 5000), "headers": [(b"x-forwarded-for", b"1.1.1.1, 2.2.2.2")]}
    assert client_ip(direct, []) == "203.0.113.9", "an untrusted peer ignores forwarded headers"

    proxied = {"client": ("10.0.0.1", 5000), "headers": [(b"x-forwarded-for", b"9.9.9.9, 8.8.8.8, 10.0.0.1")]}
    assert client_ip(proxied, ["10.0.0.1"]) == "8.8.8.8", "a trusted proxy uses the right-most untrusted entry"

    assert client_ip({"client": None, "headers": []}, []) == "unknown"


def test_client_ip_canonicalizes_and_drops_malformed_and_duplicate_forwarded_entries() -> None:
    scope = {
        "client": ("10.0.0.1", 5000),
        "headers": [
            (b"x-forwarded-for", b"not-an-ip, 2001:DB8::1"),
            (b"x-forwarded-for", b"203.0.113.5"),
        ],
    }
    assert client_ip(scope, ["10.0.0.1"]) == "203.0.113.5", (
        "every forwarded header is read and the right-most valid untrusted entry wins"
    )

    ipv6 = {"client": ("10.0.0.1", 5000), "headers": [(b"x-forwarded-for", b"2001:DB8::1")]}
    assert client_ip(ipv6, ["10.0.0.1"]) == "2001:db8::1", "an IPv6 forwarded entry is canonicalized to one spelling"

    only_malformed = {"client": ("10.0.0.1", 5000), "headers": [(b"x-forwarded-for", b"garbage")]}
    assert client_ip(only_malformed, ["10.0.0.1"]) == "10.0.0.1", "a malformed forwarded entry falls back to the peer"


def test_superseded_blob_is_not_deleted_inline(runtime: Refaat, workspace_id: str, corpus_id: str) -> None:
    runtime.ingest(workspace_id, corpus_id, _policy_document(b"Original policy content."))
    old_row = runtime.store.get_document_by_source(workspace_id, corpus_id, "policy.txt")
    assert old_row is not None
    old_path = Path(urlparse(str(old_row["blob_uri"])).path)
    assert old_path.exists(), "the first ingest wrote its blob"

    runtime.ingest(workspace_id, corpus_id, _policy_document(b"A revised, different policy content."))

    assert old_path.exists(), "the superseded blob is left for the deferred sweep, not deleted inline"


def test_gc_reclaims_a_superseded_blob(runtime: Refaat, workspace_id: str, corpus_id: str) -> None:
    runtime.ingest(workspace_id, corpus_id, _policy_document(b"Original policy content."))
    old_row = runtime.store.get_document_by_source(workspace_id, corpus_id, "policy.txt")
    assert old_row is not None
    old_path = Path(urlparse(str(old_row["blob_uri"])).path)

    runtime.ingest(workspace_id, corpus_id, _policy_document(b"A revised, different policy content."))
    current_row = runtime.store.get_document_by_source(workspace_id, corpus_id, "policy.txt")
    assert current_row is not None
    current_path = Path(urlparse(str(current_row["blob_uri"])).path)

    reclaimed = collect_unreferenced_blobs(
        blob_store=runtime.ingestion.blob_store,
        store=runtime.store,
        now=runtime.store.clock.now() + timedelta(hours=1),
        grace_seconds=0,
    )

    assert reclaimed == 1
    assert not old_path.exists(), "the orphaned blob is reclaimed by the gc"
    assert current_path.exists(), "the live blob is untouched"


def test_gc_never_deletes_a_referenced_blob(runtime: Refaat, workspace_id: str, corpus_id: str) -> None:
    runtime.ingest(workspace_id, corpus_id, _policy_document(b"Live policy content."))
    row = runtime.store.get_document_by_source(workspace_id, corpus_id, "policy.txt")
    assert row is not None
    live_path = Path(urlparse(str(row["blob_uri"])).path)
    assert live_path.exists()

    reclaimed = collect_unreferenced_blobs(
        blob_store=runtime.ingestion.blob_store,
        store=runtime.store,
        now=runtime.store.clock.now() + timedelta(hours=1),
        grace_seconds=0,
    )

    assert reclaimed == 0
    assert live_path.exists(), "a referenced blob is never deleted, even past the grace window"


def test_gc_keeps_a_blob_within_the_grace_window(runtime: Refaat, workspace_id: str, corpus_id: str) -> None:
    runtime.ingest(workspace_id, corpus_id, _policy_document(b"First policy content."))
    old_row = runtime.store.get_document_by_source(workspace_id, corpus_id, "policy.txt")
    assert old_row is not None
    old_path = Path(urlparse(str(old_row["blob_uri"])).path)

    runtime.ingest(workspace_id, corpus_id, _policy_document(b"Second, different policy content."))
    assert old_path.exists()

    reclaimed = collect_unreferenced_blobs(
        blob_store=runtime.ingestion.blob_store,
        store=runtime.store,
        now=runtime.store.clock.now(),
        grace_seconds=86_400,
    )

    assert reclaimed == 0
    assert old_path.exists(), "an orphan younger than the grace window is kept"


def test_gc_worker_job_is_registered_and_reclaims(runtime: Refaat, workspace_id: str, corpus_id: str) -> None:
    runtime.ingest(workspace_id, corpus_id, _policy_document(b"Original policy content."))
    old_row = runtime.store.get_document_by_source(workspace_id, corpus_id, "policy.txt")
    assert old_row is not None
    old_path = Path(urlparse(str(old_row["blob_uri"])).path)
    runtime.ingest(workspace_id, corpus_id, _policy_document(b"A revised, different policy content."))

    worker = DurableWorker(runtime)
    assert "blob.gc" in worker.handlers
    runtime.settings.blob_gc_grace_seconds = 0

    result = worker._gc_blobs(workspace_id, {})

    assert result == {"blobs_reclaimed": 1}
    assert not old_path.exists()


def test_replace_document_advances_corpus_version_by_one_per_change(
    runtime: Refaat, workspace_id: str, corpus_id: str
) -> None:
    start = runtime.store.get_corpus(workspace_id, corpus_id).version
    first = runtime.ingest(workspace_id, corpus_id, _doc("a.txt", b"Alpha document content."))
    second = runtime.ingest(workspace_id, corpus_id, _doc("b.txt", b"Beta document content."))

    assert second.corpus_version - first.corpus_version == 1, "each change advances the version by exactly one"
    assert runtime.store.get_corpus(workspace_id, corpus_id).version == start + 2


def test_concurrent_replace_document_has_no_lost_version_increment(
    runtime: Refaat, workspace_id: str, corpus_id: str
) -> None:
    start = runtime.store.get_corpus(workspace_id, corpus_id).version
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        barrier.wait()
        try:
            runtime.ingest(workspace_id, corpus_id, _doc(f"c{index}.txt", f"Distinct content number {index}.".encode()))
        except BaseException as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, f"both concurrent ingests succeed: {errors}"
    assert runtime.store.get_corpus(workspace_id, corpus_id).version == start + 2, (
        "the atomic increment loses no bump: two distinct documents advance the version by two"
    )


def test_event_sink_is_closed_when_a_later_build_step_fails(settings: Settings) -> None:
    closed: list[bool] = []

    class ClientOwningSink:
        def emit(self, topic: str, payload: dict[str, object], *, event_id: str | None = None) -> None:
            del topic, payload, event_id

        def close(self) -> None:
            closed.append(True)

    def _boom() -> object:
        raise RuntimeError("later build step failed")

    builder = RuntimeBuilder(settings).with_event_sink(ClientOwningSink())
    builder._reranker = _boom  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="later build step failed"):
        builder.build()

    assert closed == [True], "a sink registered before the failing build step is still closed on cleanup"


def test_event_sink_is_closed_when_plugin_discovery_fails(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[bool] = []

    class ClientOwningSink:
        def emit(self, topic: str, payload: dict[str, object], *, event_id: str | None = None) -> None:
            del topic, payload, event_id

        def close(self) -> None:
            closed.append(True)

    def _boom(self: PluginRegistry, group: str) -> object:
        raise RuntimeError("plugin discovery failed")

    # Plugin discovery runs before the custom-sink handling, so its failure must still
    # close a sink injected via with_event_sink.
    monkeypatch.setattr(PluginRegistry, "discover", _boom)
    builder = RuntimeBuilder(settings).with_event_sink(ClientOwningSink())

    with pytest.raises(RuntimeError, match="plugin discovery failed"):
        builder.build()

    assert closed == [True], "a sink injected before plugin discovery is still closed when discovery fails"


def test_finish_evaluation_run_is_monotonic(runtime: Refaat, workspace_id: str, corpus_id: str) -> None:
    store = runtime.store
    suite_id = store.create_evaluation_suite(workspace_id, "suite", [{"question": "q", "answer": "a"}])
    run_id = store.create_evaluation_run(workspace_id, suite_id, corpus_id)
    store.finish_evaluation_run(workspace_id, run_id, metrics={"score": 1.0}, results=[{"ok": True}])
    assert store.get_evaluation_run(workspace_id, run_id)["status"] == EvaluationStatus.SUCCEEDED

    store.finish_evaluation_run(workspace_id, run_id, metrics={}, results=[], error="late failure")
    run = store.get_evaluation_run(workspace_id, run_id)
    assert run["status"] == EvaluationStatus.SUCCEEDED, "a straggler failure cannot corrupt a committed success"
    assert run["error"] is None


def test_put_cache_reraises_a_non_scope_conflict(runtime: Refaat, workspace_id: str, corpus_id: str) -> None:
    store = runtime.store
    now = store.clock.now()
    response = QueryResponse(
        answer="a", citations=(), routes=(), cached=False, corpus_version=1, trace_id="t", usage={}
    )

    def entry(cache_id: str, query_hash: str) -> CacheEntry:
        return CacheEntry(
            id=cache_id,
            workspace_id=workspace_id,
            corpus_id=corpus_id,
            corpus_version=1,
            user_scope="anonymous",
            query="q",
            query_hash=query_hash,
            query_embedding=(0.0,),
            response=response,
            expires_at=now + timedelta(hours=1),
            created_at=now,
        )

    store.put_cache(entry("cache_shared", "scope_a"))
    with pytest.raises(IntegrityError):
        store.put_cache(entry("cache_shared", "scope_b"))
