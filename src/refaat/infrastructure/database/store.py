from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from sqlalchemy import (
    Connection,
    Engine,
    Table,
    and_,
    create_engine,
    delete,
    event,
    exists,
    func,
    insert,
    inspect,
    or_,
    select,
    text,
    update,
)
from sqlalchemy.engine import RowMapping
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.pool import NullPool

from refaat.clock import Clock
from refaat.core.text import sha256_text, stable_digest
from refaat.domain.audit import AuditEvent
from refaat.domain.authz import canonical_acl, permission_fingerprint
from refaat.domain.enums import EvaluationStatus, JobStatus, MessageStatus
from refaat.domain.erasure import ErasureReceipt
from refaat.domain.jobs import RESERVED_JOB_KINDS
from refaat.domain.kill_switch import KillSwitchSet
from refaat.domain.legal_hold import LegalHoldError
from refaat.domain.models import (
    AgentMessage,
    CacheEntry,
    Chunk,
    Corpus,
    Entity,
    HarnessSnapshot,
    JsonObject,
    MetadataFilter,
    QueryResponse,
    Relation,
    Workspace,
)
from refaat.domain.quota import QuotaKey
from refaat.errors import ConflictError, NotFoundError
from refaat.ids import new_id
from refaat.infrastructure.cache_payload import response_from_json, response_to_json
from refaat.infrastructure.database.schema import (
    agent_messages,
    agents,
    api_keys,
    approval_requests,
    audit_outbox,
    audit_outbox_deliveries,
    chunk_hypotheticals,
    chunks,
    corpora,
    deferred_erasures,
    derivation_manifests,
    documents,
    entities,
    entity_terms,
    erasure_ledger,
    evaluation_runs,
    evaluation_suites,
    fine_tune_runs,
    harness_snapshots,
    hold_permission_history,
    index_generations,
    jobs,
    kill_switches,
    legal_holds,
    metadata,
    model_call_cache,
    optimization_candidates,
    optimization_runs,
    outbox,
    outbox_deliveries,
    preferences,
    program_approvals,
    program_artifacts,
    program_canaries,
    program_deployments,
    program_shadow_runs,
    query_cache,
    quota_counters,
    quota_reservations,
    raptor_nodes,
    relations,
    schedules,
    session_turns,
    sessions,
    tool_approvals,
    workspaces,
)


def _turn_cites_any(metadata: Any, chunk_ids: set[str]) -> bool:
    """True when a session turn's recorded citations intersect the given chunk ids."""
    if not isinstance(metadata, Mapping):
        return False
    citations = metadata.get("citations")
    if not isinstance(citations, (list, tuple)):
        return False
    return any(str(citation) in chunk_ids for citation in citations)


def _turn_references_document(metadata: Any, document_id: str) -> bool:
    """True when a turn recorded the document id among its cited documents.

    A turn stores the stable document ids of its citations, so a turn survives a re-ingest that
    rewrote the cited chunk ids and is still matched here by the unchanged document id.
    """
    if not isinstance(metadata, Mapping):
        return False
    cited = metadata.get("cited_document_ids")
    if not isinstance(cited, (list, tuple)):
        return False
    return any(str(document) == document_id for document in cited)


# Process-wide reentrant SQLite destroy locks, keyed by the database url plus the corpus advisory
# key. Every SQLStore against one SQLite database shares one lock per corpus, so two runtimes in a
# single process serialize their destroyers. SQLite has no cross-process advisory lock, so a
# multi-process SQLite deployment cannot serialize destroyers and is unsupported for enforceable
# holds; PostgreSQL is required there.
_SQLITE_DESTROY_LOCKS: dict[tuple[str, str, str], threading.RLock] = {}
_SQLITE_DESTROY_LOCKS_GUARD = threading.Lock()


class SQLStore:
    def __init__(self, database_url: str, clock: Clock, *, lock_max_waiters: int = 64) -> None:
        kwargs: dict[str, Any] = {"pool_pre_ping": True}
        if database_url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        self.engine = create_engine(database_url, **kwargs)
        self.clock = clock
        # Enforcement of an existing hold is state-driven and unconditional: every destructive path
        # consults the legal_holds table regardless of this flag. The flag gates ONLY the management
        # write surface (place/lift), which with_legal_hold_registry() opts into.
        self._legal_hold_management = False
        # Gates ONLY the kill-switch management write surface (engage/disengage); enforcement of an
        # engaged switch is state-driven and does not depend on it.
        self._kill_switch_management = False
        # Gates the quota housekeeping surface (the reservation reaper). Enforcement of a configured
        # quota does not depend on it; a policy installed at build time reserves and admits directly.
        self._quota_management = False
        self._lock_engine: Engine | None = None
        self._lock_engine_guard = threading.Lock()
        # The process-wide SQLite destroy-lock registry is keyed by this database url, so a second
        # SQLStore over the same SQLite database shares the same per-corpus lock.
        self._sqlite_lock_scope = str(self.engine.url)
        # Bounds the concurrent NullPool lock connections a same-corpus Postgres waiter burst can
        # open, so a large burst cannot exhaust the server's connection limit; the shared QueuePool is
        # a separate engine and is unaffected.
        self._pg_lock_semaphore = threading.BoundedSemaphore(lock_max_waiters)
        # Per-thread record of the Postgres session lock connection each held corpus key is pinned to,
        # so a wider application boundary can re-enter the same lock without a second connection.
        self._pg_destroy_held = threading.local()
        if database_url.startswith("sqlite"):
            self._configure_sqlite(self.engine)

    @staticmethod
    def _configure_sqlite(engine: Engine) -> None:
        @event.listens_for(engine, "connect")
        def set_pragmas(dbapi_connection: Any, _: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

    def initialize(self) -> None:
        metadata.create_all(self.engine)

    def close(self) -> None:
        self.engine.dispose()
        if self._lock_engine is not None:
            self._lock_engine.dispose()

    def ping(self) -> bool:
        with self.engine.connect() as conn:
            conn.execute(select(1)).scalar_one()
        return True

    def enable_legal_hold_management(self) -> None:
        """Opt this runtime into the legal-hold management write surface (place/lift).

        Enforcement of an existing hold does NOT depend on this: every destructive path consults the
        legal_holds table unconditionally. This flag only lets LegalHoldService record and lift holds.
        """
        self._legal_hold_management = True

    @property
    def legal_hold_management_enabled(self) -> bool:
        return self._legal_hold_management

    def enable_kill_switch_management(self) -> None:
        """Opt this runtime into the kill-switch management write surface (engage/disengage).

        Enforcement of an engaged switch does NOT depend on this flag: a runtime with a kill-switch
        registry installed enforces engaged switches regardless. This flag only lets the management
        service record and clear switches.
        """
        self._kill_switch_management = True

    @property
    def kill_switch_management_enabled(self) -> bool:
        return self._kill_switch_management

    def enable_quota_management(self) -> None:
        """Opt this runtime into the quota housekeeping surface (the reservation reaper).

        Enforcement of a configured quota does not depend on this flag: a runtime with a quota
        policy installed reserves and admits directly. This flag only lets the worker reap leaked
        reservations for a runtime that governs quotas.
        """
        self._quota_management = True

    @property
    def quota_management_enabled(self) -> bool:
        return self._quota_management

    @staticmethod
    def _advisory_key(workspace_id: str, corpus_id: str) -> int:
        digest = hashlib.sha256(f"legal_hold:{workspace_id}:{corpus_id}".encode()).digest()[:8]
        return int.from_bytes(digest, "big", signed=True)

    def _sqlite_destroy_lock(self, key: tuple[str, str]) -> threading.RLock:
        scoped = (self._sqlite_lock_scope, key[0], key[1])
        with _SQLITE_DESTROY_LOCKS_GUARD:
            lock = _SQLITE_DESTROY_LOCKS.get(scoped)
            if lock is None:
                lock = threading.RLock()
                _SQLITE_DESTROY_LOCKS[scoped] = lock
            return lock

    def _ensure_lock_engine(self) -> Engine:
        # A dedicated NullPool engine to the SAME database, so the pinned session-advisory-lock
        # connection lives OUTSIDE the shared QueuePool and never competes with an inner destroy
        # transaction for a pool slot. Created lazily on the first Postgres destroy, under a guard.
        if self._lock_engine is None:
            with self._lock_engine_guard:
                if self._lock_engine is None:
                    self._lock_engine = create_engine(self.engine.url, poolclass=NullPool)
        return self._lock_engine

    @contextmanager
    def document_destroy_lock(self, workspace_id: str, corpus_id: str) -> Iterator[None]:
        """Serialize destructive operations on one corpus: place, delete, overwrite, and erase.

        Enforcement is unconditional, so the lock is a real lock, never a no-op. It is reentrant
        within a thread, so a wider application boundary (an ingest overwrite or a delete) can hold
        it while an inner store chokepoint re-enters the same key without a second acquisition.

        On Postgres a session-level pg_advisory_lock on a connection from the dedicated NullPool lock
        engine (created lazily), bounded by a semaphore so a same-corpus waiter burst cannot exhaust
        the server's connection limit: a blocked waiter parks on that NullPool connection, never a
        shared QueuePool slot, so an in-flight erase's inner transactions can never be starved. On
        SQLite a process-wide per-(database, corpus) reentrant lock, so two SQLStore instances over
        one SQLite database in a single process still serialize. A multi-process SQLite deployment
        cannot serialize destroyers across processes (SQLite has no cross-process advisory lock) and
        is unsupported for enforceable legal holds; Postgres is required for multi-process enforcement.
        """
        key = (workspace_id, corpus_id)
        if self.engine.dialect.name != "postgresql":
            lock = self._sqlite_destroy_lock(key)
            lock.acquire()
            try:
                yield
            finally:
                lock.release()
            return
        held = self._pg_destroy_held
        if not hasattr(held, "conns"):
            held.conns = {}
        if key in held.conns:
            # Reentrant: this thread already pins the session lock on its own connection.
            yield
            return
        advisory_key = self._advisory_key(workspace_id, corpus_id)
        # Bound concurrent lock connections so a same-corpus waiter burst cannot exhaust the server's
        # connection limit before parking on the advisory lock.
        self._pg_lock_semaphore.acquire()
        conn = None
        try:
            conn = self._ensure_lock_engine().connect().execution_options(isolation_level="AUTOCOMMIT")
            conn.exec_driver_sql("SELECT pg_advisory_lock(%s)", (advisory_key,))
        except BaseException:
            # Close the opened connection before releasing the permit, so a failure between connect and
            # lock acquisition never leaks a NullPool connection while returning the semaphore slot.
            try:
                if conn is not None:
                    conn.close()
            finally:
                self._pg_lock_semaphore.release()
            raise
        held.conns[key] = conn
        try:
            yield
        finally:
            del held.conns[key]
            try:
                conn.exec_driver_sql("SELECT pg_advisory_unlock(%s)", (advisory_key,))
            finally:
                try:
                    conn.close()
                finally:
                    self._pg_lock_semaphore.release()

    def _is_document_held(self, conn: Any, workspace_id: str, corpus_id: str, document_id: str) -> bool:
        stmt = (
            select(legal_holds.c.id)
            .where(
                legal_holds.c.workspace_id == workspace_id,
                legal_holds.c.corpus_id == corpus_id,
                legal_holds.c.lifted_at.is_(None),
                or_(legal_holds.c.document_id.is_(None), legal_holds.c.document_id == document_id),
            )
            .limit(1)
        )
        return conn.execute(stmt).first() is not None

    def is_document_held(self, workspace_id: str, corpus_id: str, document_id: str) -> bool:
        with self.engine.connect() as conn:
            return self._is_document_held(conn, workspace_id, corpus_id, document_id)

    def place_legal_hold(
        self,
        workspace_id: str,
        *,
        corpus_id: str,
        document_id: str | None,
        placed_by: str,
        reason: str | None,
        placed_at: datetime,
    ) -> str:
        hold_id = new_id("hold")
        # Acquire the corpus lock (NullPool on Postgres) BEFORE the short QueuePool insert, so a place
        # serializes against an in-flight erase without a blocked waiter holding a shared pool slot.
        with self.document_destroy_lock(workspace_id, corpus_id), self.engine.begin() as conn:
            conn.execute(
                insert(legal_holds).values(
                    id=hold_id,
                    workspace_id=workspace_id,
                    corpus_id=corpus_id,
                    document_id=document_id,
                    status="active",
                    reason=reason,
                    placed_by=placed_by,
                    placed_at=placed_at,
                    lifted_by=None,
                    lifted_at=None,
                )
            )
        return hold_id

    def lift_legal_hold(self, workspace_id: str, hold_id: str, *, lifted_by: str, lifted_at: datetime) -> bool:
        with self.engine.begin() as conn:
            result = conn.execute(
                update(legal_holds)
                .where(
                    legal_holds.c.workspace_id == workspace_id,
                    legal_holds.c.id == hold_id,
                    legal_holds.c.lifted_at.is_(None),
                )
                .values(status="lifted", lifted_by=lifted_by, lifted_at=lifted_at)
            )
        return result.rowcount == 1

    def get_legal_hold(self, workspace_id: str, hold_id: str) -> RowMapping | None:
        with self.engine.connect() as conn:
            return (
                conn.execute(
                    select(legal_holds).where(
                        legal_holds.c.workspace_id == workspace_id,
                        legal_holds.c.id == hold_id,
                    )
                )
                .mappings()
                .first()
            )

    def list_legal_holds(
        self, workspace_id: str, corpus_id: str | None = None, *, active_only: bool = True
    ) -> list[JsonObject]:
        conditions = [legal_holds.c.workspace_id == workspace_id]
        if corpus_id is not None:
            conditions.append(legal_holds.c.corpus_id == corpus_id)
        if active_only:
            conditions.append(legal_holds.c.lifted_at.is_(None))
        with self.engine.connect() as conn:
            rows = conn.execute(select(legal_holds).where(*conditions).order_by(legal_holds.c.placed_at)).mappings()
            return [dict(row) for row in rows]

    def load_engaged_kill_switches(self) -> KillSwitchSet:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(kill_switches.c.workspace_id, kill_switches.c.scope).where(
                    kill_switches.c.disengaged_at.is_(None)
                )
            ).all()
        return KillSwitchSet(frozenset((str(row[0]), str(row[1])) for row in rows))

    def engage_kill_switch(
        self, workspace_id: str, *, scope: str, engaged_by: str, reason: str | None, engaged_at: datetime
    ) -> str:
        switch_id = new_id("ks")
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    insert(kill_switches).values(
                        id=switch_id,
                        workspace_id=workspace_id,
                        scope=scope,
                        reason=reason,
                        engaged_by=engaged_by,
                        engaged_at=engaged_at,
                        disengaged_by=None,
                        disengaged_at=None,
                    )
                )
        except IntegrityError as exc:
            raise ConflictError(f"Kill switch already engaged: {workspace_id or '*'}/{scope}") from exc
        return switch_id

    def get_active_kill_switch(self, workspace_id: str, scope: str) -> RowMapping | None:
        with self.engine.connect() as conn:
            return (
                conn.execute(
                    select(kill_switches).where(
                        kill_switches.c.workspace_id == workspace_id,
                        kill_switches.c.scope == scope,
                        kill_switches.c.disengaged_at.is_(None),
                    )
                )
                .mappings()
                .first()
            )

    def disengage_kill_switch(
        self, workspace_id: str, *, scope: str, disengaged_by: str, disengaged_at: datetime
    ) -> bool:
        with self.engine.begin() as conn:
            result = conn.execute(
                update(kill_switches)
                .where(
                    kill_switches.c.workspace_id == workspace_id,
                    kill_switches.c.scope == scope,
                    kill_switches.c.disengaged_at.is_(None),
                )
                .values(disengaged_by=disengaged_by, disengaged_at=disengaged_at)
            )
        return result.rowcount == 1

    def list_kill_switches(self, workspace_id: str | None = None, *, active_only: bool = True) -> list[JsonObject]:
        conditions: list[Any] = []
        if workspace_id is not None:
            conditions.append(kill_switches.c.workspace_id == workspace_id)
        if active_only:
            conditions.append(kill_switches.c.disengaged_at.is_(None))
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(kill_switches).where(*conditions).order_by(kill_switches.c.engaged_at)
            ).mappings()
            return [dict(row) for row in rows]

    def count_active_legal_holds(self, workspace_id: str | None = None) -> int:
        conditions: list[Any] = [legal_holds.c.lifted_at.is_(None)]
        if workspace_id is not None:
            conditions.append(legal_holds.c.workspace_id == workspace_id)
        with self.engine.connect() as conn:
            return int(conn.execute(select(func.count()).select_from(legal_holds).where(*conditions)).scalar_one())

    def count_active_kill_switches(self, workspace_id: str | None = None) -> int:
        conditions: list[Any] = [kill_switches.c.disengaged_at.is_(None)]
        if workspace_id is not None:
            conditions.append(kill_switches.c.workspace_id == workspace_id)
        with self.engine.connect() as conn:
            return int(conn.execute(select(func.count()).select_from(kill_switches).where(*conditions)).scalar_one())

    def audit_outbox_backlog(self, workspace_id: str | None = None) -> tuple[int, int | None, int]:
        """Undelivered audit-outbox depth, oldest undelivered age in seconds, and max attempts.

        Reads only counts and timestamps, never a payload, last_error, or claimed_by, so a caller
        can key DEGRADED off staleness (oldest age) rather than mere depth without seeing content.
        """
        conditions: list[Any] = [audit_outbox.c.delivered_at.is_(None)]
        if workspace_id is not None:
            conditions.append(audit_outbox.c.workspace_id == workspace_id)
        with self.engine.connect() as conn:
            row = conn.execute(
                select(
                    func.count(),
                    func.min(audit_outbox.c.created_at),
                    func.coalesce(func.max(audit_outbox.c.attempts), 0),
                )
                .select_from(audit_outbox)
                .where(*conditions)
            ).one()
        depth = int(row[0])
        oldest = row[1]
        max_attempts = int(row[2])
        if oldest is None:
            return depth, None, max_attempts
        if oldest.tzinfo is None:
            oldest = oldest.replace(tzinfo=UTC)
        age_seconds = int((self.clock.now() - oldest).total_seconds())
        return depth, max(age_seconds, 0), max_attempts

    def count_stuck_audit(self, workspace_id: str | None = None) -> int:
        conditions: list[Any] = [
            audit_outbox.c.delivered_at.is_(None),
            audit_outbox.c.last_error.isnot(None),
        ]
        if workspace_id is not None:
            conditions.append(audit_outbox.c.workspace_id == workspace_id)
        with self.engine.connect() as conn:
            return int(conn.execute(select(func.count()).select_from(audit_outbox).where(*conditions)).scalar_one())

    def quota_usage_for_workspace(self, workspace_id: str) -> list[JsonObject]:
        """Live WORKSPACE-scope counters only; never a PRINCIPAL or CORPUS row, never a scope_key.

        The scope_key of a principal or corpus counter embeds a principal or corpus id, so only the
        workspace-scope rows (whose scope_key is the workspace id itself) are projected, and even
        those omit scope_key. Returns dimension, spent, reserved, and the window length.
        """
        with self.engine.connect() as conn:
            rows = (
                conn.execute(
                    select(
                        quota_counters.c.dimension,
                        quota_counters.c.spent,
                        quota_counters.c.reserved,
                        quota_counters.c.window_seconds,
                    )
                    .where(
                        quota_counters.c.workspace_id == workspace_id,
                        quota_counters.c.scope == "workspace",
                    )
                    .order_by(quota_counters.c.dimension, quota_counters.c.window_start)
                )
                .mappings()
                .all()
            )
        return [dict(row) for row in rows]

    def count_active_tool_approvals(self, workspace_id: str | None = None) -> int:
        conditions: list[Any] = [tool_approvals.c.status == "granted"]
        if workspace_id is not None:
            conditions.append(tool_approvals.c.workspace_id == workspace_id)
        with self.engine.connect() as conn:
            return int(conn.execute(select(func.count()).select_from(tool_approvals).where(*conditions)).scalar_one())

    def count_open_approval_requests(self, workspace_id: str | None = None) -> int:
        conditions: list[Any] = [approval_requests.c.status == "open"]
        if workspace_id is not None:
            conditions.append(approval_requests.c.workspace_id == workspace_id)
        with self.engine.connect() as conn:
            return int(
                conn.execute(select(func.count()).select_from(approval_requests).where(*conditions)).scalar_one()
            )

    def job_posture(self, workspace_id: str | None = None) -> tuple[int, int, int, int]:
        """Pending, running, stuck-lease, and failed job counts; never a payload, error, or owner."""
        now = self.clock.now()
        base: list[Any] = []
        if workspace_id is not None:
            base.append(jobs.c.workspace_id == workspace_id)

        def _count(*extra: Any) -> int:
            with self.engine.connect() as conn:
                return int(conn.execute(select(func.count()).select_from(jobs).where(*base, *extra)).scalar_one())

        pending = _count(jobs.c.status == JobStatus.PENDING.value)
        running = _count(jobs.c.status == JobStatus.RUNNING.value)
        stuck = _count(jobs.c.status == JobStatus.RUNNING.value, jobs.c.leased_until < now)
        failed = _count(jobs.c.status == JobStatus.FAILED.value)
        return pending, running, stuck, failed

    def schema_revision(self) -> str | None:
        """The applied alembic revision, or None when the schema was auto-created (no version row).

        The has_table guard runs before the version read so a missing table on Postgres does not
        poison the surrounding transaction; a fresh connection isolates it either way.
        """
        with self.engine.connect() as conn:
            if not inspect(conn).has_table("alembic_version"):
                return None
            row = conn.execute(text("SELECT version_num FROM alembic_version")).first()
        return str(row[0]) if row is not None else None

    @staticmethod
    def _quota_counter_id(key: QuotaKey) -> str:
        return stable_digest((key.scope, key.scope_key, key.dimension, str(key.window_start)), length=64)

    @contextmanager
    def _quota_transaction(self) -> Iterator[Connection]:
        """One serialized transaction for the atomic quota accumulate-and-check.

        On SQLite a BEGIN IMMEDIATE takes the write lock up front so two concurrent reservations
        serialize on the busy_timeout instead of deadlocking on a shared-to-write upgrade; a loser
        that still times out raises and the policy denies (fail closed). On PostgreSQL a plain
        transaction plus the guarded row UPDATE gives the row-lock guarantee.
        """
        if self.engine.dialect.name == "sqlite":
            conn = self.engine.connect().execution_options(isolation_level="AUTOCOMMIT")
            try:
                conn.exec_driver_sql("BEGIN IMMEDIATE")
                try:
                    yield conn
                except BaseException:
                    conn.exec_driver_sql("ROLLBACK")
                    raise
                else:
                    conn.exec_driver_sql("COMMIT")
            finally:
                conn.close()
        else:
            with self.engine.begin() as conn:
                yield conn

    def quota_reserve(self, *, key: QuotaKey, reservation_id: str, estimate: int, ttl_seconds: float) -> bool:
        """Atomically reserve a ceiling against a window counter; True on admit, False on exhaustion.

        A guarded ``UPDATE ... WHERE spent + reserved + estimate <= limit`` decided by ``rowcount``
        is the concurrency chokepoint: two near-exhausted concurrent reservations cannot both pass,
        since the DB serializes the row. The reservation is a durable, leased, idempotent row, so a
        re-run with the same id is an admit-as-no-op rather than a double charge.
        """
        for attempt in range(3):
            try:
                return self._quota_reserve_once(
                    key=key, reservation_id=reservation_id, estimate=estimate, ttl_seconds=ttl_seconds
                )
            except IntegrityError:
                if attempt == 2:
                    raise
        raise AssertionError("unreachable")

    def _quota_reserve_once(self, *, key: QuotaKey, reservation_id: str, estimate: int, ttl_seconds: float) -> bool:
        now = self.clock.now()
        counter_id = self._quota_counter_id(key)
        with self._quota_transaction() as conn:
            existing = conn.execute(
                select(quota_reservations.c.state).where(quota_reservations.c.reservation_id == reservation_id)
            ).first()
            if existing is not None:
                return True
            present = conn.execute(select(quota_counters.c.id).where(quota_counters.c.id == counter_id)).first()
            if present is None:
                conn.execute(
                    insert(quota_counters).values(
                        id=counter_id,
                        workspace_id=key.workspace_id,
                        scope=key.scope,
                        scope_key=key.scope_key,
                        dimension=key.dimension,
                        window_start=key.window_start,
                        window_seconds=key.window_seconds,
                        spent=0,
                        reserved=0,
                        updated_at=now,
                    )
                )
            result = conn.execute(
                update(quota_counters)
                .where(
                    quota_counters.c.id == counter_id,
                    quota_counters.c.spent + quota_counters.c.reserved + estimate <= key.limit,
                )
                .values(reserved=quota_counters.c.reserved + estimate, updated_at=now)
            )
            if result.rowcount != 1:
                return False
            conn.execute(
                insert(quota_reservations).values(
                    reservation_id=reservation_id,
                    counter_id=counter_id,
                    estimate=estimate,
                    state="open",
                    reserved_at=now,
                    expires_at=now + timedelta(seconds=ttl_seconds),
                )
            )
            return True

    def quota_commit(self, *, reservation_id: str, actual: int) -> None:
        """Reconcile an open reservation to the measured spend: charge min(actual, ceiling), release
        the reserved ceiling. The state gate makes a re-run or a commit-after-release a no-op."""
        now = self.clock.now()
        with self._quota_transaction() as conn:
            row = conn.execute(
                select(quota_reservations.c.counter_id, quota_reservations.c.estimate).where(
                    quota_reservations.c.reservation_id == reservation_id,
                    quota_reservations.c.state == "open",
                )
            ).first()
            if row is None:
                return
            counter_id = str(row[0])
            estimate = int(row[1])
            gate = conn.execute(
                update(quota_reservations)
                .where(quota_reservations.c.reservation_id == reservation_id, quota_reservations.c.state == "open")
                .values(state="committed")
            )
            if gate.rowcount != 1:
                return
            charge = min(int(actual), estimate)
            if charge < 0:
                charge = 0
            conn.execute(
                update(quota_counters)
                .where(quota_counters.c.id == counter_id)
                .values(
                    spent=quota_counters.c.spent + charge,
                    reserved=quota_counters.c.reserved - estimate,
                    updated_at=now,
                )
            )

    def quota_release(self, *, reservation_id: str) -> None:
        """Release an open reservation's ceiling with no spend recorded. Idempotent via the state gate."""
        now = self.clock.now()
        with self._quota_transaction() as conn:
            row = conn.execute(
                select(quota_reservations.c.counter_id, quota_reservations.c.estimate).where(
                    quota_reservations.c.reservation_id == reservation_id,
                    quota_reservations.c.state == "open",
                )
            ).first()
            if row is None:
                return
            counter_id = str(row[0])
            estimate = int(row[1])
            gate = conn.execute(
                update(quota_reservations)
                .where(quota_reservations.c.reservation_id == reservation_id, quota_reservations.c.state == "open")
                .values(state="released")
            )
            if gate.rowcount != 1:
                return
            conn.execute(
                update(quota_counters)
                .where(quota_counters.c.id == counter_id)
                .values(reserved=quota_counters.c.reserved - estimate, updated_at=now)
            )

    def quota_reap(self, *, now: datetime | None = None, limit: int = 100) -> int:
        """Reclaim expired open reservations by charging the ceiling to spent, not releasing to zero.

        A crash between reserve and commit leaves an open lease; charging its full estimate to spent
        is fail-closed (the crashing run burns its worst case) while returning reserved to baseline,
        so a crash never permanently blocks the tenant.
        """
        moment = now or self.clock.now()
        reclaimed = 0
        with self._quota_transaction() as conn:
            rows = conn.execute(
                select(
                    quota_reservations.c.reservation_id,
                    quota_reservations.c.counter_id,
                    quota_reservations.c.estimate,
                )
                .where(quota_reservations.c.state == "open", quota_reservations.c.expires_at < moment)
                .limit(limit)
            ).all()
            for reservation_id, counter_id, estimate in rows:
                gate = conn.execute(
                    update(quota_reservations)
                    .where(quota_reservations.c.reservation_id == reservation_id, quota_reservations.c.state == "open")
                    .values(state="expired")
                )
                if gate.rowcount != 1:
                    continue
                conn.execute(
                    update(quota_counters)
                    .where(quota_counters.c.id == str(counter_id))
                    .values(
                        spent=quota_counters.c.spent + int(estimate),
                        reserved=quota_counters.c.reserved - int(estimate),
                        updated_at=moment,
                    )
                )
                reclaimed += 1
        return reclaimed

    def quota_gc(self, *, before_window_start: int) -> int:
        """Delete fully-settled past windows: a counter older than the cutoff with no reserved amount
        and no open reservation, so a late commit can never target a removed row."""
        removed = 0
        with self._quota_transaction() as conn:
            candidates = conn.execute(
                select(quota_counters.c.id).where(
                    quota_counters.c.window_start < before_window_start,
                    quota_counters.c.reserved == 0,
                )
            ).all()
            for (counter_id,) in candidates:
                open_left = conn.execute(
                    select(quota_reservations.c.reservation_id)
                    .where(
                        quota_reservations.c.counter_id == str(counter_id),
                        quota_reservations.c.state == "open",
                    )
                    .limit(1)
                ).first()
                if open_left is not None:
                    continue
                conn.execute(delete(quota_reservations).where(quota_reservations.c.counter_id == str(counter_id)))
                conn.execute(delete(quota_counters).where(quota_counters.c.id == str(counter_id)))
                removed += 1
        return removed

    def release_job(self, job_id: str, lease_owner: str, *, run_after: datetime) -> bool:
        """Return a claimed job to PENDING without counting this attempt.

        Used when an incident freeze blocks dispatch: it undoes the attempt increment claim_job
        applied and clears the lease with a delayed run_after, so a job blocked by a freeze is retried
        after the freeze lifts rather than exhausting its retries during it.
        """
        with self.engine.begin() as conn:
            result = conn.execute(
                update(jobs)
                .where(
                    jobs.c.id == job_id,
                    jobs.c.lease_owner == lease_owner,
                    jobs.c.status == JobStatus.RUNNING,
                )
                .values(
                    status=JobStatus.PENDING,
                    attempts=jobs.c.attempts - 1,
                    lease_owner=None,
                    leased_until=None,
                    run_after=run_after,
                    updated_at=self.clock.now(),
                )
            )
        return result.rowcount == 1

    def create_workspace(self, slug: str, name: str) -> Workspace:
        now = self.clock.now()
        row: dict[str, Any] = {"id": new_id("ws"), "slug": slug, "name": name, "created_at": now}
        try:
            with self.engine.begin() as conn:
                conn.execute(insert(workspaces).values(**row))
        except IntegrityError as exc:
            raise ConflictError(f"Workspace slug already exists: {slug}") from exc
        return Workspace(**row)

    def get_workspace(self, workspace_id: str) -> Workspace:
        with self.engine.connect() as conn:
            row = conn.execute(select(workspaces).where(workspaces.c.id == workspace_id)).mappings().first()
        if row is None:
            raise NotFoundError("Workspace not found")
        return Workspace(**dict(row))

    def find_workspace_by_slug(self, slug: str) -> Workspace | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(workspaces).where(workspaces.c.slug == slug)).mappings().first()
        return Workspace(**dict(row)) if row is not None else None

    def create_api_key(self, workspace_id: str, name: str, prefix: str, digest: str, scopes: Sequence[str]) -> str:
        self.get_workspace(workspace_id)
        key_id = new_id("key")
        with self.engine.begin() as conn:
            conn.execute(
                insert(api_keys).values(
                    id=key_id,
                    workspace_id=workspace_id,
                    name=name,
                    prefix=prefix,
                    digest=digest,
                    scopes=list(scopes),
                    created_at=self.clock.now(),
                    revoked_at=None,
                    last_used_at=None,
                )
            )
        return key_id

    def get_api_key_candidate(self, prefix: str) -> RowMapping | None:
        with self.engine.connect() as conn:
            return (
                conn.execute(select(api_keys).where(api_keys.c.prefix == prefix, api_keys.c.revoked_at.is_(None)))
                .mappings()
                .first()
            )

    def touch_api_key(self, key_id: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(update(api_keys).where(api_keys.c.id == key_id).values(last_used_at=self.clock.now()))

    def revoke_api_key(self, workspace_id: str, key_id: str) -> None:
        with self.engine.begin() as conn:
            result = conn.execute(
                update(api_keys)
                .where(api_keys.c.id == key_id, api_keys.c.workspace_id == workspace_id)
                .values(revoked_at=self.clock.now())
            )
        if result.rowcount == 0:
            raise NotFoundError("API key not found")

    def create_corpus(self, workspace_id: str, name: str, settings: JsonObject | None = None) -> Corpus:
        self.get_workspace(workspace_id)
        row: dict[str, Any] = {
            "id": new_id("corpus"),
            "workspace_id": workspace_id,
            "name": name,
            "version": 0,
            "settings": settings or {},
            "created_at": self.clock.now(),
        }
        try:
            with self.engine.begin() as conn:
                conn.execute(insert(corpora).values(**row))
        except IntegrityError as exc:
            raise ConflictError(f"Corpus name already exists: {name}") from exc
        return Corpus(**row)

    def get_corpus(self, workspace_id: str, corpus_id: str) -> Corpus:
        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    select(corpora).where(
                        corpora.c.id == corpus_id,
                        corpora.c.workspace_id == workspace_id,
                    )
                )
                .mappings()
                .first()
            )
        if row is None:
            raise NotFoundError("Corpus not found")
        return Corpus(**dict(row))

    def list_corpora(self, workspace_id: str) -> list[Corpus]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(corpora).where(corpora.c.workspace_id == workspace_id).order_by(corpora.c.name)
            ).mappings()
            return [Corpus(**dict(row)) for row in rows]

    def find_corpus_by_name(self, workspace_id: str, name: str) -> Corpus | None:
        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    select(corpora).where(
                        corpora.c.workspace_id == workspace_id,
                        corpora.c.name == name,
                    )
                )
                .mappings()
                .first()
            )
        return Corpus(**dict(row)) if row is not None else None

    def get_document_by_source(self, workspace_id: str, corpus_id: str, source_id: str) -> RowMapping | None:
        with self.engine.connect() as conn:
            return (
                conn.execute(
                    select(documents).where(
                        documents.c.workspace_id == workspace_id,
                        documents.c.corpus_id == corpus_id,
                        documents.c.source_id == source_id,
                    )
                )
                .mappings()
                .first()
            )

    def get_document(self, workspace_id: str, corpus_id: str, document_id: str) -> RowMapping:
        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    select(documents).where(
                        documents.c.workspace_id == workspace_id,
                        documents.c.corpus_id == corpus_id,
                        documents.c.id == document_id,
                    )
                )
                .mappings()
                .first()
            )
        if row is None:
            raise NotFoundError("Document not found")
        return row

    def list_documents(self, workspace_id: str, corpus_id: str) -> list[JsonObject]:
        self.get_corpus(workspace_id, corpus_id)
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(documents)
                .where(documents.c.workspace_id == workspace_id, documents.c.corpus_id == corpus_id)
                .order_by(documents.c.updated_at.desc())
            ).mappings()
            return [dict(row) for row in rows]

    def referenced_blob_uris(self) -> set[str]:
        """Every blob uri a current document row points at, across all workspaces.

        The blob store is shared, so garbage collection is global: a key is live
        while any document references it. Content-addressed keys mean two documents
        can share one blob, so distinct uris are taken as the reference set.
        """
        with self.engine.connect() as conn:
            rows = conn.execute(select(documents.c.blob_uri).distinct()).all()
        return {str(row[0]) for row in rows}

    def is_blob_referenced(self, blob_uri: str) -> bool:
        with self.engine.connect() as conn:
            return (
                conn.execute(select(documents.c.id).where(documents.c.blob_uri == blob_uri).limit(1)).first()
                is not None
            )

    def replace_document(
        self,
        *,
        workspace_id: str,
        corpus_id: str,
        document_id: str,
        source_id: str,
        title: str,
        media_type: str,
        checksum: str,
        blob_uri: str,
        document_metadata: JsonObject,
        chunk_rows: Sequence[Mapping[str, Any]],
        entity_rows: Sequence[Mapping[str, Any]],
        relation_rows: Sequence[Mapping[str, Any]],
        hypothetical_rows: Sequence[Mapping[str, Any]] = (),
        entity_term_rows: Sequence[Mapping[str, Any]] = (),
        expect_existing: bool = False,
        actor: str | None = None,
        maintain_derivation_digest: bool = False,
    ) -> int:
        now = self.clock.now()
        # The per-document integrity digest, folded from the authoritative chunk set this call inserts, so
        # a later recompute from committed chunks matches. None (stored NULL) when the caller does not
        # maintain it, keeping the off path byte-identical.
        derivation_digest = (
            self._fold_chunk_digest((str(row["id"]), str(row["checksum"])) for row in chunk_rows)
            if maintain_derivation_digest
            else None
        )
        # Acquire the corpus lock (NullPool on Postgres) BEFORE the short QueuePool transaction, so a
        # blocked waiter never holds a shared pool slot. Reentrant: an ingest overwrite already holds
        # this key across its wider read-decide-mutate boundary.
        with self.document_destroy_lock(workspace_id, corpus_id), self.engine.begin() as conn:
            corpus = (
                conn.execute(
                    select(corpora).where(
                        corpora.c.id == corpus_id,
                        corpora.c.workspace_id == workspace_id,
                    )
                )
                .mappings()
                .first()
            )
            if corpus is None:
                raise NotFoundError("Corpus not found")
            existing_row = (
                conn.execute(
                    select(documents.c.id, documents.c.checksum).where(
                        documents.c.corpus_id == corpus_id,
                        documents.c.source_id == source_id,
                    )
                )
                .mappings()
                .first()
            )
            # Refuse only a genuine content overwrite of an existing, held document. A same-checksum
            # re-tag (ACL/metadata) is evidence-preserving and allowed, and a brand-new document is
            # never blocked, even under a corpus-wide hold.
            if (
                existing_row is not None
                and existing_row["checksum"] != checksum
                and self._is_document_held(conn, workspace_id, corpus_id, str(existing_row["id"]))
            ):
                raise LegalHoldError("Document is under legal hold")
            existing = existing_row["id"] if existing_row is not None else None
            if existing is None:
                if expect_existing:
                    # The caller read an existing row before the lock, but under the lock it is gone
                    # (an interleaved erase removed it). Do NOT resurrect it under the stale id; the
                    # bounded ingest retry re-reads and creates a genuinely new document.
                    raise ConflictError("Document source no longer exists")
                try:
                    conn.execute(
                        insert(documents).values(
                            id=document_id,
                            workspace_id=workspace_id,
                            corpus_id=corpus_id,
                            source_id=source_id,
                            title=title,
                            media_type=media_type,
                            checksum=checksum,
                            derivation_digest=derivation_digest,
                            blob_uri=blob_uri,
                            status="indexing",
                            error=None,
                            metadata=document_metadata,
                            created_at=now,
                            updated_at=now,
                        )
                    )
                except IntegrityError as exc:
                    raise ConflictError("Document source already exists") from exc
            else:
                existing_id = cast(str, existing)
                if existing_id != document_id:
                    # A racing first-write committed this (corpus, source) under a
                    # different id while our child rows were built against document_id;
                    # blindly UPDATE-ing would leave those rows pointing at a missing
                    # parent. Surface a ConflictError so the bounded ingest retry re-reads
                    # the winner and rebuilds against it.
                    raise ConflictError("Document source already exists")
                if self._is_document_held(conn, workspace_id, corpus_id, document_id):
                    # The document is under a hold and the content checksum is unchanged (a differing
                    # checksum was already refused above), so this is a re-tag. A re-tag may change the
                    # ACL or metadata, but it must not alter the held evidence: the incoming chunk rows
                    # must carry exactly the same (id, checksum) pairs as the stored held chunks, or the
                    # re-tag is refused so a hold cannot be used to silently rewrite held content.
                    stored_pairs = sorted(
                        (str(row[0]), str(row[1]))
                        for row in conn.execute(
                            select(chunks.c.id, chunks.c.checksum).where(chunks.c.document_id == document_id)
                        ).all()
                    )
                    # Recompute each incoming chunk checksum from its text rather than trusting the
                    # caller-declared checksum, so a re-tag cannot rewrite held chunk text while claiming
                    # the stored checksum.
                    incoming_pairs = sorted((str(row["id"]), sha256_text(str(row["text"]))) for row in chunk_rows)
                    if stored_pairs != incoming_pairs:
                        raise LegalHoldError("Document is under legal hold; a re-tag may not alter the held evidence")
                    # Ledger any permission change on held evidence in this same transaction, so who could
                    # see the held evidence stays reconstructable and the prior ACL is never silently
                    # overwritten. Fail-closed: a failed insert rolls back the whole re-tag.
                    old_metadata = conn.execute(
                        select(documents.c.metadata).where(documents.c.id == document_id)
                    ).scalar_one()
                    old_fingerprint = permission_fingerprint(old_metadata)
                    new_fingerprint = permission_fingerprint(document_metadata)
                    if old_fingerprint != new_fingerprint:
                        covering_hold_ids = [
                            str(row[0])
                            for row in conn.execute(
                                select(legal_holds.c.id).where(
                                    legal_holds.c.workspace_id == workspace_id,
                                    legal_holds.c.corpus_id == corpus_id,
                                    legal_holds.c.lifted_at.is_(None),
                                    or_(
                                        legal_holds.c.document_id.is_(None),
                                        legal_holds.c.document_id == document_id,
                                    ),
                                )
                            ).all()
                        ]
                        conn.execute(
                            insert(hold_permission_history).values(
                                id=new_id("hph"),
                                workspace_id=workspace_id,
                                corpus_id=corpus_id,
                                document_id=document_id,
                                covering_hold_ids=covering_hold_ids,
                                old_fingerprint=old_fingerprint,
                                new_fingerprint=new_fingerprint,
                                old_acl=canonical_acl(old_metadata),
                                new_acl=canonical_acl(document_metadata),
                                content_checksum=checksum,
                                actor=actor or "system",
                                created_at=now,
                            )
                        )
                conn.execute(delete(relations).where(relations.c.document_id == document_id))
                conn.execute(delete(entity_terms).where(entity_terms.c.document_id == document_id))
                conn.execute(delete(entities).where(entities.c.document_id == document_id))
                conn.execute(delete(chunk_hypotheticals).where(chunk_hypotheticals.c.document_id == document_id))
                conn.execute(delete(chunks).where(chunks.c.document_id == document_id))
                conn.execute(
                    update(documents)
                    .where(documents.c.id == document_id)
                    .values(
                        title=title,
                        media_type=media_type,
                        checksum=checksum,
                        derivation_digest=derivation_digest,
                        blob_uri=blob_uri,
                        status="indexing",
                        error=None,
                        metadata=document_metadata,
                        updated_at=now,
                    )
                )
            if chunk_rows:
                conn.execute(insert(chunks), list(chunk_rows))
            if hypothetical_rows:
                conn.execute(insert(chunk_hypotheticals), list(hypothetical_rows))
            if entity_rows:
                conn.execute(insert(entities), list(entity_rows))
            if relation_rows:
                conn.execute(insert(relations), list(relation_rows))
            if entity_term_rows:
                conn.execute(insert(entity_terms), list(entity_term_rows))
            # Increment atomically in the UPDATE, never read-then-write an absolute
            # value: two concurrent updates that both read V would both write V+1 and
            # lose one bump, so a response cached between the commits would stay valid
            # against changed content. The RETURNING value is the authoritative new
            # version this write advanced to.
            version = int(
                conn.execute(
                    update(corpora)
                    .where(corpora.c.id == corpus_id, corpora.c.workspace_id == workspace_id)
                    .values(version=corpora.c.version + 1)
                    .returning(corpora.c.version)
                ).scalar_one()
            )
            conn.execute(update(documents).where(documents.c.id == document_id).values(status="ready", updated_at=now))
            conn.execute(
                insert(outbox).values(
                    id=new_id("evt"),
                    workspace_id=workspace_id,
                    topic="document.indexed",
                    payload={
                        "workspace_id": workspace_id,
                        "document_id": document_id,
                        "corpus_id": corpus_id,
                        "corpus_version": version,
                    },
                    attempts=0,
                    available_at=now,
                    delivered_at=None,
                    last_error=None,
                    created_at=now,
                )
            )
            return version

    def delete_document(self, workspace_id: str, corpus_id: str, document_id: str) -> int:
        # Acquire the corpus lock (NullPool on Postgres) BEFORE the short QueuePool transaction, so a
        # blocked waiter never holds a shared pool slot. Reentrant: IngestionService.delete already
        # holds this key across its wider read-decide-mutate boundary.
        with self.document_destroy_lock(workspace_id, corpus_id), self.engine.begin() as conn:
            if self._is_document_held(conn, workspace_id, corpus_id, document_id):
                raise LegalHoldError("Document is under legal hold")
            result = conn.execute(
                delete(documents).where(
                    documents.c.id == document_id,
                    documents.c.workspace_id == workspace_id,
                    documents.c.corpus_id == corpus_id,
                )
            )
            if result.rowcount == 0:
                raise NotFoundError("Document not found")
            corpus_version = conn.execute(
                update(corpora)
                .where(corpora.c.id == corpus_id, corpora.c.workspace_id == workspace_id)
                .values(version=corpora.c.version + 1)
                .returning(corpora.c.version)
            ).scalar_one()
            return int(corpus_version)

    def list_document_hypothetical_questions(self, workspace_id: str, corpus_id: str, document_id: str) -> list[str]:
        """The hypothetical question texts a document produced, for erasure key recomputation."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(chunk_hypotheticals.c.question).where(
                    chunk_hypotheticals.c.workspace_id == workspace_id,
                    chunk_hypotheticals.c.corpus_id == corpus_id,
                    chunk_hypotheticals.c.document_id == document_id,
                )
            ).all()
        return [str(row[0]) for row in rows]

    def list_raptor_summaries(self, workspace_id: str, corpus_id: str) -> list[str]:
        """Every RAPTOR summary text currently stored for a corpus, for erasure key recomputation."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(raptor_nodes.c.summary).where(
                    raptor_nodes.c.workspace_id == workspace_id,
                    raptor_nodes.c.corpus_id == corpus_id,
                )
            ).all()
        return [str(row[0]) for row in rows]

    def delete_query_cache_for_corpus(self, workspace_id: str, corpus_id: str) -> int:
        """Physically delete every cached answer for a corpus. Idempotent: zero rows is success."""
        with self.engine.begin() as conn:
            result = conn.execute(
                delete(query_cache).where(
                    query_cache.c.workspace_id == workspace_id,
                    query_cache.c.corpus_id == corpus_id,
                )
            )
        return int(result.rowcount or 0)

    def delete_generation_model_cache(self, workspace_id: str) -> int:
        """Physically delete every generation model-call cache row for a workspace.

        Generation rows are workspace-scoped, so this is a complete purge of the tenant's cached
        model answers. It accepts a cold cache (zero rows) and is idempotent.
        """
        with self.engine.begin() as conn:
            result = conn.execute(
                delete(model_call_cache).where(
                    model_call_cache.c.kind == "generation",
                    model_call_cache.c.workspace_id == workspace_id,
                )
            )
        return int(result.rowcount or 0)

    def delete_embedding_model_cache(self, key_digests: Sequence[str]) -> int:
        """Delete embedding model-call cache rows for the given content-addressed keys.

        Embedding rows are global and backend-scoped (workspace scope is the reserved empty
        string), content-addressed by the embedding key digest, so deleting exactly the keys the
        erased texts hash to removes their vectors without touching another document's rows.
        """
        keys = list(dict.fromkeys(str(key) for key in key_digests))
        if not keys:
            return 0
        with self.engine.begin() as conn:
            result = conn.execute(
                delete(model_call_cache).where(
                    model_call_cache.c.kind == "embedding",
                    model_call_cache.c.workspace_id == self._EMBEDDING_SCOPE,
                    model_call_cache.c.cache_key.in_(keys),
                )
            )
        return int(result.rowcount or 0)

    def clear_session_summaries_for_corpus(self, workspace_id: str, corpus_id: str) -> int:
        """Set every session's compacted summary to NULL for a corpus.

        A compacted summary may have folded any document in the corpus and there is no
        per-document summary linkage, so erasure clears every summary in the corpus. Idempotent:
        only sessions that still carry a summary are cleared, so a re-run clears nothing.
        """
        with self.engine.begin() as conn:
            result = conn.execute(
                update(sessions)
                .where(
                    sessions.c.workspace_id == workspace_id,
                    sessions.c.corpus_id == corpus_id,
                    sessions.c.summary.is_not(None),
                )
                .values(summary=None, updated_at=self.clock.now())
            )
        return int(result.rowcount or 0)

    def delete_session_turns_for_document(
        self, workspace_id: str, corpus_id: str, document_id: str, chunk_ids: Sequence[str]
    ) -> int:
        """Delete every session turn that cites the erased document.

        A turn is matched when its recorded citations intersect the erased chunk ids or when it
        recorded the erased document id among its cited documents. The document id match is what
        catches a turn from a superseded revision, whose chunk ids no longer exist after a
        re-ingest but whose stable document id still does. Turn citations live in the turn metadata
        as JSON, which no dialect-portable predicate can match, so the match is computed in Python
        over the corpus's turns and the hits are deleted by id. Idempotent: no match deletes nothing.
        """
        wanted = {str(chunk_id) for chunk_id in chunk_ids}
        with self.engine.begin() as conn:
            rows = (
                conn.execute(
                    select(session_turns.c.id, session_turns.c.metadata)
                    .select_from(session_turns.join(sessions, session_turns.c.session_id == sessions.c.id))
                    .where(
                        sessions.c.workspace_id == workspace_id,
                        sessions.c.corpus_id == corpus_id,
                    )
                )
                .mappings()
                .all()
            )
            doomed = [
                str(row["id"])
                for row in rows
                if _turn_references_document(row["metadata"], document_id) or _turn_cites_any(row["metadata"], wanted)
            ]
            if not doomed:
                return 0
            result = conn.execute(delete(session_turns).where(session_turns.c.id.in_(doomed)))
        return int(result.rowcount or 0)

    def delete_preferences_for_document(self, workspace_id: str, corpus_id: str, document_id: str) -> int:
        """Delete learned preferences keyed to a document. Idempotent: zero rows is success."""
        with self.engine.begin() as conn:
            result = conn.execute(
                delete(preferences).where(
                    preferences.c.workspace_id == workspace_id,
                    preferences.c.corpus_id == corpus_id,
                    preferences.c.key == f"document:{document_id}",
                )
            )
        return int(result.rowcount or 0)

    def delete_outbox_for_document(self, document_id: str) -> int:
        """Delete outbox events whose payload references a document id.

        The reference lives in the payload JSON, which no dialect-portable predicate matches, so
        candidates are filtered in Python. Document ids are globally unique, so matching on the
        id alone never touches another document's events. Idempotent: zero rows is success.
        """
        with self.engine.begin() as conn:
            rows = conn.execute(select(outbox.c.id, outbox.c.payload)).mappings().all()
            doomed = [
                str(row["id"])
                for row in rows
                if isinstance(row["payload"], dict) and row["payload"].get("document_id") == document_id
            ]
            if not doomed:
                return 0
            result = conn.execute(delete(outbox).where(outbox.c.id.in_(doomed)))
        return int(result.rowcount or 0)

    def record_erasure(self, receipt: ErasureReceipt) -> None:
        """Append one erasure-request record to the ledger.

        Stores only the receipt's ids, status, per-surface counts, and timestamp; the receipt is
        redacted by construction, so no erased content can enter the ledger.
        """
        with self.engine.begin() as conn:
            conn.execute(
                insert(erasure_ledger).values(
                    request_id=receipt.request_id,
                    workspace_id=receipt.workspace_id,
                    corpus_id=receipt.corpus_id,
                    document_id=receipt.document_id,
                    status=str(receipt.status),
                    counts=receipt.counts(),
                    created_at=receipt.occurred_at,
                )
            )

    def record_deferred_erasure(
        self,
        workspace_id: str,
        corpus_id: str,
        document_id: str,
        *,
        requested_by: str,
        requested_at: datetime,
    ) -> str:
        """Record one pending deferred erasure, idempotently.

        One pending row per (workspace, corpus, document) via the partial unique index, so a repeated
        erase of a held document converges on the existing pending id rather than duplicating. The
        reentrant destroy lock serializes against an in-flight place or destroy.
        """
        with self.document_destroy_lock(workspace_id, corpus_id), self.engine.begin() as conn:
            existing = conn.execute(
                select(deferred_erasures.c.id).where(
                    deferred_erasures.c.workspace_id == workspace_id,
                    deferred_erasures.c.corpus_id == corpus_id,
                    deferred_erasures.c.document_id == document_id,
                    deferred_erasures.c.status == "pending",
                )
            ).scalar()
            if existing is not None:
                return str(existing)
            deferred_id = new_id("defer")
            conn.execute(
                insert(deferred_erasures).values(
                    id=deferred_id,
                    workspace_id=workspace_id,
                    corpus_id=corpus_id,
                    document_id=document_id,
                    status="pending",
                    requested_by=requested_by,
                    requested_at=requested_at,
                    executed_request_id=None,
                    executed_at=None,
                )
            )
        return deferred_id

    def list_pending_deferred_erasures(self, workspace_id: str, corpus_id: str) -> list[RowMapping]:
        with self.engine.connect() as conn:
            return list(
                conn.execute(
                    select(deferred_erasures)
                    .where(
                        deferred_erasures.c.workspace_id == workspace_id,
                        deferred_erasures.c.corpus_id == corpus_id,
                        deferred_erasures.c.status == "pending",
                    )
                    .order_by(deferred_erasures.c.requested_at)
                )
                .mappings()
                .all()
            )

    def mark_deferred_erasure_executed(
        self, deferred_id: str, workspace_id: str, *, executed_request_id: str, executed_at: datetime
    ) -> bool:
        """Mark one pending deferred erasure executed.

        Conditional on status='pending', so a concurrent processor cannot double-mark: only the writer
        that actually transitioned the row returns True. Scoped by workspace so an id can never touch
        another workspace's row.
        """
        with self.engine.begin() as conn:
            result = conn.execute(
                update(deferred_erasures)
                .where(
                    deferred_erasures.c.id == deferred_id,
                    deferred_erasures.c.workspace_id == workspace_id,
                    deferred_erasures.c.status == "pending",
                )
                .values(status="executed", executed_request_id=executed_request_id, executed_at=executed_at)
            )
        return result.rowcount == 1

    def list_pending_deferred_erasure_scopes(self) -> list[tuple[str, str]]:
        """Distinct (workspace, corpus) pairs with a pending deferred erasure, for the sweep job.

        The sweep re-drives every scope's pending requests, so a lift-time execution that failed or
        was interrupted is eventually completed (at-least-once), independent of any future lift.
        """
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(deferred_erasures.c.workspace_id, deferred_erasures.c.corpus_id)
                .where(deferred_erasures.c.status == "pending")
                .distinct()
            ).all()
        return [(str(row[0]), str(row[1])) for row in rows]

    def get_erasure_record(self, request_id: str) -> JsonObject | None:
        with self.engine.connect() as conn:
            row = (
                conn.execute(select(erasure_ledger).where(erasure_ledger.c.request_id == request_id)).mappings().first()
            )
        return dict(row) if row is not None else None

    @staticmethod
    def _chunk_from_row(row: RowMapping) -> Chunk:
        return Chunk(
            id=str(row["id"]),
            corpus_id=str(row["corpus_id"]),
            document_id=str(row["document_id"]),
            ordinal=int(row["ordinal"]),
            text=str(row["text"]),
            token_count=int(row["token_count"]),
            checksum=str(row["checksum"]),
            metadata=cast(JsonObject, row["metadata"]),
            embedding=tuple(float(value) for value in row["embedding"]),
            image_ref=(str(row["image_ref"]) if row["image_ref"] is not None else None),
            media_type=(str(row["media_type"]) if row["media_type"] is not None else None),
        )

    def document_stats(self, workspace_id: str, corpus_id: str, document_id: str) -> JsonObject:
        self.get_document(workspace_id, corpus_id, document_id)
        with self.engine.connect() as conn:
            return {
                "chunks": int(
                    conn.execute(
                        select(func.count()).select_from(chunks).where(chunks.c.document_id == document_id)
                    ).scalar_one()
                ),
                "entities": int(
                    conn.execute(
                        select(func.count()).select_from(entities).where(entities.c.document_id == document_id)
                    ).scalar_one()
                ),
                "relations": int(
                    conn.execute(
                        select(func.count()).select_from(relations).where(relations.c.document_id == document_id)
                    ).scalar_one()
                ),
            }

    def list_chunks(self, workspace_id: str, corpus_id: str, *, limit: int | None = None) -> list[Chunk]:
        with self.engine.connect() as conn:
            query = select(chunks).where(
                chunks.c.workspace_id == workspace_id,
                chunks.c.corpus_id == corpus_id,
            )
            if limit is not None:
                query = query.limit(limit)
            rows = conn.execute(query).mappings()
            return [self._chunk_from_row(row) for row in rows]

    def list_document_chunks(
        self,
        workspace_id: str,
        corpus_id: str,
        document_id: str,
    ) -> list[Chunk]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(chunks).where(
                    chunks.c.workspace_id == workspace_id,
                    chunks.c.corpus_id == corpus_id,
                    chunks.c.document_id == document_id,
                )
            ).mappings()
            return [self._chunk_from_row(row) for row in rows]

    def filter_chunk_ids(
        self,
        workspace_id: str,
        corpus_id: str,
        metadata_filter: MetadataFilter | None,
    ) -> set[str] | None:
        if metadata_filter is None:
            return None
        return {
            chunk.id for chunk in self.list_chunks(workspace_id, corpus_id) if metadata_filter.matches(chunk.metadata)
        }

    def get_chunks(self, workspace_id: str, corpus_id: str, chunk_ids: Sequence[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(chunks).where(
                    chunks.c.workspace_id == workspace_id,
                    chunks.c.corpus_id == corpus_id,
                    chunks.c.id.in_(chunk_ids),
                )
            ).mappings()
            by_id = {str(row["id"]): self._chunk_from_row(row) for row in rows}
        return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]

    def replace_raptor_nodes(
        self,
        workspace_id: str,
        corpus_id: str,
        node_rows: Sequence[Mapping[str, Any]],
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                delete(raptor_nodes).where(
                    raptor_nodes.c.workspace_id == workspace_id,
                    raptor_nodes.c.corpus_id == corpus_id,
                )
            )
            if node_rows:
                conn.execute(insert(raptor_nodes), list(node_rows))

    def list_raptor_nodes(
        self, workspace_id: str, corpus_id: str
    ) -> list[tuple[str, tuple[float, ...], tuple[str, ...]]]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(
                    raptor_nodes.c.id,
                    raptor_nodes.c.embedding,
                    raptor_nodes.c.leaf_chunk_ids,
                ).where(
                    raptor_nodes.c.workspace_id == workspace_id,
                    raptor_nodes.c.corpus_id == corpus_id,
                )
            ).all()
        return [
            (
                str(row[0]),
                tuple(float(value) for value in row[1]),
                tuple(str(chunk_id) for chunk_id in row[2]),
            )
            for row in rows
        ]

    def list_hypotheticals(self, workspace_id: str, corpus_id: str) -> list[tuple[str, tuple[float, ...]]]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(chunk_hypotheticals.c.chunk_id, chunk_hypotheticals.c.embedding).where(
                    chunk_hypotheticals.c.workspace_id == workspace_id,
                    chunk_hypotheticals.c.corpus_id == corpus_id,
                )
            ).all()
        return [(str(row[0]), tuple(float(value) for value in row[1])) for row in rows]

    def list_graph(self, workspace_id: str, corpus_id: str) -> tuple[list[Entity], list[Relation]]:
        with self.engine.connect() as conn:
            entity_rows = conn.execute(
                select(entities).where(
                    entities.c.workspace_id == workspace_id,
                    entities.c.corpus_id == corpus_id,
                )
            ).mappings()
            relation_rows = conn.execute(
                select(relations).where(
                    relations.c.workspace_id == workspace_id,
                    relations.c.corpus_id == corpus_id,
                )
            ).mappings()
            entity_values = [
                Entity(
                    id=str(row["id"]),
                    corpus_id=str(row["corpus_id"]),
                    document_id=str(row["document_id"]),
                    name=str(row["name"]),
                    normalized_name=str(row["normalized_name"]),
                    kind=str(row["kind"]),
                    metadata=cast(JsonObject, row["metadata"]),
                )
                for row in entity_rows
            ]
            relation_values = [
                Relation(
                    id=str(row["id"]),
                    corpus_id=str(row["corpus_id"]),
                    document_id=str(row["document_id"]),
                    source_entity_id=str(row["source_entity_id"]),
                    target_entity_id=str(row["target_entity_id"]),
                    kind=str(row["kind"]),
                    weight=float(row["weight"]),
                    metadata=cast(JsonObject, row["metadata"]),
                )
                for row in relation_rows
            ]
        return entity_values, relation_values

    def graph_index_present(self, workspace_id: str, corpus_id: str) -> bool:
        # Usable only when complete: any non-empty-named entity missing an in-scope term row means a
        # partial index (pre-flag or unchanged-reingest doc) that would under-return, so read as absent
        # and full-scan. The EXISTS is scoped to (workspace, corpus) exactly as graph_neighborhood reads
        # its term rows, so readiness and the read use one predicate and cannot diverge. Relies on the
        # write path writing an entity's complete, in-scope token set atomically with the entity, so
        # one in-scope term row implies full coverage; it is not a guard against a corrupted term table.
        uncovered = (
            select(entities.c.id)
            .where(
                entities.c.workspace_id == workspace_id,
                entities.c.corpus_id == corpus_id,
                entities.c.normalized_name != "",
                ~exists().where(
                    entity_terms.c.entity_id == entities.c.id,
                    entity_terms.c.workspace_id == workspace_id,
                    entity_terms.c.corpus_id == corpus_id,
                ),
            )
            .limit(1)
        )
        with self.engine.connect() as conn:
            return conn.execute(uncovered).first() is None

    def max_entity_name_length(self, workspace_id: str, corpus_id: str) -> int:
        with self.engine.connect() as conn:
            value = conn.execute(
                select(func.max(func.length(entities.c.normalized_name))).where(
                    entities.c.workspace_id == workspace_id,
                    entities.c.corpus_id == corpus_id,
                )
            ).scalar()
        return int(value or 0)

    def graph_neighborhood(
        self,
        workspace_id: str,
        corpus_id: str,
        *,
        term_candidates: frozenset[str],
        name_substrings: frozenset[str],
    ) -> tuple[list[Entity], list[Relation]]:
        """Fetch only the query-relevant subgraph for the opt-in term-indexed graph read.

        Candidate generation is ACL-agnostic: graph_context re-applies its exact predicate and its
        authorized-documents filter over the result, so over-return is harmless and the only obligation
        is to never under-return a name graph_context would match. Candidates are the entities whose
        normalized_name matches a query substring or whose token index matches a query term, plus every
        relation touching a candidate and both endpoint entities of those relations (the 1-hop frontier),
        so by_id resolves every relation post-filter. Every result is ordered by id, deterministic across
        dialects and batches.
        """
        batch = 800
        names = sorted(name_substrings)
        terms = sorted(term_candidates)
        candidate_ids: set[str] = set()
        entity_rows: dict[str, RowMapping] = {}
        relation_rows: dict[str, RowMapping] = {}
        with self.engine.connect() as conn:
            for start in range(0, len(names), batch):
                chunk = names[start : start + batch]
                if not chunk:
                    continue
                id_rows = conn.execute(
                    select(entities.c.id).where(
                        entities.c.workspace_id == workspace_id,
                        entities.c.corpus_id == corpus_id,
                        entities.c.normalized_name.in_(chunk),
                    )
                ).scalars()
                candidate_ids.update(str(value) for value in id_rows)
            for start in range(0, len(terms), batch):
                chunk = terms[start : start + batch]
                if not chunk:
                    continue
                id_rows = conn.execute(
                    select(entity_terms.c.entity_id).where(
                        entity_terms.c.workspace_id == workspace_id,
                        entity_terms.c.corpus_id == corpus_id,
                        entity_terms.c.term.in_(chunk),
                    )
                ).scalars()
                candidate_ids.update(str(value) for value in id_rows)
            if not candidate_ids:
                return [], []
            candidates = sorted(candidate_ids)
            for start in range(0, len(candidates), batch):
                chunk = candidates[start : start + batch]
                rows = conn.execute(
                    select(relations).where(
                        relations.c.workspace_id == workspace_id,
                        relations.c.corpus_id == corpus_id,
                        or_(
                            relations.c.source_entity_id.in_(chunk),
                            relations.c.target_entity_id.in_(chunk),
                        ),
                    )
                ).mappings()
                for row in rows:
                    relation_rows[str(row["id"])] = row
            entity_ids = set(candidate_ids)
            for row in relation_rows.values():
                entity_ids.add(str(row["source_entity_id"]))
                entity_ids.add(str(row["target_entity_id"]))
            all_ids = sorted(entity_ids)
            for start in range(0, len(all_ids), batch):
                chunk = all_ids[start : start + batch]
                rows = conn.execute(
                    select(entities).where(
                        entities.c.workspace_id == workspace_id,
                        entities.c.corpus_id == corpus_id,
                        entities.c.id.in_(chunk),
                    )
                ).mappings()
                for row in rows:
                    entity_rows[str(row["id"])] = row
        entity_values = [
            Entity(
                id=str(row["id"]),
                corpus_id=str(row["corpus_id"]),
                document_id=str(row["document_id"]),
                name=str(row["name"]),
                normalized_name=str(row["normalized_name"]),
                kind=str(row["kind"]),
                metadata=cast(JsonObject, row["metadata"]),
            )
            for row in sorted(entity_rows.values(), key=lambda item: str(item["id"]))
        ]
        relation_values = [
            Relation(
                id=str(row["id"]),
                corpus_id=str(row["corpus_id"]),
                document_id=str(row["document_id"]),
                source_entity_id=str(row["source_entity_id"]),
                target_entity_id=str(row["target_entity_id"]),
                kind=str(row["kind"]),
                weight=float(row["weight"]),
                metadata=cast(JsonObject, row["metadata"]),
            )
            for row in sorted(relation_rows.values(), key=lambda item: str(item["id"]))
        ]
        return entity_values, relation_values

    @staticmethod
    def _serialize_response(response: QueryResponse) -> JsonObject:
        return response_to_json(response)

    @staticmethod
    def _deserialize_response(data: Mapping[str, Any]) -> QueryResponse:
        return response_from_json(data)

    def get_exact_cache(
        self, workspace_id: str, corpus_id: str, version: int, user_scope: str, query_hash: str
    ) -> CacheEntry | None:
        now = self.clock.now()
        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    select(query_cache).where(
                        query_cache.c.workspace_id == workspace_id,
                        query_cache.c.corpus_id == corpus_id,
                        query_cache.c.corpus_version == version,
                        query_cache.c.user_scope == user_scope,
                        query_cache.c.query_hash == query_hash,
                        query_cache.c.expires_at > now,
                    )
                )
                .mappings()
                .first()
            )
        return self._cache_from_row(row) if row is not None else None

    def list_cache_candidates(
        self, workspace_id: str, corpus_id: str, version: int, user_scope: str, limit: int
    ) -> list[CacheEntry]:
        now = self.clock.now()
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(query_cache)
                .where(
                    query_cache.c.workspace_id == workspace_id,
                    query_cache.c.corpus_id == corpus_id,
                    query_cache.c.corpus_version == version,
                    query_cache.c.user_scope == user_scope,
                    query_cache.c.expires_at > now,
                )
                .order_by(query_cache.c.created_at.desc())
                .limit(limit)
            ).mappings()
            return [self._cache_from_row(row) for row in rows]

    def _cache_from_row(self, row: RowMapping) -> CacheEntry:
        return CacheEntry(
            id=str(row["id"]),
            workspace_id=str(row["workspace_id"]),
            corpus_id=str(row["corpus_id"]),
            corpus_version=int(row["corpus_version"]),
            user_scope=str(row["user_scope"]),
            query=str(row["query"]),
            query_hash=str(row["query_hash"]),
            query_embedding=tuple(float(value) for value in row["query_embedding"]),
            response=self._deserialize_response(cast(Mapping[str, Any], row["response"])),
            expires_at=cast(datetime, row["expires_at"]),
            created_at=cast(datetime, row["created_at"]),
        )

    def put_cache(self, entry: CacheEntry) -> None:
        # Best-effort on the identical-scope race only: a peer that wins the same scope
        # makes the lost write harmless. A different violation (id collision, foreign
        # key) is a real error and must surface, so suppress only when the scope is cached.
        try:
            self._retry_on_conflict(lambda: self._put_cache_once(entry))
        except IntegrityError:
            if not self._scope_is_cached(entry):
                raise

    def _scope_is_cached(self, entry: CacheEntry) -> bool:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(query_cache.c.id)
                .where(
                    query_cache.c.workspace_id == entry.workspace_id,
                    query_cache.c.corpus_id == entry.corpus_id,
                    query_cache.c.corpus_version == entry.corpus_version,
                    query_cache.c.user_scope == entry.user_scope,
                    query_cache.c.query_hash == entry.query_hash,
                )
                .limit(1)
            ).first()
        return row is not None

    def _put_cache_once(self, entry: CacheEntry) -> None:
        values = {
            "id": entry.id,
            "workspace_id": entry.workspace_id,
            "corpus_id": entry.corpus_id,
            "corpus_version": entry.corpus_version,
            "user_scope": entry.user_scope,
            "query": entry.query,
            "query_hash": entry.query_hash,
            "query_embedding": list(entry.query_embedding),
            "response": self._serialize_response(entry.response),
            "expires_at": entry.expires_at,
            "created_at": entry.created_at,
        }
        with self.engine.begin() as conn:
            conn.execute(
                delete(query_cache).where(
                    query_cache.c.workspace_id == entry.workspace_id,
                    query_cache.c.corpus_id == entry.corpus_id,
                    query_cache.c.corpus_version == entry.corpus_version,
                    query_cache.c.user_scope == entry.user_scope,
                    query_cache.c.query_hash == entry.query_hash,
                )
            )
            conn.execute(insert(query_cache).values(**values))

    def purge_expired_cache(self) -> int:
        with self.engine.begin() as conn:
            result = conn.execute(delete(query_cache).where(query_cache.c.expires_at <= self.clock.now()))
        return int(result.rowcount or 0)

    # Embeddings are content-addressed and backend-scoped, never tenant-scoped
    # (an embedding of identical text under the same backend is identical), so they
    # share one reserved global scope; generation rows carry the real workspace id.
    # The vector value leaks nothing cross-tenant; the accepted residual is a
    # same-text existence side channel (a hit is faster and charges nothing).
    _EMBEDDING_SCOPE = ""

    def get_model_call(self, workspace_id: str, model_id: str, cache_key: str) -> JsonObject | None:
        payload = self._get_model_cache_payload("generation", workspace_id, model_id, cache_key)
        return cast(JsonObject, payload) if payload is not None else None

    def put_model_call(
        self, workspace_id: str, model_id: str, cache_key: str, payload: JsonObject, ttl_seconds: int
    ) -> None:
        self._put_model_cache("generation", workspace_id, model_id, cache_key, dict(payload), ttl_seconds)

    def get_embedding(self, model_id: str, cache_key: str) -> tuple[float, ...] | None:
        payload = self._get_model_cache_payload("embedding", self._EMBEDDING_SCOPE, model_id, cache_key)
        if payload is None:
            return None
        return tuple(float(value) for value in payload["vector"])

    def put_embedding(self, model_id: str, cache_key: str, vector: tuple[float, ...], ttl_seconds: int) -> None:
        self._put_model_cache(
            "embedding", self._EMBEDDING_SCOPE, model_id, cache_key, {"vector": list(vector)}, ttl_seconds
        )

    def _get_model_cache_payload(
        self, kind: str, workspace_id: str, model_id: str, cache_key: str
    ) -> Mapping[str, Any] | None:
        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    select(model_call_cache.c.payload).where(
                        model_call_cache.c.kind == kind,
                        model_call_cache.c.workspace_id == workspace_id,
                        model_call_cache.c.model_id == model_id,
                        model_call_cache.c.cache_key == cache_key,
                        model_call_cache.c.expires_at > self.clock.now(),
                    )
                )
                .mappings()
                .first()
            )
        return cast(Mapping[str, Any], row["payload"]) if row is not None else None

    def _put_model_cache(
        self, kind: str, workspace_id: str, model_id: str, cache_key: str, payload: JsonObject, ttl_seconds: int
    ) -> None:
        now = self.clock.now()
        values = {
            "id": new_id("modelcall"),
            "kind": kind,
            "workspace_id": workspace_id,
            "model_id": model_id,
            "cache_key": cache_key,
            "payload": payload,
            "expires_at": now + timedelta(seconds=ttl_seconds),
            "created_at": now,
        }
        self._retry_on_conflict(lambda: self._put_model_cache_once(kind, workspace_id, model_id, cache_key, values))

    def _put_model_cache_once(
        self, kind: str, workspace_id: str, model_id: str, cache_key: str, values: Mapping[str, Any]
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                delete(model_call_cache).where(
                    model_call_cache.c.kind == kind,
                    model_call_cache.c.workspace_id == workspace_id,
                    model_call_cache.c.model_id == model_id,
                    model_call_cache.c.cache_key == cache_key,
                )
            )
            conn.execute(insert(model_call_cache).values(**values))

    def purge_expired_model_call_cache(self) -> int:
        with self.engine.begin() as conn:
            result = conn.execute(delete(model_call_cache).where(model_call_cache.c.expires_at <= self.clock.now()))
        return int(result.rowcount or 0)

    def create_session(
        self,
        workspace_id: str,
        corpus_id: str,
        *,
        user_id: str | None,
        goal: str | None,
        metadata_value: JsonObject | None = None,
    ) -> str:
        self.get_corpus(workspace_id, corpus_id)
        session_id = new_id("session")
        now = self.clock.now()
        with self.engine.begin() as conn:
            conn.execute(
                insert(sessions).values(
                    id=session_id,
                    workspace_id=workspace_id,
                    corpus_id=corpus_id,
                    user_id=user_id,
                    goal=goal,
                    summary=None,
                    metadata=metadata_value or {},
                    created_at=now,
                    updated_at=now,
                )
            )
        return session_id

    def get_session(self, workspace_id: str, session_id: str) -> JsonObject:
        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    select(sessions).where(
                        sessions.c.id == session_id,
                        sessions.c.workspace_id == workspace_id,
                    )
                )
                .mappings()
                .first()
            )
        if row is None:
            raise NotFoundError("Session not found")
        return dict(row)

    def append_session_turn(
        self, workspace_id: str, session_id: str, role: str, content: str, metadata_value: JsonObject | None = None
    ) -> str:
        self.get_session(workspace_id, session_id)
        turn_id = new_id("turn")
        now = self.clock.now()
        with self.engine.begin() as conn:
            conn.execute(
                insert(session_turns).values(
                    id=turn_id,
                    session_id=session_id,
                    role=role,
                    content=content,
                    metadata=metadata_value or {},
                    created_at=now,
                )
            )
            conn.execute(update(sessions).where(sessions.c.id == session_id).values(updated_at=now))
        return turn_id

    def list_session_turns(self, workspace_id: str, session_id: str) -> list[JsonObject]:
        self.get_session(workspace_id, session_id)
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(session_turns)
                .where(session_turns.c.session_id == session_id)
                .order_by(session_turns.c.created_at, session_turns.c.id)
            ).mappings()
            return [dict(row) for row in rows]

    def compact_session(
        self,
        workspace_id: str,
        session_id: str,
        *,
        summary: str,
        remove_turn_ids: Sequence[str],
        auth_scope: str | None = None,
        summary_cited_documents: Sequence[str] | None = None,
    ) -> None:
        session = self.get_session(workspace_id, session_id)
        with self.engine.begin() as conn:
            if remove_turn_ids:
                conn.execute(
                    delete(session_turns).where(
                        session_turns.c.session_id == session_id,
                        session_turns.c.id.in_(remove_turn_ids),
                    )
                )
            values: dict[str, Any] = {"summary": summary, "updated_at": self.clock.now()}
            metadata_updates: dict[str, Any] = {}
            if auth_scope is not None:
                # Bind the compacted summary to the writer principal's authorization scope so a
                # later read can surface it only for the same scope.
                metadata_updates["summary_auth_scope"] = auth_scope
            if summary_cited_documents is not None:
                # The document ids the summary folded, recomputed against current ACLs at read time so
                # a router or egress ceiling counts the summary from the latest tags. Stored whenever a
                # consumer is installed, independent of the access policy, matching the graph path.
                metadata_updates["summary_cited_documents"] = list(summary_cited_documents)
            if metadata_updates:
                existing_metadata = dict(cast(JsonObject, session.get("metadata") or {}))
                existing_metadata.update(metadata_updates)
                values["metadata"] = existing_metadata
            conn.execute(update(sessions).where(sessions.c.id == session_id).values(**values))

    def get_preferences(self, workspace_id: str, corpus_id: str, user_scope: str) -> JsonObject:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(preferences).where(
                    preferences.c.workspace_id == workspace_id,
                    preferences.c.corpus_id == corpus_id,
                    preferences.c.user_scope == user_scope,
                )
            ).mappings()
            return {
                str(row["key"]): {
                    "value": row["value"],
                    "confidence": float(row["confidence"]),
                    "observations": int(row["observations"]),
                }
                for row in rows
            }

    @staticmethod
    def _retry_on_conflict(operation: Callable[[], None]) -> None:
        for attempt in range(3):
            try:
                operation()
                return
            except IntegrityError:
                if attempt == 2:
                    raise

    @staticmethod
    def _retry_returning(operation: Callable[[], JsonObject]) -> JsonObject:
        """Retry an insert that recomputes a contended sequence on a unique violation.

        Two concurrent canary writers can each read the same ``max(sequence)`` and
        pick the same next value; the ``uq_program_canaries_sequence`` constraint
        rejects the loser, whose transaction rolled back, so a retry recomputes the
        now-higher max and both writers land strictly increasing sequences.
        """
        for attempt in range(5):
            try:
                return operation()
            except IntegrityError:
                if attempt == 4:
                    raise
        raise AssertionError("unreachable")

    def observe_preference(
        self,
        *,
        workspace_id: str,
        corpus_id: str,
        user_scope: str,
        key: str,
        value: Any,
        positive: bool,
    ) -> None:
        self._retry_on_conflict(
            lambda: self._observe_preference_once(
                workspace_id=workspace_id,
                corpus_id=corpus_id,
                user_scope=user_scope,
                key=key,
                value=value,
                positive=positive,
            )
        )

    def _observe_preference_once(
        self,
        *,
        workspace_id: str,
        corpus_id: str,
        user_scope: str,
        key: str,
        value: Any,
        positive: bool,
    ) -> None:
        now = self.clock.now()
        with self.engine.begin() as conn:
            row = (
                conn.execute(
                    select(preferences).where(
                        preferences.c.workspace_id == workspace_id,
                        preferences.c.corpus_id == corpus_id,
                        preferences.c.user_scope == user_scope,
                        preferences.c.key == key,
                    )
                )
                .mappings()
                .first()
            )
            if row is None:
                conn.execute(
                    insert(preferences).values(
                        id=new_id("pref"),
                        workspace_id=workspace_id,
                        corpus_id=corpus_id,
                        user_scope=user_scope,
                        key=key,
                        value=value,
                        confidence=0.6 if positive else 0.4,
                        observations=1,
                        updated_at=now,
                    )
                )
                return
            observations = min(int(row["observations"]) + 1, 10_000)
            previous = float(row["confidence"])
            target = 1.0 if positive else 0.0
            confidence = max(0.05, min(0.95, previous + (target - previous) / min(observations, 20)))
            conn.execute(
                update(preferences)
                .where(preferences.c.id == row["id"])
                .values(
                    value=value,
                    confidence=confidence,
                    observations=observations,
                    updated_at=now,
                )
            )

    def enqueue_job(
        self,
        workspace_id: str,
        kind: str,
        payload: JsonObject,
        *,
        run_after: datetime | None = None,
        max_attempts: int = 5,
        dedup_key: str | None = None,
    ) -> str:
        if kind in RESERVED_JOB_KINDS and not dedup_key:
            # The chokepoint guard: a reserved kind carries a run-scoped dedup invariant, so it can never be
            # enqueued without a dedup key by any caller. The generic endpoints reject it earlier; this makes
            # the invariant hold for every current and future call site, not only the ones reviewed today.
            raise ValueError(f"Job kind {kind!r} requires a dedup key")
        self.get_workspace(workspace_id)
        job_id = new_id("job")
        now = self.clock.now()
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    insert(jobs).values(
                        id=job_id,
                        workspace_id=workspace_id,
                        kind=kind,
                        payload=payload,
                        status=JobStatus.PENDING,
                        attempts=0,
                        max_attempts=max_attempts,
                        run_after=run_after or now,
                        lease_owner=None,
                        leased_until=None,
                        heartbeat_at=None,
                        last_error=None,
                        result=None,
                        dedup_key=dedup_key,
                        created_at=now,
                        updated_at=now,
                    )
                )
        except IntegrityError:
            # A concurrent enqueue already holds the active-job slot for this
            # dedup_key; converge on it rather than duplicating the run's work.
            if dedup_key is not None:
                existing = self._active_dedup_job(dedup_key)
                if existing is not None:
                    return existing
            raise
        return job_id

    def _active_dedup_job(self, dedup_key: str) -> str | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(jobs.c.id)
                .where(
                    jobs.c.dedup_key == dedup_key,
                    jobs.c.status.in_((JobStatus.PENDING, JobStatus.RUNNING)),
                )
                .order_by(jobs.c.created_at.desc())
                .limit(1)
            ).first()
        return str(row[0]) if row is not None else None

    def _insert_ignoring_conflict(self, table: Table, values: JsonObject) -> bool:
        """Insert one row, treating a duplicate as an already-present no-op.

        The insert runs in its own transaction so a losing concurrent writer's
        IntegrityError rolls back only this insert, never a sibling write, and the
        caller learns whether the row was newly created.
        """
        try:
            with self.engine.begin() as conn:
                conn.execute(insert(table).values(**values))
        except IntegrityError:
            return False
        return True

    def get_job(self, workspace_id: str, job_id: str) -> JsonObject:
        with self.engine.connect() as conn:
            row = (
                conn.execute(select(jobs).where(jobs.c.id == job_id, jobs.c.workspace_id == workspace_id))
                .mappings()
                .first()
            )
        if row is None:
            raise NotFoundError("Job not found")
        return dict(row)

    def claim_job(self, worker_id: str, lease_seconds: int) -> JsonObject | None:
        now = self.clock.now()
        leased_until = now + timedelta(seconds=lease_seconds)
        with self.engine.begin() as conn:
            candidate = conn.execute(
                select(jobs.c.id)
                .where(
                    jobs.c.run_after <= now,
                    or_(
                        jobs.c.status == JobStatus.PENDING,
                        and_(jobs.c.status == JobStatus.RUNNING, jobs.c.leased_until < now),
                    ),
                    jobs.c.attempts < jobs.c.max_attempts,
                )
                .order_by(jobs.c.run_after, jobs.c.created_at)
                .limit(1)
            ).scalar_one_or_none()
            if candidate is None:
                return None
            result = conn.execute(
                update(jobs)
                .where(
                    jobs.c.id == candidate,
                    or_(jobs.c.status == JobStatus.PENDING, jobs.c.leased_until < now),
                )
                .values(
                    status=JobStatus.RUNNING,
                    attempts=jobs.c.attempts + 1,
                    lease_owner=worker_id,
                    leased_until=leased_until,
                    heartbeat_at=now,
                    updated_at=now,
                )
            )
            if result.rowcount != 1:
                return None
            row = conn.execute(select(jobs).where(jobs.c.id == candidate)).mappings().one()
            return dict(row)

    def heartbeat_job(self, job_id: str, worker_id: str, lease_seconds: int) -> bool:
        now = self.clock.now()
        with self.engine.begin() as conn:
            result = conn.execute(
                update(jobs)
                .where(
                    jobs.c.id == job_id,
                    jobs.c.status == JobStatus.RUNNING,
                    jobs.c.lease_owner == worker_id,
                )
                .values(
                    heartbeat_at=now,
                    leased_until=now + timedelta(seconds=lease_seconds),
                    updated_at=now,
                )
            )
            return result.rowcount == 1

    def complete_job(self, job_id: str, worker_id: str, result_value: JsonObject) -> None:
        with self.engine.begin() as conn:
            result = conn.execute(
                update(jobs)
                .where(
                    jobs.c.id == job_id,
                    jobs.c.status == JobStatus.RUNNING,
                    jobs.c.lease_owner == worker_id,
                )
                .values(
                    status=JobStatus.SUCCEEDED,
                    result=result_value,
                    leased_until=None,
                    updated_at=self.clock.now(),
                )
            )
        if result.rowcount != 1:
            raise ConflictError("Job lease is no longer owned by this worker")

    def fail_job(
        self,
        job_id: str,
        worker_id: str,
        error: str,
        retry_delay_seconds: int,
        *,
        terminal: bool = False,
    ) -> None:
        now = self.clock.now()
        with self.engine.begin() as conn:
            row = (
                conn.execute(
                    select(jobs.c.attempts, jobs.c.max_attempts).where(
                        jobs.c.id == job_id,
                        jobs.c.status == JobStatus.RUNNING,
                        jobs.c.lease_owner == worker_id,
                    )
                )
                .mappings()
                .first()
            )
            if row is None:
                raise ConflictError("Job lease is no longer owned by this worker")
            terminal = terminal or int(row["attempts"]) >= int(row["max_attempts"])
            # Fold ownership into the write itself: another worker could reclaim the
            # job between the read above and this update once the lease expires, so a
            # blind update by id alone would clobber a job it no longer owns. Zero
            # rows means the lease was lost, the same conflict the worker swallows.
            outcome = conn.execute(
                update(jobs)
                .where(
                    jobs.c.id == job_id,
                    jobs.c.status == JobStatus.RUNNING,
                    jobs.c.lease_owner == worker_id,
                )
                .values(
                    status=JobStatus.FAILED if terminal else JobStatus.PENDING,
                    last_error=error[:8000],
                    run_after=now + timedelta(seconds=retry_delay_seconds),
                    lease_owner=None,
                    leased_until=None,
                    updated_at=now,
                )
            )
            if outcome.rowcount != 1:
                raise ConflictError("Job lease is no longer owned by this worker")

    def reap_abandoned_jobs(self, limit: int = 50) -> list[JsonObject]:
        """Terminalize RUNNING jobs whose lease expired after exhausting all attempts.

        A process killed on every attempt leaves ``attempts == max_attempts`` with
        an expired lease; ``claim_job`` excludes it (attempts are used up), so it
        would stay RUNNING forever. This reaps those jobs to FAILED and returns the
        affected rows so the caller can terminalize any dependent aggregate (an
        optimization run) that would otherwise stay RUNNING.
        """
        now = self.clock.now()
        with self.engine.begin() as conn:
            rows = (
                conn.execute(
                    select(jobs.c.id, jobs.c.kind, jobs.c.workspace_id, jobs.c.payload)
                    .where(
                        jobs.c.status == JobStatus.RUNNING,
                        jobs.c.leased_until < now,
                        jobs.c.attempts >= jobs.c.max_attempts,
                    )
                    .order_by(jobs.c.updated_at)
                    .limit(limit)
                )
                .mappings()
                .all()
            )
            reaped: list[JsonObject] = []
            for row in rows:
                # Re-assert the reaped predicate in the write: a slow lease owner can
                # complete a job after its lease expired but before this update, and a
                # blind update by id would then overwrite a SUCCEEDED job with FAILED.
                outcome = conn.execute(
                    update(jobs)
                    .where(
                        jobs.c.id == row["id"],
                        jobs.c.status == JobStatus.RUNNING,
                        jobs.c.leased_until < now,
                        jobs.c.attempts >= jobs.c.max_attempts,
                    )
                    .values(
                        status=JobStatus.FAILED,
                        last_error="Job abandoned after exhausting all attempts",
                        lease_owner=None,
                        leased_until=None,
                        updated_at=now,
                    )
                )
                if outcome.rowcount == 1:
                    reaped.append(dict(row))
        return reaped

    def create_schedule(
        self,
        workspace_id: str,
        name: str,
        job_kind: str,
        payload: JsonObject,
        interval_seconds: int,
        *,
        enabled: bool = True,
    ) -> str:
        if interval_seconds < 1:
            raise ValueError("interval_seconds must be positive")
        schedule_id = new_id("schedule")
        now = self.clock.now()
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    insert(schedules).values(
                        id=schedule_id,
                        workspace_id=workspace_id,
                        name=name,
                        job_kind=job_kind,
                        payload=payload,
                        interval_seconds=interval_seconds,
                        next_run_at=now + timedelta(seconds=interval_seconds),
                        enabled=enabled,
                        created_at=now,
                    )
                )
        except IntegrityError as exc:
            raise ConflictError(f"Schedule name already exists: {name}") from exc
        return schedule_id

    def dispatch_due_schedules(self, limit: int = 50) -> int:
        now = self.clock.now()
        dispatched = 0
        with self.engine.begin() as conn:
            candidates = conn.execute(
                select(schedules.c.id, schedules.c.interval_seconds)
                # A reserved kind has a run-scoped dedup invariant no periodic schedule can satisfy, so it is
                # never dispatched from a schedule regardless of how the row was created. This is the effect-
                # point guard; the schedule and job endpoints reject it earlier at creation.
                .where(
                    schedules.c.enabled.is_(True),
                    schedules.c.next_run_at <= now,
                    schedules.c.job_kind.notin_(tuple(RESERVED_JOB_KINDS)),
                )
                .order_by(schedules.c.next_run_at)
                .limit(limit)
            ).all()
            for schedule_id, interval_seconds in candidates:
                claimed = (
                    conn.execute(
                        update(schedules)
                        .where(
                            schedules.c.id == schedule_id,
                            schedules.c.enabled.is_(True),
                            schedules.c.next_run_at <= now,
                        )
                        .values(next_run_at=now + timedelta(seconds=int(interval_seconds)))
                        .returning(schedules.c.workspace_id, schedules.c.job_kind, schedules.c.payload)
                    )
                    .mappings()
                    .first()
                )
                if claimed is None:
                    continue
                conn.execute(
                    insert(jobs).values(
                        id=new_id("job"),
                        workspace_id=claimed["workspace_id"],
                        kind=claimed["job_kind"],
                        payload=claimed["payload"],
                        status=JobStatus.PENDING,
                        attempts=0,
                        max_attempts=5,
                        run_after=now,
                        lease_owner=None,
                        leased_until=None,
                        heartbeat_at=None,
                        last_error=None,
                        result=None,
                        created_at=now,
                        updated_at=now,
                    )
                )
                dispatched += 1
        return dispatched

    def create_agent(
        self,
        workspace_id: str,
        name: str,
        kind: str,
        *,
        goal: str | None,
        state: JsonObject | None = None,
        retained: bool = True,
        parent_agent_id: str | None = None,
        owner_scope: str | None = None,
    ) -> str:
        agent_id = new_id("agent")
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    insert(agents).values(
                        id=agent_id,
                        workspace_id=workspace_id,
                        name=name,
                        kind=kind,
                        goal=goal,
                        state=state or {},
                        retained=retained,
                        parent_agent_id=parent_agent_id,
                        owner_scope=owner_scope,
                        last_heartbeat_at=self.clock.now(),
                        created_at=self.clock.now(),
                    )
                )
        except IntegrityError as exc:
            raise ConflictError(f"Agent name already exists: {name}") from exc
        return agent_id

    def get_agent(self, workspace_id: str, agent_id: str) -> JsonObject:
        with self.engine.connect() as conn:
            row = (
                conn.execute(select(agents).where(agents.c.id == agent_id, agents.c.workspace_id == workspace_id))
                .mappings()
                .first()
            )
        if row is None:
            raise NotFoundError("Agent not found")
        return dict(row)

    def get_agent_message(self, workspace_id: str, message_id: str) -> JsonObject:
        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    select(agent_messages).where(
                        agent_messages.c.id == message_id, agent_messages.c.workspace_id == workspace_id
                    )
                )
                .mappings()
                .first()
            )
        if row is None:
            raise NotFoundError("Agent message not found")
        return dict(row)

    def heartbeat_agent(self, workspace_id: str, agent_id: str, state: JsonObject | None = None) -> None:
        values: JsonObject = {"last_heartbeat_at": self.clock.now()}
        if state is not None:
            values["state"] = state
        with self.engine.begin() as conn:
            result = conn.execute(
                update(agents).where(agents.c.id == agent_id, agents.c.workspace_id == workspace_id).values(**values)
            )
        if result.rowcount == 0:
            raise NotFoundError("Agent not found")

    def send_agent_message(
        self,
        workspace_id: str,
        sender_agent_id: str,
        recipient_agent_id: str,
        kind: str,
        payload: JsonObject,
        correlation_id: str | None = None,
    ) -> AgentMessage:
        self.get_agent(workspace_id, sender_agent_id)
        self.get_agent(workspace_id, recipient_agent_id)
        value = AgentMessage(
            id=new_id("msg"),
            workspace_id=workspace_id,
            sender_agent_id=sender_agent_id,
            recipient_agent_id=recipient_agent_id,
            kind=kind,
            payload=payload,
            correlation_id=correlation_id,
            created_at=self.clock.now(),
        )
        with self.engine.begin() as conn:
            conn.execute(
                insert(agent_messages).values(
                    id=value.id,
                    workspace_id=value.workspace_id,
                    sender_agent_id=value.sender_agent_id,
                    recipient_agent_id=value.recipient_agent_id,
                    kind=value.kind,
                    payload=value.payload,
                    correlation_id=value.correlation_id,
                    status=MessageStatus.PENDING,
                    created_at=value.created_at,
                    acknowledged_at=None,
                )
            )
        return value

    def receive_agent_messages(
        self, workspace_id: str, recipient_agent_id: str, *, limit: int = 50
    ) -> list[AgentMessage]:
        self.get_agent(workspace_id, recipient_agent_id)
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(agent_messages)
                .where(
                    agent_messages.c.workspace_id == workspace_id,
                    agent_messages.c.recipient_agent_id == recipient_agent_id,
                    agent_messages.c.status == MessageStatus.PENDING,
                )
                .order_by(agent_messages.c.created_at)
                .limit(limit)
            ).mappings()
            return [
                AgentMessage(
                    id=str(row["id"]),
                    workspace_id=str(row["workspace_id"]),
                    sender_agent_id=str(row["sender_agent_id"]),
                    recipient_agent_id=str(row["recipient_agent_id"]),
                    kind=str(row["kind"]),
                    payload=cast(JsonObject, row["payload"]),
                    correlation_id=cast(str | None, row["correlation_id"]),
                    created_at=cast(datetime, row["created_at"]),
                )
                for row in rows
            ]

    def acknowledge_agent_message(self, workspace_id: str, message_id: str) -> None:
        with self.engine.begin() as conn:
            result = conn.execute(
                update(agent_messages)
                .where(
                    agent_messages.c.id == message_id,
                    agent_messages.c.workspace_id == workspace_id,
                )
                .values(status=MessageStatus.ACKNOWLEDGED, acknowledged_at=self.clock.now())
            )
        if result.rowcount == 0:
            raise NotFoundError("Agent message not found")

    def active_harness(self, workspace_id: str, corpus_id: str | None) -> HarnessSnapshot | None:
        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    select(harness_snapshots)
                    .where(
                        harness_snapshots.c.workspace_id == workspace_id,
                        harness_snapshots.c.corpus_id == corpus_id,
                        harness_snapshots.c.active.is_(True),
                    )
                    .order_by(harness_snapshots.c.created_at.desc())
                    .limit(1)
                )
                .mappings()
                .first()
            )
        return self._harness_from_row(row) if row is not None else None

    @staticmethod
    def _harness_from_row(row: RowMapping) -> HarnessSnapshot:
        return HarnessSnapshot(
            id=str(row["id"]),
            workspace_id=str(row["workspace_id"]),
            corpus_id=cast(str | None, row["corpus_id"]),
            parent_id=cast(str | None, row["parent_id"]),
            base_prompt_hash=str(row["base_prompt_hash"]),
            supplemental=cast(JsonObject, row["supplemental"]),
            evidence=cast(JsonObject, row["evidence"]),
            active=bool(row["active"]),
            created_at=cast(datetime, row["created_at"]),
        )

    def create_harness_snapshot(
        self,
        *,
        workspace_id: str,
        corpus_id: str | None,
        parent_id: str | None,
        base_prompt_hash: str,
        supplemental: JsonObject,
        evidence: JsonObject,
        activate: bool,
    ) -> HarnessSnapshot:
        snapshot = HarnessSnapshot(
            id=new_id("harness"),
            workspace_id=workspace_id,
            corpus_id=corpus_id,
            parent_id=parent_id,
            base_prompt_hash=base_prompt_hash,
            supplemental=supplemental,
            evidence=evidence,
            active=activate,
            created_at=self.clock.now(),
        )
        with self.engine.begin() as conn:
            if activate:
                conn.execute(
                    update(harness_snapshots)
                    .where(
                        harness_snapshots.c.workspace_id == workspace_id,
                        harness_snapshots.c.corpus_id == corpus_id,
                        harness_snapshots.c.active.is_(True),
                    )
                    .values(active=False)
                )
            conn.execute(
                insert(harness_snapshots).values(
                    id=snapshot.id,
                    workspace_id=snapshot.workspace_id,
                    corpus_id=snapshot.corpus_id,
                    parent_id=snapshot.parent_id,
                    base_prompt_hash=snapshot.base_prompt_hash,
                    supplemental=snapshot.supplemental,
                    evidence=snapshot.evidence,
                    active=snapshot.active,
                    created_at=snapshot.created_at,
                )
            )
        return snapshot

    def activate_harness_snapshot(self, workspace_id: str, snapshot_id: str) -> HarnessSnapshot:
        with self.engine.begin() as conn:
            row = (
                conn.execute(
                    select(harness_snapshots).where(
                        harness_snapshots.c.id == snapshot_id,
                        harness_snapshots.c.workspace_id == workspace_id,
                    )
                )
                .mappings()
                .first()
            )
            if row is None:
                raise NotFoundError("Harness snapshot not found")
            conn.execute(
                update(harness_snapshots)
                .where(
                    harness_snapshots.c.workspace_id == workspace_id,
                    harness_snapshots.c.corpus_id == row["corpus_id"],
                )
                .values(active=False)
            )
            conn.execute(update(harness_snapshots).where(harness_snapshots.c.id == snapshot_id).values(active=True))
        return replace(self._harness_from_row(row), active=True)

    def create_evaluation_suite(self, workspace_id: str, name: str, cases: list[JsonObject]) -> str:
        suite_id = new_id("suite")
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    insert(evaluation_suites).values(
                        id=suite_id,
                        workspace_id=workspace_id,
                        name=name,
                        cases=cases,
                        created_at=self.clock.now(),
                    )
                )
        except IntegrityError as exc:
            raise ConflictError(f"Evaluation suite already exists: {name}") from exc
        return suite_id

    def get_evaluation_suite(self, workspace_id: str, suite_id: str) -> JsonObject:
        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    select(evaluation_suites).where(
                        evaluation_suites.c.id == suite_id,
                        evaluation_suites.c.workspace_id == workspace_id,
                    )
                )
                .mappings()
                .first()
            )
        if row is None:
            raise NotFoundError("Evaluation suite not found")
        return dict(row)

    def create_evaluation_run(self, workspace_id: str, suite_id: str, corpus_id: str) -> str:
        self.get_evaluation_suite(workspace_id, suite_id)
        self.get_corpus(workspace_id, corpus_id)
        run_id = new_id("eval")
        with self.engine.begin() as conn:
            conn.execute(
                insert(evaluation_runs).values(
                    id=run_id,
                    workspace_id=workspace_id,
                    suite_id=suite_id,
                    corpus_id=corpus_id,
                    status=EvaluationStatus.RUNNING,
                    metrics={},
                    results=[],
                    error=None,
                    started_at=self.clock.now(),
                    finished_at=None,
                )
            )
        return run_id

    def finish_evaluation_run(
        self,
        workspace_id: str,
        run_id: str,
        *,
        metrics: JsonObject,
        results: list[JsonObject],
        error: str | None = None,
    ) -> None:
        with self.engine.begin() as conn:
            result = conn.execute(
                update(evaluation_runs)
                .where(
                    evaluation_runs.c.id == run_id,
                    evaluation_runs.c.workspace_id == workspace_id,
                    # Monotonic: a committed SUCCEEDED is terminal, a straggler cannot corrupt it.
                    evaluation_runs.c.status != EvaluationStatus.SUCCEEDED,
                )
                .values(
                    # error is not None marks the failure, not truthiness: an empty
                    # error string is still a failure and must not flip to SUCCEEDED.
                    status=EvaluationStatus.FAILED if error is not None else EvaluationStatus.SUCCEEDED,
                    metrics=metrics,
                    results=results,
                    error=error,
                    finished_at=self.clock.now(),
                )
            )
            if result.rowcount == 0:
                already = conn.execute(
                    select(evaluation_runs.c.id).where(
                        evaluation_runs.c.id == run_id,
                        evaluation_runs.c.workspace_id == workspace_id,
                    )
                ).first()
                if already is None:
                    raise NotFoundError("Evaluation run not found")

    def get_evaluation_run(self, workspace_id: str, run_id: str) -> JsonObject:
        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    select(evaluation_runs).where(
                        evaluation_runs.c.id == run_id,
                        evaluation_runs.c.workspace_id == workspace_id,
                    )
                )
                .mappings()
                .first()
            )
        if row is None:
            raise NotFoundError("Evaluation run not found")
        return dict(row)

    def create_optimization_run(
        self,
        workspace_id: str,
        *,
        program: str,
        corpus_id: str,
        dataset_name: str,
        dataset_version: str,
        dataset_cases: list[JsonObject],
        search_space: list[JsonObject],
        bundle_id: str,
        budget: int | None,
        plan_hash: str,
        program_definition_hash: str | None = None,
        program_immutable_policy_hash: str | None = None,
    ) -> str:
        self.get_corpus(workspace_id, corpus_id)
        run_id = new_id("opt")
        now = self.clock.now()
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    insert(optimization_runs).values(
                        id=run_id,
                        workspace_id=workspace_id,
                        program=program,
                        corpus_id=corpus_id,
                        dataset_name=dataset_name,
                        dataset_version=dataset_version,
                        dataset_cases=dataset_cases,
                        search_space=search_space,
                        bundle_id=bundle_id,
                        program_definition_hash=program_definition_hash,
                        program_immutable_policy_hash=program_immutable_policy_hash,
                        budget=budget,
                        candidate_total=len(search_space),
                        plan_hash=plan_hash,
                        # A run is created PENDING and advanced to RUNNING only when the
                        # worker actually starts it, so a crash between the create and the
                        # enqueue leaves a visibly PENDING run, never a falsely RUNNING one.
                        status=EvaluationStatus.PENDING,
                        result={},
                        best_artifact_id=None,
                        error=None,
                        started_at=now,
                        updated_at=now,
                        finished_at=None,
                    )
                )
        except IntegrityError as exc:
            # The partial-unique index rejects a second non-terminal run for the same
            # plan: a concurrent enqueue already won. Surface a conflict the caller
            # converges on by re-reading that winner.
            raise ConflictError("A non-terminal optimization run already exists for this plan") from exc
        return run_id

    def get_optimization_run_by_plan(self, workspace_id: str, program: str, plan_hash: str) -> JsonObject | None:
        """The newest non-terminal run for an identical plan, so enqueue is idempotent.

        Two enqueues of the same materialized plan reuse the in-flight run and its
        job instead of creating a duplicate; a terminal (SUCCEEDED or FAILED) run
        never blocks a fresh enqueue.
        """
        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    select(optimization_runs)
                    .where(
                        optimization_runs.c.workspace_id == workspace_id,
                        optimization_runs.c.program == program,
                        optimization_runs.c.plan_hash == plan_hash,
                        optimization_runs.c.status.in_((EvaluationStatus.PENDING, EvaluationStatus.RUNNING)),
                    )
                    .order_by(optimization_runs.c.started_at.desc())
                    .limit(1)
                )
                .mappings()
                .first()
            )
        return dict(row) if row is not None else None

    def mark_optimization_run_running(self, workspace_id: str, run_id: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                update(optimization_runs)
                .where(
                    optimization_runs.c.id == run_id,
                    optimization_runs.c.workspace_id == workspace_id,
                    optimization_runs.c.status != EvaluationStatus.SUCCEEDED,
                )
                .values(status=EvaluationStatus.RUNNING, updated_at=self.clock.now())
            )

    def get_optimization_run(self, workspace_id: str, run_id: str) -> JsonObject:
        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    select(optimization_runs).where(
                        optimization_runs.c.id == run_id,
                        optimization_runs.c.workspace_id == workspace_id,
                    )
                )
                .mappings()
                .first()
            )
        if row is None:
            raise NotFoundError("Optimization run not found")
        return dict(row)

    def record_optimization_candidate(self, run_id: str, ordinal: int, candidate: JsonObject) -> None:
        with self.engine.begin() as conn:
            existing = conn.execute(
                select(optimization_candidates.c.id).where(
                    optimization_candidates.c.run_id == run_id,
                    optimization_candidates.c.ordinal == ordinal,
                )
            ).first()
            if existing is not None:
                return
            try:
                conn.execute(
                    insert(optimization_candidates).values(
                        id=new_id("optc"),
                        run_id=run_id,
                        ordinal=ordinal,
                        tunables=candidate["tunables"],
                        artifact_id=candidate["artifact_id"],
                        definition_hash=candidate.get("definition_hash"),
                        immutable_policy_hash=candidate.get("immutable_policy_hash"),
                        hard_constraints_passed=bool(candidate["hard_constraints_passed"]),
                        score=float(candidate["score"]),
                        aggregate=candidate["aggregate"],
                        mean_tokens=float(candidate["mean_tokens"]),
                        created_at=self.clock.now(),
                    )
                )
            except IntegrityError:
                return

    def get_optimization_candidate(self, run_id: str, ordinal: int) -> JsonObject | None:
        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    select(optimization_candidates).where(
                        optimization_candidates.c.run_id == run_id,
                        optimization_candidates.c.ordinal == ordinal,
                    )
                )
                .mappings()
                .first()
            )
        return dict(row) if row is not None else None

    def list_optimization_candidates(self, run_id: str) -> list[JsonObject]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(optimization_candidates)
                .where(optimization_candidates.c.run_id == run_id)
                .order_by(optimization_candidates.c.ordinal)
            ).mappings()
            return [dict(row) for row in rows]

    def finish_optimization_run(
        self,
        workspace_id: str,
        run_id: str,
        *,
        result: JsonObject,
        best_artifact_id: str | None,
        error: str | None = None,
    ) -> None:
        """Finalize a run as SUCCEEDED or FAILED, never overwriting a prior success.

        Only SUCCEEDED is treated as terminal here: a reaper-set FAILED is
        intentionally not terminal, so a late resumer that completed the plan may
        still flip a FAILED run to SUCCEEDED. That is by design and benign because
        the winning artifact is registered before this call, so best_artifact_id
        always resolves regardless of which state the row lands in.
        """
        with self.engine.begin() as conn:
            outcome = conn.execute(
                update(optimization_runs)
                .where(
                    optimization_runs.c.id == run_id,
                    optimization_runs.c.workspace_id == workspace_id,
                    # Monotonic on the terminal SUCCEEDED state: a committed success is
                    # authoritative and a later racing resumer's FAILED (or re-SUCCEEDED)
                    # must never overwrite it.
                    optimization_runs.c.status != EvaluationStatus.SUCCEEDED,
                )
                .values(
                    # An empty error string is still a failure: distinguish
                    # error=None (success) from error is not None (failure), so a
                    # RuntimeError with an empty message never flips a run to
                    # SUCCEEDED and short-circuits the retry.
                    status=EvaluationStatus.FAILED if error is not None else EvaluationStatus.SUCCEEDED,
                    result=result,
                    best_artifact_id=best_artifact_id,
                    error=error,
                    updated_at=self.clock.now(),
                    finished_at=self.clock.now(),
                )
            )
        if outcome.rowcount == 0:
            # A run already SUCCEEDED is left untouched (idempotent); only a truly
            # missing run is an error.
            run = self.get_optimization_run(workspace_id, run_id)
            if run["status"] != EvaluationStatus.SUCCEEDED:
                raise NotFoundError("Optimization run not found")

    def fail_optimization_run(self, workspace_id: str, run_id: str, error: str) -> bool:
        """Terminalize a non-terminal run as FAILED, leaving a terminal run untouched.

        Used by the abandoned-job reaper so a run whose job exhausted its attempts
        does not stay RUNNING forever. Idempotent: a SUCCEEDED or already FAILED run
        is left as is.
        """
        now = self.clock.now()
        with self.engine.begin() as conn:
            outcome = conn.execute(
                update(optimization_runs)
                .where(
                    optimization_runs.c.id == run_id,
                    optimization_runs.c.workspace_id == workspace_id,
                    optimization_runs.c.status.in_((EvaluationStatus.PENDING, EvaluationStatus.RUNNING)),
                )
                .values(
                    status=EvaluationStatus.FAILED,
                    error=error,
                    updated_at=now,
                    finished_at=now,
                )
            )
        return outcome.rowcount == 1

    def reenqueue_orphan_optimization_runs(self, grace_seconds: int, limit: int = 50) -> list[str]:
        """Re-enqueue a job for a PENDING run that has no live job.

        A crash between ``create_optimization_run`` and ``enqueue_job`` leaves a
        PENDING run with no job, which nothing would ever pick up. This finds those
        runs (older than the grace window so an in-flight enqueue is never reaped)
        and enqueues a fresh ``optimization.run`` job. Resume is idempotent, so even
        a duplicate job converges: the SUCCEEDED short-circuit and the per-ordinal
        insert-if-absent make a second run a no-op. Returns the repaired run ids.
        """
        cutoff = self.clock.now() - timedelta(seconds=grace_seconds)
        with self.engine.begin() as conn:
            pending = conn.execute(
                select(optimization_runs.c.id, optimization_runs.c.workspace_id)
                .where(
                    optimization_runs.c.status == EvaluationStatus.PENDING,
                    optimization_runs.c.updated_at < cutoff,
                )
                .order_by(optimization_runs.c.updated_at)
                .limit(limit)
            ).all()
            if not pending:
                return []
            live_payloads = conn.execute(
                select(jobs.c.payload).where(
                    jobs.c.kind == "optimization.run",
                    jobs.c.status.in_((JobStatus.PENDING, JobStatus.RUNNING)),
                )
            ).scalars()
            referenced = {
                str(payload["run_id"])
                for payload in live_payloads
                if isinstance(payload, dict) and payload.get("run_id")
            }
            orphans = [
                (str(run_id), str(workspace_id)) for run_id, workspace_id in pending if str(run_id) not in referenced
            ]
        repaired: list[str] = []
        now = self.clock.now()
        for run_id, workspace_id in orphans:
            # The referenced snapshot only closes the common case; the dedup index is
            # the real serialization. A legitimate enqueue that landed after the
            # snapshot makes this insert a no-op, so exactly one job exists per run.
            created = self._insert_ignoring_conflict(
                jobs,
                {
                    "id": new_id("job"),
                    "workspace_id": workspace_id,
                    "kind": "optimization.run",
                    "payload": {"run_id": run_id},
                    "status": JobStatus.PENDING,
                    "attempts": 0,
                    "max_attempts": 5,
                    "run_after": now,
                    "lease_owner": None,
                    "leased_until": None,
                    "heartbeat_at": None,
                    "last_error": None,
                    "result": None,
                    "dedup_key": run_id,
                    "created_at": now,
                    "updated_at": now,
                },
            )
            if created:
                repaired.append(run_id)
        return repaired

    def reenqueue_orphan_fine_tune_runs(self, grace_seconds: int, limit: int = 50) -> list[str]:
        """Re-drive a RUNNING fine-tune run that has no live poll job.

        The worker completes a poll job and then, in a separate transaction, enqueues
        the next delayed poll. A crash in that window leaves the run RUNNING with a
        provider job reference but no pending or running fine_tune.run job, so nothing
        polls it again. This finds those runs (older than the grace window so an
        in-flight enqueue is never reaped) and re-enqueues a poll. The run-scoped dedup
        index serializes it, so a concurrent legitimate enqueue makes this a no-op and
        exactly one poll job exists per run. Returns the re-driven run ids.
        """
        cutoff = self.clock.now() - timedelta(seconds=grace_seconds)
        with self.engine.begin() as conn:
            running = conn.execute(
                select(fine_tune_runs.c.id, fine_tune_runs.c.workspace_id)
                .where(
                    fine_tune_runs.c.status == EvaluationStatus.RUNNING,
                    fine_tune_runs.c.provider_job_ref.isnot(None),
                    fine_tune_runs.c.updated_at < cutoff,
                )
                .order_by(fine_tune_runs.c.updated_at)
                .limit(limit)
            ).all()
            if not running:
                return []
            live_payloads = conn.execute(
                select(jobs.c.payload).where(
                    jobs.c.kind == "fine_tune.run",
                    jobs.c.status.in_((JobStatus.PENDING, JobStatus.RUNNING)),
                )
            ).scalars()
            referenced = {
                str(payload["run_id"])
                for payload in live_payloads
                if isinstance(payload, dict) and payload.get("run_id")
            }
            orphans = [
                (str(run_id), str(workspace_id)) for run_id, workspace_id in running if str(run_id) not in referenced
            ]
        repaired: list[str] = []
        now = self.clock.now()
        for run_id, workspace_id in orphans:
            created = self._insert_ignoring_conflict(
                jobs,
                {
                    "id": new_id("job"),
                    "workspace_id": workspace_id,
                    "kind": "fine_tune.run",
                    "payload": {"run_id": run_id},
                    "status": JobStatus.PENDING,
                    "attempts": 0,
                    "max_attempts": 5,
                    "run_after": now,
                    "lease_owner": None,
                    "leased_until": None,
                    "heartbeat_at": None,
                    "last_error": None,
                    "result": None,
                    "dedup_key": run_id,
                    "created_at": now,
                    "updated_at": now,
                },
            )
            if created:
                repaired.append(run_id)
        return repaired

    def create_fine_tune_run(
        self,
        workspace_id: str,
        *,
        dataset_version: str,
        base_model: str,
        hyperparameters: JsonObject,
        requester: str,
        provenance: JsonObject,
        content_key: str,
    ) -> str:
        self.get_workspace(workspace_id)
        run_id = new_id("ft")
        now = self.clock.now()
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    insert(fine_tune_runs).values(
                        id=run_id,
                        workspace_id=workspace_id,
                        dataset_version=dataset_version,
                        base_model=base_model,
                        hyperparameters=hyperparameters,
                        # A run is created PENDING and never approved: governance requires an
                        # explicit approval before the job may execute the training request.
                        status=EvaluationStatus.PENDING,
                        approved=False,
                        approved_by=None,
                        requester=requester,
                        result_model_ref=None,
                        provenance=provenance,
                        content_key=content_key,
                        created_at=now,
                        updated_at=now,
                    )
                )
        except IntegrityError as exc:
            raise ConflictError("A non-terminal fine-tune run already exists for this request") from exc
        return run_id

    def get_fine_tune_run(self, workspace_id: str, run_id: str) -> JsonObject:
        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    select(fine_tune_runs).where(
                        fine_tune_runs.c.id == run_id,
                        fine_tune_runs.c.workspace_id == workspace_id,
                    )
                )
                .mappings()
                .first()
            )
        if row is None:
            raise NotFoundError("Fine-tune run not found")
        return dict(row)

    def approve_fine_tune_run(self, workspace_id: str, run_id: str, approver: str) -> JsonObject:
        """Approve the training data of a PENDING run, gating execution.

        Only a PENDING run can be approved: an approval must precede any execution,
        so a run cannot be approved after it has already started or terminalized.
        """
        with self.engine.begin() as conn:
            outcome = conn.execute(
                update(fine_tune_runs)
                .where(
                    fine_tune_runs.c.id == run_id,
                    fine_tune_runs.c.workspace_id == workspace_id,
                    fine_tune_runs.c.status == EvaluationStatus.PENDING,
                    fine_tune_runs.c.approved.is_(False),
                )
                .values(approved=True, approved_by=approver, updated_at=self.clock.now())
            )
        if outcome.rowcount == 0:
            run = self.get_fine_tune_run(workspace_id, run_id)
            raise ConflictError(f"Fine-tune run cannot be approved in status {run['status']!r}")
        return self.get_fine_tune_run(workspace_id, run_id)

    def mark_fine_tune_run_running(self, workspace_id: str, run_id: str) -> None:
        # A compare-and-set: only a PENDING or already-RUNNING run may start, so a terminalized run never
        # silently re-runs. RUNNING re-entry is required for a delayed poll of a long provider job and for
        # crash recovery. At-most-one concurrent runner is guaranteed one layer up (the fine_tune.run job
        # carries dedup_key=run_id under a partial-unique index, and the worker lease owns it), and the
        # provider Idempotency-Key guarantees at most one provider job even across a create-then-crash retry.
        with self.engine.begin() as conn:
            outcome = conn.execute(
                update(fine_tune_runs)
                .where(
                    fine_tune_runs.c.id == run_id,
                    fine_tune_runs.c.workspace_id == workspace_id,
                    fine_tune_runs.c.status.in_((EvaluationStatus.PENDING, EvaluationStatus.RUNNING)),
                )
                .values(status=EvaluationStatus.RUNNING, updated_at=self.clock.now())
            )
        if outcome.rowcount == 0:
            run = self.get_fine_tune_run(workspace_id, run_id)
            raise ConflictError(f"Fine-tune run cannot start in status {run['status']!r}")

    def set_fine_tune_provider_job_ref(self, workspace_id: str, run_id: str, provider_job_ref: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                update(fine_tune_runs)
                .where(
                    fine_tune_runs.c.id == run_id,
                    fine_tune_runs.c.workspace_id == workspace_id,
                )
                .values(provider_job_ref=provider_job_ref, updated_at=self.clock.now())
            )

    def finish_fine_tune_run(
        self,
        workspace_id: str,
        run_id: str,
        *,
        result_model_ref: str | None = None,
        provenance: JsonObject | None = None,
        error: str | None = None,
    ) -> None:
        """Finalize a run as SUCCEEDED with its model reference, or FAILED with an error.

        Never overwrites a committed SUCCEEDED run, so a re-run after a crash is
        idempotent. An empty error string still counts as a failure: the outcome is
        decided on ``error is not None``, so a raised error with no message never
        flips a run to SUCCEEDED.
        """
        now = self.clock.now()
        values: JsonObject = {
            "status": EvaluationStatus.FAILED if error is not None else EvaluationStatus.SUCCEEDED,
            "updated_at": now,
        }
        if error is None:
            values["result_model_ref"] = result_model_ref
        if provenance is not None:
            values["provenance"] = provenance
        with self.engine.begin() as conn:
            if error is not None:
                # Record the inspectable error inside the provenance payload, since the
                # run keeps its provenance column; a failed run stays fully inspectable.
                existing = conn.execute(
                    select(fine_tune_runs.c.provenance).where(
                        fine_tune_runs.c.id == run_id,
                        fine_tune_runs.c.workspace_id == workspace_id,
                    )
                ).scalar_one_or_none()
                merged = dict(existing) if isinstance(existing, dict) else {}
                merged["error"] = error
                if provenance is not None:
                    merged.update(provenance)
                values["provenance"] = merged
            outcome = conn.execute(
                update(fine_tune_runs)
                .where(
                    fine_tune_runs.c.id == run_id,
                    fine_tune_runs.c.workspace_id == workspace_id,
                    fine_tune_runs.c.status != EvaluationStatus.SUCCEEDED,
                )
                .values(**values)
            )
        if outcome.rowcount == 0:
            run = self.get_fine_tune_run(workspace_id, run_id)
            if run["status"] != EvaluationStatus.SUCCEEDED:
                raise NotFoundError("Fine-tune run not found")

    def fail_fine_tune_run(self, workspace_id: str, run_id: str, error: str) -> bool:
        """Terminalize a non-terminal run as FAILED, leaving a terminal run untouched.

        Used by the abandoned-job reaper and the exhausted-job path so a run whose
        job exhausted its attempts does not stay RUNNING forever. Idempotent.
        """
        now = self.clock.now()
        with self.engine.begin() as conn:
            existing = conn.execute(
                select(fine_tune_runs.c.provenance).where(
                    fine_tune_runs.c.id == run_id,
                    fine_tune_runs.c.workspace_id == workspace_id,
                    fine_tune_runs.c.status.in_((EvaluationStatus.PENDING, EvaluationStatus.RUNNING)),
                )
            ).scalar_one_or_none()
            if existing is None:
                return False
            merged = dict(existing) if isinstance(existing, dict) else {}
            merged["error"] = error
            outcome = conn.execute(
                update(fine_tune_runs)
                .where(
                    fine_tune_runs.c.id == run_id,
                    fine_tune_runs.c.workspace_id == workspace_id,
                    fine_tune_runs.c.status.in_((EvaluationStatus.PENDING, EvaluationStatus.RUNNING)),
                )
                .values(status=EvaluationStatus.FAILED, provenance=merged, updated_at=now)
            )
        return outcome.rowcount == 1

    def pending_outbox(self, limit: int = 100) -> list[JsonObject]:
        now = self.clock.now()
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(outbox)
                .where(outbox.c.delivered_at.is_(None), outbox.c.available_at <= now)
                .order_by(outbox.c.created_at)
                .limit(limit)
            ).mappings()
            return [dict(row) for row in rows]

    def claim_outbox(self, worker_id: str, lease_seconds: int, limit: int = 100) -> list[JsonObject]:
        now = self.clock.now()
        claimed_until = now + timedelta(seconds=lease_seconds)
        unclaimed = or_(outbox.c.claimed_until.is_(None), outbox.c.claimed_until < now)
        with self.engine.begin() as conn:
            candidates = [
                str(row[0])
                for row in conn.execute(
                    select(outbox.c.id)
                    .where(outbox.c.delivered_at.is_(None), outbox.c.available_at <= now, unclaimed)
                    .order_by(outbox.c.created_at)
                    .limit(limit)
                ).all()
            ]
            if not candidates:
                return []
            rows = conn.execute(
                update(outbox)
                .where(outbox.c.id.in_(candidates), outbox.c.delivered_at.is_(None), unclaimed)
                .values(claimed_by=worker_id, claimed_until=claimed_until)
                .returning(outbox)
            ).mappings()
            return sorted((dict(row) for row in rows), key=lambda event: event["created_at"])

    def mark_outbox_delivered(self, event_id: str, worker_id: str) -> bool:
        # Only the current lease owner may finalize an event: a reclaim after the
        # lease expired hands the event to another worker, so a stale owner must not
        # be able to mark it delivered. Gate on claimed_by plus an unexpired lease.
        now = self.clock.now()
        with self.engine.begin() as conn:
            result = conn.execute(
                update(outbox)
                .where(
                    outbox.c.id == event_id,
                    outbox.c.delivered_at.is_(None),
                    outbox.c.claimed_by == worker_id,
                    outbox.c.claimed_until >= now,
                )
                .values(delivered_at=now)
            )
        return result.rowcount == 1

    def purge_delivered_outbox(self, retention_days: int) -> int:
        """Delete fully delivered outbox events older than the retention window.

        Their per-sink ledger rows in ``outbox_deliveries`` cascade away with the
        event, so the ledger stays bounded. Only rows with ``delivered_at`` set are
        removed, so an undelivered or in-flight event is never dropped.
        """
        cutoff = self.clock.now() - timedelta(days=retention_days)
        with self.engine.begin() as conn:
            result = conn.execute(
                delete(outbox).where(outbox.c.delivered_at.is_not(None), outbox.c.delivered_at < cutoff)
            )
        return int(result.rowcount or 0)

    def delivered_outbox_sinks(self, event_id: str) -> set[str]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(outbox_deliveries.c.sink_name).where(outbox_deliveries.c.event_id == event_id)
            ).all()
        return {str(row[0]) for row in rows}

    def record_outbox_delivery(self, event_id: str, sink_name: str) -> None:
        """Persist that a sink delivered an event, once. Idempotent.

        A concurrent or redelivered publish may record the same pair; the unique
        constraint makes the duplicate a no-op rather than an error, so the ledger
        stays a faithful record of which sinks have succeeded for an event.
        """
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    insert(outbox_deliveries).values(
                        id=new_id("obd"),
                        event_id=event_id,
                        sink_name=sink_name,
                        delivered_at=self.clock.now(),
                    )
                )
        except IntegrityError:
            pass

    def mark_outbox_failed(self, event_id: str, worker_id: str, error: str, delay_seconds: int) -> None:
        now = self.clock.now()
        with self.engine.begin() as conn:
            conn.execute(
                update(outbox)
                .where(
                    outbox.c.id == event_id,
                    outbox.c.delivered_at.is_(None),
                    outbox.c.claimed_by == worker_id,
                    outbox.c.claimed_until >= now,
                )
                .values(
                    attempts=outbox.c.attempts + 1,
                    last_error=error[:4000],
                    available_at=now + timedelta(seconds=delay_seconds),
                    claimed_by=None,
                    claimed_until=None,
                )
            )

    def _insert_audit_outbox(self, conn: Any, workspace_id: str, event: AuditEvent) -> str:
        # Persist one audit event on an open connection, so a caller can enqueue it inside the same
        # transaction as the state change it records (durable at-least-once, atomic with the change).
        now = self.clock.now()
        event_id = new_id("aevt")
        conn.execute(
            insert(audit_outbox).values(
                id=event_id,
                workspace_id=workspace_id,
                event_type=event.event_type,
                payload=event.to_json(),
                attempts=0,
                available_at=now,
                delivered_at=None,
                last_error=None,
                created_at=now,
            )
        )
        return event_id

    def enqueue_audit_event(self, workspace_id: str, event: AuditEvent) -> str:
        """Persist an audit event for durable at-least-once delivery. Returns the row id."""
        with self.engine.begin() as conn:
            return self._insert_audit_outbox(conn, workspace_id, event)

    @staticmethod
    def _fold_chunk_digest(pairs: Iterable[tuple[str, str]]) -> str:
        # The order-independent XOR fold over (chunk_id, checksum) pairs, shared so a per-document digest
        # and the corpus rollup are folded by identical bytes.
        acc = bytearray(32)
        for chunk_id, checksum in pairs:
            digest = hashlib.sha256(f"{chunk_id}\x00{checksum}".encode()).digest()
            for i in range(32):
                acc[i] ^= digest[i]
        return acc.hex()

    def derivation_rollup(self, workspace_id: str, corpus_id: str) -> tuple[dict[str, int], str]:
        # An order-independent checksum folded by XOR over streamed chunk (id, checksum) pairs, so the
        # digest is row-order stable and the rollup stays bounded in memory on a large corpus.
        tables = {
            "documents": documents,
            "chunks": chunks,
            "hypotheticals": chunk_hypotheticals,
            "raptor_nodes": raptor_nodes,
            "entities": entities,
            "relations": relations,
        }
        with self.engine.connect() as conn:
            counts = {
                name: int(
                    conn.execute(
                        select(func.count())
                        .select_from(table)
                        .where(table.c.workspace_id == workspace_id, table.c.corpus_id == corpus_id)
                    ).scalar_one()
                )
                for name, table in tables.items()
            }
            stream = conn.execution_options(stream_results=True).execute(
                select(chunks.c.id, chunks.c.checksum).where(
                    chunks.c.workspace_id == workspace_id, chunks.c.corpus_id == corpus_id
                )
            )
            chunk_body = self._fold_chunk_digest((str(chunk_id), str(checksum)) for chunk_id, checksum in stream)
        content_checksum = sha256_text("|".join([chunk_body, *(f"{name}={counts[name]}" for name in sorted(counts))]))
        return counts, content_checksum

    def upsert_derivation_manifest(
        self,
        workspace_id: str,
        corpus_id: str,
        *,
        derivation_key: str,
        corpus_version: int,
        config: JsonObject,
        embedding_model: str,
        embedding_dimensions: int,
        row_counts: JsonObject,
        content_checksum: str,
        status: str,
    ) -> None:
        now = self.clock.now()
        values = {
            "corpus_version": corpus_version,
            "config": config,
            "embedding_model": embedding_model,
            "embedding_dimensions": embedding_dimensions,
            "row_counts": row_counts,
            "content_checksum": content_checksum,
            "status": status,
            "updated_at": now,
        }
        with self.engine.begin() as conn:
            existing = conn.execute(
                select(derivation_manifests.c.id).where(
                    derivation_manifests.c.workspace_id == workspace_id,
                    derivation_manifests.c.corpus_id == corpus_id,
                    derivation_manifests.c.derivation_key == derivation_key,
                )
            ).scalar_one_or_none()
            if existing is None:
                conn.execute(
                    insert(derivation_manifests).values(
                        id=new_id("drv"),
                        workspace_id=workspace_id,
                        corpus_id=corpus_id,
                        derivation_key=derivation_key,
                        created_at=now,
                        **values,
                    )
                )
            else:
                # Monotonic: only advance to a corpus_version at least the stored one, so a slow stale
                # refresh that raced a fresher one cannot overwrite it.
                conn.execute(
                    update(derivation_manifests)
                    .where(
                        derivation_manifests.c.id == existing,
                        derivation_manifests.c.corpus_version <= corpus_version,
                    )
                    .values(**values)
                )

    def get_derivation_manifest(self, workspace_id: str, corpus_id: str, derivation_key: str) -> JsonObject | None:
        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    select(derivation_manifests).where(
                        derivation_manifests.c.workspace_id == workspace_id,
                        derivation_manifests.c.corpus_id == corpus_id,
                        derivation_manifests.c.derivation_key == derivation_key,
                    )
                )
                .mappings()
                .first()
            )
        return dict(row) if row is not None else None

    def list_derivation_manifests(self, workspace_id: str, corpus_id: str) -> list[JsonObject]:
        with self.engine.connect() as conn:
            rows = (
                conn.execute(
                    select(derivation_manifests)
                    .where(
                        derivation_manifests.c.workspace_id == workspace_id,
                        derivation_manifests.c.corpus_id == corpus_id,
                    )
                    .order_by(
                        derivation_manifests.c.updated_at.desc(),
                        derivation_manifests.c.derivation_key,
                    )
                )
                .mappings()
                .all()
            )
        return [dict(row) for row in rows]

    def get_active_index_generation(self, workspace_id: str, corpus_id: str) -> str | None:
        # The active generation's content identity for the answer-cache fold. Degrades to None (never
        # ""), swallowing a missing-table/read error, so the off path folds nothing and, unlike the
        # kill-switch mirror, the hot query path it runs on never raises.
        try:
            with self.engine.connect() as conn:
                row = (
                    conn.execute(
                        select(
                            index_generations.c.derivation_key,
                            index_generations.c.content_checksum,
                        ).where(
                            index_generations.c.workspace_id == workspace_id,
                            index_generations.c.corpus_id == corpus_id,
                            index_generations.c.status == "active",
                        )
                    )
                    .mappings()
                    .first()
                )
        except SQLAlchemyError:
            return None
        if row is None:
            return None
        key = str(row["derivation_key"])
        checksum = str(row["content_checksum"])
        if not key or not checksum:
            return None
        return sha256_text(f"{key}:{checksum}")

    def document_derivation_digest(self, workspace_id: str, corpus_id: str, document_id: str) -> str:
        # Recompute the per-document digest from committed chunks so verify can compare it to the stored
        # column and localize a drifted document.
        with self.engine.connect() as conn:
            stream = conn.execution_options(stream_results=True).execute(
                select(chunks.c.id, chunks.c.checksum).where(
                    chunks.c.workspace_id == workspace_id,
                    chunks.c.corpus_id == corpus_id,
                    chunks.c.document_id == document_id,
                )
            )
            return self._fold_chunk_digest((str(chunk_id), str(checksum)) for chunk_id, checksum in stream)

    def get_document_derivation_digest(
        self, workspace_id: str, corpus_id: str, document_id: str
    ) -> tuple[bool, str | None]:
        # (exists, stored digest): existence distinguishes a missing document (no_document) from a present
        # row whose digest is NULL (no_digest, e.g. feature off or a pre-migration row).
        with self.engine.connect() as conn:
            row = conn.execute(
                select(documents.c.derivation_digest).where(
                    documents.c.workspace_id == workspace_id,
                    documents.c.corpus_id == corpus_id,
                    documents.c.id == document_id,
                )
            ).first()
        if row is None:
            return (False, None)
        return (True, str(row[0]) if row[0] is not None else None)

    def list_document_derivation_digests(self, workspace_id: str, corpus_id: str) -> dict[str, str | None]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(documents.c.id, documents.c.derivation_digest).where(
                    documents.c.workspace_id == workspace_id,
                    documents.c.corpus_id == corpus_id,
                )
            ).all()
        return {str(row[0]): (str(row[1]) if row[1] is not None else None) for row in rows}

    def list_hold_permission_history(self, workspace_id: str, corpus_id: str, document_id: str) -> list[JsonObject]:
        with self.engine.connect() as conn:
            rows = (
                conn.execute(
                    select(hold_permission_history)
                    .where(
                        hold_permission_history.c.workspace_id == workspace_id,
                        hold_permission_history.c.corpus_id == corpus_id,
                        hold_permission_history.c.document_id == document_id,
                    )
                    .order_by(hold_permission_history.c.created_at, hold_permission_history.c.id)
                )
                .mappings()
                .all()
            )
        return [dict(row) for row in rows]

    def claim_audit_outbox(self, worker_id: str, lease_seconds: int, limit: int = 100) -> list[JsonObject]:
        now = self.clock.now()
        claimed_until = now + timedelta(seconds=lease_seconds)
        unclaimed = or_(audit_outbox.c.claimed_until.is_(None), audit_outbox.c.claimed_until < now)
        with self.engine.begin() as conn:
            candidates = [
                str(row[0])
                for row in conn.execute(
                    select(audit_outbox.c.id)
                    .where(audit_outbox.c.delivered_at.is_(None), audit_outbox.c.available_at <= now, unclaimed)
                    .order_by(audit_outbox.c.created_at)
                    .limit(limit)
                ).all()
            ]
            if not candidates:
                return []
            rows = conn.execute(
                update(audit_outbox)
                .where(audit_outbox.c.id.in_(candidates), audit_outbox.c.delivered_at.is_(None), unclaimed)
                .values(claimed_by=worker_id, claimed_until=claimed_until)
                .returning(audit_outbox)
            ).mappings()
            return sorted((dict(row) for row in rows), key=lambda event: event["created_at"])

    def mark_audit_delivered(self, event_id: str, worker_id: str) -> bool:
        # Only the current lease owner may finalize an event; gate on claimed_by plus an
        # unexpired lease so a stale owner cannot mark an event another worker now owns.
        now = self.clock.now()
        with self.engine.begin() as conn:
            result = conn.execute(
                update(audit_outbox)
                .where(
                    audit_outbox.c.id == event_id,
                    audit_outbox.c.delivered_at.is_(None),
                    audit_outbox.c.claimed_by == worker_id,
                    audit_outbox.c.claimed_until >= now,
                )
                .values(delivered_at=now)
            )
        return result.rowcount == 1

    def delivered_audit_sinks(self, event_id: str) -> set[str]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(audit_outbox_deliveries.c.sink_name).where(audit_outbox_deliveries.c.event_id == event_id)
            ).all()
        return {str(row[0]) for row in rows}

    def record_audit_delivery(self, event_id: str, sink_name: str) -> None:
        """Persist that a sink delivered an audit event, once. Idempotent: a redelivery hits the
        unique constraint and is a no-op, so the ledger stays a faithful at-least-once record."""
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    insert(audit_outbox_deliveries).values(
                        id=new_id("aobd"),
                        event_id=event_id,
                        sink_name=sink_name,
                        delivered_at=self.clock.now(),
                    )
                )
        except IntegrityError:
            pass

    def mark_audit_failed(self, event_id: str, worker_id: str, error: str, delay_seconds: int) -> None:
        now = self.clock.now()
        with self.engine.begin() as conn:
            conn.execute(
                update(audit_outbox)
                .where(
                    audit_outbox.c.id == event_id,
                    audit_outbox.c.delivered_at.is_(None),
                    audit_outbox.c.claimed_by == worker_id,
                    audit_outbox.c.claimed_until >= now,
                )
                .values(
                    attempts=audit_outbox.c.attempts + 1,
                    last_error=error[:4000],
                    available_at=now + timedelta(seconds=delay_seconds),
                    claimed_by=None,
                    claimed_until=None,
                )
            )

    def purge_delivered_audit_outbox(self, retention_days: int) -> int:
        """Delete fully delivered audit events older than the retention window. Their per-sink
        ledger rows cascade away, so the ledger stays bounded; an in-flight event is never dropped."""
        cutoff = self.clock.now() - timedelta(days=retention_days)
        with self.engine.begin() as conn:
            result = conn.execute(
                delete(audit_outbox).where(
                    audit_outbox.c.delivered_at.is_not(None), audit_outbox.c.delivered_at < cutoff
                )
            )
        return int(result.rowcount or 0)

    def save_program_artifact(
        self, workspace_id: str, program: str, artifact_id: str, definition_hash: str, document: JsonObject
    ) -> None:
        with self.engine.begin() as conn:
            existing = conn.execute(
                select(program_artifacts.c.row_id).where(
                    program_artifacts.c.workspace_id == workspace_id,
                    program_artifacts.c.artifact_id == artifact_id,
                )
            ).first()
            if existing is not None:
                return
            try:
                conn.execute(
                    insert(program_artifacts).values(
                        row_id=new_id("pa"),
                        workspace_id=workspace_id,
                        artifact_id=artifact_id,
                        program=program,
                        definition_hash=definition_hash,
                        document=document,
                        created_at=self.clock.now(),
                    )
                )
            except IntegrityError:
                return

    def get_program_artifact(self, workspace_id: str, artifact_id: str) -> JsonObject | None:
        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    select(program_artifacts.c.document).where(
                        program_artifacts.c.workspace_id == workspace_id,
                        program_artifacts.c.artifact_id == artifact_id,
                    )
                )
                .mappings()
                .first()
            )
        return dict(row["document"]) if row is not None else None

    def record_deployment(
        self, workspace_id: str, program: str, environment: str, artifact_id: str, action: str
    ) -> JsonObject:
        deployment_id = new_id("dep")
        with self.engine.begin() as conn:
            max_sequence = conn.execute(
                select(func.coalesce(func.max(program_deployments.c.sequence), 0)).where(
                    program_deployments.c.workspace_id == workspace_id,
                    program_deployments.c.program == program,
                    program_deployments.c.environment == environment,
                )
            ).scalar_one()
            sequence = int(max_sequence) + 1
            conn.execute(
                insert(program_deployments).values(
                    id=deployment_id,
                    workspace_id=workspace_id,
                    program=program,
                    environment=environment,
                    artifact_id=artifact_id,
                    action=action,
                    sequence=sequence,
                    created_at=self.clock.now(),
                )
            )
        return {
            "id": deployment_id,
            "program": program,
            "environment": environment,
            "artifact_id": artifact_id,
            "action": action,
            "sequence": sequence,
        }

    def current_deployment(self, workspace_id: str, program: str, environment: str) -> JsonObject | None:
        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    select(program_deployments)
                    .where(
                        program_deployments.c.workspace_id == workspace_id,
                        program_deployments.c.program == program,
                        program_deployments.c.environment == environment,
                    )
                    .order_by(program_deployments.c.sequence.desc())
                    .limit(1)
                )
                .mappings()
                .first()
            )
        return dict(row) if row is not None else None

    def deployment_history(self, workspace_id: str, program: str, environment: str) -> list[JsonObject]:
        with self.engine.connect() as conn:
            rows = (
                conn.execute(
                    select(program_deployments)
                    .where(
                        program_deployments.c.workspace_id == workspace_id,
                        program_deployments.c.program == program,
                        program_deployments.c.environment == environment,
                    )
                    .order_by(program_deployments.c.sequence.desc())
                )
                .mappings()
                .all()
            )
        return [dict(row) for row in rows]

    def record_approval(
        self, workspace_id: str, program: str, environment: str, artifact_id: str, approver: str
    ) -> JsonObject:
        approval_id = new_id("appr")
        with self.engine.begin() as conn:
            conn.execute(
                insert(program_approvals).values(
                    id=approval_id,
                    workspace_id=workspace_id,
                    program=program,
                    environment=environment,
                    artifact_id=artifact_id,
                    approver=approver,
                    created_at=self.clock.now(),
                )
            )
        return {
            "id": approval_id,
            "program": program,
            "environment": environment,
            "artifact_id": artifact_id,
            "approver": approver,
        }

    def is_approved(self, workspace_id: str, program: str, environment: str, artifact_id: str) -> bool:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(program_approvals.c.id).where(
                    program_approvals.c.workspace_id == workspace_id,
                    program_approvals.c.program == program,
                    program_approvals.c.environment == environment,
                    program_approvals.c.artifact_id == artifact_id,
                )
            ).first()
        return row is not None

    def new_tool_approval_id(self) -> str:
        return new_id("tappr")

    def record_tool_approval(
        self,
        workspace_id: str,
        *,
        fingerprint: str,
        capability_name: str,
        approver_id: str,
        approver_roles: Sequence[str],
        granted_at: datetime,
        approval_id: str | None = None,
        kind: str = "normal",
        reason: str | None = None,
    ) -> str:
        # A caller-reserved id lets the grant be audited before the row is inserted (emit-then-insert).
        approval_id = approval_id or new_id("tappr")
        with self.engine.begin() as conn:
            conn.execute(
                insert(tool_approvals).values(
                    id=approval_id,
                    workspace_id=workspace_id,
                    fingerprint=fingerprint,
                    capability_name=capability_name,
                    approver_id=approver_id,
                    approver_roles=sorted(approver_roles),
                    granted_at=granted_at,
                    status="granted",
                    kind=kind,
                    reason=reason,
                )
            )
        return approval_id

    def list_active_tool_approvals(self, workspace_id: str, fingerprint: str) -> list[JsonObject]:
        with self.engine.connect() as conn:
            rows = (
                conn.execute(
                    select(
                        tool_approvals.c.id,
                        tool_approvals.c.fingerprint,
                        tool_approvals.c.approver_id,
                        tool_approvals.c.approver_roles,
                        tool_approvals.c.granted_at,
                        tool_approvals.c.kind,
                    )
                    .where(
                        tool_approvals.c.workspace_id == workspace_id,
                        tool_approvals.c.fingerprint == fingerprint,
                        tool_approvals.c.status == "granted",
                    )
                    .order_by(tool_approvals.c.granted_at, tool_approvals.c.id)
                )
                .mappings()
                .all()
            )
        return [dict(row) for row in rows]

    def consume_tool_approvals(
        self,
        workspace_id: str,
        fingerprint: str,
        approval_ids: Sequence[str],
        *,
        audit_event: AuditEvent | None = None,
    ) -> bool:
        # Single-use: atomically consume exactly the approvals that satisfied the decision, all or
        # nothing, so a run is bound to its own evidence and a concurrent cancel denies rather than
        # letting an unrelated same-fingerprint row absorb the consume. When an audit_event is passed it
        # is persisted in the SAME transaction, so a consumed approval can never exist without its audit
        # record. Byte-identical when audit_event is None.
        ids = tuple(dict.fromkeys(str(approval_id) for approval_id in approval_ids))
        if not ids:
            return False
        with self.engine.connect() as conn:
            transaction = conn.begin()
            result = conn.execute(
                update(tool_approvals)
                .where(
                    tool_approvals.c.workspace_id == workspace_id,
                    tool_approvals.c.fingerprint == fingerprint,
                    tool_approvals.c.status == "granted",
                    tool_approvals.c.id.in_(ids),
                )
                .values(status="consumed", consumed_at=self.clock.now())
            )
            if result.rowcount != len(ids):
                transaction.rollback()
                return False
            if audit_event is not None:
                self._insert_audit_outbox(conn, workspace_id, audit_event)
            transaction.commit()
            return True

    def cancel_tool_approval(
        self,
        workspace_id: str,
        approval_id: str,
        *,
        audit_event_factory: Callable[[str, str], AuditEvent] | None = None,
    ) -> dict[str, str] | None:
        # When a factory is passed the cancel audit is written in the SAME transaction as the cancel, so a
        # cancelled approval is never left unaudited. The factory receives the row's fingerprint and
        # capability name (known only after the update). Byte-identical when no factory is passed.
        with self.engine.begin() as conn:
            row = conn.execute(
                update(tool_approvals)
                .where(
                    tool_approvals.c.workspace_id == workspace_id,
                    tool_approvals.c.id == approval_id,
                    tool_approvals.c.status == "granted",
                )
                .values(status="cancelled", cancelled_at=self.clock.now())
                .returning(tool_approvals.c.fingerprint, tool_approvals.c.capability_name)
            ).one_or_none()
            if row is None:
                return None
            fingerprint = str(row.fingerprint)
            capability_name = str(row.capability_name)
            if audit_event_factory is not None:
                self._insert_audit_outbox(conn, workspace_id, audit_event_factory(fingerprint, capability_name))
        return {"fingerprint": fingerprint, "capability_name": capability_name}

    def record_approval_request(
        self,
        workspace_id: str,
        *,
        fingerprint: str,
        capability_name: str,
        requested_by: str,
        created_at: datetime,
        request_id: str | None = None,
    ) -> str:
        request_id = request_id or new_id("areq")
        with self.engine.begin() as conn:
            conn.execute(
                insert(approval_requests).values(
                    id=request_id,
                    workspace_id=workspace_id,
                    fingerprint=fingerprint,
                    capability_name=capability_name,
                    requested_by=requested_by,
                    created_at=created_at,
                    status="open",
                )
            )
        return request_id

    def earliest_open_request_anchor(self, workspace_id: str, fingerprint: str) -> datetime | None:
        with self.engine.connect() as conn:
            anchor = conn.execute(
                select(func.min(approval_requests.c.created_at)).where(
                    approval_requests.c.workspace_id == workspace_id,
                    approval_requests.c.fingerprint == fingerprint,
                    approval_requests.c.status == "open",
                )
            ).scalar()
        if not isinstance(anchor, datetime):
            return None
        # SQLite returns a naive datetime; the escalation core fails closed on a naive anchor, so
        # coerce to timezone-aware UTC to match the store's always-UTC writes.
        return anchor.replace(tzinfo=UTC) if anchor.tzinfo is None else anchor

    def _next_canary_sequence(self, conn: Any, workspace_id: str, program: str, environment: str) -> int:
        max_sequence = conn.execute(
            select(func.coalesce(func.max(program_canaries.c.sequence), 0)).where(
                program_canaries.c.workspace_id == workspace_id,
                program_canaries.c.program == program,
                program_canaries.c.environment == environment,
            )
        ).scalar_one()
        return int(max_sequence) + 1

    def set_canary(
        self,
        workspace_id: str,
        program: str,
        environment: str,
        candidate_artifact_id: str,
        percent: int,
        compare: bool,
    ) -> JsonObject:
        canary_id = new_id("can")

        def attempt() -> JsonObject:
            with self.engine.begin() as conn:
                sequence = self._next_canary_sequence(conn, workspace_id, program, environment)
                conn.execute(
                    insert(program_canaries).values(
                        id=canary_id,
                        workspace_id=workspace_id,
                        program=program,
                        environment=environment,
                        candidate_artifact_id=candidate_artifact_id,
                        percent=percent,
                        compare=compare,
                        action="set",
                        sequence=sequence,
                        created_at=self.clock.now(),
                    )
                )
            return {
                "id": canary_id,
                "program": program,
                "environment": environment,
                "candidate_artifact_id": candidate_artifact_id,
                "percent": percent,
                "compare": compare,
                "action": "set",
                "sequence": sequence,
            }

        return self._retry_returning(attempt)

    def clear_canary(self, workspace_id: str, program: str, environment: str) -> JsonObject:
        canary_id = new_id("can")

        def attempt() -> JsonObject:
            with self.engine.begin() as conn:
                sequence = self._next_canary_sequence(conn, workspace_id, program, environment)
                conn.execute(
                    insert(program_canaries).values(
                        id=canary_id,
                        workspace_id=workspace_id,
                        program=program,
                        environment=environment,
                        candidate_artifact_id=None,
                        percent=0,
                        compare=False,
                        action="clear",
                        sequence=sequence,
                        created_at=self.clock.now(),
                    )
                )
            return {
                "id": canary_id,
                "program": program,
                "environment": environment,
                "candidate_artifact_id": None,
                "percent": 0,
                "compare": False,
                "action": "clear",
                "sequence": sequence,
            }

        return self._retry_returning(attempt)

    def current_canary(self, workspace_id: str, program: str, environment: str) -> JsonObject | None:
        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    select(program_canaries)
                    .where(
                        program_canaries.c.workspace_id == workspace_id,
                        program_canaries.c.program == program,
                        program_canaries.c.environment == environment,
                    )
                    .order_by(
                        program_canaries.c.sequence.desc(),
                        program_canaries.c.created_at.desc(),
                        program_canaries.c.id.desc(),
                    )
                    .limit(1)
                )
                .mappings()
                .first()
            )
        if row is None or row["action"] == "clear":
            return None
        return dict(row)

    def canary_history(self, workspace_id: str, program: str, environment: str) -> list[JsonObject]:
        with self.engine.connect() as conn:
            rows = (
                conn.execute(
                    select(program_canaries)
                    .where(
                        program_canaries.c.workspace_id == workspace_id,
                        program_canaries.c.program == program,
                        program_canaries.c.environment == environment,
                    )
                    .order_by(program_canaries.c.sequence.desc())
                )
                .mappings()
                .all()
            )
        return [dict(row) for row in rows]

    def record_shadow_run(
        self,
        workspace_id: str,
        program: str,
        environment: str,
        active_artifact_id: str,
        candidate_artifact_id: str,
        served: str,
        bucket: int,
        query_hash: str,
        comparison: JsonObject | None,
        idempotency_key: str | None = None,
    ) -> str:
        """Record a shadow comparison row and its outbox event.

        When ``idempotency_key`` is set (the durable batch path), the row and event
        ids are content-addressed from the key and the write is insert-if-absent, so
        an at-least-once redelivery of the same batch upserts rather than duplicating
        rows, events, or spend. The served request path passes no key and keeps its
        random-id behavior byte-identical.
        """
        now = self.clock.now()
        if idempotency_key is not None:
            run_id = f"shd_{hashlib.sha256(idempotency_key.encode()).hexdigest()[:56]}"
            event_id = f"evt_{hashlib.sha256(idempotency_key.encode()).hexdigest()[:56]}"
        else:
            run_id = new_id("shd")
            event_id = new_id("evt")
        shadow_values = {
            "workspace_id": workspace_id,
            "program": program,
            "environment": environment,
            "active_artifact_id": active_artifact_id,
            "candidate_artifact_id": candidate_artifact_id,
            "served": served,
            "bucket": bucket,
            "query_hash": query_hash,
            "comparison": comparison,
            "created_at": now,
        }
        event_values = {
            "workspace_id": workspace_id,
            "topic": "program.shadow_compared",
            "payload": {
                "workspace_id": workspace_id,
                "program": program,
                "environment": environment,
                "shadow_run_id": run_id,
                "served": served,
                "candidate_artifact_id": candidate_artifact_id,
            },
            "attempts": 0,
            "available_at": now,
            "delivered_at": None,
            "last_error": None,
            "created_at": now,
        }
        if idempotency_key is None:
            with self.engine.begin() as conn:
                conn.execute(insert(program_shadow_runs).values(id=run_id, **shadow_values))
                conn.execute(insert(outbox).values(id=event_id, **event_values))
            return run_id
        # Keyed path: two independent atomic upserts, never an early return. The row
        # and its outbox event are re-ensured on every call, so a retry after the
        # retention purge dropped either one re-creates it, and a losing concurrent
        # first writer's conflict is a no-op rather than a surfaced error.
        self._insert_ignoring_conflict(program_shadow_runs, {"id": run_id, **shadow_values})
        self._insert_ignoring_conflict(outbox, {"id": event_id, **event_values})
        return run_id

    def purge_expired_shadow_runs(self, retention_days: int) -> int:
        cutoff = self.clock.now() - timedelta(days=retention_days)
        with self.engine.begin() as conn:
            result = conn.execute(delete(program_shadow_runs).where(program_shadow_runs.c.created_at < cutoff))
        return int(result.rowcount or 0)

    def shadow_runs(self, workspace_id: str, program: str, environment: str, limit: int = 50) -> list[JsonObject]:
        with self.engine.connect() as conn:
            rows = (
                conn.execute(
                    select(program_shadow_runs)
                    .where(
                        program_shadow_runs.c.workspace_id == workspace_id,
                        program_shadow_runs.c.program == program,
                        program_shadow_runs.c.environment == environment,
                    )
                    .order_by(program_shadow_runs.c.created_at.desc())
                    .limit(limit)
                )
                .mappings()
                .all()
            )
        return [dict(row) for row in rows]

    def stats(self, workspace_id: str, corpus_id: str) -> JsonObject:
        self.get_corpus(workspace_id, corpus_id)
        with self.engine.connect() as conn:
            return {
                "documents": int(
                    conn.execute(
                        select(func.count())
                        .select_from(documents)
                        .where(
                            documents.c.workspace_id == workspace_id,
                            documents.c.corpus_id == corpus_id,
                        )
                    ).scalar_one()
                ),
                "chunks": int(
                    conn.execute(
                        select(func.count())
                        .select_from(chunks)
                        .where(
                            chunks.c.workspace_id == workspace_id,
                            chunks.c.corpus_id == corpus_id,
                        )
                    ).scalar_one()
                ),
                "entities": int(
                    conn.execute(
                        select(func.count())
                        .select_from(entities)
                        .where(
                            entities.c.workspace_id == workspace_id,
                            entities.c.corpus_id == corpus_id,
                        )
                    ).scalar_one()
                ),
                "relations": int(
                    conn.execute(
                        select(func.count())
                        .select_from(relations)
                        .where(
                            relations.c.workspace_id == workspace_id,
                            relations.c.corpus_id == corpus_id,
                        )
                    ).scalar_one()
                ),
            }
