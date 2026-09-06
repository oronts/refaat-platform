from __future__ import annotations

from datetime import timedelta

import pytest

from refaat.application.runtime import Refaat
from refaat.application.worker import DurableWorker
from refaat.domain.models import CacheEntry, QueryResponse
from refaat.ids import new_id

pytestmark = pytest.mark.integration


def test_purge_expired_cache_removes_only_expired(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    store = runtime.store
    now = store.clock.now()
    response = QueryResponse(
        answer="cached",
        citations=(),
        routes=("cache",),
        cached=False,
        corpus_version=1,
        trace_id="trace",
        usage={},
    )

    def entry(query_hash: str, expires_at: object) -> CacheEntry:
        return CacheEntry(
            id=new_id("cache"),
            workspace_id=project.workspace_id,
            corpus_id=project.corpus_id,
            corpus_version=1,
            user_scope="anonymous",
            query="q",
            query_hash=query_hash,
            query_embedding=(0.0,),
            response=response,
            expires_at=expires_at,
            created_at=now - timedelta(hours=2),
        )

    store.put_cache(entry("expired", now - timedelta(hours=1)))
    store.put_cache(entry("live", now + timedelta(hours=1)))

    assert store.purge_expired_cache() == 1
    assert store.purge_expired_cache() == 0
    assert store.get_exact_cache(project.workspace_id, project.corpus_id, 1, "anonymous", "live") is not None


def test_cache_purge_job_runs_through_the_worker(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    job_id = runtime.store.enqueue_job(project.workspace_id, "cache.purge", {})
    assert DurableWorker(runtime).run_once() is True
    job = runtime.store.get_job(project.workspace_id, job_id)
    assert job["status"] == "succeeded"
    assert "purged" in job["result"]
