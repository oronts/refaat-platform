"""The tool-deadline helper distinguishes started, completed, and abandoned, and the meter is atomic."""

from __future__ import annotations

import threading
import time
from contextvars import ContextVar

import refaat.application.tool_timeout as tool_timeout
from refaat.application.budget import CostMeter
from refaat.application.tool_timeout import run_tool_within_deadline


def _reset_pool() -> None:
    tool_timeout._SEMAPHORE = None


def test_completed_call_returns_the_observation() -> None:
    _reset_pool()
    outcome = run_tool_within_deadline(lambda: {"ok": True}, timeout=1.0, max_workers=4)
    assert outcome.started
    assert outcome.completed
    assert outcome.observation == {"ok": True}
    assert outcome.error is None


def test_a_raised_error_is_carried_back() -> None:
    _reset_pool()

    def _boom() -> dict[str, object]:
        raise ValueError("boom")

    outcome = run_tool_within_deadline(_boom, timeout=1.0, max_workers=4)
    assert outcome.started
    assert outcome.completed
    assert isinstance(outcome.error, ValueError)


def test_a_hang_is_started_but_not_completed() -> None:
    _reset_pool()
    release = threading.Event()
    try:
        outcome = run_tool_within_deadline(lambda: release.wait(5.0), timeout=0.05, max_workers=4)
        assert outcome.started
        assert not outcome.completed
        assert outcome.observation is None
    finally:
        release.set()


def test_a_full_pool_reports_never_started() -> None:
    _reset_pool()
    release = threading.Event()
    try:
        first = run_tool_within_deadline(lambda: release.wait(5.0), timeout=0.05, max_workers=1)
        assert first.started
        assert not first.completed
        # The single slot is held by the still-running first worker, so the next call cannot start.
        second = run_tool_within_deadline(lambda: {"ok": True}, timeout=0.05, max_workers=1)
        assert not second.started
        assert not second.completed
    finally:
        release.set()


def test_a_full_pool_scrubs_the_untouched_bundle() -> None:
    _reset_pool()
    release = threading.Event()
    calls = {"n": 0}
    try:
        first = run_tool_within_deadline(lambda: release.wait(5.0), timeout=0.05, max_workers=1)
        assert first.started
        # The slot is held, so this never starts; on_finish must still scrub the bundle nothing touched.
        second = run_tool_within_deadline(
            lambda: {"ok": True}, timeout=0.05, max_workers=1, on_finish=lambda: calls.__setitem__("n", calls["n"] + 1)
        )
        assert not second.started
        assert calls["n"] == 1
    finally:
        release.set()


def test_contextvars_are_visible_inside_the_worker() -> None:
    _reset_pool()
    marker: ContextVar[str] = ContextVar("marker", default="unset")
    token = marker.set("caller-value")
    try:
        outcome = run_tool_within_deadline(lambda: {"seen": marker.get()}, timeout=1.0, max_workers=4)
    finally:
        marker.reset(token)
    assert outcome.observation == {"seen": "caller-value"}


def test_on_finish_runs_exactly_once_after_an_abandoned_worker_completes() -> None:
    _reset_pool()
    release = threading.Event()
    finished = threading.Event()
    calls = {"n": 0}

    def _on_finish() -> None:
        calls["n"] += 1
        finished.set()

    try:
        outcome = run_tool_within_deadline(lambda: release.wait(5.0), timeout=0.05, max_workers=4, on_finish=_on_finish)
        assert outcome.started
        assert not outcome.completed
        assert calls["n"] == 0  # deferred: the worker is still running, so the bundle is not scrubbed yet
        release.set()
        assert finished.wait(5.0)
        assert calls["n"] == 1
    finally:
        release.set()


def test_cost_meter_charge_is_atomic_under_concurrency() -> None:
    meter = CostMeter(token_ceiling=None)
    iterations = 2000
    barrier = threading.Barrier(2)

    def _charge() -> None:
        barrier.wait()
        for _ in range(iterations):
            meter.charge({"prompt_tokens": 1, "completion_tokens": 0})

    threads = [threading.Thread(target=_charge) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert meter.tokens_used == 2 * iterations


def test_completed_beats_the_deadline_even_when_slow() -> None:
    _reset_pool()
    outcome = run_tool_within_deadline(lambda: (time.sleep(0.02), {"ok": True})[1], timeout=1.0, max_workers=4)
    assert outcome.completed
    assert outcome.observation == {"ok": True}
