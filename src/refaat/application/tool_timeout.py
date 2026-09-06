"""A caller deadline for one governed tool call, distinguishing three outcomes.

A plain skill's ``run`` is arbitrary blocking code: Refaat can stop WAITING for it at a deadline, but
it cannot TERMINATE it (Python has no safe thread kill), and its I/O may already have reached an
external system. So the deadline reports one of three outcomes and never conflates them:

- ``started`` False: the bounded worker pool was full, so the skill provably never ran (a confirmed
  no-effect). The caller is free to retry.
- ``started`` True, ``completed`` True: the skill returned or raised within the deadline; the result
  (or its exception) is carried back verbatim, exactly as the inline path.
- ``started`` True, ``completed`` False: the deadline elapsed while the skill was still running. It is
  NOT terminated, so the outcome is unknown; the worker keeps running and, when it finally returns,
  runs ``on_finish`` exactly once (the deferred secret scrub) so a still-live bundle is never scrubbed
  out from under the worker.

The worker pool is a process-global bounded semaphore sized on first activation (first-writer-wins,
the same in-process boundary the edge rate limiter documents); a Redis-backed cross-process bound is a
follow-up. This module is imported lazily by the governed loop, so the disabled path never touches it.
"""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_SEMAPHORE_LOCK = threading.Lock()
_SEMAPHORE: threading.BoundedSemaphore | None = None


def _acquire_slot(max_workers: int) -> threading.BoundedSemaphore | None:
    # Return the exact semaphore acquired so the worker releases that one, never whatever the global
    # currently is: a worker that outlives a pool swap must not over-release the new semaphore.
    global _SEMAPHORE
    with _SEMAPHORE_LOCK:
        if _SEMAPHORE is None:
            _SEMAPHORE = threading.BoundedSemaphore(max(1, max_workers))
        semaphore = _SEMAPHORE
    return semaphore if semaphore.acquire(blocking=False) else None


def _once(fn: Callable[[], None] | None) -> Callable[[], None]:
    lock = threading.Lock()
    state = {"done": False}

    def _call() -> None:
        if fn is None:
            return
        with lock:
            if state["done"]:
                return
            state["done"] = True
        fn()

    return _call


@dataclass
class ToolRunOutcome:
    started: bool
    completed: bool
    observation: Any | None = None
    error: BaseException | None = None


def run_tool_within_deadline(
    invoke: Callable[[], Any],
    *,
    timeout: float,
    max_workers: int,
    on_finish: Callable[[], None] | None = None,
) -> ToolRunOutcome:
    """Run ``invoke`` on a worker thread under a caller deadline; see the module docstring.

    on_finish is owned here in every path: if no worker runs (pool full, or the thread cannot start) it
    is called synchronously since the bundle is untouched; if a worker runs it is called in the worker's
    finally. The caller therefore never scrubs the bundle itself and cannot scrub it out from under a
    live worker, and a scrub that raises still cannot leak the slot (the slot is released first).
    """
    finish_once = _once(on_finish)
    semaphore = _acquire_slot(max_workers)
    if semaphore is None:
        finish_once()
        return ToolRunOutcome(started=False, completed=False)
    # Copy on the calling thread so the worker sees the caller's active_meter, approval context, and
    # in_unserved_run() bindings; a fresh thread would otherwise reset every ContextVar to its default.
    ctx = contextvars.copy_context()
    done = threading.Event()
    box: dict[str, Any] = {}

    def _worker() -> None:
        try:
            box["observation"] = ctx.run(invoke)
        except BaseException as exc:  # carried back to the caller thread verbatim
            box["error"] = exc
        finally:
            semaphore.release()
            done.set()
            finish_once()

    try:
        threading.Thread(target=_worker, name="refaat-tool-deadline", daemon=True).start()
    except BaseException:
        # The slot was acquired but no worker will run it, so free it and scrub the untouched bundle
        # here rather than leak the slot; a thread-start failure degrades to a confirmed no-effect.
        semaphore.release()
        finish_once()
        return ToolRunOutcome(started=False, completed=False)
    if done.wait(timeout):
        return ToolRunOutcome(started=True, completed=True, observation=box.get("observation"), error=box.get("error"))
    return ToolRunOutcome(started=True, completed=False)


__all__ = ["ToolRunOutcome", "run_tool_within_deadline"]
