from __future__ import annotations

import signal
from types import SimpleNamespace
from typing import Any

from refaat import worker as worker_module


def test_worker_entrypoint_stops_and_closes(monkeypatch) -> None:
    state: dict[str, Any] = {"signals": {}}

    class Runtime:
        def close(self) -> None:
            state["closed"] = True

    runtime = Runtime()

    class Builder:
        def build(self) -> Runtime:
            return runtime

    class Worker:
        def __init__(self, value: Runtime) -> None:
            assert value is runtime

        def run_forever(self, stop) -> None:
            handler = state["signals"][signal.SIGTERM]
            handler(signal.SIGTERM, SimpleNamespace())
            state["stopped"] = stop.is_set()

    def register(signum: int, handler: object) -> None:
        state["signals"][signum] = handler

    monkeypatch.setattr(worker_module, "RuntimeBuilder", Builder)
    monkeypatch.setattr(worker_module, "DurableWorker", Worker)
    monkeypatch.setattr(worker_module.signal, "signal", register)

    worker_module.run()

    assert set(state["signals"]) == {signal.SIGINT, signal.SIGTERM}
    assert state["stopped"] is True
    assert state["closed"] is True
