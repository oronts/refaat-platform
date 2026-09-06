from __future__ import annotations

import signal
import threading

from refaat.application.runtime import RuntimeBuilder
from refaat.application.worker import DurableWorker


def run() -> None:
    runtime = RuntimeBuilder().build()
    stop = threading.Event()

    def request_stop(signum: int, frame: object) -> None:
        del signum, frame
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        DurableWorker(runtime).run_forever(stop)
    finally:
        runtime.close()


if __name__ == "__main__":
    run()
