#!/usr/bin/env python3
"""Offline regression benchmark: a hot path must not become super-linear.

Absolute timings are environment sensitive, so the offline gate asserts a relative signal instead:
the citation filter, on the query pipeline's hot path, is timed at two input sizes ten times apart,
and the ratio of the larger to the smaller run must stay well below the quadratic ratio. A linear
implementation lands near the size ratio; an accidental O(N*M) rewrite lands near its square, which
the bound catches. The minimum over several trials is used so that scheduler noise, which can only
add time, does not inflate the ratio.

Set REFAAT_BENCHMARK_SCALE=1 for a larger, slower size tier; the default tier is the required gate.
Fail-closed: a ratio over the bound, or an unexpectedly slow small run, exits non-zero.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable

from refaat.core.citations import filter_cited_chunks
from refaat.domain.models import Chunk

TRIALS = 9
REPEATS = 200
RATIO_BOUND = 50.0
SMALL = 200
LARGE = 2000


def _chunks(count: int) -> list[Chunk]:
    return [
        Chunk(
            id=f"chunk-{index}",
            corpus_id="c",
            document_id="d",
            ordinal=index,
            text="",
            token_count=0,
            checksum="",
            metadata={},
            embedding=(),
        )
        for index in range(count)
    ]


def _min_time(fn: Callable[[], object]) -> float:
    best = float("inf")
    for _ in range(TRIALS):
        start = time.perf_counter()
        for _ in range(REPEATS):
            fn()
        best = min(best, time.perf_counter() - start)
    return best


def _case(size: int) -> Callable[[], object]:
    allowed = _chunks(size)
    # Half the cited ids hit and half miss, so both the membership hits and misses are exercised.
    cited = tuple(f"chunk-{index}" for index in range(0, size * 2, 1))
    return lambda: filter_cited_chunks(cited, allowed)


def main(argv: list[str] | None = None) -> int:
    large_size = LARGE * 5 if os.environ.get("REFAAT_BENCHMARK_SCALE") == "1" else LARGE

    small_time = _min_time(_case(SMALL))
    large_time = _min_time(_case(large_size))

    size_ratio = large_size / SMALL
    ratio = large_time / small_time if small_time > 0 else float("inf")

    print(f"benchmarks: citation filter size x{size_ratio:.0f}, time x{ratio:.1f} (bound {RATIO_BOUND:.0f})")

    if ratio > RATIO_BOUND:
        print(f"benchmarks: FAIL, the citation filter scaled x{ratio:.1f}, over the x{RATIO_BOUND:.0f} bound")
        print("A near-quadratic scaling suggests a hot-path regression; investigate the citation filter.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
