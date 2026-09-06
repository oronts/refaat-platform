from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import benchmarks_check as bench


def test_citation_filter_is_not_super_linear() -> None:
    assert bench.main() == 0


def test_bound_separates_linear_from_quadratic() -> None:
    # The bound is meaningful only if it sits between the linear size ratio and its square, so a
    # linear implementation passes while a quadratic rewrite (near the square) fails.
    linear = bench.LARGE / bench.SMALL
    assert linear < bench.RATIO_BOUND < linear * linear
