# Performance benchmarks

The `benchmarks` gate guards a hot path against an algorithmic regression, offline and without a flaky absolute timing.

It times the citation filter, on the query pipeline's hot path, at two input sizes ten times apart and asserts the ratio of the larger to the smaller run stays well below the quadratic ratio. A linear implementation lands near the size ratio; an accidental quadratic rewrite lands near its square, which the bound catches. The minimum over several trials is used, so scheduler noise, which can only add time, does not inflate the ratio. The bound is deliberately generous, so the gate is stable across machines and only a genuine super-linear regression fails it.

Set `REFAAT_BENCHMARK_SCALE=1` to run a larger, slower size tier. The default tier is the required gate; the larger tier is an opt-in for a dedicated performance run.

Absolute throughput and latency under a real workload, and scale tests under authorization, are deployment-specific and are measured out of band, not in the offline gate.
