# Evidence-backed harness improvement

The immutable base prompt requires evidence-only factual claims and exact chunk citations. Its SHA-256 hash is stored in every harness snapshot.

A snapshot may contain only:

- bounded semantic, lexical, and graph weights;
- bounded `top_k`;
- the fusion strategy (`rrf` or `linear`), the RRF constant `rrf_k`, and the autocut toggle and threshold;
- the BM25 knobs `bm25_k1` (0 to 5) and `bm25_b` (0 to 1) for the offline lexical ranker;
- the `query_expansion` toggle (boolean) and `max_query_variants` (integer 1 to 8) for multi-query retrieval;
- an exact-match metadata `filters` object (scalar for equality, list for membership) applied before scoring;
- answer-style data;
- a short supplemental instruction suffix.

Unknown keys, out-of-range values, and snapshots without evidence are rejected. A `filters` value must be an object of scalar-or-list members with a bounded field count and string length; membership matching is exact-type, so `1` never matches `"1"` and a boolean never matches an integer. An over-restrictive filter that selects no chunks degrades to the standard insufficient-evidence answer with no citations. Activation deactivates the prior snapshot for the same workspace/corpus but never deletes it. Rollback is activation of an earlier snapshot.

`propose_from_metrics` requires at least ten evaluation cases. It makes small deterministic proposals when citation recall or answer hit rate is below thresholds. A proposal is not automatically activated; an administrator records evidence and creates/activates a snapshot.

This design deliberately excludes self-modification of code, base safety rules, credentials, tools, or authorization policy.
