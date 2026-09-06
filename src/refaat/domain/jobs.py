from __future__ import annotations

# Job kinds whose enqueue carries a run-scoped dedup or approval invariant. They must be enqueued only
# through their governed path (which sets a dedup key), never through a generic or scheduled enqueue that
# would create a second, unprotected job. Enforced at every enqueue chokepoint (the generic job and
# schedule endpoints, and schedule dispatch).
RESERVED_JOB_KINDS = frozenset({"fine_tune.run", "optimization.run"})
