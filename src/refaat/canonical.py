from __future__ import annotations

import hashlib
import json

from refaat.domain.models import JsonObject


def canonical_json(payload: JsonObject) -> str:
    """Serialize a JSON object to the one canonical form used for content addressing.

    Sorted keys and compact separators make the bytes reproducible across runs and
    processes, so any two callers that hash the same object agree on the digest.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_digest(payload: JsonObject) -> str:
    """The sha256 hex digest of a JSON object in its canonical serialization."""
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()
