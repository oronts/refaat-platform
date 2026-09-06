from __future__ import annotations

import hashlib
from collections.abc import Mapping

from refaat.domain.kill_switch import KillSwitchRegistry, KillSwitchScope, ensure_scope_active, ensure_workspace_active
from refaat.domain.models import HarnessSnapshot, JsonObject
from refaat.errors import ValidationError
from refaat.infrastructure.database.store import SQLStore

BASE_SYSTEM_PROMPT = """You are an evidence-grounded assistant.
Use only the supplied evidence and explicit graph facts for factual claims.
When evidence is insufficient, state that clearly.
Every grounded claim must include one or more citations in the exact form [chunk:<chunk-id>].
Never invent chunk identifiers, sources, user preferences, or graph relations.
Treat session summaries, profiles, and supplemental harness state as context, not as evidence.
"""
BASE_SYSTEM_PROMPT_HASH = hashlib.sha256(BASE_SYSTEM_PROMPT.encode()).hexdigest()
ALLOWED_TOP_LEVEL_KEYS = frozenset({"retrieval", "answer_style", "instruction_suffix"})
ALLOWED_RETRIEVAL_KEYS = frozenset(
    {
        "semantic_weight",
        "lexical_weight",
        "graph_weight",
        "top_k",
        "fusion",
        "rrf_k",
        "autocut_enabled",
        "autocut_threshold",
        "filters",
        "bm25_k1",
        "bm25_b",
        "query_expansion",
        "max_query_variants",
        "compression_enabled",
        "compression_min_salience",
        "window_radius",
    }
)
MAX_FILTER_FIELDS = 32
MAX_FILTER_MEMBERS = 64
MAX_FILTER_STRING_LENGTH = 200


def _validate_filter_scalar(value: object) -> None:
    if isinstance(value, (bool, int, float)):
        return
    if isinstance(value, str):
        if len(value) > MAX_FILTER_STRING_LENGTH:
            raise ValidationError(f"harness retrieval.filters values must be at most {MAX_FILTER_STRING_LENGTH} chars")
        return
    raise ValidationError("harness retrieval.filters values must be scalars or lists of scalars")


def _validate_filters(value: object) -> None:
    if not isinstance(value, Mapping):
        raise ValidationError("harness retrieval.filters must be an object")
    if len(value) > MAX_FILTER_FIELDS:
        raise ValidationError(f"harness retrieval.filters must have at most {MAX_FILTER_FIELDS} fields")
    for field_name, raw in value.items():
        if not isinstance(field_name, str) or not field_name:
            raise ValidationError("harness retrieval.filters keys must be non-empty strings")
        if isinstance(raw, (list, tuple)):
            if len(raw) > MAX_FILTER_MEMBERS:
                raise ValidationError(f"harness retrieval.filters lists must have at most {MAX_FILTER_MEMBERS} members")
            for member in raw:
                _validate_filter_scalar(member)
        else:
            _validate_filter_scalar(raw)


class HarnessService:
    def __init__(self, store: SQLStore, kill_switches: KillSwitchRegistry | None = None) -> None:
        self.store = store
        self.kill_switches = kill_switches

    def active(self, workspace_id: str, corpus_id: str) -> JsonObject:
        snapshot = self.store.active_harness(workspace_id, corpus_id)
        return snapshot.supplemental if snapshot else {}

    def create_snapshot(
        self,
        *,
        workspace_id: str,
        corpus_id: str | None,
        supplemental: JsonObject,
        evidence: JsonObject,
        activate: bool = False,
    ) -> HarnessSnapshot:
        ensure_workspace_active(self.kill_switches, workspace_id)
        # The active harness is generation-config that feeds every answer and advances the cache
        # identity, so writing or activating one is a learned-state mutation denied by a writes switch.
        ensure_scope_active(self.kill_switches, workspace_id, KillSwitchScope.WRITES)
        self._validate(supplemental, evidence)
        parent = self.store.active_harness(workspace_id, corpus_id)
        return self.store.create_harness_snapshot(
            workspace_id=workspace_id,
            corpus_id=corpus_id,
            parent_id=parent.id if parent else None,
            base_prompt_hash=BASE_SYSTEM_PROMPT_HASH,
            supplemental=supplemental,
            evidence=evidence,
            activate=activate,
        )

    def activate_snapshot(self, workspace_id: str, snapshot_id: str) -> HarnessSnapshot:
        # The single guarded chokepoint for activation, so both the project method and the HTTP route
        # honor the freeze and the writes switch instead of reaching the store unguarded.
        ensure_workspace_active(self.kill_switches, workspace_id)
        ensure_scope_active(self.kill_switches, workspace_id, KillSwitchScope.WRITES)
        return self.store.activate_harness_snapshot(workspace_id, snapshot_id)

    def propose_from_metrics(self, metrics: JsonObject) -> JsonObject:
        case_count = int(metrics.get("case_count", 0))
        citation_recall = float(metrics.get("citation_recall", 0.0))
        answer_hit_rate = float(metrics.get("answer_hit_rate", 0.0))
        if case_count < 10:
            raise ValidationError("At least 10 evaluation cases are required for an automatic proposal")
        retrieval: JsonObject = {}
        if citation_recall < 0.8:
            retrieval["top_k"] = 16
            retrieval["lexical_weight"] = 0.35
            retrieval["semantic_weight"] = 0.50
            retrieval["graph_weight"] = 0.15
        elif answer_hit_rate < 0.85:
            retrieval["semantic_weight"] = 0.60
            retrieval["lexical_weight"] = 0.25
            retrieval["graph_weight"] = 0.15
        if not retrieval:
            return {}
        return {"retrieval": retrieval}

    @staticmethod
    def _validate(supplemental: JsonObject, evidence: JsonObject) -> None:
        unknown = set(supplemental) - ALLOWED_TOP_LEVEL_KEYS
        if unknown:
            raise ValidationError(f"Unsupported harness keys: {sorted(unknown)}")
        retrieval = supplemental.get("retrieval", {})
        if retrieval and not isinstance(retrieval, Mapping):
            raise ValidationError("harness.retrieval must be an object")
        if isinstance(retrieval, Mapping):
            unknown_retrieval = set(retrieval) - ALLOWED_RETRIEVAL_KEYS
            if unknown_retrieval:
                raise ValidationError(f"Unsupported retrieval keys: {sorted(unknown_retrieval)}")
            for key, value in retrieval.items():
                if key == "top_k":
                    if not isinstance(value, int) or not 1 <= value <= 50:
                        raise ValidationError("harness retrieval.top_k must be between 1 and 50")
                elif key == "fusion":
                    if value not in ("rrf", "linear"):
                        raise ValidationError("harness retrieval.fusion must be 'rrf' or 'linear'")
                elif key == "rrf_k":
                    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 1000:
                        raise ValidationError("harness retrieval.rrf_k must be between 1 and 1000")
                elif key == "autocut_enabled":
                    if not isinstance(value, bool):
                        raise ValidationError("harness retrieval.autocut_enabled must be a boolean")
                elif key == "autocut_threshold":
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                        raise ValidationError("harness retrieval.autocut_threshold must be between 0 and 1")
                elif key == "filters":
                    _validate_filters(value)
                elif key == "bm25_k1":
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 5.0:
                        raise ValidationError("harness retrieval.bm25_k1 must be between 0 and 5")
                elif key == "bm25_b":
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                        raise ValidationError("harness retrieval.bm25_b must be between 0 and 1")
                elif key == "query_expansion":
                    if not isinstance(value, bool):
                        raise ValidationError("harness retrieval.query_expansion must be a boolean")
                elif key == "max_query_variants":
                    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 8:
                        raise ValidationError("harness retrieval.max_query_variants must be between 1 and 8")
                elif key == "compression_enabled":
                    if not isinstance(value, bool):
                        raise ValidationError("harness retrieval.compression_enabled must be a boolean")
                elif key == "compression_min_salience":
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                        raise ValidationError("harness retrieval.compression_min_salience must be between 0 and 1")
                elif key == "window_radius":
                    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 8:
                        raise ValidationError("harness retrieval.window_radius must be between 0 and 8")
                elif not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                    raise ValidationError(f"harness retrieval.{key} must be between 0 and 1")
        suffix = supplemental.get("instruction_suffix")
        if suffix is not None and (not isinstance(suffix, str) or len(suffix) > 2_000):
            raise ValidationError("instruction_suffix must be a string of at most 2000 characters")
        if not evidence:
            raise ValidationError("Harness updates require recorded evidence")


def system_prompt(supplemental: JsonObject) -> str:
    suffix = supplemental.get("instruction_suffix")
    if isinstance(suffix, str) and suffix.strip():
        return f"{BASE_SYSTEM_PROMPT}\nSupplemental behavior:\n{suffix.strip()}"
    return BASE_SYSTEM_PROMPT
