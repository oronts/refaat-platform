from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from refaat.core.text import normalize_text, split_clauses, tokens
from refaat.domain.models import JsonObject

# A capitalized word token, used as a deterministic proper-noun proxy for the
# entity signal. The router never touches the graph store during feature
# extraction (that would cost a query round-trip before the strategy is known),
# so entity presence is inferred from surface form only.
_CAPITALIZED = re.compile(r"[A-ZÀ-ÖØ-Þ][\w'-]*")


@dataclass(frozen=True, slots=True)
class QueryFeatures:
    token_count: int
    clause_count: int
    entity_count: int


@dataclass(frozen=True, slots=True)
class RouteStrategy:
    label: str
    fusion: str
    autocut_enabled: bool
    query_expansion: bool
    top_k: int | None

    def as_json(self) -> JsonObject:
        return {
            "label": self.label,
            "fusion": self.fusion,
            "autocut_enabled": self.autocut_enabled,
            "query_expansion": self.query_expansion,
            "top_k": self.top_k,
        }


@dataclass(frozen=True, slots=True)
class RouterConfig:
    """Thresholds that map query features onto a retrieval strategy.

    ``simple_top_k`` and ``complex_top_k`` are optional overrides: left ``None``
    the routed strategy keeps the inner program's ``top_k`` so an explicitly set
    inner budget flows through untouched, and only the retrieval regime (fusion,
    autocut, sub-query decomposition) is switched per query.
    """

    long_query_tokens: int = 24
    multi_clause_threshold: int = 2
    entity_threshold: int = 2
    simple_top_k: int | None = None
    complex_top_k: int | None = None

    def as_json(self) -> JsonObject:
        return {
            "long_query_tokens": self.long_query_tokens,
            "multi_clause_threshold": self.multi_clause_threshold,
            "entity_threshold": self.entity_threshold,
            "simple_top_k": self.simple_top_k,
            "complex_top_k": self.complex_top_k,
        }

    @classmethod
    def from_json(cls, value: JsonObject | None) -> RouterConfig:
        if not value:
            return cls()
        defaults = cls()
        return cls(
            long_query_tokens=_as_int(value.get("long_query_tokens"), defaults.long_query_tokens),
            multi_clause_threshold=_as_int(value.get("multi_clause_threshold"), defaults.multi_clause_threshold),
            entity_threshold=_as_int(value.get("entity_threshold"), defaults.entity_threshold),
            simple_top_k=_as_optional_int(value.get("simple_top_k")),
            complex_top_k=_as_optional_int(value.get("complex_top_k")),
        )

    @property
    def cache_id(self) -> str:
        canonical = json.dumps(self.as_json(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return "router:" + hashlib.sha256(canonical.encode()).hexdigest()[:32]


def _as_int(value: object, default: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return default


def _as_optional_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def extract_features(query: str) -> QueryFeatures:
    normalized = normalize_text(query)
    capitalized = _CAPITALIZED.findall(normalized)
    # Drop one leading capital as the sentence-initial word so a lowercase
    # question with no proper nouns scores zero entities.
    entity_count = max(0, len(capitalized) - 1)
    return QueryFeatures(
        token_count=len(tokens(normalized)),
        clause_count=len(split_clauses(normalized)),
        entity_count=entity_count,
    )


def route(features: QueryFeatures, config: RouterConfig) -> RouteStrategy:
    complex_query = (
        features.clause_count >= config.multi_clause_threshold
        or features.token_count >= config.long_query_tokens
        or features.entity_count >= config.entity_threshold
    )
    if complex_query:
        return RouteStrategy(
            label="deep",
            fusion="rrf",
            autocut_enabled=False,
            query_expansion=True,
            top_k=config.complex_top_k,
        )
    return RouteStrategy(
        label="direct",
        fusion="linear",
        autocut_enabled=True,
        query_expansion=False,
        top_k=config.simple_top_k,
    )


def route_query(query: str, config: RouterConfig) -> RouteStrategy:
    return route(extract_features(query), config)
