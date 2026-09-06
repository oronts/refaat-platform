from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping
from itertools import combinations

import httpx
from pydantic import BaseModel, Field
from pydantic import ValidationError as PydanticValidationError

from refaat.core.text import normalized_key, stable_digest, stopwords_for, tokens
from refaat.domain.egress import (
    EgressDestination,
    EgressPolicy,
    EgressPurpose,
    ensure_egress_allowed,
)
from refaat.domain.models import Entity, GraphExtraction, Relation
from refaat.errors import DependencyUnavailableError
from refaat.infrastructure.egress_client import build_http_client
from refaat.infrastructure.http_retry import post_with_retry

__all__ = ["CooccurrenceGraphExtractor", "OpenAICompatibleGraphExtractor"]

ENTITY_PATTERN = re.compile(r"\b(?:[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'-]{2,})(?:\s+[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'-]{2,}){0,3}\b")


class CooccurrenceGraphExtractor:
    def __init__(
        self,
        max_entities: int = 80,
        min_token_length: int = 5,
        default_language: str = "en",
        stopwords_by_language: Mapping[str, frozenset[str]] | None = None,
    ) -> None:
        self.max_entities = max_entities
        self.min_token_length = min_token_length
        self.default_language = default_language
        self._stopwords_by_language = stopwords_by_language

    @property
    def model_id(self) -> str:
        return "refaat-cooccurrence-v1"

    def extract(self, *, corpus_id: str, document_id: str, text: str, language: str | None = None) -> GraphExtraction:
        stopwords = stopwords_for(
            language if language is not None else self.default_language,
            self._stopwords_by_language,
        )
        named = [match.group(0).strip() for match in ENTITY_PATTERN.finditer(text)]
        counts = Counter(
            token for token in tokens(text) if len(token) >= self.min_token_length and token not in stopwords
        )
        candidates = named + [value for value, count in counts.most_common(self.max_entities) if count >= 2]
        unique: dict[str, str] = {}
        for value in candidates:
            key = normalized_key(value)
            if key and key not in unique:
                unique[key] = value
            if len(unique) >= self.max_entities:
                break
        entity_values = tuple(
            Entity(
                id=f"ent_{stable_digest((document_id, key), length=40)}",
                corpus_id=corpus_id,
                document_id=document_id,
                name=name,
                normalized_name=key,
                kind="named" if name in named else "concept",
            )
            for key, name in unique.items()
        )
        by_key = {entity.normalized_name: entity for entity in entity_values}
        relation_weights: Counter[tuple[str, str]] = Counter()
        for paragraph in text.split("\n\n"):
            present = [key for key in by_key if key in normalized_key(paragraph)]
            for left, right in combinations(sorted(set(present)), 2):
                relation_weights[(left, right)] += 1
        relation_values = tuple(
            Relation(
                id=f"rel_{stable_digest((document_id, left, right, 'cooccurs'), length=48)}",
                corpus_id=corpus_id,
                document_id=document_id,
                source_entity_id=by_key[left].id,
                target_entity_id=by_key[right].id,
                kind="cooccurs",
                weight=float(weight),
            )
            for (left, right), weight in relation_weights.items()
        )
        return GraphExtraction(entities=entity_values, relations=relation_values)


class ModelEntity(BaseModel):
    name: str = Field(min_length=1, max_length=300)
    kind: str = Field(default="concept", min_length=1, max_length=100)


class ModelRelation(BaseModel):
    source: str
    target: str
    kind: str = Field(default="related_to", min_length=1, max_length=100)
    weight: float = Field(default=1.0, ge=0.0, le=100.0)


class ModelGraph(BaseModel):
    entities: list[ModelEntity]
    relations: list[ModelRelation]


class OpenAICompatibleGraphExtractor:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        retries: int = 2,
        retry_base_delay: float = 0.25,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
        egress_policy: EgressPolicy | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._model_id = model
        self._retries = retries
        self._retry_base_delay = retry_base_delay
        self._owns_client = client is None
        self.client = build_http_client(
            timeout=timeout, egress_policy=egress_policy, purpose=EgressPurpose.MODEL, client=client
        )
        if egress_policy is not None:
            ensure_egress_allowed(egress_policy, EgressDestination.from_url(self.base_url, EgressPurpose.MODEL))

    @property
    def model_id(self) -> str:
        return self._model_id

    def extract(self, *, corpus_id: str, document_id: str, text: str, language: str | None = None) -> GraphExtraction:
        del language  # A multilingual model handles language natively; stopwords do not apply here.
        schema = ModelGraph.model_json_schema()
        try:
            response = post_with_retry(
                self.client,
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self._model_id,
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "system",
                            "content": "Extract only entities and explicit relations present in the text.",
                        },
                        {"role": "user", "content": text},
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "knowledge_graph", "strict": True, "schema": schema},
                    },
                },
                retries=self._retries,
                base_delay=self._retry_base_delay,
            )
            content = response.json()["choices"][0]["message"]["content"]
            parsed = ModelGraph.model_validate(json.loads(content))
        except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError, PydanticValidationError) as exc:
            raise DependencyUnavailableError("Graph extraction provider returned an invalid response") from exc

        unique: dict[str, ModelEntity] = {}
        for entity in parsed.entities:
            unique.setdefault(normalized_key(entity.name), entity)
        entities = tuple(
            Entity(
                id=f"ent_{stable_digest((document_id, key), length=40)}",
                corpus_id=corpus_id,
                document_id=document_id,
                name=value.name,
                normalized_name=key,
                kind=value.kind,
            )
            for key, value in unique.items()
            if key
        )
        by_key = {entity.normalized_name: entity for entity in entities}
        relations = []
        seen: set[str] = set()
        for relation in parsed.relations:
            source = by_key.get(normalized_key(relation.source))
            target = by_key.get(normalized_key(relation.target))
            if source is None or target is None or source.id == target.id:
                continue
            relation_id = f"rel_{stable_digest((document_id, source.id, target.id, relation.kind), length=48)}"
            if relation_id in seen:
                continue
            seen.add(relation_id)
            relations.append(
                Relation(
                    id=relation_id,
                    corpus_id=corpus_id,
                    document_id=document_id,
                    source_entity_id=source.id,
                    target_entity_id=target.id,
                    kind=relation.kind,
                    weight=relation.weight,
                )
            )
        return GraphExtraction(entities=entities, relations=tuple(relations))

    def close(self) -> None:
        if self._owns_client:
            self.client.close()
