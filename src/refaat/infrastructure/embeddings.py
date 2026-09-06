from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import httpx

from refaat.canonical import canonical_digest
from refaat.core.math import normalize_vector
from refaat.core.text import tokens
from refaat.domain.egress import (
    EgressDestination,
    EgressPolicy,
    EgressPurpose,
    ensure_egress_allowed,
)
from refaat.errors import DependencyUnavailableError
from refaat.infrastructure.egress_client import build_http_client
from refaat.infrastructure.http_retry import post_with_retry


class DeterministicEmbedder:
    def __init__(self, dimensions: int = 384, model_id: str = "refaat-hash-v1") -> None:
        if dimensions < 32:
            raise ValueError("dimensions must be at least 32")
        self._dimensions = dimensions
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * self._dimensions
        values = tokens(text)
        if not values:
            return tuple(vector)
        features = values + [f"{values[index]}::{values[index + 1]}" for index in range(len(values) - 1)]
        for feature in features:
            digest = hashlib.blake2b(feature.encode(), digest_size=16).digest()
            bucket = int.from_bytes(digest[:8], "big") % self._dimensions
            sign = 1.0 if digest[8] & 1 else -1.0
            weight = 1.0 + math.log1p(len(feature)) / 8.0
            vector[bucket] += sign * weight
        return normalize_vector(vector)


class OpenAICompatibleEmbedder:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        dimensions: int,
        send_dimensions: bool = True,
        retries: int = 2,
        retry_base_delay: float = 0.25,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
        egress_policy: EgressPolicy | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._model_id = model
        self._dimensions = dimensions
        self._send_dimensions = send_dimensions
        self._retries = retries
        self._retry_base_delay = retry_base_delay
        self.timeout = timeout
        self._owns_client = client is None
        self.client = build_http_client(
            timeout=timeout, egress_policy=egress_policy, purpose=EgressPurpose.MODEL, client=client
        )
        if egress_policy is not None:
            ensure_egress_allowed(egress_policy, EgressDestination.from_url(self.base_url, EgressPurpose.MODEL))

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def cache_identity(self) -> str:
        # Digest the structured fields, not a ':'-joined string: a base URL may itself
        # contain ':' (a port), so raw concatenation could alias two distinct backends.
        return "openai-compat-embed:" + canonical_digest(
            {
                "base_url": self.base_url,
                "model": self._model_id,
                "dimensions": self._dimensions,
                "send_dimensions": self._send_dimensions,
            }
        )

    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        body: dict[str, object] = {"model": self._model_id, "input": list(texts)}
        if self._send_dimensions:
            body["dimensions"] = self._dimensions
        try:
            response = post_with_retry(
                self.client,
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
                retries=self._retries,
                base_delay=self._retry_base_delay,
            )
            payload = response.json()
            ordered = sorted(payload.get("data", []), key=lambda item: int(item["index"]))
            vectors = [tuple(float(value) for value in item["embedding"]) for item in ordered]
        except (httpx.HTTPError, KeyError, ValueError, TypeError, AttributeError) as exc:
            raise DependencyUnavailableError("Embedding provider request failed") from exc
        if len(vectors) != len(texts) or any(len(vector) != self._dimensions for vector in vectors):
            raise DependencyUnavailableError("Embedding provider returned an invalid response")
        return vectors

    def close(self) -> None:
        if self._owns_client:
            self.client.close()
