from __future__ import annotations

import base64
from collections.abc import Iterator
from typing import Any

import httpx

from refaat.canonical import canonical_digest
from refaat.core.citations import extract_cited_ids
from refaat.core.evidence_framing import render_untrusted_evidence
from refaat.domain.egress import (
    EgressDestination,
    EgressPolicy,
    EgressPurpose,
    ensure_egress_allowed,
)
from refaat.domain.models import GenerationImage, GenerationRequest, GenerationResult
from refaat.errors import DependencyUnavailableError
from refaat.infrastructure.egress_client import build_http_client
from refaat.infrastructure.generation import _sse_content
from refaat.infrastructure.http_retry import post_with_retry

_CAPTION_SYSTEM_PROMPT = (
    "You describe an image for a retrieval index. Write one factual sentence naming what the image "
    "shows. Do not invent text, numbers, or details you cannot see. Reply with the description only."
)

_VISION_SYSTEM_PROMPT = (
    "You answer strictly from the supplied evidence and images. Every claim must cite the evidence it "
    "comes from using the marker [chunk:<id>] with the exact chunk id shown next to that evidence. Cite "
    "an image only through the [chunk:<id>] marker of the image evidence block it belongs to. Never "
    "invent a chunk id."
)


def _data_uri(image: GenerationImage) -> str:
    encoded = base64.b64encode(image.content).decode("ascii")
    return f"data:{image.media_type};base64,{encoded}"


class OpenAIVisionCaptioner:
    """Real multimodal captioner used only when a vision model is configured.

    Never constructed in the offline gate: the ingestion service builds an honest
    deterministic descriptor when no captioner is present, and degrades back to it if
    this call fails. This adapter is the only place that sends image bytes to a model,
    reading its key from config like every other adapter.
    """

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
        self._model = model
        self._retries = retries
        self._retry_base_delay = retry_base_delay
        self._owns_client = client is None
        self.client = build_http_client(
            timeout=timeout, egress_policy=egress_policy, purpose=EgressPurpose.MODEL, client=client
        )
        if egress_policy is not None:
            ensure_egress_allowed(egress_policy, EgressDestination.from_url(self.base_url, EgressPurpose.MODEL))

    @property
    def provider_id(self) -> str:
        return "vision-caption:" + canonical_digest({"base_url": self.base_url, "model": self._model})

    def caption(self, content: bytes, *, media_type: str, title: str) -> str:
        image = GenerationImage(chunk_id="", media_type=media_type, content=content)
        body: dict[str, Any] = {
            "model": self._model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": _CAPTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Title: {title}" if title else "Describe this image."},
                        {"type": "image_url", "image_url": {"url": _data_uri(image)}},
                    ],
                },
            ],
        }
        try:
            response = post_with_retry(
                self.client,
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
                retries=self._retries,
                base_delay=self._retry_base_delay,
            )
            payload: dict[str, Any] = response.json()
            return str(payload["choices"][0]["message"]["content"]).strip()
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise DependencyUnavailableError("Image captioning provider request failed") from exc

    def close(self) -> None:
        if self._owns_client:
            self.client.close()


class OpenAIVisionGenerator:
    """Real vision generator used only when a multimodal model is configured.

    Answers over the actual image content carried on the request while still returning
    citations to the real stored image-descriptor chunks: the model is instructed to
    cite the [chunk:<id>] markers of the evidence blocks, and the downstream pipeline
    filters those ids to retrieved chunks and resolves them against stored rows, so the
    raw image is never a citation. A text-only request (no images) answers exactly like
    the text generator. Never constructed in the offline gate.
    """

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
        self._model = model
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
        return self._model

    @property
    def cache_identity(self) -> str:
        return "vision-gen:" + canonical_digest({"base_url": self.base_url, "model": self._model})

    def _messages(self, request: GenerationRequest) -> list[dict[str, Any]]:
        context = render_untrusted_evidence(request.context.hits)
        graph = "\n".join(request.context.graph_facts) or "None"
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"Question:\n{request.query}\n\n"
                    f"Evidence:\n{context}\n\n"
                    f"Graph facts:\n{graph}\n\n"
                    f"Session summary:\n{request.context.session_summary or 'None'}"
                ),
            }
        ]
        for image in request.images:
            content.append({"type": "text", "text": f"Image evidence [chunk:{image.chunk_id}]:"})
            content.append({"type": "image_url", "image_url": {"url": _data_uri(image)}})
        return [
            {"role": "system", "content": request.system_prompt or _VISION_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def generate(self, request: GenerationRequest) -> GenerationResult:
        body = {"model": self._model, "temperature": 0, "messages": self._messages(request)}
        try:
            response = post_with_retry(
                self.client,
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
                retries=self._retries,
                base_delay=self._retry_base_delay,
            )
            payload: dict[str, Any] = response.json()
            answer = str(payload["choices"][0]["message"]["content"])
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise DependencyUnavailableError("Multimodal generation provider request failed") from exc
        usage = payload.get("usage", {})
        return GenerationResult(answer=answer, cited_chunk_ids=extract_cited_ids(answer), usage=usage)

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        try:
            with self.client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self._model, "temperature": 0, "stream": True, "messages": self._messages(request)},
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    delta = _sse_content(line)
                    if delta:
                        yield delta
        except httpx.HTTPError as exc:
            raise DependencyUnavailableError("Multimodal generation provider request failed") from exc

    def close(self) -> None:
        if self._owns_client:
            self.client.close()
