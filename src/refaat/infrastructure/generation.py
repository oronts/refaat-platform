from __future__ import annotations

import json
from collections.abc import Generator, Iterator, Sequence
from typing import Any

import httpx

from refaat.canonical import canonical_digest
from refaat.core.citations import extract_cited_ids
from refaat.core.evidence_framing import render_untrusted_evidence
from refaat.core.guidance import render_guidance
from refaat.core.text import split_sentences, tokens
from refaat.domain.egress import (
    EgressDestination,
    EgressPolicy,
    EgressPurpose,
    ensure_egress_allowed,
)
from refaat.domain.models import (
    GenerationRequest,
    GenerationResult,
    ToolCallDelta,
    ToolProposal,
    ToolSpec,
)
from refaat.errors import DependencyUnavailableError
from refaat.infrastructure.egress_client import build_http_client
from refaat.infrastructure.http_retry import post_with_retry

_NO_EVIDENCE = "I could not find grounded evidence for this question in the selected corpus."


def _canonical_mapping(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sse_content(line: str) -> str:
    if not line.startswith("data:"):
        return ""
    data = line[len("data:") :].strip()
    if not data or data == "[DONE]":
        return ""
    try:
        payload = json.loads(data)
        return str(payload["choices"][0]["delta"].get("content") or "")
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return ""


def _sse_usage(line: str) -> dict[str, Any] | None:
    """Extract the token usage from a streamed chunk, if present.

    With ``stream_options.include_usage`` an OpenAI-compatible provider sends a
    final chunk that carries the real token counts and no content, so a streamed
    answer can report the same tokens and cost as a non-streamed call.
    """
    if not line.startswith("data:"):
        return None
    data = line[len("data:") :].strip()
    if not data or data == "[DONE]":
        return None
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return None
    usage = payload.get("usage") if isinstance(payload, dict) else None
    return dict(usage) if isinstance(usage, dict) else None


def _arguments_fragment(raw: object) -> str:
    """Coerce a provider ``function.arguments`` value to a string fragment.

    A compliant provider streams the arguments as opaque JSON string slices, but
    some deliver a whole object. Serializing a non-string to canonical JSON keeps
    the accumulator join total and lets schema validation report a wrong field
    type, mirroring how the non-streaming adapter accepts a dict.
    """
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    return json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sse_tool_call_delta(line: str) -> ToolCallDelta | None:
    """Map one SSE line to a neutral :class:`ToolCallDelta`, or None to skip.

    Collapses to the first ``index == 0`` tool call and ignores any parallel
    calls, matching ``propose_tool_call``'s ``calls[0]`` behavior. A
    ``finish_reason == "tool_calls"`` chunk terminates the call; a keep-alive or
    unrelated chunk maps to None. A terminal chunk that also co-locates a closing
    arguments fragment folds both into one delta so the last fragment is never
    dropped.
    """
    if not line.startswith("data:"):
        return None
    data = line[len("data:") :].strip()
    if not data or data == "[DONE]":
        return None
    try:
        choice = json.loads(data)["choices"][0]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return None
    done = choice.get("finish_reason") == "tool_calls"
    calls = (choice.get("delta") or {}).get("tool_calls") or []
    first = next((call for call in calls if call.get("index", 0) == 0), None)
    if first is None:
        return ToolCallDelta(tool=None, done=True) if done else None
    function = first.get("function") or {}
    return ToolCallDelta(
        tool=function.get("name") or None,
        arguments_fragment=_arguments_fragment(function.get("arguments")),
        done=done,
    )


class ExtractiveGenerator:
    def __init__(self, max_sentences: int = 7) -> None:
        self.max_sentences = max_sentences

    @property
    def model_id(self) -> str:
        return "refaat-extractive-v1"

    def _select(self, request: GenerationRequest) -> list[tuple[str, str]]:
        # Evidence-only by contract: tunable instructions and demonstrations on
        # the request are intentionally ignored here so the offline generator
        # can only ever cite real retrieved chunks, never demonstration text.
        query_terms = set(tokens(request.query))
        candidates: list[tuple[float, str, str]] = []
        for rank, hit in enumerate(request.context.hits):
            for sentence in split_sentences(hit.chunk.text):
                overlap = len(query_terms & set(tokens(sentence)))
                score = hit.score + overlap / max(len(query_terms), 1) - rank * 0.01
                candidates.append((score, sentence, hit.chunk.id))
        candidates.sort(key=lambda value: (-value[0], value[2], value[1]))
        selected: list[tuple[str, str]] = []
        seen: set[str] = set()
        for _, sentence, chunk_id in candidates:
            normalized = sentence.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            selected.append((sentence, chunk_id))
            if len(selected) >= self.max_sentences:
                break
        return selected

    def generate(self, request: GenerationRequest) -> GenerationResult:
        # Honesty boundary: the extractive generator cannot constrain its own
        # decode, so request.response_schema is intentionally ignored here.
        # Structural typing offline is enforced by StructuredValidator, not the
        # decoder, so this call returns byte-identical output with or without a
        # schema on the request.
        selected = self._select(request)
        if not selected:
            return GenerationResult(
                answer=_NO_EVIDENCE,
                cited_chunk_ids=(),
                usage={"provider": "extractive", "sentences": 0},
            )
        answer = " ".join(f"{sentence} [chunk:{chunk_id}]" for sentence, chunk_id in selected)
        cited = tuple(dict.fromkeys(chunk_id for _, chunk_id in selected))
        return GenerationResult(
            answer=answer,
            cited_chunk_ids=cited,
            usage={"provider": "extractive", "sentences": len(selected)},
        )

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        selected = self._select(request)
        if not selected:
            yield _NO_EVIDENCE
            return
        for index, (sentence, chunk_id) in enumerate(selected):
            prefix = "" if index == 0 else " "
            yield f"{prefix}{sentence} [chunk:{chunk_id}]"


class OpenAICompatibleGenerator:
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

    @property
    def cache_identity(self) -> str:
        # Digest the structured fields, not a ':'-joined string: a base URL may itself
        # contain ':' (a port), so raw concatenation could alias two distinct backends.
        return "openai-compat-gen:" + canonical_digest({"base_url": self.base_url, "model": self._model_id})

    def _post_chat(self, json_body: dict[str, Any]) -> dict[str, Any]:
        try:
            response = post_with_retry(
                self.client,
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=json_body,
                retries=self._retries,
                base_delay=self._retry_base_delay,
            )
            payload: dict[str, Any] = response.json()
            return payload
        except (httpx.HTTPError, ValueError) as exc:
            raise DependencyUnavailableError("Generation provider request failed") from exc

    def generate(self, request: GenerationRequest) -> GenerationResult:
        body: dict[str, Any] = {"model": self._model_id, "temperature": 0, "messages": self._messages(request)}
        if request.response_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "structured_output", "schema": request.response_schema, "strict": True},
            }
        payload = self._post_chat(body)
        try:
            content = str(payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise DependencyUnavailableError("Generation provider request failed") from exc
        answer = self._structured_answer(content) if request.response_schema is not None else content
        usage = payload.get("usage", {})
        return GenerationResult(answer=answer, cited_chunk_ids=extract_cited_ids(answer), usage=usage)

    @staticmethod
    def _structured_answer(content: str) -> str:
        # A schema-capable provider returns the object as the message content;
        # the grounded answer text lives in its "answer" field, and citation
        # markers in that text are filtered to retrieved chunks downstream.
        try:
            parsed = json.loads(content)
            return str(parsed["answer"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise DependencyUnavailableError("Structured generation response was not valid JSON") from exc

    def propose_tool_call(self, query: str, tools: Sequence[ToolSpec], used: frozenset[str]) -> ToolProposal | None:
        del used
        if not tools:
            return None
        tool_schemas = [
            {
                "type": "function",
                "function": {"name": spec.name, "description": spec.description, "parameters": spec.parameters},
            }
            for spec in tools
        ]
        payload = self._post_chat(
            {
                "model": self._model_id,
                "temperature": 0,
                "messages": [{"role": "user", "content": query}],
                "tools": tool_schemas,
                "tool_choice": "auto",
            }
        )
        try:
            message = payload["choices"][0]["message"]
            calls = message.get("tool_calls") or []
            if not calls:
                return None
            function = calls[0]["function"]
            name = str(function["name"])
            raw = function.get("arguments")
            arguments = json.loads(raw) if isinstance(raw, str) else raw
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise DependencyUnavailableError("Tool-calling provider request failed") from exc
        if not isinstance(arguments, dict):
            arguments = {}
        return ToolProposal(tool=name, arguments=arguments)

    def stream(self, request: GenerationRequest) -> Generator[str, None, dict[str, Any]]:
        usage: dict[str, Any] = {}
        try:
            with self.client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self._model_id,
                    "temperature": 0,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "messages": self._messages(request),
                },
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    delta = _sse_content(line)
                    if delta:
                        yield delta
                        continue
                    chunk_usage = _sse_usage(line)
                    if chunk_usage:
                        usage = chunk_usage
        except httpx.HTTPError as exc:
            raise DependencyUnavailableError("Generation provider request failed") from exc
        return usage

    def stream_tool_call(self, query: str, tools: Sequence[ToolSpec], used: frozenset[str]) -> Iterator[ToolCallDelta]:
        del used
        if not tools:
            return
        tool_schemas = [
            {
                "type": "function",
                "function": {"name": spec.name, "description": spec.description, "parameters": spec.parameters},
            }
            for spec in tools
        ]
        try:
            with self.client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self._model_id,
                    "temperature": 0,
                    "stream": True,
                    "messages": [{"role": "user", "content": query}],
                    "tools": tool_schemas,
                    "tool_choice": "auto",
                },
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    delta = _sse_tool_call_delta(line)
                    if delta is not None:
                        yield delta
        except httpx.HTTPError as exc:
            raise DependencyUnavailableError("Tool-calling provider request failed") from exc

    def _messages(self, request: GenerationRequest) -> list[dict[str, str]]:
        context = render_untrusted_evidence(request.context.hits)
        graph = "\n".join(request.context.graph_facts)
        # Render the mappings in one canonical (sorted) form so equal cache key
        # implies equal prompt bytes: two profiles with the same entries but a
        # different insertion order must not produce two different prompts.
        profile = _canonical_mapping(request.context.profile)
        supplemental = _canonical_mapping(request.context.supplemental_harness)
        guidance_block = ""
        if request.instructions or request.demonstrations:
            guidance = render_guidance(request.instructions, request.demonstrations)
            guidance_block = f"Formatting guidance (never cite these):\n```\n{guidance}\n```\n\n"
        return [
            {"role": "system", "content": request.system_prompt},
            {
                "role": "user",
                "content": (
                    f"{guidance_block}"
                    f"Question:\n{request.query}\n\n"
                    f"Evidence:\n{context}\n\n"
                    f"Graph facts:\n{graph or 'None'}\n\n"
                    f"Session summary:\n{request.context.session_summary or 'None'}\n\n"
                    f"Bounded user profile:\n{profile}\n\n"
                    f"Supplemental harness:\n{supplemental}"
                ),
            },
        ]

    def close(self) -> None:
        if self._owns_client:
            self.client.close()
