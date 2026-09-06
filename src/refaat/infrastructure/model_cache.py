from __future__ import annotations

import json
from collections.abc import Generator as AbcGenerator
from collections.abc import Iterator, Sequence
from typing import Any

from refaat.application.budget import CostMeter, active_meter
from refaat.application.services import in_shadow_execution, in_unserved_run
from refaat.canonical import canonical_digest
from refaat.core.text import sha256_bytes, tokens
from refaat.domain.models import (
    GenerationRequest,
    GenerationResult,
    JsonObject,
    SearchHit,
    ToolCallDelta,
    ToolProposal,
    ToolSpec,
)
from refaat.ports import CacheIdentified, Embedder, Generator, ModelCallCache


def _chargeable_meter() -> CostMeter | None:
    """The active run meter, or None on the served degraded path.

    The served degraded path (a bare ``shadow_execution`` scope that suppresses a
    transient-outage answer's writes) must not spend the served run's budget, so it
    charges nothing. An unserved deployment run charges its OWN isolated meter (bound
    by the deployment router), so its diagnostic spend is bounded yet never draws down
    the served run's budget. Outside any shadow scope this is the plain active meter.
    """
    if in_shadow_execution() and not in_unserved_run():
        return None
    return active_meter()


def cache_identity_of(adapter: object) -> str | None:
    """The provider fingerprint of a cacheable adapter, or None for one without.

    Wrapping is a positive opt-in: only an adapter that exposes ``cache_identity``
    is ever cached, so the deterministic offline stack and any custom deterministic
    adapter stay unwrapped and can never be served a stale cached value.
    """
    return adapter.cache_identity if isinstance(adapter, CacheIdentified) else None


def resolved_cache_identity(adapter: object) -> str | None:
    """The backend cache identity of a raw adapter OR a caching wrapper.

    The caching wrappers deliberately do not expose ``cache_identity`` (so wrapping is never a positive
    re-cache opt-in), but they hold the inner backend identity, which a scope key such as the durable
    optimization plan digest must fold to keep two same-model different-backend runs apart.
    """
    for attr in ("cache_identity", "embedding_cache_identity", "generation_cache_identity"):
        value = getattr(adapter, attr, None)
        if isinstance(value, str):
            return value
    return None


def _estimate_tokens(text: str) -> int:
    return len(tokens(text))


def _request_prompt_tokens(request: GenerationRequest) -> int:
    """Deterministic prompt-token estimate for a streamed miss.

    The provider sends no usage over a stream, so the prompt is estimated from
    the request the caller assembled: the system prompt, the query, every hit's
    chunk text, the graph facts, the session summary, and any guidance. Charging
    this up front means an early-closed stream still pays for the prompt it sent.
    """
    context = request.context
    parts: list[str] = [request.system_prompt, request.query]
    parts.extend(hit.chunk.text for hit in context.hits)
    parts.extend(context.graph_facts)
    if context.session_summary:
        parts.append(context.session_summary)
    if request.instructions:
        parts.append(request.instructions)
    for demonstration in request.demonstrations:
        parts.append(demonstration.question)
        parts.append(demonstration.answer)
    return sum(_estimate_tokens(part) for part in parts)


def _tool_manifest_tokens(tools: Sequence[ToolSpec]) -> int:
    total = 0
    for tool in tools:
        total += _estimate_tokens(tool.name) + _estimate_tokens(tool.description)
        total += _estimate_tokens(json.dumps(tool.parameters, sort_keys=True, ensure_ascii=False))
    return total


def _proposal_tokens(proposal: ToolProposal) -> int:
    return _estimate_tokens(proposal.tool) + _estimate_tokens(
        json.dumps(proposal.arguments, sort_keys=True, ensure_ascii=False)
    )


def _tool_call_delta_tokens(delta: ToolCallDelta) -> int:
    total = _estimate_tokens(delta.tool) if delta.tool else 0
    if delta.arguments_fragment:
        total += _estimate_tokens(delta.arguments_fragment)
    return total


def _embedding_key(identity: str, text: str) -> str:
    return canonical_digest({"identity": identity, "text": text})


def embedding_cache_key(identity: str, text: str) -> str:
    """The embedding cache key for a text under a backend identity.

    Public alias of the internal keying used by :class:`CachingEmbedder`, so erasure can
    recompute the exact rows an embedded text produced and delete them without drift.
    """
    return _embedding_key(identity, text)


def _hit_payload(hit: SearchHit) -> JsonObject:
    chunk = hit.chunk
    return {
        "id": chunk.id,
        "corpus_id": chunk.corpus_id,
        "document_id": chunk.document_id,
        "ordinal": chunk.ordinal,
        "text": chunk.text,
        "token_count": chunk.token_count,
        "checksum": chunk.checksum,
        "metadata": chunk.metadata,
        # Two chunks can share id/text/checksum yet carry different embedding vectors
        # after an embedder swap and re-ingest, so a custom generator that reads the
        # raw vector must not be served a stale entry; fold the embedding into the key.
        "embedding": list(chunk.embedding),
        # Key on the exact score, not a 6-decimal rounding: a custom generator can
        # read repr(hit.score), so two hits that differ only past the 6th decimal
        # are genuinely distinct provider inputs and must not alias to one entry.
        # Scores are deterministic given the embeddings, so exact keying never
        # fragments the offline cache.
        "score": hit.score,
        "semantic_score": hit.semantic_score,
        "lexical_score": hit.lexical_score,
        "graph_score": hit.graph_score,
        "reasons": list(hit.reasons),
    }


def _generation_key(identity: str, workspace_id: str, request: GenerationRequest) -> str:
    """A canonical digest of the COMPLETE generation input.

    Every field the provider can read is folded in, so equal key implies equal
    provider input: the backend ``identity`` (base URL plus model), the tenant
    ``workspace_id``, the system prompt and query, each hit's full chunk (including
    ``document_id``, which the built-in generator renders into the prompt) and its
    scores, the graph facts, session summary, profile, supplemental harness,
    instructions, demonstrations, and response schema. Two requests that differ in
    any one of these get distinct keys, so no distinct provider input can ever share
    a cache entry.
    """
    context = request.context
    payload: JsonObject = {
        "identity": identity,
        "workspace_id": workspace_id,
        "system_prompt": request.system_prompt,
        "query": request.query,
        # The context carries its own query (a transform can restate it), which the
        # provider can read; fold it in so two contexts differing only there separate.
        "context_query": context.query,
        "hits": [_hit_payload(hit) for hit in context.hits],
        "graph_facts": list(context.graph_facts),
        "session_summary": context.session_summary,
        "profile": context.profile,
        "supplemental_harness": context.supplemental_harness,
        "context_token_count": context.token_count,
        "routes": list(context.routes),
        "instructions": request.instructions,
        "demonstrations": [demo.as_json() for demo in request.demonstrations],
        "response_schema": request.response_schema,
    }
    # Fold attached image content in only when present, keyed by a content digest so
    # two requests with different pixels never share a vision cache entry. A text-only
    # request carries no images, so its digest is byte-identical to before.
    if request.images:
        payload["images"] = [
            {"chunk_id": image.chunk_id, "media_type": image.media_type, "content": sha256_bytes(image.content)}
            for image in request.images
        ]
    return canonical_digest(payload)


class CachingEmbedder:
    """Transparent content-addressed cache and cost meter in front of a real Embedder.

    Delegates ``model_id`` and ``dimensions`` unchanged so the answer-cache scope is
    untouched, and keys on the inner adapter's ``cache_identity`` so two backends
    sharing a bare model name never collide. ``cache`` may be None so the wrapper
    meters embedding spend without caching when only a budget is configured. A cache
    hit is byte-identical to a miss because decode is deterministic; a real miss
    charges the active meter with a token estimate of the embedded text.
    """

    def __init__(
        self, inner: Embedder, cache: ModelCallCache | None, *, ttl_seconds: int, close_inner: bool = True
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._ttl = ttl_seconds
        self._identity = cache_identity_of(inner) if cache is not None else None
        self._close_inner = close_inner

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    @property
    def embedding_cache_identity(self) -> str | None:
        """The backend identity this wrapper writes embedding rows under, or None when uncached.

        Distinct from the ``cache_identity`` marker on purpose, so wrapping this embedder is never
        turned into a positive cache opt-in; erasure reads it to recompute the exact rows to purge.
        """
        return self._identity

    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        text_list = list(texts)
        identity = self._identity
        if self._cache is None or identity is None:
            vectors = self._inner.embed(text_list)
            self._charge(text_list)
            return vectors
        keys = [_embedding_key(identity, text) for text in text_list]
        cached = [self._cache.get_embedding(identity, key) for key in keys]
        pending: dict[str, str] = {}
        for key, text, hit in zip(keys, text_list, cached, strict=True):
            if hit is None and key not in pending:
                pending[key] = text
        computed: dict[str, tuple[float, ...]] = {}
        if pending:
            pending_keys = list(pending)
            pending_texts = [pending[key] for key in pending_keys]
            vectors = self._inner.embed(pending_texts)
            shadow = in_shadow_execution()
            for key, vector in zip(pending_keys, vectors, strict=True):
                if not shadow:
                    self._cache.put_embedding(identity, key, vector, self._ttl)
                computed[key] = vector
            self._charge(pending_texts)
        return [hit if hit is not None else computed[key] for key, hit in zip(keys, cached, strict=True)]

    def _charge(self, texts: Sequence[str]) -> None:
        if not texts:
            return
        meter = _chargeable_meter()
        if meter is not None:
            meter.charge({"prompt_tokens": sum(_estimate_tokens(text) for text in texts)}, model_id=self.model_id)

    def close(self) -> None:
        if self._close_inner:
            _close(self._inner)


class CachingGenerator:
    """Transparent cache and cost meter in front of a real Generator.

    Delegates ``model_id`` unchanged so the citation contract and answer-cache
    identity hold, and keys on the inner adapter's ``cache_identity`` plus the
    request ``workspace_id`` so a tenant only ever reads its own rows and no two
    backends collide. An empty ``workspace_id`` is never cacheable: it delegates
    live (still metered), so an internal caller that forgot to thread its tenant
    can never cross-serve. On a real miss it calls the inner adapter, charges the active
    meter (never raising; token and cost are known only after the call), and stores
    the result. A hit returns byte-identical output and charges nothing, because it
    was already paid. ``cache`` may be None so the wrapper meters without caching
    when only a budget is configured; ``stream`` and ``propose_tool_call`` delegate
    live and charge a token estimate of their real call.
    """

    def __init__(
        self, inner: Generator, cache: ModelCallCache | None, *, ttl_seconds: int, close_inner: bool = True
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._ttl = ttl_seconds
        self._identity = cache_identity_of(inner) if cache is not None else None
        self._close_inner = close_inner

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    @property
    def generation_cache_identity(self) -> str | None:
        # The backend identity this wrapper caches under, exposed under a name distinct from
        # cache_identity so wrapping is never turned into a positive cache opt-in.
        return self._identity

    def generate(self, request: GenerationRequest) -> GenerationResult:
        identity = self._identity
        cache = self._cache
        # Fail-safe: an empty workspace is never cacheable. An internal caller that
        # forgot to thread its tenant would otherwise share scope "" across tenants
        # and cross-serve, so an unscoped request drops to the uncacheable path
        # below, delegating live (still metered) without ever touching a cache row.
        if cache is not None and identity is not None and request.workspace_id:
            key = _generation_key(identity, request.workspace_id, request)
            hit = cache.get_model_call(request.workspace_id, identity, key)
            if hit is not None:
                return _result_from_payload(hit)
            result = self._inner.generate(request)
            self._charge(result.usage)
            if not in_shadow_execution():
                cache.put_model_call(request.workspace_id, identity, key, _payload_from_result(result), self._ttl)
            return result
        result = self._inner.generate(request)
        self._charge(result.usage)
        return result

    def stream(self, request: GenerationRequest) -> AbcGenerator[str, None, dict[str, Any]]:
        meter = _chargeable_meter()
        # The stream port yields text and forwards the provider's real usage as the
        # generator's return value when it is available. A streamed miss is still
        # metered with a deterministic prompt-plus-completion estimate (the prompt up
        # front so an early-closed stream still pays for its prompt, then each
        # completion piece), while the real usage is forwarded to the caller for
        # reporting. The metering divergence from generate is documented in
        # docs/streaming.md.
        if meter is not None:
            meter.charge({"prompt_tokens": _request_prompt_tokens(request)}, model_id=self.model_id)
        inner = self._inner.stream(request)
        usage: dict[str, Any] = {}
        while True:
            try:
                piece = next(inner)
            except StopIteration as stop:
                if isinstance(stop.value, dict):
                    usage = stop.value
                break
            if meter is not None:
                meter.charge({"completion_tokens": _estimate_tokens(piece)}, model_id=self.model_id)
            yield piece
        return usage

    def propose_tool_call(self, query: str, tools: Sequence[ToolSpec], used: frozenset[str]) -> ToolProposal | None:
        propose = getattr(self._inner, "propose_tool_call", None)
        if propose is None:
            return None
        proposal: ToolProposal | None = propose(query, tools, used)
        meter = _chargeable_meter()
        if meter is not None:
            prompt = _estimate_tokens(query) + _tool_manifest_tokens(tools)
            completion = _proposal_tokens(proposal) if proposal is not None else 0
            meter.charge({"prompt_tokens": prompt, "completion_tokens": completion}, model_id=self.model_id)
        return proposal

    def stream_tool_call(self, query: str, tools: Sequence[ToolSpec], used: frozenset[str]) -> Iterator[ToolCallDelta]:
        """Metered proxy for a streamed tool call, mirroring ``propose_tool_call``.

        A streamed tool-call planner built over the wrapped generator routes
        through the same single meter, so streamed planning is metered like every
        other model path. The prompt estimate (query plus serialized manifest) is
        charged up front and each streamed delta's fragment as it arrives. The
        deterministic offline stack exposes no streaming tool generator, so this is
        never reached there and the offline path stays byte-identical.
        """
        stream = getattr(self._inner, "stream_tool_call", None)
        if stream is None:
            return
        meter = _chargeable_meter()
        if meter is not None:
            meter.charge(
                {"prompt_tokens": _estimate_tokens(query) + _tool_manifest_tokens(tools)}, model_id=self.model_id
            )
        for delta in stream(query, tools, used):
            if meter is not None:
                meter.charge({"completion_tokens": _tool_call_delta_tokens(delta)}, model_id=self.model_id)
            yield delta

    def _charge(self, usage: JsonObject) -> None:
        meter = _chargeable_meter()
        if meter is not None:
            meter.charge(usage, model_id=self.model_id)

    def close(self) -> None:
        if self._close_inner:
            _close(self._inner)


def _payload_from_result(result: GenerationResult) -> JsonObject:
    return {"answer": result.answer, "cited_chunk_ids": list(result.cited_chunk_ids), "usage": result.usage}


def _result_from_payload(payload: JsonObject) -> GenerationResult:
    return GenerationResult(
        answer=str(payload["answer"]),
        cited_chunk_ids=tuple(str(value) for value in payload["cited_chunk_ids"]),
        usage=dict(payload.get("usage", {})),
    )


def _close(inner: object) -> None:
    close = getattr(inner, "close", None)
    if callable(close):
        close()
