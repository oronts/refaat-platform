"""In-memory per-provider circuit breaker and the transparent adapter wrappers.

``InMemoryCircuitBreaker`` holds a per-endpoint state machine under a lock with an injectable clock.
``BreakerGenerator`` and ``BreakerEmbedder`` sit innermost, before any caching wrapper, and bracket
each provider-hitting call: a fast-fail when the endpoint circuit is open, otherwise the call runs and
its outcome is recorded. Only a genuine transport or 5xx fault contributes to opening; a client-side or
parse fault does not. Per-process state is the honest boundary, the same one the edge rate limiter
documents; a cross-process breaker is a follow-up, and the per-call request timeout keeps every provider
call bounded regardless of breaker state.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from collections.abc import Generator as AbcGenerator
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlsplit

import httpx

from refaat.domain.models import GenerationRequest, GenerationResult
from refaat.domain.resilience import CallOutcome, CircuitBreakerConfig, CircuitState
from refaat.errors import DependencyUnavailableError, ProviderCircuitOpenError
from refaat.ports import Embedder, Generator


def provider_endpoint(base_url: str) -> str:
    parts = urlsplit(base_url)
    host = (parts.hostname or "").lower()
    port = parts.port if parts.port is not None else (443 if parts.scheme == "https" else 80)
    path = parts.path.rstrip("/")
    return f"{parts.scheme}://{host}:{port}{path}"


def _contributes(exc: DependencyUnavailableError, count_429: bool) -> bool:
    # Only a genuine unreachable-or-failing endpoint trips the breaker. A client-side protocol fault,
    # an unsupported scheme, a parse failure, or a no-cause error is not a provider outage.
    cause = exc.__cause__
    if isinstance(cause, httpx.HTTPStatusError):
        code = cause.response.status_code
        if code == 429:
            return count_429
        return 500 <= code <= 599
    return isinstance(cause, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError))


@dataclass
class _EndpointState:
    state: CircuitState = CircuitState.CLOSED
    failures: int = 0
    opened_at: float = 0.0
    cycle: int = 0
    probe_in_flight: bool = False
    probe_deadline: float = 0.0


class InMemoryCircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig, *, now: Callable[[], float] = time.monotonic) -> None:
        self._config = config
        self.reset_timeout_seconds = config.reset_timeout_seconds
        self._now = now
        self._lock = threading.Lock()
        self._states: dict[str, _EndpointState] = {}

    def acquire(self, endpoint: str) -> int | None:
        with self._lock:
            state = self._states.setdefault(endpoint, _EndpointState())
            now = self._now()
            if state.state is CircuitState.CLOSED:
                return state.cycle
            if state.state is CircuitState.OPEN:
                if now - state.opened_at < self._config.reset_timeout_seconds:
                    return None
                state.state = CircuitState.HALF_OPEN
                state.probe_in_flight = True
                state.probe_deadline = now + self._config.half_open_deadline_seconds
                return state.cycle
            # HALF_OPEN: admit exactly one probe. A probe that never reported past its deadline is
            # assumed dead and a fresh probe is admitted, so a lost record cannot wedge the endpoint.
            if state.probe_in_flight and now < state.probe_deadline:
                return None
            # Replacing an abandoned probe: bump the cycle so its stale token can never govern this one.
            if state.probe_in_flight:
                state.cycle += 1
            state.probe_in_flight = True
            state.probe_deadline = now + self._config.half_open_deadline_seconds
            return state.cycle

    def record(self, endpoint: str, token: int, outcome: CallOutcome) -> None:
        with self._lock:
            state = self._states.get(endpoint)
            if state is None or token != state.cycle:
                return  # a stale token from a superseded cycle never governs the current one
            now = self._now()
            if outcome is CallOutcome.SUCCESS:
                self._states[endpoint] = _EndpointState(cycle=state.cycle)
                return
            if state.state is CircuitState.HALF_OPEN:
                # a probe that did not cleanly succeed re-opens with a fresh cooldown and a new cycle
                state.state = CircuitState.OPEN
                state.opened_at = now
                state.failures = 0
                state.probe_in_flight = False
                state.cycle += 1
                return
            if state.state is CircuitState.OPEN:
                return
            if outcome is CallOutcome.INCONCLUSIVE:
                return
            state.failures += 1
            if state.failures >= self._config.failure_threshold:
                state.state = CircuitState.OPEN
                state.opened_at = now
                state.failures = 0
                state.cycle += 1

    def state_of(self, endpoint: str) -> CircuitState:
        with self._lock:
            state = self._states.get(endpoint)
            return state.state if state is not None else CircuitState.CLOSED


def _classify(exc: DependencyUnavailableError, count_429: bool) -> CallOutcome:
    return CallOutcome.FAILURE if _contributes(exc, count_429) else CallOutcome.INCONCLUSIVE


class _BreakerBase:
    # Optional provider methods to bracket through the breaker, set on the subclass (e.g. tool-call egress).
    _GUARDED_UNARY: frozenset[str] = frozenset()
    _GUARDED_STREAMING: frozenset[str] = frozenset()

    def __init__(
        self,
        inner: Any,
        breaker: InMemoryCircuitBreaker,
        endpoint: str,
        *,
        count_429: bool,
        retry_after_seconds: float,
    ) -> None:
        self._inner = inner
        self._breaker = breaker
        self._endpoint = endpoint
        self._count_429 = count_429
        self._retry_after_seconds = retry_after_seconds
        # A real instance attribute (only when the inner has one) so the static CacheIdentified check
        # sees it and caching stays installed over the breaker; __getattr__ forwarding is invisible to it.
        identity = getattr(inner, "cache_identity", None)
        if identity is not None:
            self.cache_identity = identity

    def __getattr__(self, name: str) -> Any:
        # Forward optional inner members, guarding the tool-call ones; a missing member raises so hasattr
        # still mirrors the inner exactly.
        if name.startswith("_"):
            raise AttributeError(name)
        attr = getattr(self._inner, name)
        if name in self._GUARDED_UNARY:
            return self._guard_unary(attr)
        if name in self._GUARDED_STREAMING:
            return self._guard_streaming(attr)
        return attr

    def _acquire(self) -> int:
        token = self._breaker.acquire(self._endpoint)
        if token is None:
            raise ProviderCircuitOpenError(retry_after_seconds=self._retry_after_seconds)
        return token

    def _guard_unary(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        def call(*args: Any, **kwargs: Any) -> Any:
            token = self._acquire()
            outcome = CallOutcome.INCONCLUSIVE
            try:
                result = fn(*args, **kwargs)
                outcome = CallOutcome.SUCCESS
                return result
            except DependencyUnavailableError as exc:
                outcome = _classify(exc, self._count_429)
                raise
            finally:
                self._breaker.record(self._endpoint, token, outcome)

        return call

    def _guard_streaming(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        def call(*args: Any, **kwargs: Any) -> AbcGenerator[Any, None, Any]:
            token = self._acquire()
            return self._guarded_stream(fn(*args, **kwargs), token)

        return call

    def _guarded_stream(self, inner_stream: Any, token: int) -> AbcGenerator[Any, None, Any]:
        outcome = CallOutcome.INCONCLUSIVE
        try:
            result = yield from inner_stream
            outcome = CallOutcome.SUCCESS
            return result
        except DependencyUnavailableError as exc:
            outcome = _classify(exc, self._count_429)
            raise
        finally:
            self._breaker.record(self._endpoint, token, outcome)


class BreakerGenerator(_BreakerBase):
    _GUARDED_UNARY = frozenset({"propose_tool_call"})
    _GUARDED_STREAMING = frozenset({"stream_tool_call"})

    @property
    def model_id(self) -> str:
        return cast(str, self._inner.model_id)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        token = self._acquire()
        outcome = CallOutcome.INCONCLUSIVE
        try:
            result: GenerationResult = self._inner.generate(request)
            outcome = CallOutcome.SUCCESS
            return result
        except DependencyUnavailableError as exc:
            outcome = _classify(exc, self._count_429)
            raise
        finally:
            self._breaker.record(self._endpoint, token, outcome)

    def stream(self, request: GenerationRequest) -> AbcGenerator[str, None, dict[str, Any]]:
        token = self._acquire()
        return self._guarded_stream(self._inner.stream(request), token)


class BreakerEmbedder(_BreakerBase):
    @property
    def model_id(self) -> str:
        return cast(str, self._inner.model_id)

    @property
    def dimensions(self) -> int:
        return cast(int, self._inner.dimensions)

    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        token = self._acquire()
        outcome = CallOutcome.INCONCLUSIVE
        try:
            vectors: list[tuple[float, ...]] = self._inner.embed(texts)
            outcome = CallOutcome.SUCCESS
            return vectors
        except DependencyUnavailableError as exc:
            outcome = _classify(exc, self._count_429)
            raise
        finally:
            self._breaker.record(self._endpoint, token, outcome)


def build_circuit_breaker(config: CircuitBreakerConfig) -> InMemoryCircuitBreaker:
    return InMemoryCircuitBreaker(config)


def wrap_generator_breaker(generator: Generator, breaker: InMemoryCircuitBreaker, *, count_429: bool) -> Generator:
    base_url = getattr(generator, "base_url", None)
    if base_url is None:
        return generator
    return BreakerGenerator(
        generator,
        breaker,
        provider_endpoint(base_url),
        count_429=count_429,
        retry_after_seconds=breaker.reset_timeout_seconds,
    )


def wrap_embedder_breaker(embedder: Embedder, breaker: InMemoryCircuitBreaker, *, count_429: bool) -> Embedder:
    base_url = getattr(embedder, "base_url", None)
    if base_url is None:
        return embedder
    return BreakerEmbedder(
        embedder,
        breaker,
        provider_endpoint(base_url),
        count_429=count_429,
        retry_after_seconds=breaker.reset_timeout_seconds,
    )


__all__ = [
    "BreakerEmbedder",
    "BreakerGenerator",
    "InMemoryCircuitBreaker",
    "build_circuit_breaker",
    "provider_endpoint",
    "wrap_embedder_breaker",
    "wrap_generator_breaker",
]
