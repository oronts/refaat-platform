"""The per-provider circuit breaker opens on genuine faults, fast-fails while open, and probes once."""

from __future__ import annotations

import math
from typing import Any

import httpx
import pytest

from refaat.domain.resilience import CallOutcome, CircuitBreakerConfig, CircuitState
from refaat.errors import DependencyUnavailableError, ProviderCircuitOpenError
from refaat.infrastructure.model_cache import cache_identity_of
from refaat.infrastructure.resilience import (
    BreakerEmbedder,
    BreakerGenerator,
    InMemoryCircuitBreaker,
    _contributes,
    provider_endpoint,
)
from refaat.ports import CacheIdentified

_EP = "https://api.provider.test/v1"


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _breaker(
    clock: _Clock, *, threshold: int = 3, reset: float = 30.0, half_open: float = 60.0
) -> InMemoryCircuitBreaker:
    config = CircuitBreakerConfig(
        failure_threshold=threshold, reset_timeout_seconds=reset, half_open_deadline_seconds=half_open
    )
    return InMemoryCircuitBreaker(config, now=clock)


def _fail(breaker: InMemoryCircuitBreaker, endpoint: str = _EP, outcome: CallOutcome = CallOutcome.FAILURE) -> None:
    token = breaker.acquire(endpoint)
    assert token is not None
    breaker.record(endpoint, token, outcome)


def test_opens_after_threshold_then_fast_fails() -> None:
    clock = _Clock()
    breaker = _breaker(clock, threshold=3)
    for _ in range(3):
        _fail(breaker)
    assert breaker.state_of(_EP) is CircuitState.OPEN
    assert breaker.acquire(_EP) is None


def test_inconclusive_never_trips() -> None:
    clock = _Clock()
    breaker = _breaker(clock, threshold=3)
    for _ in range(10):
        _fail(breaker, outcome=CallOutcome.INCONCLUSIVE)
    assert breaker.state_of(_EP) is CircuitState.CLOSED
    assert breaker.acquire(_EP) is not None


def test_success_resets_the_streak() -> None:
    clock = _Clock()
    breaker = _breaker(clock, threshold=3)
    _fail(breaker)
    _fail(breaker)
    _fail(breaker, outcome=CallOutcome.SUCCESS)
    _fail(breaker)
    _fail(breaker)
    assert breaker.state_of(_EP) is CircuitState.CLOSED


def test_half_open_admits_one_probe_after_cooldown() -> None:
    clock = _Clock()
    breaker = _breaker(clock, threshold=1, reset=30.0)
    _fail(breaker)
    assert breaker.acquire(_EP) is None  # still cooling down
    clock.t = 31.0
    token = breaker.acquire(_EP)
    assert token is not None
    assert breaker.state_of(_EP) is CircuitState.HALF_OPEN
    assert breaker.acquire(_EP) is None  # a probe is already out


def test_probe_success_closes() -> None:
    clock = _Clock()
    breaker = _breaker(clock, threshold=1, reset=30.0)
    _fail(breaker)
    clock.t = 31.0
    token = breaker.acquire(_EP)
    assert token is not None
    breaker.record(_EP, token, CallOutcome.SUCCESS)
    assert breaker.state_of(_EP) is CircuitState.CLOSED
    assert breaker.acquire(_EP) is not None


def test_probe_failure_reopens_with_fresh_cooldown() -> None:
    clock = _Clock()
    breaker = _breaker(clock, threshold=1, reset=30.0)
    _fail(breaker)
    clock.t = 31.0
    token = breaker.acquire(_EP)
    assert token is not None
    breaker.record(_EP, token, CallOutcome.FAILURE)
    assert breaker.state_of(_EP) is CircuitState.OPEN
    assert breaker.acquire(_EP) is None  # a fresh cooldown started at t=31


def test_probe_inconclusive_reopens() -> None:
    clock = _Clock()
    breaker = _breaker(clock, threshold=1, reset=30.0)
    _fail(breaker)
    clock.t = 31.0
    token = breaker.acquire(_EP)
    assert token is not None
    breaker.record(_EP, token, CallOutcome.INCONCLUSIVE)
    assert breaker.state_of(_EP) is CircuitState.OPEN


def test_lost_probe_self_heals_past_the_deadline() -> None:
    clock = _Clock()
    breaker = _breaker(clock, threshold=1, reset=30.0, half_open=60.0)
    _fail(breaker)
    clock.t = 31.0
    first = breaker.acquire(_EP)
    assert first is not None  # a probe admitted but never recorded (a lost worker)
    assert breaker.acquire(_EP) is None  # within the probe deadline, no second probe
    clock.t = 31.0 + 61.0
    assert breaker.acquire(_EP) is not None  # past the deadline, a fresh probe is admitted


def test_stale_token_never_governs_a_newer_cycle() -> None:
    clock = _Clock()
    breaker = _breaker(clock, threshold=1)
    stale = breaker.acquire(_EP)  # CLOSED, cycle 0
    assert stale is not None
    _fail(breaker)  # opens, cycle -> 1
    assert breaker.state_of(_EP) is CircuitState.OPEN
    breaker.record(_EP, stale, CallOutcome.SUCCESS)  # a stale success from cycle 0
    assert breaker.state_of(_EP) is CircuitState.OPEN  # ignored, the open circuit stands


def test_endpoints_are_isolated() -> None:
    clock = _Clock()
    breaker = _breaker(clock, threshold=1)
    _fail(breaker, endpoint="https://a.test/v1")
    assert breaker.state_of("https://a.test/v1") is CircuitState.OPEN
    assert breaker.acquire("https://b.test/v1") is not None


def test_provider_endpoint_keying() -> None:
    assert provider_endpoint("https://api.openai.com/v1") == provider_endpoint("https://api.openai.com:443/v1")
    assert provider_endpoint("https://gw.test/openai") != provider_endpoint("https://gw.test/anthropic")
    assert provider_endpoint("https://API.Test/v1/") == provider_endpoint("https://api.test/v1")


def _reraise(cause: BaseException) -> DependencyUnavailableError:
    try:
        raise DependencyUnavailableError("provider failed") from cause
    except DependencyUnavailableError as exc:
        return exc


def _status_error(code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", _EP)
    response = httpx.Response(code, request=request)
    return httpx.HTTPStatusError("status", request=request, response=response)


@pytest.mark.parametrize(
    ("cause", "count_429", "expected"),
    [
        (_status_error(503), False, True),
        (_status_error(500), False, True),
        (_status_error(400), False, False),
        (_status_error(401), False, False),
        (_status_error(429), False, False),
        (_status_error(429), True, True),
        (httpx.ConnectError("down"), False, True),
        (httpx.ReadTimeout("slow"), False, True),
        (httpx.RemoteProtocolError("bad"), False, True),
        (httpx.LocalProtocolError("client bug"), False, False),
        (ValueError("parse"), False, False),
    ],
)
def test_contributes_classifies_causes(cause: BaseException, count_429: bool, expected: bool) -> None:
    assert _contributes(_reraise(cause), count_429) is expected


def test_no_cause_error_does_not_contribute() -> None:
    assert _contributes(DependencyUnavailableError("no cause"), False) is False


class _FakeGenerator:
    base_url = _EP
    model_id = "fake-model"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, request: Any) -> dict[str, Any]:
        self.calls += 1
        raise _reraise(httpx.ConnectError("down"))


def test_wrapper_opens_then_fast_fails_without_touching_the_provider() -> None:
    clock = _Clock()
    breaker = _breaker(clock, threshold=2)
    inner = _FakeGenerator()
    wrapped = BreakerGenerator(inner, breaker, _EP, count_429=False, retry_after_seconds=30.0)
    for _ in range(2):
        with pytest.raises(DependencyUnavailableError):
            wrapped.generate(object())
    assert inner.calls == 2
    # Now open: the next call fast-fails and never reaches the provider.
    with pytest.raises(ProviderCircuitOpenError):
        wrapped.generate(object())
    assert inner.calls == 2


class _StreamingGenerator:
    base_url = _EP
    model_id = "fake-model"

    def stream(self, request: Any) -> Any:
        def _gen() -> Any:
            yield "hello "
            yield "world"
            return {"prompt_tokens": 3, "completion_tokens": 5}

        return _gen()


def test_streamed_usage_return_value_is_forwarded() -> None:
    clock = _Clock()
    breaker = _breaker(clock)
    wrapped = BreakerGenerator(_StreamingGenerator(), breaker, _EP, count_429=False, retry_after_seconds=30.0)
    stream = wrapped.stream(object())
    chunks = []
    try:
        while True:
            chunks.append(next(stream))
    except StopIteration as stop:
        usage = stop.value
    assert chunks == ["hello ", "world"]
    assert usage == {"prompt_tokens": 3, "completion_tokens": 5}
    assert breaker.state_of(_EP) is CircuitState.CLOSED


class _FakeEmbedder:
    base_url = _EP
    model_id = "fake-embed"
    dimensions = 3

    def embed(self, texts: Any) -> list[tuple[float, ...]]:
        raise _reraise(httpx.ConnectError("down"))


def test_embedder_wrapper_opens_and_fast_fails() -> None:
    clock = _Clock()
    breaker = _breaker(clock, threshold=1)
    wrapped = BreakerEmbedder(_FakeEmbedder(), breaker, _EP, count_429=False, retry_after_seconds=30.0)
    with pytest.raises(DependencyUnavailableError):
        wrapped.embed(["x"])
    with pytest.raises(ProviderCircuitOpenError):
        wrapped.embed(["x"])


class _CacheIdentifiedGenerator:
    base_url = _EP
    model_id = "fake-model"

    @property
    def cache_identity(self) -> str:
        return "prov:abc"

    def generate(self, request: Any) -> dict[str, Any]:
        return {"answer": "ok"}


def test_wrapper_is_transparent_to_cache_identity() -> None:
    # The breaker sits innermost, before caching; wrapping must not disable model-call caching, so the
    # wrapper must satisfy the runtime_checkable CacheIdentified check and expose the inner identity.
    clock = _Clock()
    breaker = _breaker(clock)
    wrapped = BreakerGenerator(_CacheIdentifiedGenerator(), breaker, _EP, count_429=False, retry_after_seconds=30.0)
    assert isinstance(wrapped, CacheIdentified)
    assert cache_identity_of(wrapped) == "prov:abc"


def test_wrapper_over_a_non_identified_inner_stays_uncached() -> None:
    clock = _Clock()
    breaker = _breaker(clock)
    wrapped = BreakerGenerator(_FakeGenerator(), breaker, _EP, count_429=False, retry_after_seconds=30.0)
    assert not isinstance(wrapped, CacheIdentified)
    assert cache_identity_of(wrapped) is None


def test_expired_probe_token_cannot_govern_the_replacement_cycle() -> None:
    # An abandoned half-open probe must not close (or reopen) the circuit once a
    # replacement probe has been admitted under a fresh cycle.
    clock = _Clock()
    breaker = _breaker(clock, threshold=1, reset=30.0, half_open=60.0)
    _fail(breaker)  # OPEN, cycle -> 1
    clock.t = 31.0
    probe_a = breaker.acquire(_EP)  # HALF_OPEN, token = cycle 1
    assert probe_a is not None
    clock.t = 92.0  # past probe A's deadline (31 + 60 = 91)
    probe_b = breaker.acquire(_EP)  # replacement admitted under a bumped cycle
    assert probe_b is not None
    assert probe_b != probe_a
    breaker.record(_EP, probe_a, CallOutcome.SUCCESS)  # the abandoned probe reports late
    assert breaker.state_of(_EP) is CircuitState.HALF_OPEN  # ignored, the replacement still governs
    breaker.record(_EP, probe_b, CallOutcome.SUCCESS)  # only the live probe can close
    assert breaker.state_of(_EP) is CircuitState.CLOSED


class _ToolCallGenerator:
    base_url = _EP
    model_id = "fake-model"

    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self._fail = fail

    def propose_tool_call(self, query: str, tools: Any, used: Any) -> Any:
        self.calls += 1
        if self._fail:
            raise _reraise(httpx.ConnectError("down"))
        return {"tool": "search"}

    def stream_tool_call(self, query: str, tools: Any, used: Any) -> Any:
        self.calls += 1
        if self._fail:
            raise _reraise(httpx.ConnectError("down"))
        yield {"delta": 1}


def test_tool_call_planning_fast_fails_on_an_open_circuit() -> None:
    clock = _Clock()
    breaker = _breaker(clock, threshold=1)
    inner = _ToolCallGenerator()
    wrapped = BreakerGenerator(inner, breaker, _EP, count_429=False, retry_after_seconds=30.0)
    _fail(breaker)  # open the endpoint by another route
    with pytest.raises(ProviderCircuitOpenError):
        wrapped.propose_tool_call("q", (), frozenset())
    with pytest.raises(ProviderCircuitOpenError):
        # stream_tool_call fast-fails on the acquire, before the inner stream is constructed
        wrapped.stream_tool_call("q", (), frozenset())
    assert inner.calls == 0


def test_tool_call_failures_contribute_to_opening() -> None:
    clock = _Clock()
    breaker = _breaker(clock, threshold=1)
    wrapped = BreakerGenerator(_ToolCallGenerator(fail=True), breaker, _EP, count_429=False, retry_after_seconds=30.0)
    with pytest.raises(DependencyUnavailableError):
        wrapped.propose_tool_call("q", (), frozenset())
    assert breaker.state_of(_EP) is CircuitState.OPEN
    with pytest.raises(ProviderCircuitOpenError):
        wrapped.propose_tool_call("q", (), frozenset())


def test_tool_call_capability_is_not_faked_for_a_non_supporting_inner() -> None:
    clock = _Clock()
    breaker = _breaker(clock)
    wrapped = BreakerGenerator(_FakeGenerator(), breaker, _EP, count_429=False, retry_after_seconds=30.0)
    assert not hasattr(wrapped, "propose_tool_call")
    assert not hasattr(wrapped, "stream_tool_call")


@pytest.mark.parametrize(
    "config",
    [
        {"failure_threshold": 0, "reset_timeout_seconds": 1.0, "half_open_deadline_seconds": 1.0},
        {"failure_threshold": 1, "reset_timeout_seconds": 0.0, "half_open_deadline_seconds": 1.0},
        {"failure_threshold": 1, "reset_timeout_seconds": 1.0, "half_open_deadline_seconds": -1.0},
        {"failure_threshold": 1, "reset_timeout_seconds": math.nan, "half_open_deadline_seconds": 1.0},
        {"failure_threshold": 1, "reset_timeout_seconds": 1.0, "half_open_deadline_seconds": math.inf},
    ],
)
def test_invalid_config_is_rejected_at_construction(config: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="must be"):
        CircuitBreakerConfig(**config)
