from __future__ import annotations

import httpx
import pytest

from refaat.domain.models import ToolCallDelta, ToolSpec
from refaat.errors import DependencyUnavailableError
from refaat.infrastructure.generation import OpenAICompatibleGenerator
from refaat.programs.toolcall import _ToolCallAccumulator


def _sse(*lines: str) -> bytes:
    return "".join(f"{line}\n" for line in lines).encode()


def _generator(body: bytes) -> OpenAICompatibleGenerator:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    return OpenAICompatibleGenerator(
        base_url="http://provider",
        api_key="key",
        model="test-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _reduce_stream(generator: OpenAICompatibleGenerator, tools: list[ToolSpec]) -> object:
    accumulator = _ToolCallAccumulator()
    for delta in generator.stream_tool_call("q", tools, frozenset()):
        accumulator.feed(delta)
    return accumulator.result()


def test_stream_tool_call_maps_sse_chunks_to_neutral_deltas() -> None:
    body = _sse(
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"lookup"}}]}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"q\\":"}}]}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"x\\"}"}}]}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"name":"other","arguments":"drop"}}]}}]}',
        'data: {"choices":[{"finish_reason":"tool_calls"}]}',
        "data: [DONE]",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    generator = OpenAICompatibleGenerator(
        base_url="http://provider",
        api_key="key",
        model="test-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    tools = [ToolSpec(name="lookup", description="lookup a fact")]
    deltas = list(generator.stream_tool_call("find x", tools, frozenset()))

    assert deltas == [
        ToolCallDelta(tool="lookup", arguments_fragment=""),
        ToolCallDelta(tool=None, arguments_fragment='{"q":'),
        ToolCallDelta(tool=None, arguments_fragment='"x"}'),
        ToolCallDelta(tool=None, done=True),
    ]
    # Provider-neutrality: argument fragments stay raw, unparsed strings; a
    # parallel index != 0 call is dropped, never surfaced.
    fragments = [delta.arguments_fragment for delta in deltas]
    assert all(isinstance(fragment, str) for fragment in fragments)
    assert "".join(fragments) == '{"q":"x"}'


def test_stream_preserves_a_co_located_terminal_arguments_fragment() -> None:
    body = _sse(
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"lookup"}}]}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"topic\\":"}}]}}]}',
        'data: {"choices":[{"finish_reason":"tool_calls",'
        '"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"rollback\\"}"}}]}}]}',
        "data: [DONE]",
    )
    proposal = _reduce_stream(_generator(body), [ToolSpec(name="lookup", description="lookup a fact")])
    assert proposal is not None
    assert proposal.arguments == {"topic": "rollback"}  # type: ignore[attr-defined]


def test_stream_malformed_arguments_raises_like_the_non_streaming_path() -> None:
    body = _sse(
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"lookup"}}]}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"topic\\": "}}]}}]}',
        'data: {"choices":[{"finish_reason":"tool_calls"}]}',
        "data: [DONE]",
    )
    with pytest.raises(DependencyUnavailableError):
        _reduce_stream(_generator(body), [ToolSpec(name="lookup", description="lookup a fact")])


def test_stream_non_string_arguments_object_is_serialized_not_a_type_error() -> None:
    body = _sse(
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
        '"function":{"name":"lookup","arguments":{"topic":5}}}]}}]}',
        'data: {"choices":[{"finish_reason":"tool_calls"}]}',
        "data: [DONE]",
    )
    proposal = _reduce_stream(_generator(body), [ToolSpec(name="lookup", description="lookup a fact")])
    assert proposal is not None
    assert proposal.arguments == {"topic": 5}  # type: ignore[attr-defined]
