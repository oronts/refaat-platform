# Streaming responses

A grounded answer can be streamed as it is produced instead of waiting for the whole response. Streaming is available on the generator port, the HTTP API, and the SDK. It reuses the same retrieval, caching, and citation pipeline as a non-streamed query, so a streamed answer and a non-streamed answer share the answer cache.

## What streams and what is authoritative

Two kinds of server-sent event are emitted:

- `delta` events carry incremental answer text as it is produced.
- one terminal `final` event carries the authoritative result: the full grounded answer, validated citations, routes, the cache flag, the corpus version, the trace id, and usage.

The deltas are a live preview. The `final` event is authoritative: it applies the same citation validation and extractive fallback as a non-streamed query, so the answer and citations in the `final` event are always grounded. A client that needs strict grounding renders the deltas for responsiveness and reconciles with the `final` event. When the streamed preview was not grounded and the authoritative answer is the extractive fallback, the `final` event sets `replaced` to true, so a client replaces the previewed text with the grounded answer; in the common case the preview equals the final answer and `replaced` is false. Streamed responses report streaming usage, so `usage.provider` is `stream` and token-level provider counts are not available mid-stream. On a cache hit the cached answer is emitted as a single delta followed by the `final` event.

## HTTP

```
POST /v1/corpora/{corpus_id}/query/stream
Content-Type: application/json
Accept: text/event-stream

{"query": "How does rollback work?"}
```

The response is `text/event-stream`. Each event carries an `event:` type and a `data:` JSON payload:

```
event: delta
data: {"type": "delta", "text": "Rollback restores the prior artifact"}

event: final
data: {"type": "final", "answer": "...", "citations": [...], "routes": [...], "cached": false, "replaced": false, "corpus_version": 3, "trace_id": "...", "usage": {...}}
```

The endpoint uses the read scope, the same as the non-streaming query. It is a POST because the query is a request body; a browser consumes it with `fetch` and a `ReadableStream` rather than `EventSource`.

## SDK

```python
for event in client.stream_query(corpus_id, "How does rollback work?"):
    if event["type"] == "delta":
        print(event["text"], end="")
    else:
        citations = event["citations"]
```

## Streaming tool calls

A governed ReAct program streams its trajectory as it runs, then delegates the final grounded answer to the same query pipeline above. The `final` event and its citations are authoritative and equal to running the program without streaming. The joined `delta` preview equals that answer on the deterministic offline extractive path, where the preview is never replaced; on a model generation path the preview is a live approximation reconciled by the authoritative `final` event.

```
POST /v1/corpora/{corpus_id}/programs/{name}/run/stream
```

Three event kinds are emitted, in order:

- `tool_call_delta` events carry provider-neutral fragments of a streamed tool call: a `tool` name once, then opaque `arguments_fragment` slices. A fragment is buffered and only parsed once the call is complete, so a partial argument is never validated, authorized, or executed.
- one `tool_step` event per governed iteration reports the tool, whether it was authorized, and the reason. An authorized tool runs exactly once, with the fully complete argument dict. The raw tool observation is not streamed to the caller: it never passed the output governor, so it stays out of every caller-facing egress path.
- `delta` events then the single terminal `final` event carry the grounded answer. The `final` event is authoritative and equals the non-streaming answer; the joined `delta` preview equals it on the offline extractive path and is reconciled by `final` when a model preview is replaced. The `final` event also carries `react_steps`, the recorded trajectory.

The deterministic keyword planner and the offline extractive generator are the default and are never streamed as a model: their trajectory emits completed `tool_step` events only, with no `tool_call_delta`. Only a real function-calling provider that supports streamed tool calls emits `tool_call_delta` events, when one is injected. A tool observation never joins the answer and never reaches the caller, so no fragment, partial argument, or observation is ever citable or egressed; the streamed `react_steps` records the tool decisions only.

If the stream fails after the first byte has been flushed, the response status is already 200, so the endpoint emits a terminal `error` event rather than truncating the stream silently.

```python
for event in client.stream_program(corpus_id, "governed-react", "How does rollback work?"):
    if event["type"] == "tool_call_delta":
        buffer += event["arguments_fragment"]
    elif event["type"] == "tool_step":
        record(event)
    elif event["type"] == "delta":
        print(event["text"], end="")
```

## Budgeting a stream

When a run token or cost budget is configured the whole stream is metered against one run meter. The meter is a plain object resolved once for the stream and re-bound around each synchronous unit of work, rather than held open across every yield. A streamed response is driven across the ASGI threadpool, which copies the context on every iteration, and a budget scope whose context variable set and reset straddled a yield would break across that boundary: the reset would raise and the meter set in one step would be invisible in the next. Re-binding per unit keeps the one meter visible for every model call in the stream and torn down once, with no cross-context error. With no budget configured and no outer meter, nothing is installed and the stream is byte-identical to a build without budgeting.

Streamed and non-streamed metering diverge on purpose, because the streaming generator port yields answer text only and surfaces no provider usage. A non-streamed generation charges the meter with the provider's real reported usage. A streamed generation has no such usage to read, so it charges a deterministic estimate instead: the prompt is estimated up front from the assembled request, so an early-closed stream still pays for the prompt it sent, then each produced completion piece is charged as an estimate. The same query can therefore meter a slightly different token count streamed versus non-streamed. Both are prompt plus completion; only the streamed side is an estimate. On the deterministic offline stack neither path is metered, so there is no divergence there.

## Notes

- The offline extractive generator streams sentence by sentence; an OpenAI-compatible generator streams provider content deltas parsed from the upstream `text/event-stream`.
- Streaming does not retry mid-stream. Transient-failure retry applies to the non-streaming generation path.
