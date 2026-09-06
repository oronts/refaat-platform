# Output governance

Model output is untrusted. The output-governance seam reviews a produced answer plus its
resolved citations after generation and before the response is returned or cached, and can
refuse or redact it. It is an opt-in governance layer. With no governor installed the query
and RLM pipelines are byte-identical to the default, on the non-streaming, streaming, and
RLM paths.

## Where it runs

A governor is a pure, model-independent function. It runs at the single seam where the
grounded answer and its citations are already resolved: on the query path this is
`_finalize`, after the citations are built and immediately before the response is persisted
or cached; on the RLM path it is after the final result is assembled and before the session
is recorded. Because the citations are resolved first, the governor inspects the final
grounded output rather than a raw model span.

The governor receives an `OutputContext` (the answer, its citations, the request principal,
the workspace, and the corpus) and returns an `OutputDecision`:

- `allowed=True` with no `redacted_answer`: serve the answer unchanged.
- `allowed=True` with a `redacted_answer`: serve the redacted text, adding an
  `output_denied` route marker.
- `allowed=False`: return a governed no-output response (a fixed message, no citations, an
  `output_denied` route). A denial is never cached and never becomes a session turn, so a
  governor or ACL change is re-evaluated on the next query.

The `OutputDecision.reason` is a bounded machine code, never raw offending text, so it is
safe to record in the audit trail.

## The reference governor

`infrastructure.output_governance.PatternDenylistGovernor` is the shipped reference
governor. It refuses or redacts a produced answer that contains a configured literal or
regular expression:

```python
from refaat.application.runtime import RuntimeBuilder
from refaat.infrastructure.output_governance import PatternDenylistGovernor

governor = PatternDenylistGovernor(["Project Bluejay"], mode="refuse")
runtime = RuntimeBuilder().with_output_governor(governor).build()
```

A plain string is escaped to a literal match. A caller who needs a real regular expression
passes a compiled `re.Pattern`. In `redact` mode each match is replaced with a placeholder
and the answer is served. It can also be enabled from configuration:
`REFAAT_OUTPUT_DENYLIST_PATTERNS` (a non-empty list auto-installs the governor) with
`REFAAT_OUTPUT_GOVERNANCE_MODE` set to `refuse` or `redact`. An empty list installs nothing.

### What the reference governor does not catch

The pattern denylist catches only a configured literal or regex appearing in the final
answer text. It is honestly not a semantic output filter. It does not detect:

- reconstructed or paraphrased PII,
- prompt injection,
- an unconfigured secret the operator did not list,
- unsafe generated URL, HTML, or SQL.

The promise is a bounded blast radius on a known string, not injection prevention. Compose
it with a real DLP or classification governor, implemented against the `OutputGovernor`
protocol, for those threats. Patterns are trusted operator configuration, never client
input, so an expensive regex is an operator concern and not a client-reachable ReDoS vector.

## Streaming is buffered on a governed flow

Because a governor may withhold or rewrite the whole answer, a streamed governed query
cannot emit tokens as they arrive: an un-recallable token could leak content the governor
would refuse. When a governor is installed the streaming path buffers. It consumes the whole
generation without yielding, runs the governor in `_finalize`, then emits exactly one delta
with the governed answer or the refusal, followed by the final event. With no governor
installed the streaming path stays incremental and byte-identical.

Agentic intermediate streaming is a separate surface. A governed ReAct program's
intermediate fragments (its `tool_call_delta` and `tool_step` events) are not reviewed by
this seam and are caller-facing; the program's final answer is governed because it routes
through the query pipeline. Gating the intermediate agentic fragments is a future option,
not part of this seam.

## Evaluated at write time only

The governor runs when an answer is produced. A cache hit returns before `_finalize`, so the
governor does not re-run on a hit. The governor identity (`cache_id`) is folded into the
answer-cache key only when a governor is installed, so an answer produced under one governor
(or none) is never served after the governor is installed, removed, or changed. Because of
this, a governor must be a pure function of the answer plus whatever it folds into its
`cache_id`; a governance decision that depends on the citations is not re-evaluated on a
cache hit.

## Byte-identical when unconfigured

The governor is `None` by default. When no governor is installed the review is skipped, no
route is added, the answer-cache key omits the governor term, and no artifact id or program
definition changes. Output governance composes with the authorization and model-routing
seams but is independent of them: authorization filters which evidence a principal may see,
model routing gates whether evidence reaches the model, and output governance reviews what
the model produced.
