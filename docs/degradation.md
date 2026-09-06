# Degradation on provider outage

A generation provider can fail with a hard outage. By default that outage propagates and the
request refuses with a 503. The degradation seam makes that behavior a first-class, governed
choice: an installed policy decides whether a served answer may be degraded to a grounded, cited
fallback, abstain, or refuse. It is opt-in. With no policy installed the query and RLM answer
paths are byte-identical to the default, on the non-streaming, streaming, and RLM paths.

## Scope

This policy governs a generation PROVIDER OUTAGE at two request seams:

- the query answer seam (the generation call in the query and streaming paths),
- the RLM answer seam (the per-node generation call).

It does not touch anything else. In particular, the following are out of scope and unchanged:

- the query pipeline's citation-miss extractive fallback (when the model answered but its
  citations did not resolve). That is a grounding miss, not a provider outage, and it stays
  exactly as before.
- the internal degrade-to-deterministic helpers (the reranker, the judge, HyPE, and the
  query-transform catalog all degrade to their deterministic default on a provider failure). Those
  are internal quality lifts, not the served answer seam, and they are unchanged.

## The three modes

An installed policy inspects a described `DependencyFailure` and returns a `DegradationDecision`
with one of three modes:

- `SERVE_DEGRADED`: serve a grounded, cited fallback answer marked with `degraded=True`. This
  requires that grounding survives, meaning retrieved evidence is present so the extractive
  fallback can produce a cited answer. The served answer still goes through the same citation and
  output-governance pipeline as a normal answer.
- `ABSTAIN`: return an honest no-answer with empty citations.
- `REFUSE`: let the outage surface as an error (the 503), which is the legacy behavior.

`DependencyFailure.grounding_survives` is the safety input: a policy must never serve a degraded
answer when grounding does not survive.

## The reference policies

Two reference policies ship, selected by configuration or injected directly:

- `RefuseAllDegradationPolicy` (config `refuse_all`): every outage refuses. This is the
  fail-closed posture, identical in effect to the default except that it is an explicit choice.
- `SafeDefaultsDegradationPolicy` (config `safe_defaults`): serve a grounded, cited fallback when
  one survives, otherwise abstain. It never serves an ungrounded answer.

```python
from refaat.application.runtime import RuntimeBuilder
from refaat.domain.degradation import SafeDefaultsDegradationPolicy

runtime = RuntimeBuilder().with_degradation_policy(SafeDefaultsDegradationPolicy()).build()
```

From configuration, set `REFAAT_DEGRADATION_POLICY` to `none` (default), `refuse_all`, or
`safe_defaults`. An injected policy wins over the config selector.

## Transient outages never poison durable state

A provider outage is transient, so a degraded or abstained response is never written to the
answer cache and never becomes a session turn. On the query path the response is built under a
shadow scope that suppresses those writes. On the RLM path an abstained run returns before the
session is recorded. This means a later identical request, once the provider recovers, is served a
fresh full answer rather than a cached degraded one.

The degradation policy identity is folded into the answer-cache key only when a policy is
installed, so an answer produced under one policy (or none) is never served after the policy is
installed, removed, or changed. With no policy the cache key omits the term and is byte-identical.

## Streaming

On the streaming path an outage is degradable only before the first token has left. When a policy
is installed and the provider fails before any delta, the seam resolves the decision and emits the
degraded final event, the abstain final event, or, for a refuse, the terminal error event exactly
as today. An outage after the first token has already been emitted cannot be retracted, so it
propagates as before. With no policy installed the streaming path re-raises on an outage, matching
today's terminal error behavior.

## Audit

When a policy is installed and a degradation decision is actually made, one redacted
`degradation.decided` audit event is recorded through the installed audit sink. It carries the
dependency class, the mode, a machine reason code, and the bounded provider id, alongside the
standard redacted envelope of ids, counts, and timestamps. It never carries a query or evidence
text. Emission is best-effort: a degradation is not protection-decreasing, so a sink outage does
not also crash the served path. With no policy installed no event is emitted.

## Byte-identical when unconfigured

The policy is `None` by default. When no policy is installed the outage propagates uncaught, the
answer-cache key omits the degradation term, no route or field changes on a normal answer, and no
artifact id or program definition changes. The `degraded` field on the query and RLM responses
defaults to `False`, so serialization is unchanged when a response is not degraded.
