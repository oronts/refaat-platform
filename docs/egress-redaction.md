# Generation-evidence egress redaction

A generation model provider is a data-egress boundary. The context-transform seam rewrites the
evidence a generation model is about to see, replacing operator-configured spans with a placeholder
before the request reaches the provider. It is an opt-in, defense-in-depth control. With no transform
installed the query and RLM pipelines are byte-identical to the default.

This control covers the generation model's evidence input only. Read the "What this does not cover"
section below before relying on it: it is not a complete data-loss-prevention boundary.

## Where it runs

The redaction runs at a single shared seam, `domain.context_transform.redact_for_egress`, applied on
every path that sends corpus evidence to a real generator, after the model-routing gate and
immediately before the generate or stream call:

- The query path (`QueryService.query`).
- Both streaming branches (`QueryService.stream_query`, plain and buffered-governed).
- The RLM node egress (`RecursiveContextService`), on every recursive generation.
- The corrective-RAG relevance judge (`ModelConfidenceEvaluator`).

A transform receives a `GenerationRequest` and returns a `TransformResult`: the rewritten request,
the per-pattern redaction counts, and the number of images dropped. The seam adds one fail-safe of
its own: under an active transform, image egress is refused. A text redactor cannot inspect pixels,
so any image on the request is dropped and counted rather than sent to a vision model.

With no transform installed the seam returns the same request object untouched, so the path is
byte-identical.

## The reference redactor

`infrastructure.context_transform.PatternRedactor` is the shipped reference transform. It replaces
each configured literal or regular expression appearing in the redacted fields with a placeholder:

```python
from refaat.application.runtime import RuntimeBuilder
from refaat.infrastructure.context_transform import PatternRedactor

runtime = (
    RuntimeBuilder(settings)
    .with_context_transform(PatternRedactor(["ACME-INTERNAL-TOKEN"], placeholder="[REDACTED]"))
    .build()
)
```

A plain string is escaped to a literal, so redacting `a.b` does not match `axb`. A caller who wants a
real regular expression passes a compiled `re.Pattern`. Patterns are trusted operator configuration,
never client input.

The default redacted fields are the retrieved chunk bodies (`evidence`), the graph facts
(`graph_facts`), and the session summary (`session_summary`). The user query is not redacted by
default because it is the user's own input, not corpus evidence; an operator can add `"query"` to the
field set to redact it too.

When several patterns match overlapping spans, resolution is deterministic: the leftmost span wins,
and at an equal start the earlier-listed pattern wins; a later overlapping span is discarded. Counts
are over the surviving spans. Idempotency holds for patterns that do not themselves match the
placeholder.

### Configuration

The reference redactor auto-installs from settings when patterns are configured. An empty list
installs nothing, so the path stays byte-identical:

```bash
REFAAT_EGRESS_REDACTION_PATTERNS='["ACME-INTERNAL-TOKEN", "PROJECT-BLUEJAY"]'
REFAAT_EGRESS_REDACTION_PLACEHOLDER='[REDACTED]'
REFAAT_EGRESS_REDACTION_FIELDS='["evidence", "graph_facts", "session_summary"]'
```

An injected transform (`with_context_transform`) takes precedence over the settings-driven redactor.

## Citations and grounding

Redaction changes only the text the generation model reads. The chunk id, metadata, and embedding
stay intact, so a citation still resolves to the original stored chunk. The authorized local reader
who receives the answer still sees the real evidence in the citation quote. Redaction is an egress
control against the model provider, not a control that hides evidence from the user.

The in-process extractive fallback is never redacted. When a model answer's citations do not resolve,
the pipeline falls back to the local extractive generator against the original request, so the
fallback answer stays grounded in and cites the real evidence. The fallback runs in-process and is
not a provider egress.

## Provenance

Redaction provenance is recorded in the response usage under `egress_redaction`. It carries only the
opaque pattern-id digests and per-pattern counts, and the dropped-image count. It never carries a raw
pattern string or a matched span, so provenance can be logged without leaking the secret it protects.

## The answer cache

An installed transform folds its `cache_id` into the answer-cache identity, so an answer produced
under one transform, or under none, is never served after the transform is installed, removed, or
changed. With no transform installed the cache identity is unchanged.

## What this does not cover

This control redacts the generation model's evidence input only. It is defense-in-depth against the
generation provider, not a complete data-loss-prevention boundary. The following are separate
provider egresses that this control does not touch. If your threat model includes them, route them
through a trusted, self-hosted, or zero-retention provider, and treat the source data accordingly.

- Embeddings. The embedding provider receives the raw chunk text at ingestion and the raw query text
  at query time. Neither is redacted by this seam.
- Fine-tuning. Training data sent to a fine-tuning provider is a separate egress.
- Ingestion-time model calls. HyPE hypothetical-question generation, RAPTOR summarization, and the
  vision captioner send corpus-derived text or pixels to a model at ingestion time.
- Evaluation-time judging. The `GeneratorJudge` sends an answer and its evidence to a judge model.

The reference redactor also catches only a configured span. It does not detect reconstructed or
paraphrased sensitive content, an unconfigured secret, or prompt injection. Compose it with a real
classification or DLP control for those.

## Coverage guard

A test enumerates every generation-port call site in the source tree and asserts a pinned
classification, so a new generation-model evidence egress cannot be added without being classified
and, when it sends corpus evidence, wired to the redaction seam. The guard is honestly scoped to the
generation port; the embedding and fine-tuning egresses reach their providers through a different
mechanism and are out of the guard by design, in line with the coverage limits above.
