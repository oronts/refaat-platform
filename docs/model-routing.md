# Data-classification model routing

The model-routing seam decides whether the retrieved evidence may be sent to a
model, before any generation happens. It is an opt-in governance layer. With no
router installed the query pipeline is byte-identical to the default, on both the
non-streaming and streaming paths.

## How the classification is derived

The data classification is derived deterministically from the retrieved evidence,
never by a model. For each retrieved hit the runtime reads the chunk ACL tag via
`domain.authz.access_tag_from_metadata` and takes its `Classification`. A chunk
that carries no ACL defaults to `PUBLIC`, and an empty context is `PUBLIC`. The
classification passed to the router is the maximum across every piece of evidence
that reaches the model, not only the retrieved hits: the graph facts and the
session summary are also carried into the prompt, so the router counts the
maximum classification of the documents backing the graph facts and of the
session summary as well. The most sensitive piece of evidence sets the level.
This derivation runs only when a router or residency gate is installed to consume
it, and it counts a graph fact's backing document whether or not per-principal
authorization is active, so a router or residency gate installed without an access
policy still sees graph evidence.

The router receives that classification plus the request principal and returns an
allow or deny verdict. The model never chooses the classification and never
decides the verdict.

## The ceiling router

`domain.routing.ClassificationCeilingRouter` is the shipped reference router. It
allows generation when the evidence classification is at or below a fixed ceiling
and denies otherwise:

```python
from refaat.application.runtime import RuntimeBuilder
from refaat.domain.authz import Classification
from refaat.domain.routing import ClassificationCeilingRouter

runtime = RuntimeBuilder().with_model_router(ClassificationCeilingRouter(ceiling=Classification.INTERNAL)).build()
```

With an `INTERNAL` ceiling, `PUBLIC` and `INTERNAL` evidence is answered normally
while `CONFIDENTIAL` and `RESTRICTED` evidence is refused. A router may implement
its own contract through the `ModelRouter` protocol; a verdict may also carry a
`model_id` to steer an admitted classification to a specific model.

## The refusal

When the router denies the classification, the pipeline returns a governed
no-generation response: a fixed refusal message, no citations, and a
distinguishable `routing_denied` route. The refusal is produced before the
generation call, so over-classified evidence is never sent to the model. The check
is applied on `query()`, `stream_query()`, and the RLM `run()` path, so no request
shape can bypass routing. The detailed router reason is never surfaced to the
caller; a generic reason is returned so a prober cannot learn the evidence
sensitivity from the refusal.

## The residency router

`domain.routing.ResidencyRouter` gates the active generation endpoint on
operator-declared data residency. It maps the derived evidence classification to a
company `ResidencyPolicy` rule, then admits generation only when the provider's
declared region is in the rule's allowed regions, its data-retention term is at or
below the rule maximum, and (when the rule gates purpose) its purpose is permitted.
Retention terms are ordered `zero`, `transient`, `limited`, `retained`, from least
to most retained, so a provider is admitted only when its retention is at or below
the rule maximum.

You can install it from configuration. When `residency_policy` is set, the runtime
builds a `ResidencyRouter` from the per-classification rules plus the active
provider tags, with no code change:

```bash
REFAAT_RESIDENCY_POLICY='{"confidential": {"allowed_regions": ["eu"], "max_retention": "transient"}}'
REFAAT_RESIDENCY_PROVIDER_REGION=eu
REFAAT_RESIDENCY_PROVIDER_RETENTION=zero
```

Region and purpose admission is exact-match on opaque operator strings. A case or
whitespace mismatch (for example `EU` against `eu`) does not match, so it silently
refuses, which is fail-closed by design. A policy with no rule for a classification
refuses that classification, so enabling a residency policy without a `PUBLIC` rule
refuses untagged and `PUBLIC` evidence by design. An untagged provider, meaning an
empty region or an unset retention, fails closed.

## Residency zones versus classification

Classification is a sensitivity level; a residency zone is a placement origin. Two
documents can share a classification yet originate in different jurisdictions, so a
rule can additionally gate the residency zone of the evidence. Set `allowed_zones`
on a rule and tag each document's ACL with a `residency_zone`:

```bash
REFAAT_RESIDENCY_POLICY='{"confidential": {"allowed_regions": ["eu"], "max_retention": "zero", "allowed_zones": ["eu"]}}'
```

```json
{"acl": {"classification": "confidential", "residency_zone": "eu"}}
```

When a rule sets `allowed_zones`, generation is admitted only when every declared
evidence zone is in the set. Evidence tagged with a zone outside the set, evidence
with no zone, and evidence with a malformed ACL or zone are all denied, so the gate
is fail-closed for unzoned and broken evidence. A rule that leaves `allowed_zones`
unset does not gate zones and is byte-identical to before. The zone rides the
existing ACL metadata, so there is no new column and no ingest parameter; a zone
change re-tags and re-versions the document exactly like any other ACL change. The
caller-facing refusal is the same generic message, so the zone value never leaks.

This gates the model egress in the runtime. Keeping data physically at rest in a
given region (a separate database, vector index, and object store per region) is a
deployment concern under the single-composition-root architecture, running one
instance per region, not a runtime feature.

## Composing routers

`domain.routing.CompositeModelRouter` runs several routers together. It denies on
the first member that denies and allows only when every member allows, so a
classification ceiling and a residency policy can gate the same request. When both
an injected router and a `residency_policy` are configured, the runtime composes
them automatically into a `CompositeModelRouter`. The composite cache identity folds
its members, so a change to any member reroutes.

## The RLM path is gated too

The model router now gates the RLM generation path, not only `query()` and
`stream_query()`. The RLM egress unit is every piece of evidence that reaches the
model: the authorized corpus chunks, the graph facts, and the session summary. The
router is consulted once on the maximum classification across all three before any
sub-generation, so a graph fact or a summary whose backing evidence exceeds the
ceiling refuses the run just as an over-classified chunk would. A denial returns the
same governed refusal shape and records no session turn.

Both the query and RLM paths recompute the session summary's classification and
residency from the current ACLs of the documents it folded, so a document
re-classified after compaction is reflected at read time. A cited document that no
longer resolves, for example one erased after the summary was written, is treated as
unresolved provenance: it fails closed to the maximum classification and a zone no
operator set admits, so a summary that still carries its content can never lower the
ceiling.

## Deploying a tightened ceiling

The answer cache is keyed on the corpus version and the installed router identity,
not on the classification derivation itself, so an answer generated and cached
under a looser ceiling is still served from cache after the ceiling is tightened by
a code change alone, until it expires or the corpus version advances. When you
deploy a change that makes the ceiling stricter, flush the answer cache or advance
the corpus version so previously admitted answers are regenerated under the new
ceiling rather than served from a pre-change cache entry.

## Honest boundary

The residency router trusts operator-declared tags. It does not verify a provider's
actual hosting region, its real retention, or its purpose handling, and it is not a
prompt-injection defense. It is a governance gate with a bounded blast radius: it
prevents the runtime from sending over-classified or wrong-region evidence to the
configured generation endpoint, given honest tags. Verifying the provider's real
behavior, and any contractual residency guarantee, remain a deployment
responsibility.

## Byte-identical when unconfigured

The router is `None` by default, and `residency_policy` is unset by default. When no
router is installed and no residency policy is configured, the routing check is
skipped entirely, no route is added, and no cache identity or artifact id changes.
Routing is composable with the authorization enforcement seam but independent of
it: routing gates generation by classification and residency, while authorization
filters which evidence a principal may see at all.
