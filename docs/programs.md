# Typed programs and artifacts

Refaat Programs is a native, custom layer (no third-party framework dependency) for declaring AI behavior as typed, versioned, content-addressed Python programs. A program adds a stable typed contract, tunable parameters, validators, and an immutable artifact on top of the runtime. Execution reuses the runtime, so RAG, GraphRAG, CAG, version-safe caching, and citation validation all apply unchanged. There is no second retrieval, generation, or cache implementation.

## Signatures

A signature declares typed inputs and outputs. Fields carry Refaat-aware metadata such as `grounded`, `required`, `sensitive`, and `tunable`, and the class exposes a stable content hash used to key artifacts and caches.

```python
from refaat.programs import InputField, OutputField, Signature


class GroundedAnswer(Signature):
    question: str = InputField(description="The user question")

    answer: str = OutputField(description="Answer grounded in evidence", grounded=True)
    citations: list[str] = OutputField(description="Canonical citations", required=True)
```

## The built-in grounded-rag program

Every project registers a built-in `grounded-rag` program, so you can run a program without writing one:

```python
from refaat import Refaat

with Refaat() as runtime:
    project = runtime.project(workspace="oronts", corpus="engineering")
    project.ingest("./docs")

    result = project.programs.get("grounded-rag").run(question="How does rollback work?")

    print(result.answer)
    print(result.grounded, len(result.citations))
    print(result.artifact_id)
```

The result is a typed `ProgramResult` with the answer, canonical citations, routes, cache status, corpus version, trace id, validator outcomes, usage, and the content-addressed artifact id.

## Custom programs

Subclass a program to give it a name, an input signature, and tunables. The immutable grounding policy and citation contract are never tunable; only the declared tunables can later be optimized.

```python
from refaat.programs import GroundedRagProgram, InputField, Signature, Tunables


class IncidentQuestion(Signature):
    question: str = InputField(description="An incident-response question")


class IncidentRunbook(GroundedRagProgram):
    name = "incident-runbook"
    input_signature = IncidentQuestion


assistant = project.programs.register(IncidentRunbook(tunables=Tunables(top_k=6, minimum_citation_coverage=0.5)))
result = assistant.run(question="Who handles a Sev1 incident?")
```

## Artifacts

A program produces an immutable, content-addressed artifact. Artifacts are canonical JSON only: no pickle, credentials, or executable payload. Identical definitions resolve to the same `artifact_id`, and any change to a signature or tunable produces a new id.

```python
artifact = project.programs.get("grounded-rag").artifact
# {"schema_version": "1", "program": "grounded-rag", "backend": "native",
#  "definition_hash": "...", "immutable_policy_hash": "...", "signatures": {...},
#  "tunables": {...}, "validators": ["grounding"], "dependency_versions": {"refaat": "0.1.0"}}
```

## Validators

Validators are deterministic checks run after execution. The built-in `GroundingValidator` reads `minimum_citation_coverage` as a real fraction: when it is above zero, at least that share of the answer's sentences must carry a resolved citation marker (so `0.5` and `0.9` are different bars, not the same on/off toggle). At the default floor of `0.0` any single resolved citation satisfies it and a deliberately empty answer is trivially grounded, so a program that leaves the floor at zero is unchanged. `ProgramResult.grounded` is true only when every validator passes.

## Command line

The built-in program is available from the CLI against any workspace and corpus:

```bash
refaat program list <workspace_id> <corpus_id>
refaat program run <workspace_id> <corpus_id> grounded-rag "How does rollback work?"
```

## HTTP

The same operations are available over HTTP and execute the same application service:

- `GET /v1/corpora/{corpus_id}/programs` lists programs.
- `POST /v1/corpora/{corpus_id}/programs/{name}/run` runs a program and returns the typed result with citations.
- `POST /v1/corpora/{corpus_id}/programs/{name}/evaluate` evaluates a program over a dataset.
- `POST /v1/corpora/{corpus_id}/programs/{name}/promote` and `POST /v1/programs/{name}/rollback` manage deployments; promotion requires the admin scope.
- `GET /v1/programs/{name}/deployments/{environment}` resolves the deployed artifact.

The full request and response shapes are in the committed OpenAPI document.

## Tracing

When OpenTelemetry is enabled, the program layer emits spans `refaat.program.run`, `refaat.program.evaluate`, and `refaat.program.optimize`, each tagged with the program name and, for evaluation, the case count. With telemetry disabled the spans are a zero-cost no-op, so the program layer runs offline and in CI without change.

## Bounded inference policy

`BoundedInferencePolicy` wraps any program in a bounded multi-attempt loop without touching retrieval, generation, caching, or citations. In `best_of` mode it runs a fixed number of attempts and selects the best result; in `refine` mode it re-runs until a grounded result appears, up to a hard attempt cap, then selects the best of what it collected.

```python
from refaat.programs import BoundedInferencePolicy, GroundedRagProgram

policy = BoundedInferencePolicy(GroundedRagProgram(), mode="best_of", attempts=3)
result = policy.execute(project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None)
```

Selection is a pure function of `ProgramResult` fields: `default_result_score` ranks grounded results first, then by citation count, then by a bounded length tiebreak, and `select_best` keeps the earliest attempt on a tie. The mode and attempt cap are folded into the content-addressed artifact, so best-of-3 and best-of-5 are distinct, reproducible artifacts. On the deterministic offline stack repeated attempts are identical, so the selection is verified over crafted result sets; the sampling benefit appears once the inner program calls a sampling model.

## KNN few-shot selection

`KNNFewShotProgram` selects few-shot demonstrations per query instead of pinning a fixed set. Given an inner program and a labeled `pool` of `Demonstration` values, it embeds the query and each pool question, keeps the `k` nearest by cosine similarity in a deterministic order, sanitizes the selected demonstrations, and runs a clone of the inner program with those demonstrations folded into its tunables. The embedder defaults to the runtime embedder, so the selector stays provider-neutral and needs no extra configuration offline.

```python
from refaat.domain.models import Demonstration
from refaat.programs import GroundedRagProgram, KNNFewShotProgram

pool = [
    Demonstration(question="how does rollback work", answer="Rollback restores the previously approved snapshot."),
    Demonstration(question="who approves a release", answer="The release owner approves it."),
]
program = KNNFewShotProgram(GroundedRagProgram(), pool, k=2)
result = program.execute(project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None)
```

Each selected demonstration passes through the shipped sanitizer, so a real corpus chunk id planted in a pool demonstration is rewritten to the reserved example sentinel and can never be echoed back as a resolved citation. The inner program is never mutated, because `with_tunables` returns a clone. The program name is `f"{inner.name}+knn"`, and the pool digest, `k`, and the inner artifact id are pinned into the content-addressed artifact, so distinct pools or `k` values are distinct, reproducible artifacts, and each query's selection folds into the answer-cache identity through the guidance digest. A non-empty pool and a `k` of at least one are required; otherwise construction raises `ValidationError`.

## Governed ReAct

`GovernedReActProgram` runs a bounded think-act-observe loop in which every tool call is authorized before it runs. A deterministic `KeywordToolPlanner` proposes the next tool, a pure `ToolGovernor` admits it only when the caller holds the tool's required scopes and the tool's side-effect class (`read_only`, `write`, `external`) does not exceed the policy ceiling, and only then does the runtime execute the tool's `skill.run(context, arguments)`. No host code and no generated code runs. The loop is hard-capped by `max_iterations`, and the final answer is the runtime's grounded query, so retrieval, generation, caching, and citations are unchanged. The governed steps, authorized and denied, are recorded in the result outputs, and the full governance policy is folded into the content-addressed artifact.

```python
from refaat.programs import (
    GovernedReActProgram,
    KeywordToolPlanner,
    ReActPolicy,
    SideEffectClass,
    ToolGrant,
)

policy = ReActPolicy(
    max_iterations=4,
    granted_scopes=("read:corpus",),
    side_effect_ceiling=SideEffectClass.READ_ONLY,
    grants=(ToolGrant("lookup", required_scopes=("read:corpus",), side_effect=SideEffectClass.READ_ONLY),),
)
planner = KeywordToolPlanner({"lookup": ("rollback",)})
program = GovernedReActProgram(policy=policy, planner=planner, skills={"lookup": lookup_skill})
result = program.execute(project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None)
```

The planner is model-free, so a governed run is reproducible offline; a model-backed planner can drop in behind the same interface without changing the governance or the artifact contract.

## Program as a governed tool

`ProgramSkill` wraps a promoted `BoundProgram` so a `GovernedReActProgram` can call an entire program as one governed tool. Its `run` re-enters the wrapped program's full pipeline with the retrieved query and returns the sub-program's grounded answer, citations, and pinned inner artifact id as the observation, reusing `ProgramResult.to_json` so there is one serialization body. The inner artifact id is read from the artifact property, never indexed out of `to_json`, because the artifact id is the digest of `to_json` and is not one of its keys.

`GovernedReActProgram.from_registry` builds the skills map by resolving each policy grant from a `SkillRegistry`, with no change to the governed loop body: a wrapped program is looked up by grant name exactly like a plain skill. When a grant resolves to a `ProgramSkill`, its inner program's content-addressed artifact id is folded into the governed artifact, so a program-planned-as-tool trajectory is a distinct, honest artifact and changing the wrapped program forces re-approval. A plain or plugin skill instead contributes its class identity (module and qualname) when the policy declares capabilities, so a capability-bearing governed program binds each approval to the skill class identity and swapping the class forces re-approval; a governed program with no capabilities folds no skill identity and stays byte-identical. The binding is to a class identity, not to code, version, or instance configuration, so those are registration-trust residuals (see the approvals residual note).

```python
from refaat.domain.enums import SideEffectClass
from refaat.domain.models import ReActPolicy, ToolGrant
from refaat.programs import GovernedReActProgram, KeywordToolPlanner
from refaat.skills import ProgramSkill, SkillRegistry

registry = SkillRegistry()
registry.register(ProgramSkill(project.programs.get("grounded-rag"), name="incident_playbook"))

policy = ReActPolicy(
    max_iterations=4,
    granted_scopes=("read",),
    side_effect_ceiling=SideEffectClass.READ_ONLY,
    grants=(ToolGrant("incident_playbook", required_scopes=("read",), side_effect=SideEffectClass.READ_ONLY),),
)
planner = KeywordToolPlanner({"incident_playbook": ("incident",)})
react = GovernedReActProgram.from_registry(policy=policy, planner=planner, registry=registry)
result = react.execute(
    project, inputs={"question": "how does an incident rollback work"}, user_id=None, session_id=None
)
```

## Trajectory evaluation

`trajectory_bundle` scores a governed agent run end to end through the unchanged `ProgramEvaluator` and `MetricBundle`, since the trajectory metrics read `react_steps`, `usage`, and grounding through the same `Metric` protocol every other metric uses. It composes the shipped `Grounded` and `CitationRecall` metrics with four trajectory metrics:

- `ToolSafety` is a hard structural gate: a denied governed step must carry no observation. In a real trace a denied step is never executed, so this holds by construction; it is a regression tripwire, not a live discriminator.
- `ToolAuthorization` is plan coverage: the fraction of steps handled correctly, where a correctly denied over-privileged tool counts toward coverage rather than against it. A trajectory that correctly refuses an over-privileged tool is never ranked below one where every planned tool happened to be authorized; the score drops only on a governance breach.
- `StepBudget` and `TokenCost` share one monotonic `_budget_score` body: a value at the budget scores 1.0 and a value past it decays. The step and token budgets are bundle construction parameters, not evaluation-case fields.

Grounding and tool safety are hard constraints, so a trajectory that fails to ground or leaks an observation on a denied step never passes, independent of the weighted objective score.

```python
from refaat.evaluation import Dataset, ProgramEvaluator, trajectory_bundle

dataset = Dataset.from_records("incident-eval", [{"id": "e1", "query": "how does an incident rollback work"}])
report = ProgramEvaluator(project.programs.register(react)).evaluate(dataset, trajectory_bundle())
```

## Program ensemble

`EnsembleProgram` composes several member programs and combines their results with a pure, deterministic policy: `best` reuses `select_best` (grounded first, then the result score), and `majority` picks the result whose citation set is most common, with ties broken by `select_best`. Members are never mutated, the member artifact ids and the policy are pinned into the ensemble's content-addressed artifact, and the ensemble usage sums each member's context tokens so evaluation reflects the ensemble's cost rather than a single member's.

```python
from refaat.programs import EnsembleProgram, GroundedRagProgram, Tunables

ensemble = EnsembleProgram(
    [GroundedRagProgram(tunables=Tunables(top_k=1)), GroundedRagProgram(tunables=Tunables(top_k=3))],
    policy="best",
)
result = ensemble.execute(project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None)
```

## Multi-agent orchestration and handoffs

`MultiAgentOrchestrator` routes a task to specialist member programs and merges their results, built on the same selection and cost machinery as the ensemble. A `TaskRouter` maps specialist names to keyword tuples and returns every specialist whose keyword substring-matches the lowercased task, in sorted-key order, falling back to the full sorted specialist set when nothing matches, so a merge is always possible and the order never depends on caller dict insertion order. Each selected member answers through its own `project.ask` path, so retrieval, caching, and the citation contract are unchanged and no answer-cache identity shifts, and members are never mutated.

Merging goes through an injected `Reducer` or, when none is given, the shipped `select_best` (grounded first, then the result score, earliest index winning ties). A reducer carries a stable `name` and must return one of the input results or a result whose citations are drawn only from the member results; the orchestrator rejects any merged result that widens the citation set beyond the union of the members. Member homogeneity is enforced at construction: all members must share both input and output signatures by content hash, and every router name must be a member.

The member artifact ids, the router keywords, and a stable reducer identity are pinned into the content-addressed artifact, so two orchestrators over the same members with distinct routers or distinct named reducers are distinct, reproducible artifacts, and the orchestrator id is distinct from every member and from an ensemble over the same members. The optional handoff mailbox records typed messages over the shipped `AgentService` with a deterministic correlation id; it is a runtime record only and is deliberately excluded from the artifact identity, so the orchestrator id is byte-identical whether handoff is on or off. Handoff defaults to off, so the deterministic offline path never needs the agent store. The orchestrator usage sums each member's context tokens, and it composes under `BoundedInferencePolicy` and `ProgramEvaluator` as an ordinary program.

```python
from refaat.programs import GroundedRagProgram, MultiAgentOrchestrator, TaskRouter, Tunables

orchestrator = MultiAgentOrchestrator(
    {
        "ops": GroundedRagProgram(tunables=Tunables(top_k=1)),
        "billing": GroundedRagProgram(tunables=Tunables(top_k=3)),
    },
    TaskRouter({"ops": ("rollback", "deploy"), "billing": ("invoice", "refund")}),
)
result = orchestrator.execute(project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None)
```

## Structured and typed output

`StructuredOutputProgram` enforces a program's typed output `Signature` as a schema. Each output field's annotation is compiled through one explicit annotation-to-JSON-type table (`str`, `int`, `float`, `bool`, `list[str]`) into a closed JSON-Schema object, and the produced or parsed output is checked with the same pure-Python `validate_object` body that validates tool arguments, so there is exactly one validation path and no jsonschema dependency on the runtime path. An unsupported annotation fails loud at compile time.

A `StructuredValidator` projects the runtime response into the structured object the program layer already exposes (the grounded answer plus its citation chunk ids) and checks it against the compiled schema. It only adds structural typing: grounding and the citation contract are unchanged, so a structured output that carries a citations field still resolves to real corpus chunk ids. On a schema violation the result is ungrounded and the shipped `refine` policy retries up to a hard attempt cap; the shipped retry loop and selection are reused verbatim.

```python
from refaat.programs import GroundedRagProgram, StructuredOutputProgram

program = StructuredOutputProgram(GroundedRagProgram(), max_attempts=3)
result = program.execute(project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None)
```

The schema folds into two identities, each conditional on a schema being in force, so schema-free programs keep byte-identical ids: the output-schema digest, the attempt cap, and the inner artifact id are pinned into the content-addressed artifact, and the schema digest folds into the answer-cache identity. On the offline extractive stack the deterministic output is validated so the machinery is genuinely exercised. A schema-capable provider additionally receives a schema-constrained generation request (`response_format` with the compiled JSON schema); the extractive generator ignores the schema on purpose (honesty boundary), because it cannot constrain its own decode, so structural typing offline is enforced by the validator rather than the decoder.

## Per-query routing

`RouterProgram` chooses a retrieval strategy per query from deterministic query features, then answers through an inner grounded program. Feature extraction is model-free: the number of query tokens, the clause count from the shipped `split_clauses`, and a surface entity proxy. A `RouterConfig` holds the thresholds (long-query token count, multi-clause threshold, entity threshold, and optional per-route `top_k` overrides), and a fixed routing table maps the features onto a `RouteStrategy`. A complex query (multi-clause, long, or entity-rich) routes to RRF fusion with sub-query decomposition on and autocut off; a simple query routes to linear fusion with decomposition off and autocut on.

Routing only re-shapes retrieval. Grounding, generation, caching, and the citation contract stay the runtime's, because the router forwards the full inner tunables into `project.ask` and layers only the routed subset on top, so a citation still resolves to a real corpus chunk. Because routing is offline-deterministic there is no honesty boundary: the same query and config always pick the same strategy.

```python
from refaat.programs import RouterProgram

program = RouterProgram()
result = program.execute(project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None)
```

The router pins its inner artifact id and the `RouterConfig` content id into the content-addressed artifact, merged on top of the base pinning, so two routers with distinct thresholds are distinct, reproducible artifacts and a schema-enforced router still carries its schema pin. The routed retrieval overrides already enter the answer-cache identity through the resolved retrieval policy (fusion, top_k, autocut, query expansion); a stable `router_id` folds in as a defensive extra, omitted when absent, so every non-routed cache scope stays byte-identical. The router thresholds default from `REFAAT_ROUTER_LONG_QUERY_TOKENS`, `REFAAT_ROUTER_MULTI_CLAUSE_THRESHOLD`, and `REFAAT_ROUTER_ENTITY_THRESHOLD` when a program carries no explicit `RouterConfig`.

## Corrective RAG

`CorrectiveRagProgram` gates retrieval quality and broadens only when a narrow retrieval looks weak. It first retrieves with a single query variant and no expansion, then a `ConfidenceEvaluator` labels the context `correct` or `low`. On `correct` the answer comes from a narrow `project.ask`; on `low` the program re-asks with query expansion on, which activates the runtime-configured `QueryTransform` (step-back offline) and RRF-merges the extra variant by chunk id. Either branch answers through the runtime's unchanged grounded query, so retrieval breadth is the only lever and generation, caching, and the citation contract are the runtime's. The gate decision (label, score, corrective_applied, rationale) is recorded in `outputs.corrective_rag` but never reaches the generator and is never citeable.

The default `DeterministicConfidenceEvaluator` is model-free and offline-deterministic. It reads only the non-normalized per-hit `semantic_score`, never the fused score (RRF and linear fusion normalize the fused score so the top hit saturates to 1.0 regardless of true similarity). It labels a context `correct` only when it has hits, its top semantic score meets `top_semantic_threshold`, and at least `min_grounded_chunks` clear `semantic_floor`. `ModelConfidenceEvaluator` wraps a generator that judges sufficiency and may flip only the label; the measured score is preserved, and a missing dependency degrades to the deterministic label rather than fabricating a verdict.

```python
from refaat.programs import CorrectiveRagProgram, DeterministicConfidenceEvaluator

program = CorrectiveRagProgram(
    evaluator=DeterministicConfidenceEvaluator(top_semantic_threshold=0.35, min_grounded_chunks=1),
    corrective_max_query_variants=4,
)
result = program.execute(project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None)
```

No new answer-cache identity is introduced: the only answer-changing state is retrieval breadth (`query_expansion`, `max_query_variants`) and the configured `query_transform.cache_id`, both already folded into the cache scope. The evaluator `cache_id`, the thresholds, and the corrective breadth are folded into the content-addressed artifact, so two corrective programs with different gates are distinct, reproducible artifacts, and being a new program name it disturbs no existing artifact id.

## Self-RAG

`SelfRagProgram` maps the three Self-RAG reflections onto shipped machinery rather than reimplementing them. A `NecessityGate` decides whether the query needs retrieval (the Retrieve reflection); the answer always flows through the grounded `project.ask` pipeline, and when the gate says retrieve the program raises retrieval breadth, so no ungrounded, uncited answer is ever returned. IsSupported is the shipped `GroundingValidator` over the result. When the answer is not supported the retry is the shipped `BoundedInferencePolicy` refine (retry until grounded, hard-capped), not a hand-rolled loop. IsUseful is a deterministic heuristic: a grounded answer whose trimmed length reaches `usefulness_min_answer_chars`. The three reflections (retrieve, is_supported, is_useful, rationale, refine_attempts) are recorded in `outputs.self_rag` but never reach the generator and are never citeable.

The default `DeterministicNecessityGate` is model-free and offline-deterministic: it deems retrieval necessary when the query carries at least `min_content_terms` content terms, so a content-rich query retrieves and a trivial one skips, a live discriminator rather than a constant. `ModelNecessityGate` wraps a generator that judges necessity for the query alone and degrades to the deterministic decision on a missing dependency, so nothing is fabricated offline.

```python
from refaat.programs import DeterministicNecessityGate, SelfRagProgram

program = SelfRagProgram(
    necessity_gate=DeterministicNecessityGate(min_content_terms=2),
    refine_attempt_cap=3,
    usefulness_min_answer_chars=1,
)
result = program.execute(project, inputs={"question": "how does rollback work"}, user_id=None, session_id=None)
```

No new answer-cache identity is introduced: the only answer-changing state is retrieval breadth and the refine re-run, both already folded into the shipped cache scope. The necessity gate `cache_id`, the refine attempt cap, the usefulness threshold, and the retrieve breadth are folded into the content-addressed artifact, so two Self-RAG programs with different reflection policies are distinct, reproducible artifacts, and being a new program name it disturbs no existing artifact id.

## Where this is going

Programs are the first layer of a native programming and optimization foundation. Versioned datasets and metrics, native retrieval and graph optimization, the immutable artifact registry, and promotion with rollback all ship today and are documented on their own pages. The full plan and its phased delivery are described in the repository review and in [Roadmap and non-goals](roadmap.md).
