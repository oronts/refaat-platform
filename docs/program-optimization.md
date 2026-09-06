# Program optimization

Refaat Optimize is a native, custom optimizer for the tunable parameters of a program. The first strategy is retrieval-policy search: it is deterministic and model-free, so it never spends a model call on a weak configuration. It reuses [program evaluation](program-evaluation.md) to score each candidate, and it never mutates the active program. It returns ranked, immutable candidate artifacts and a recommendation for explicit promotion.

## Search space

A search space is a list of tunable configurations to evaluate. `retrieval_search_space` builds the cartesian product of the values you provide.

```python
from refaat.optimization import retrieval_search_space

space = retrieval_search_space(
    top_k=(4, 8, 12, 16),
    token_budget=(None, 1500),
    minimum_citation_coverage=(0.0,),
)
```

`default_search_space()` returns a small grid over `top_k`. `context_packing_search_space()` builds a grid over `token_budget` and `top_k` to find the cheapest context that preserves answer quality. `fusion_search_space(fusion=("rrf", "linear"), rrf_k=(10, 60), top_k=(8,))` grids over the fusion strategy and the RRF rank constant, so the optimizer can compare ranking regimes and pick the fusion that scores best at the lowest token cost.

## Running the optimizer

```python
result = project.programs.get("grounded-rag").optimize(dataset, search_space=space, budget=8)

print(result.best.tunables, result.best.score)
print(result.improvement)  # best score minus the first candidate's score
for candidate in result.candidates:  # ranked: compliant first, then score descending
    print(candidate.tunables, candidate.hard_constraints_passed, candidate.score, candidate.artifact_id)
```

Each candidate is evaluated with the same metric bundle production uses. Candidates that fail a hard constraint are ranked below compliant ones regardless of score. Every candidate also reports `mean_tokens`, the average context size across the evaluation cases, so token and cost reductions are measurable; among candidates with an equal score, the optimizer ranks the one with the lower token cost first. Every candidate carries a content-addressed artifact id, so a chosen configuration is reproducible and can be promoted deliberately. The optimizer accepts an optional `budget` to cap how many candidates it evaluates.

The result also exposes `pareto_front`: the compliant candidates that no other compliant candidate dominates on both answer score and token cost. It lets you choose a quality-versus-cost tradeoff deliberately rather than only taking the single best-scoring candidate.

## Command line

```bash
refaat program optimize <workspace_id> <corpus_id> grounded-rag cases.json
```

## Chunking on a shadow index

Chunking cannot be tuned as a program parameter, because changing the chunk size requires re-indexing. `ChunkingOptimizer` evaluates chunking configurations on an ephemeral shadow index: each candidate re-ingests the corpus documents into a throwaway runtime built with a different chunk size and overlap, evaluates the program there, and discards the runtime. The active corpus and its index are never modified, so the winning configuration is a recommendation to re-ingest deliberately.

```python
from refaat.optimization import ChunkingOptimizer

result = ChunkingOptimizer(project).optimize(dataset, chunk_sizes=(256, 420, 600), overlaps=(32, 64))
print(result.best.chunk_size, result.best.mean_tokens)
```

Every candidate reports the same score and `mean_tokens` signals as retrieval optimization, and the result exposes a `pareto_front`, so you can trade answer quality against token cost. Re-indexing embeds the new chunks, so this runs a real embedding pass per candidate and is not a hot-path operation.

## Graph extraction on a shadow index

`GraphOptimizer` mirrors `ChunkingOptimizer` for the co-occurrence graph: each candidate re-ingests the corpus documents into a throwaway runtime built with a different `max_entities` and `min_token_length`, evaluates the program there, records the resulting `entity_count` and `relation_count`, and discards the runtime. The active corpus and its graph are never modified.

```python
from refaat.optimization import GraphOptimizer

result = GraphOptimizer(project).optimize(dataset, max_entities=(40, 80, 120), min_token_lengths=(4, 6))
print(result.best.max_entities, result.best.entity_count)
```

Candidates rank by answer score and `mean_tokens` with a `pareto_front`, the same as retrieval and chunking optimization. Structural graph statistics are reported for observability; scoring against labeled gold graphs (entity and relation F1) needs a labeled dataset the offline corpus does not provide, so it is a follow-up.

## Reproducing an injected production runtime

The shadow index rebuilds a throwaway runtime, so it reproduces the injected production composition rather than silently falling back to the default stack. It re-applies the stateless read-only components a production runtime injected (embedder, generator, graph extractor, reranker, cross-encoder scorer, query transform, judge, HyPE generator, chunking strategy, stopwords, plugin pins), borrowing the client-owning model adapters so the shadow never closes a production connection. It refuses to reproduce anything stateful or governing: an injected search backend (reproducing it could write to a production cluster, and retrieval optimization already runs against the live index), the store-bound RAPTOR builder, and the governance stack (access policy, audit sink, model router, secrets, kill switches, quotas, egress, output governor, and injected connectors, extractors, skills, and event sinks). Each result carries a `shadow_composition` describing what was reproduced and what was omitted and why, present only when a component was injected so the default deterministic run is unchanged. Under an active access policy with strict authorization the optimizer fails closed, because it takes no principal; run it with strict authorization off to sweep over the shadow corpus. `GraphOptimizer` refuses to run when a custom graph extractor is injected, and `ChunkingOptimizer` refuses when a custom chunking strategy is injected, because each sweep tunes only its built-in mechanism.

## Metadata filtering and BM25 tuning

The offline lexical ranker is BM25, extracted into `refaat.core.sparse.bm25_scores`. Its `bm25_k1` and `bm25_b` parameters, and an exact-match metadata `filters` object, are harness-sourced retrieval-policy fields surfaced in `RetrievalPolicy.as_json()`, so they enter the answer-cache identity through `_context_hash` with no service change. Filtering runs before both semantic and lexical scoring: only chunks whose metadata satisfies every predicate are scored, and an empty result degrades to the standard insufficient-evidence answer. The default `bm25_k1=1.2` and `bm25_b=0.75` reproduce the previous fixed lexical scorer bit for bit.

## Labeled few-shot demonstration selection

`LabeledFewShotOptimizer` picks K labeled demonstrations from the dataset and pins them, plus optional free-text instructions, into the program's tunables. Selection is deterministic: only cases carrying a `reference_answer` are eligible, they are id-sorted, and each answer is passed through `sanitize_demonstration_answer`, which rewrites every `[chunk:...]` token to `[chunk:example]`. A real corpus chunk id can therefore never be carried from a label into a provider prompt and forged into a citation.

```python
from refaat.optimization import LabeledFewShotOptimizer

result = LabeledFewShotOptimizer(project).optimize(dataset, shots=(1, 2, 4), instructions="Answer in one sentence.")
print(result.best.tunables["demonstrations"])
```

Evaluation is held out: every case eligible to be a demonstration is excluded from scoring, so a demonstration is never silently reused as its own test. Each shot count yields a distinct content-addressed artifact id because the demonstrations enter `Tunables.as_json`. Guidance also folds into the answer-cache identity through `_context_hash`, so changing instructions or demonstrations, or reordering them, busts the cache.

The `demonstrations` and `instructions` tunables are emitted by `Tunables.as_json` only when non-empty, so an unguided program's artifact id is byte-identical to its pre-guidance value. On the offline deterministic stack the `ExtractiveGenerator` is evidence-only and does not itself consume guidance, so the shared build-and-rank helper folds a deterministic `render_guidance` token estimate into each candidate's `mean_tokens`. That estimate is the token count of the actual guidance block a guidance-consuming backend would prepend, and it is 0 for every guidance-free candidate, so the Pareto token axis is load-bearing for LabeledFewShot on any backend and separates higher-shot candidates by their real prompt cost.

## Instruction search (MIPRO-style and GEPA-style)

`InstructionOptimizer` searches over a program's shipped `instructions` tunable, the free-text guidance an evaluation-ready backend prepends to the prompt. It is deterministic and model-free, so it proposes and ranks candidate instructions with no model call, and it never mutates the active program.

`optimize_mipro` is the templated proposer. `propose_instructions` counts the dataset's salient terms, reusing `core.text.tokens` and the `core.text.QUERY_STOPWORDS` filter predicate, meaning a token qualifies when its length is at least three and it is not a query stopword, with ties broken alphabetically. It then enumerates a small cartesian set of task-framing and constraint-framing templates, each anchored to those salient terms, so distinct datasets yield distinct candidates. The first proposal is always `None`, the unguided baseline, and only the non-None members are passed through `sanitize_demonstration_answer` before becoming `Tunables.instructions`, so an instruction can never inject a `[chunk:...]` citation.

```python
from refaat.optimization import InstructionOptimizer

result = InstructionOptimizer(project).optimize_mipro(dataset, max_candidates=4)
guided = next(candidate for candidate in result.candidates if "instructions" in candidate.tunables)
print(guided.tunables["instructions"])
```

`optimize_gepa` is the reflective, evolutionary proposer. It evaluates the baseline on the held-out split, collects the cases that fail the grounding hard constraint, and `reflect_instruction` breeds a mutant by appending a targeted reminder derived from those failures. The loop chains each mutant over a growing window of failing cases, is idempotent so it stops at a fixed point, and is bounded by a fixed maximum round count, so the evolutionary search is finite and reproducible. When nothing fails the hard constraint, which is common on the offline evidence-only stack, it reflects on the full held-out set so the loop always has a signal.

```python
result = InstructionOptimizer(project).optimize_gepa(dataset, rounds=3)
```

Both entry points hold out an evaluation split the same way the few-shot optimizers do: the cases are id-sorted and split, so the evaluation set is a strict subset of the dataset and its version differs from the full set. The unguided variant is held at index 0, so it is always in the ranked set, and because instructions are omitted from `Tunables.as_json` when empty, its content-addressed artifact id is byte-identical to the program's pre-slice value. A non-empty instruction enters `Tunables.as_json`, so a guided candidate has a distinct artifact id, and the same instruction folds into the answer-cache identity through `_context_hash` and the shipped `guidance_digest`, so changing the instruction busts the cache.

Honesty boundary: on the offline deterministic stack the `ExtractiveGenerator` is evidence-only, so an instruction does not change the answer. A guided candidate therefore has the same objective score as the baseline, and the offline tests assert that the search machinery ran, meaning distinct reproducible candidates including the unguided baseline, a GEPA loop that breeds at least one ranked mutant, and the winning instruction folding into a distinct artifact id and the cache identity, not that the instruction improved quality. On a guidance-consuming backend the same instruction can lift the answer, and the ranking then reflects it.

## Bootstrapped few-shot demonstrations

`BootstrapFewShotOptimizer` mirrors `LabeledFewShotOptimizer` and differs only in how the demonstrations are produced. Instead of copying labeled reference answers, it runs the active program as a deterministic teacher over the labeled cases, keeps only the runs whose result passes a metric gate, sanitizes them, and hands the demonstration set to the same build-and-rank helper. The default gate is `Grounded` plus `AnswerContains`, so a teacher run becomes a demonstration only when it is grounded and its answer contains the case's expected terms. A case is eligible only when it is labeled, meaning it carries expected terms or a reference answer, because the gate is vacuous on an unlabeled case.

```python
from refaat.optimization import BootstrapFewShotOptimizer

result = BootstrapFewShotOptimizer(project).optimize(dataset, max_demos=4)
guided = next(candidate for candidate in result.candidates if "demonstrations" in candidate.tunables)
print(guided.tunables["demonstrations"])
```

The teacher runs through `BoundProgram.run`, the exact runtime path evaluation uses, so the demonstrations are produced by the same retrieval, generation, caching, and citation pipeline production uses. Every kept answer is passed through `sanitize_demonstration_answer`, so a real corpus chunk id in the teacher's own answer is rewritten to `[chunk:example]` and can never be carried into a later provider prompt to forge a citation. The bootstrapped cases are held out: a case that becomes a demonstration is excluded from the evaluation set, so a demonstration is never scored as its own test.

The baseline candidate is the zero-shot variant with empty demonstrations, held at index 0, so it is always in the ranked set. The `RANDOM_SEARCH` mode evaluates several deterministic random subsets of the bootstrapped pool instead of the single full set:

```python
from refaat.optimization import RANDOM_SEARCH, BootstrapFewShotOptimizer

result = BootstrapFewShotOptimizer(project).optimize(dataset, mode=RANDOM_SEARCH, trials=8, seed=0)
```

Honesty boundary: on the offline deterministic stack the `ExtractiveGenerator` is evidence-only, so a bootstrapped demonstration does not change the answer. A bootstrapped candidate therefore has the same objective score as the baseline but a strictly higher `mean_tokens`, so the zero-shot baseline ranks first and `result.best` is the baseline. Offline, the tests assert the machinery ran, meaning metric-filtered selection, sanitization, and a distinct bootstrapped artifact id in the ranked set, not that bootstrapping wins. On a guidance-consuming backend the same demonstrations can lift the answer, and the ranking then reflects it.

## Router threshold search

`RouterOptimizer` searches a `RouterProgram`'s threshold configurations. Each grid entry is a `RouterConfig` written into the `router` tunable of a program clone, so every candidate is an immutable, content-addressed artifact ranked by the shared `build_ranked_result` body, exactly like every other native optimizer. A plain grounded program is wrapped in a `RouterProgram` automatically, so the search always evaluates real routing.

```python
from refaat.core.router import RouterConfig
from refaat.optimization import RouterOptimizer

grid = [
    RouterConfig(long_query_tokens=16, multi_clause_threshold=2),
    RouterConfig(long_query_tokens=32, multi_clause_threshold=3),
]
result = RouterOptimizer(project).optimize(program, dataset, grid=grid)
best = next(candidate for candidate in result.candidates if "router" in candidate.tunables)
print(best.tunables["router"])
```

Each candidate carries a distinct `router` config, so the ranked set holds distinct artifact ids. The active program is never mutated: its `router` tunable stays unset, and the candidates are evaluation-only clones.

## Durable and resumable runs

A large search can be enqueued as a durable run instead of a synchronous call. `enqueue_optimization` persists the materialized search space and the dataset, then the shared durable worker evaluates each candidate and checkpoints it. If the worker is killed mid-run, the job lease expires, the worker re-claims it, and the run resumes by skipping the already-checkpointed candidates. The durable path deliberately evaluates the full materialized search space with no live run-budget early stop, because a token or cost run budget accrues provider spend that is known only after each model call and cannot be replayed deterministically across a resume. Because every candidate evaluation is otherwise deterministic and independent, a resumed run produces a result byte-identical to the synchronous `optimize()` whenever that synchronous run is not truncated by a live run budget, which is always the case offline and whenever no `run_token_budget` or `run_cost_budget_usd` ceiling is configured.

```python
from refaat.optimization import retrieval_search_space

run_id = project.programs.get("grounded-rag").enqueue_optimization(
    dataset,
    search_space=retrieval_search_space(top_k=(4, 8, 12)),
)
# a DurableWorker(runtime) processes the "optimization.run" job
run = runtime.store.get_optimization_run(project.workspace_id, run_id)
print(run["status"], run["best_artifact_id"])
```

The same run is available over the CLI (`refaat program optimize-job` and `refaat program optimization <run_id>`), the HTTP API (`POST /v1/programs/{name}/optimizations` and `GET /v1/optimizations/{run_id}`), and both SDK clients (`enqueue_optimization` and `get_optimization_run`). At completion the run registers the winning and Pareto-front artifacts as immutable registry entries. It never promotes them and never mutates the active program, so moving one into an environment stays the operator's explicit, approval-gated promotion.

## Guarantees

- The active program is never modified. Optimization produces candidates only.
- Every candidate is immutable and content-addressed.
- Ranking applies hard constraints first, then the weighted objective score.
- The search is deterministic on the deterministic local stack, so a run reproduces its result.

## Where this is going

Retrieval-policy, graph, chunking, context-packing, and router optimization all ship today, alongside the durable and resumable run and the immutable artifact registry with promotion and rollback. Further optimizers and evaluation-driven search are described in the repository review and in [Roadmap and non-goals](roadmap.md).
