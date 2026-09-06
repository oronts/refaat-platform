# Program evaluation

Refaat Eval is a native, custom evaluation engine for the typed programs described in [Typed programs and artifacts](programs.md). It scores a program against a versioned dataset using a bundle of metrics with hard constraints and weighted objectives. Evaluation reuses the program's own runtime path, so it exercises exactly the retrieval, generation, caching, and citation behavior that production uses. It has no third-party dependency.

## Datasets

A dataset is an immutable, content-addressed set of cases. Building the same records twice yields the same `version`, and any change to a case changes it, so a score is always tied to an exact dataset.

```python
from refaat.evaluation import Dataset

dataset = Dataset.from_records(
    "handbook-regression",
    [
        {"query": "What is required before a production release?", "expected_terms": ["release"]},
        {
            "query": "How is an incident escalated?",
            "expected_terms": ["incident"],
            "expected_document_ids": ["doc_..."],
        },
        {"query": "How is data protected?", "reference_answer": "Data is encrypted at rest and in transit."},
    ],
)
```

Each case carries a query, optional `expected_terms`, optional `expected_document_ids`, an optional `reference_answer`, and tags.

## Metrics and bundles

Metrics are deterministic and work offline. The default set is `Grounded`, `AnswerContains`, `CitationRecall`, and `TokenF1`. Three deeper metrics are also available: `SemanticSimilarity` (term-frequency cosine between the answer and the reference answer), `Faithfulness` (fraction of answer tokens supported by the cited evidence), and `ContextPrecision` (fraction of cited documents that were expected). A `MetricBundle` groups metrics into hard constraints, which must pass on every case, and weighted objectives, which combine into a single score.

```python
from refaat.evaluation import MetricBundle, Grounded, AnswerContains, CitationRecall, TokenF1

bundle = MetricBundle(
    name="grounded-rag",
    metrics=(Grounded(), AnswerContains(), CitationRecall(), TokenF1()),
    hard_constraints=frozenset({"grounded"}),
    objectives={"answer_contains": 0.5, "citation_recall": 0.3, "token_f1": 0.2},
)
```

`default_bundle()` returns this configuration. `extended_bundle()` returns a larger bundle that adds `ContextPrecision`, `SemanticSimilarity`, and `Faithfulness` as weighted objectives on top of the default hard constraint. Hard constraints are applied first: a candidate that fails a hard constraint is not compliant regardless of its average score.

## Model-judge metric

`ModelJudge` is an LLM-as-judge quality metric. It scores an answer by asking a model to judge how well the answer is supported by its cited evidence, and it is a weighted objective, never a hard constraint, so a low judge score never fails a case on its own.

The metric is safe offline by construction. With no `JudgeModel` configured it degrades to a deterministic proxy that delegates the score to the shipped `Faithfulness` computation, so the offline gate never calls a model and never fabricates a judge score. A judge that reports a missing dependency degrades to the same proxy rather than guessing. When a `JudgeModel` is injected its verdict drives the score, and the judge rationale is a diagnostic string on the metric detail only, never a citation.

```python
from refaat.evaluation import judge_bundle

# offline: model_judge is the deterministic Faithfulness proxy
report = project.programs.get("grounded-rag").evaluate(dataset, judge_bundle())

# with a real judge injected
report = project.programs.get("grounded-rag").evaluate(dataset, judge_bundle(project.judge_model))
```

`judge_bundle(judge)` returns the grounded-rag metrics plus a `model_judge` objective. A `JudgeModel` is supplied to the runtime through `RuntimeBuilder.with_judge_model(...)` or the `judge_provider` setting, and surfaces on a project as `project.judge_model`. The offline default configures no judge, so `project.judge_model` is `None` and the metric stays on its proxy.

## Evaluating a program

```python
report = project.programs.get("grounded-rag").evaluate(dataset, bundle)

print(report.hard_constraints_passed, report.score)
print(report.aggregate)  # mean score per metric
for case in report.cases:
    print(case.query, case.passed, [(m.name, m.score) for m in case.metrics])
```

The `EvaluationReport` records the program, the content-addressed dataset version, per-case metric results, the mean score per metric, whether all hard constraints passed, and a single weighted score. `report.to_json()` serializes the whole report.

## Command line

```bash
refaat program eval <workspace_id> <corpus_id> grounded-rag cases.json
```

`cases.json` is a JSON array of case records in the same shape accepted by `Dataset.from_records`.

## Durable evaluation

Evaluation can run as a durable background job on the worker, so it survives a worker restart and its result is persisted with the job.

```python
job_id = project.programs.get("grounded-rag").enqueue_evaluation(dataset)
# a worker process runs it; fetch the job later and read its result field
```

The job uses the default metric bundle and stores the full report as the job result, retrievable through the jobs API.

## Where this is going

This is the second layer of the native programming and optimization foundation. The extended metric set ships today, and the few-shot and instruction optimizers already hold out a train and test split. Further retrieval and grounding metrics and cross-run baseline comparison are described in the repository review and in [Roadmap and non-goals](roadmap.md).
