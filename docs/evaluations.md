# Evaluations

Evaluation suites are versionable JSON cases with a query, expected answer terms, and expected document IDs. Runs use the same query service as production and record per-case routes and cited documents.

Current built-in metrics are:

- `answer_hit_rate`: every expected term appears in the answer;
- `citation_recall`: expected documents are represented in citations;
- `grounded_rate`: the response carries citations when it contains an answer;
- `case_count`.

These deterministic metrics are suitable for regression gates and harness evidence. Add domain-specific judges as plugins rather than replacing the raw results. Model judges should be calibrated against human labels and isolated from the model under test when possible.

Runs can execute synchronously or through the durable worker. Cache scope includes the evaluation run identity, avoiding cross-run contamination while still exercising CAG behavior inside a run.
