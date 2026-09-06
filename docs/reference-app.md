# Reference application

`examples/16_reference_app.py` is the capstone example. It composes the entire Refaat library end to end over one fabricated knowledge base, the Northwind Logistics operations and product handbook, and runs it as two faces of one runtime.

Run it:

```bash
uv run python examples/16_reference_app.py
```

It emits a single JSON object with genuine library outputs and self-verifies every capability before printing. No paid model, no network beyond the loopback API server, and a deterministic stack (deterministic embedder, extractive generator, co-occurrence graph, SQLite).

## The corpus

One fabricated corpus of sixteen documents drives both faces. Each document carries `team`, `system`, `severity`, and `doc_type` metadata, section headings, and recurring named entities (the Cargo Gateway, the Freight Ledger, the Depot Console, the Manifest Store, the Vault Service, the On-Call Dispatcher, the Incident Commander, the Rollback Runbook, the Escalation Runbook, SKU NW-100, and SKU NW-220). The repetition is deliberate so vector search, lexical search, the co-occurrence graph, reranking, and citations are all genuinely exercised, not just wired.

## Face 1: the live enterprise service

Face 1 boots the real Refaat FastAPI application under a real Uvicorn subprocess on a loopback port, bootstraps a workspace, and drives it over HTTP and the SDK.

- Ingest the corpus through `RefaatClient`, then query for a grounded answer with canonical `[chunk:<id>]` citations that resolve to real retrieved chunk ids.
- Retrieve to inspect the fusion routes.
- Stream an uncached incident query so the extractive generator emits more than one answer delta, then confirm the authoritative grounded final event.
- Repeat a query to prove the version-safe answer cache returns a `cag:exact` hit.
- Retrieve and answer a relationship question so the co-occurrence graph facts appear.
- Exercise the program and registry HTTP surface: list programs, run the grounded program, evaluate it, then approve and promote it to production and resolve the deployment, so registry lineage is real.
- Wrap the live client with LangChain (`as_langchain_retriever` and `as_langchain_tool`) and drive a real LangGraph `StateGraph` whose single node is `as_langgraph_node`. Both run for real: the LangChain retriever returns documents and the tool returns a grounded answer, and the compiled graph is invoked and its final state carries a grounded answer with citations. `langchain-core` and `langgraph` are required to run this example.

## Face 2: the in-process agentic core

Face 2 builds an in-process `Refaat` over the same deterministic corpus (its own temporary database, so there is no shared-database write race) and exercises the Python-native capabilities that are not on the HTTP surface.

- Enable the lexical reranker so the retrieve routes include `rerank`.
- Promote a `grounded-rag` program into the registry and wrap it as a `ProgramSkill`, pinned to the promoted artifact id.
- Run a `GovernedReActProgram` over a multi-step incident question. It calls the program-as-skill tool and a read-only digest tool, and an over-privileged external paging tool is correctly denied under a read-only ceiling. The authorized and denied tools are captured.
- Run a `StructuredOutputProgram` and capture whether the typed output validated.
- Run `RetrievalOptimizer` over a small retrieval search space, capture the Pareto front size, and assert the active program was never mutated.
- Score the governed run through `ProgramEvaluator` with the trajectory bundle (grounding, tool safety, step budget, and token cost).
- Promote and roll back through the registry to prove the lineage resolves back to the base artifact.

## What the run asserts

Before it prints, the example asserts non-vacuous invariants: the citations resolve to real chunk ids, the repeat query is a cache hit, the denied tool is the external paging tool, the Pareto front is non-empty, the optimizer did not mutate the active program, the evaluation grounding is a real number, and the compiled LangGraph result carries a grounded answer with citations. Every emitted value is a real library output, not a hardcoded number.
