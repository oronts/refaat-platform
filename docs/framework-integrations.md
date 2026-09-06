# Framework integrations

Refaat framework adapters translate the outer retrieval and answering interfaces. The core remains the authority for
identity, cache scope, fusion, graph traversal, preferences, budgeting, and citations. Two shapes are exposed: retrievers
that return canonical evidence, and grounded tools that return a cited answer from the same runtime. A framework-neutral
`RefaatRetriever` and `RefaatAnswerer` back both.

## LangChain

`as_langchain_retriever` returns a native `BaseRetriever`. Each `Document` contains the canonical chunk text and metadata
fields for Refaat chunk ID, document ID, and score.

`as_langchain_tool` returns a native `StructuredTool` that runs a grounded query and returns the answer with canonical
citation ids, so a LangChain agent can call Refaat for a cited answer rather than only fetching chunks.

```python
from refaat.integrations import as_langchain_tool

tool = as_langchain_tool(client, corpus_id)
answer = tool.invoke({"question": "How does rollback work?"})
```

## LlamaIndex

`as_llamaindex_retriever` returns a native `BaseRetriever` producing `NodeWithScore` values whose node IDs are canonical
Refaat chunk IDs.

## LangGraph

`as_langgraph_node` returns a plain node callable. A LangGraph node is a function that receives the graph state and
returns a partial state update, so this needs no LangGraph dependency to construct. Drop it into a `StateGraph` with
`add_node`, and it reads the question from the state and writes the grounded answer and citations back.

```python
from refaat.integrations import as_langgraph_node

node = as_langgraph_node(client, corpus_id)  # reads state["question"], writes state["answer"] and state["citations"]
state = node({"question": "How does rollback work?"})

# in a graph:
# from langgraph.graph import StateGraph
# graph.add_node("refaat", node)
```

The state keys are configurable with `question_key`, `answer_key`, and `citations_key`.

### State machine layer

Beyond the single node, Refaat ships dependency-free node and router primitives plus two lazy graph
assemblers, so a real branching state machine can orchestrate the same runtime without a hard LangGraph
dependency. Only the two assemblers import LangGraph, lazily, and raise a clear installation error when it
is absent.

An answer and its citations are produced only by the reason node (`client.query`) or the program node
(`BoundProgram.run`), both routing through the one grounded pipeline. The retrieve node's evidence is used
only for branching. It is never turned into an answer or a citation, and the abstain node emits an empty
citation list, so retrieved evidence can never leak into a citation.

Primitives:

- `as_langgraph_retrieve_node(client, corpus_id)` writes the retrieved chunks and routes for branching, and
  writes no answer or citation.
- `refaat_evidence_router(state, min_documents=1)` routes to `reason` when enough evidence was retrieved,
  otherwise to `abstain`.
- `as_langgraph_abstain_node(message="")` is an honest terminal that writes the message and an empty
  citation list.
- `as_langgraph_program_node(program)` runs a governed program and surfaces its answer, citations,
  `react_steps`, and `grounded` flag into state.
- `refaat_trajectory_router(state)` routes to `review` when any recorded step was not authorized, otherwise
  to `done`.
- `RefaatGraphState` is a `TypedDict` for the shared state, usable as `StateGraph(RefaatGraphState)`.

Assemblers:

```python
from refaat.integrations import build_grounded_graph, build_react_subgraph

graph = build_grounded_graph(client, corpus_id, min_documents=1)
final = graph.invoke({"question": "How does rollback work?"})

subgraph = build_react_subgraph(bound_program)  # in-process over a local runtime
trajectory = subgraph.invoke({"question": "How does rollback work?"})
```

`build_grounded_graph` wires retrieve to a conditional edge that reaches the grounded reason node or the
abstain node. `build_react_subgraph` wraps a governed program as a compiled graph, so it composes as a node
in a larger `StateGraph`; it runs in-process over a runtime built by `RuntimeBuilder.build`, since there is
no HTTP route for the governed ReAct program.

Each state key is written by exactly one node, so the default last-write-wins reducer is safe. If you fan
out and write the same key from multiple nodes, supply a reducer for that key.

## CrewAI and custom agents

`as_crewai_tool` returns a callable that emits citation-tagged retrieval context:

```text
[chunk:<canonical-id>] grounded text
```

`as_crewai_answer_tool` returns a callable that runs a grounded query and returns the answer with its citation ids.
Neither adapter depends on CrewAI internals, so both work as plain callable tools in other frameworks.

## Process boundary

Use the HTTP SDK for language/process isolation and the local runtime for an embedded Python application. Do not run
multiple processes against the same SQLite database. Use PostgreSQL, a shared object store, and shared rate-limit state
for multi-process deployments.

Optional framework packages are imported lazily. Attempting to construct an adapter without its dependency raises a
clear installation error instead of making the base Refaat package depend on every agent ecosystem.
