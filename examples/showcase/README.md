# Oronts Knowledge Terminal

A self-contained web app that shows the whole Refaat retrieval and evidence stack over a
curated Oronts knowledge base. It is one FastAPI service, one dark terminal page, one
image. Every answer and every citation comes from Refaat: nothing is fabricated, and each
returned citation resolves to a real stored chunk.

The app runs fully offline and deterministic by default (SQLite plus in-process vector
search, hash embeddings, extractive generation, a co-occurrence graph). Set an OpenAI key
to switch generation and embeddings to a real model. There is no login and no external
host: all page assets are local, so the UI works with no network.

## What it demonstrates

One prompt box, one mode selector, one live-streaming answer pane, a citations panel with
the verbatim quote and source document for each cited chunk, a routes strip that lights up
which retrieval stages fired, a small entities and relations view for the graph mode, and a
speak button that reads the grounded answer aloud through the browser text-to-speech
engine.

Modes wired to the HTTP surface: plain RAG, GraphRAG, corrective RAG, Self-RAG, the
governed ReAct agent, a LangGraph state machine, and structured typed output. The
capabilities list also names the model-lift features that activate only with a real model.

It also surfaces real economics and observability. A cost and tokens panel reports the
prompt, completion, and total tokens plus the USD cost of the current answer, computed
from the response usage the runtime reports and the configured model prices, so offline
the extractive generator reports zero tokens and zero cost honestly and real numbers
appear only when a model is set. A saved panel reports the tokens, USD, and latency a
cache hit avoided, measured from the real fresh run versus cached run delta. An
observability dialog shows a real captured OpenTelemetry trace of a program run (the
genuine `refaat.program.run` and `refaat.program.evaluate` spans with their real
durations and attributes), the real run-budget meter, the session economics, and the
runtime event log read from the outbox rows written at ingestion. Nothing in these panels
is a mock: every number comes from a real runtime value.

## One-command run with Docker

Build the single image from the repository root, then run it with a mounted data volume:

```bash
docker build -f examples/showcase/Dockerfile -t oronts-terminal .
docker run --rm -p 8000:8000 -v oronts-terminal-data:/data oronts-terminal
```

Open http://127.0.0.1:8000. To use a real model, pass the key:

```bash
docker run --rm -p 8000:8000 -v oronts-terminal-data:/data \
  -e REFAAT_OPENAI_API_KEY=sk-... oronts-terminal
```

## Compose alternative

```bash
docker compose -f examples/showcase/docker-compose.yml up --build
```

One service, one data volume, no other containers. No Postgres, no pgvector, no Redis, no
search cluster.

## Run without Docker

From the repository root, with the project installed:

```bash
PYTHONPATH=examples uvicorn showcase.app.main:create_app --factory --port 8000
```

## Environment

| Variable | Default | Purpose |
|---|---|---|
| `DATA_DIR` | `./data/showcase` | SQLite database and blob storage. The container volume. |
| `REFAAT_OPENAI_API_KEY` | empty | When set, generation and embeddings use the OpenAI REST API. Empty stays offline. |
| `REFAAT_OPENAI_BASE_URL` | OpenAI | Any OpenAI-compatible endpoint. |
| `REFAAT_GENERATION_MODEL` | `gpt-4o-mini` | Generation model when online. |
| `REFAAT_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model when online. |
| `REFAAT_HYPE_ENABLED` | `true` | Ingestion-time hypothetical prompt embeddings. |
| `REFAAT_RAPTOR_ENABLED` | `true` | Ingestion-time recursive summary tree. |
| `REFAAT_CONTEXTUAL_RETRIEVAL` | `true` | Prefix each chunk with its title before embedding. |
| `PORT` | `8000` | Server port inside the container. |

See `.env.example` for a copy-ready template.

## Offline and real-model modes

With no key the stack is deterministic end to end, which is why the app can be driven in
CI by an in-process client with no network and no paid model. The same code path switches
to a real model when a key is present: storage stays on SQLite and vector search stays in
process, so the image never needs an external service in either mode. The only external
dependency is the OpenAI REST API, and only when a key is set.

## HTTP surface

| Method and path | Purpose |
|---|---|
| `GET /` | The single-page UI. |
| `GET /healthz` | Liveness plus the active mode. |
| `POST /api/query` | Answer a question in a selected mode, with resolved citations and routes. |
| `GET /api/query/stream` | Server-sent events: answer deltas then an authoritative grounded final event. |
| `GET /api/stats` | Corpus and graph statistics plus the capability catalog, model-lift counts, and session economics. |
| `GET /api/observability` | A real captured OpenTelemetry trace, the run-budget meter, session economics, and the runtime event log. |
| `GET /api/corpus` | The knowledge documents and their metadata. |
| `GET /api/features` | The demonstrated capabilities and which are model-lift. |
| `GET /api/graph` | Entities and relations for a query, for the graph view. |
| `POST /api/program/{name}/run` | Run a registered program by name. |
| `GET /api/registry` | The registered programs and knowledge document count. |

Every citation the API returns is checked against the stored-chunk universe before it
leaves the server, so the grounding contract holds over HTTP.

## About

The Oronts Knowledge Terminal is built and maintained by Oronts as a reference
demonstration of a production RAG system on top of Refaat. It ships inside the open-source
Refaat repository and is licensed under the Apache License 2.0. See the repository
[README](../../README.md) and [LICENSE](../../LICENSE).
