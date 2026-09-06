# GraphRAG guide

Refaat builds graph state during ingestion and uses it as one route inside adaptive retrieval. It does not require application developers to maintain a separate GraphRAG pipeline.

## Graph construction

Each ingested document can produce:

- entities with stable IDs, normalized names, kinds, document ownership, and metadata;
- directed relations with source/target entity IDs, kind, weight, and metadata.

The default co-occurrence extractor is deterministic and suitable for local development and regression tests. The OpenAI-compatible graph adapter requests typed JSON and validates the response before persistence.

The co-occurrence extractor is language-aware. It drops high-frequency function words before counting candidate concepts, using a per-language stopword set selected from the document's `language` metadata tag (`metadata={"language": "de"}`), falling back to the `default_language` setting. English is the default, with curated German, French, Spanish, and Arabic sets in `core/text.py`. An untagged document, or one tagged `en`, resolves to the exact English set, so its graph is unchanged. The tokenizer is Unicode-aware for space-delimited scripts; word segmentation for space-free scripts such as CJK is a documented future enhancement.

## Query expansion

At retrieval time Refaat:

1. tokenizes and normalizes the query;
2. finds matching graph entities;
3. traverses bounded relations;
4. creates readable graph facts;
5. boosts documents associated with relevant entities and relations;
6. reserves bounded token space for graph facts;
7. fuses graph, lexical, semantic, and optional profile signals.

A graph fact may look like:

```text
Release Manager —cooccurs→ Security Team
```

The fact is contextual structure. Final citations still bind to document chunks.

## Inspect graph-aware context

```python
context = project.retrieve(
    "How are the Release Manager and Security Team involved in a production release?",
    top_k=6,
)

print(context.routes)
print(context.graph_facts)
for hit in context.hits:
    print(hit.graph_score, hit.chunk.document_id, hit.reasons)
```

When graph facts are present, the packet includes the `graph` route. A query can still succeed through lexical and semantic retrieval when the graph has no relevant entities.

## Designing source data

Graph extraction works best when documents use stable, explicit names for systems, teams, controls, products, and processes. Avoid assuming that entity normalization can resolve every alias. For enterprise deployments, maintain an ontology or model-assisted extraction contract and evaluate alias, relation, and cross-document behavior.

## Rebuild behavior

Document replacement removes the prior document's entities and relations and writes the replacement graph in the same versioned ingestion flow. Deletion removes graph state and advances the corpus version. Cache entries from the previous version cannot be reused.

## Limits

The built-in graph is designed for evidence expansion, not unrestricted graph analytics. It does not expose a Cypher endpoint, cross-workspace traversal, or automatic truth reconciliation. Add a custom `GraphExtractor` or store adapter when the application requires an external graph database or a domain ontology.

Run `examples/01_local_rag_graph_cache.py` for a complete graph-aware workflow.
