# ADR 0002: Canonical evidence before framework adapters

- Status: accepted
- Date: 2026-08-07

## Context

Agent frameworks expose different retriever, tool, memory, and document abstractions. Implementing retrieval separately for each framework causes ranking, authorization, citation, and cache behavior to drift.

## Decision

Refaat owns a canonical query, retrieval plan, evidence item, context packet, citation, and grounded response model. Lexical, vector, graph, session, profile, skill, and cache paths normalize into that contract. LangChain, LlamaIndex, CrewAI, FastAPI, the CLI, and the Python SDK only translate at the boundary.

## Consequences

Framework upgrades cannot silently change evidence semantics. New integrations stay thin. Evaluation runs can compare every entry point against the same expected evidence and citation rules.
