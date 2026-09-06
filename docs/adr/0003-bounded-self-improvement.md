# ADR 0003: Self-improvement is bounded, evaluated, and reversible

- Status: accepted
- Date: 2026-08-07

## Context

An autonomous system that rewrites its own grounding policy can degrade silently, weaken safety constraints, or optimize for noisy feedback. Long-running systems still need a controlled way to improve retrieval and interaction behavior.

## Decision

The immutable base system policy is hash-locked. Improvement proposals may change only an allowlisted supplemental schema such as retrieval weights, context allocations, compaction thresholds, and response-style preferences. A proposal records evidence and evaluation results, creates a new immutable snapshot, and requires explicit activation policy. Every activation has an audit trail and rollback target.

User feedback updates bounded preference state, never the base policy. Evaluation failures prevent automatic activation.

## Consequences

Refaat can learn without unreviewed prompt mutation. Regressions remain attributable and reversible. Adding a new mutable field requires a schema and policy change, not free-form model output.
