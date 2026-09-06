# OGuardAI

OGuardAI is a product of Oronts GmbH, the Munich software engineering company led by Refaat Al Ktifan. It is live at oguard.ai and it is the guardrail layer that sits between enterprise systems and LLM providers.

## What it is

OGuardAI is a semantic data protection runtime for AI systems. At its center is a policy engine with PII detection and reversible tokenization that sits between an application and any LLM. Sensitive values are detected and tokenized before a prompt leaves the enterprise boundary, and the tokenization is reversible, so protected data never reaches the model provider in the clear while the application still gets a usable result.

## Architecture

The runtime is a Rust core with a Python NER sidecar. The Rust core carries the policy engine and the tokenization path, and the Python named entity recognition sidecar handles detection. OGuardAI is self-hosted and provider neutral, so it does not tie a deployment to a single LLM vendor. It ships with SDKs, Kubernetes deployment, and observability, which makes it deployable inside an existing platform rather than a black box.

## Where it fits

OGuardAI is built for GDPR-strict deployments. It gives a governed, auditable point of control between the systems Oronts builds and the LLM providers behind them, which is the same honesty and control principle that runs through the Production AI pillar and the RAG practice. Because it is provider neutral and self-hosted, it aligns with the Oronts promise of code you own and no vendor lock-in, and it supports work that must hold to GDPR and the EU AI Act.

OGuardAI stands alongside the other Oronts products, PimTx, Vendure Data Hub, Pimcore Asset Pilot, Vendure Customer Intelligence, and the Oronts AI Assistant.
