# Oronts Engineering Principles

Oronts GmbH is a founder-led software studio in Munich, started in 2023 by Refaat Al Ktifan. Its engineering principles follow one mission: help companies escape fragile systems and AI hype and replace them with software they actually own and control. The recurring brand promise is code you own and no vendor lock-in.

## Software You Own

Every engagement targets 100 percent client code ownership and IP from the first commit, with zero vendor lock-in. The values are explicit: no black-box dependencies, no forced upgrades or surprise pricing, no code you cannot take with you. As Refaat Al Ktifan puts it, if a client cannot run and modify and deploy the software without Oronts, the work has failed.

## Durable Design Over Quick Patches

Good architecture is defined by what it makes easy to change. Oronts favors framework-native patterns and domain-driven design over quick patches, backed by pragmatic test-driven development, documentation, and architecture decision records. AI-assisted development is used with senior review on everything, and CI and CD run automated quality gates. This discipline carries through the 90-Day Production Pilot, where repos land in the client org on day 1 and each phase ends in an artifact the client keeps.

## Security and Privacy by Default

Systems are built to GDPR, the EU AI Act, and OWASP, with encryption in transit and at rest, least-privilege IAM, and audit logging. OGuardAI, the Oronts semantic data protection runtime, reflects this stance, sitting between an application and any LLM to enforce policy, detect PII, and tokenize reversibly.

## Honesty About What Is Real

Oronts is honest about what is real versus simulated. RAG and agentic AI systems ship governed, observable, and grounded with citations. The live proof is the Oronts AI Assistant running in production on the website, built on Mastra, five agents and four tools with streaming responses, and it never invents prices.
