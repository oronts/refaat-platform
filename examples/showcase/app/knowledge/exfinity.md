# Exfinity: A Client Build Delivered by Oronts

Exfinity is a multi-tenant AI commerce platform that Oronts GmbH designed and built for Exfinity GmbH. It is live at exfinity.ai and has been in delivery since 2025. Exfinity is the one named third-party client on the public Oronts site. It is a client build, not an Oronts owned product.

## What the platform is

Exfinity spans eight core services. At its center is an eight agent runtime built on Mastra, running over Anthropic Claude and OpenAI. The platform exposes an enterprise MCP layer of 27 production tools, each subject to tenant-scoped policy enforcement, so every tenant operates within its own boundary. Discovery is handled by an OpenSearch layer that combines BM25 keyword search with vector search for hybrid retrieval.

## The cloud platform

Exfinity runs on AWS. The infrastructure uses ECS Fargate, Lambda, API Gateway, DynamoDB, OpenSearch, S3, SQS, EventBridge, Cognito, CloudFront, ElastiCache, and MSK, provisioned with Terraform. This is a production, multi-tenant deployment engineered for scale and tenant isolation.

## Role and framing

Refaat Al Ktifan, Founder and CEO of Oronts, is the Lead AI Solution Architect on Exfinity. The engagement shows the Oronts practice at work end to end: agentic AI on Mastra, provider-agnostic use of Claude and OpenAI, an MCP tool platform with policy enforcement, hybrid search, and a full AWS build.

Exfinity is presented consistently as a client build that Oronts delivered for Exfinity GmbH. It is never framed as an Oronts product. The Oronts owned products remain OGuardAI, PimTx, Vendure Customer Intelligence, the Oronts AI Assistant, and the open source plugins Vendure Data Hub and Pimcore Asset Pilot. Exfinity sits apart from them as delivered client work.
