# Hierarchical retrieval (RAPTOR)

RAPTOR (Recursive Abstractive Processing for Tree Organized Retrieval) adds a persisted
multi level summary tree over a corpus so a query can match a high level summary and then
resolve down to the real leaf chunks that summary covers. It is an opt in, ingestion time
technique that never changes the citation contract: a summary is never a chunk and never a
citation, and every returned hit is a real stored chunk.

## Why a tree

Plain retrieval matches a query against individual chunks. Some questions are about a theme
that spans several chunks, where no single chunk is the best match but the group is. RAPTOR
builds summary nodes over clusters of related chunks, embeds each summary in the shared
vector space, and lets a query match those summaries. A summary match is mapped back to the
node's real leaf chunks, so the answer is still grounded in and cited from the corpus.

## How the tree is built

At ingestion, when `raptor_enabled` is set, `RaptorTreeBuilder` rebuilds the corpus tree as a
pure function of the current stored leaf chunks:

1. Deterministic embedding similarity clustering groups the leaf chunks. Items are processed
   in key order, so the grouping is reproducible and independent of input order.
2. Each cluster is summarized. The offline default is `DeterministicExtractiveSummarizer`, a
   model free summary built from the most salient sentences of the cluster. A configured
   model uses `ModelSummarizer`, which degrades to the extractive summary on a dependency
   failure or a blank generation, so a tree build never fails on the model path.
3. Each summary is embedded in the shared vector space and persisted as a node.
4. The process recurses up to a small maximum depth, summarizing the previous level's nodes.

Every rebuild deletes the corpus's prior nodes and writes the freshly derived set, so re
ingesting unchanged content reproduces byte identical nodes and never duplicates them. A node
id is content addressed from the corpus, the level, and the sorted set of real leaf chunks it
covers, in the reserved `raptor_summary_` namespace.

## Query time

When RAPTOR is enabled the retriever also matches the query embedding against the stored
summary node embeddings. A summary match resolves to the node's stored leaf chunk ids, the
best score per leaf is kept, the metadata filter is applied for parity with the semantic
path, and the result is merged into the fused candidates by chunk id through the shipped
reciprocal rank fusion. The retrieval adds a `raptor` route so the contribution is visible.

## The citation contract

A summary node id uses the reserved `raptor_summary_` prefix, a namespace a real chunk id
(always `chunk_`) can never take, so a node id can never collide with a stored chunk. Citation
resolution resolves every cited id against the chunks table, so a summary node id is absent
there and is dropped: a summary can ride as an intermediate hit but can never be cited. A
summary hit is always mapped back to its real leaf chunks, and those real stored chunks are
what get cited.

## Byte identical when disabled

RAPTOR is off by default. With `raptor_enabled` unset, ingestion builds no nodes, the
`raptor_nodes` table stays empty, the query path runs the same search as before with no layer
match and no `raptor` route, and the answer cache identity is unchanged. The migration builds
the empty table but nothing writes to it.

## Configuration

| Setting | Default | Effect |
| --- | --- | --- |
| `raptor_enabled` | `false` | Build the summary tree at ingestion and match summaries at query time. |

Inject a custom tree builder with `RuntimeBuilder.with_raptor_builder(...)` when you want a
specific summarizer, clustering depth, or cluster size.

## Honesty boundary

The clustering and the extractive summaries are deterministic and fully offline tested. The
model summary is the quality lift and activates only when a real generation model is
configured. The structure, the persistence, the query time match, and the citation contract
are the same on both paths.
