# Adaptive RAG, CAG, KAG, and GraphRAG

Refaat treats retrieval, cache, graph, and durable knowledge as routes inside one context plan rather than separate
products.

## Retrieval routes

- **RAG:** lexical and semantic candidates are normalized and fused.
- **CAG:** exact and high-threshold semantic answer entries are checked before retrieval.
- **KAG:** graph facts, explicit user preferences, session summaries, goals, skills, and supplemental harness state can
  affect context construction. Preferences and harness state are never promoted to factual evidence.
- **GraphRAG:** query entities expand through bounded graph relations. Associated documents receive a graph score and
  explicit graph facts enter the packet.

Every cache lookup includes workspace, corpus version, and a hashed effective-context scope derived from identity/session scope, immutable base policy, embedding and generation model IDs, effective retrieval policy, active harness state, current user profile, and current session summary. Document replacement or deletion advances the corpus version, invalidating stale answers without a global flush.

## Language awareness

The co-occurrence graph extractor is language-aware. `core/text.py` ships a per-language stopword registry as sensible defaults: English is the default, with curated German, French, Spanish, and Arabic sets. The graph extractor selects a document's stopword set from its `language` metadata tag (`metadata={"language": "de"}`), falling back to the `REFAAT_DEFAULT_LANGUAGE` setting (`en` by default), so a tagged document builds its graph from content words rather than function words. The tokenizer is Unicode-aware for space-delimited scripts (Latin, Cyrillic, Arabic, and Latin diacritics). An English or untagged corpus resolves to the exact English set, so its graph and cache identity are unchanged. A parallel per-language query-stopword registry ships alongside it as scaffolding, but the lexical and query-transform path does not yet consume it: query-term filtering uses the English set regardless of a document's language. Known limitations, tracked as future enhancements: language-aware lexical filtering, word segmentation for space-free scripts such as CJK (a run tokenizes as one token), and non-Latin sentence splitting.

The registry is the default, not a hardcoded list: each project can override a shipped language or add a new one without editing the library. Set the `REFAAT_STOPWORDS_BY_LANGUAGE` setting (`stopwords_by_language`, a mapping of short language code to a list of stopwords) or call `RuntimeBuilder.with_stopwords({...})` at composition time. Codes normalize to their primary subtag and words are casefolded to match the tokenizer, and an override merges onto the shipped defaults so untouched languages keep their curated sets. This is additive: with nothing configured the runtime passes the module default to the extractor and every unconfigured corpus is byte-identical. Use case: a project ingesting Italian content supplies its own Italian set with `with_stopwords({"it": ["della", "sono", "come", "questo"]})`, so Italian documents build their graph from content words while the graph of an untagged English corpus is unchanged. This override affects the graph co-occurrence stopwords; it is not a claim of linguistic completeness.

## Term-indexed graph read (opt-in)

By default the query and RLM paths load the whole corpus graph on every request and `graph_context`
selects the query-relevant subgraph in memory. That is bounded per document by the entity cap for the
built-in extractor but grows with the corpus. Setting `REFAAT_GRAPH_TERM_INDEX_ENABLED=true` switches
both paths to a term-indexed read that fetches only the query-relevant subgraph from an `entity_terms`
index, so the graph read scales with the neighborhood a query touches rather than the corpus.

The contract:

- **Byte-identical when off.** With the flag off (the default) no `entity_terms` rows are written, both
  paths read through the full scan, and nothing enters any cache or artifact identity. The three pinned
  artifact ids are unmoved and every existing test passes unchanged.
- **A faithful equivalent, not a reorder, when on.** `graph_context` is unchanged and stays the sole
  arbiter of matched entities, boosts, facts, and authorization. The index read only generates
  candidates: the entities whose normalized name is a substring of the query or whose tokens match a
  query term, plus every relation touching a candidate and both endpoint entities (the one-hop
  frontier), so it never under-returns a name the full scan would match. Over-return is harmless because
  `graph_context` re-applies its exact predicate. For a query yielding at most the twenty-fact prompt
  cap, the boosts and the offline answer and citations are identical to the full scan. Beyond that cap
  the surviving twenty facts can differ, since the index read is deterministically ordered by id while
  the full scan has no defined order; this reorder is the documented effect of enabling the flag, never
  a change to the default output.
- **Authorization is unchanged.** `graph_context` applies the authorized-documents filter first, over
  the fetched subgraph exactly as over the full scan, so an unauthorized entity can never influence a
  boost or a fact. Candidate generation is deliberately ACL-agnostic and the subgraph is a strict subset
  of the corpus graph, so the authorized output is a subset of the full-scan authorized output.
- **A bounded read, not a bounded prompt.** The prompt fact cap bounds prompt material; the term index
  bounds the database read to the query neighborhood. The read is not a fixed ceiling: correctness
  requires fetching the whole matched neighborhood, so a high-degree entity matching a large
  neighborhood still reads that neighborhood, which is at most the full graph and usually far smaller. A
  query longer than `REFAAT_GRAPH_TERM_INDEX_MAX_QUERY_CHARS` (256 by default) falls back to the full
  scan so the substring candidate set can never blow up.
- **Fail-safe, never an empty graph.** The read falls back to the full scan whenever the flag is off,
  the query exceeds the char cap, or the corpus is not fully indexed, so it never serves an empty subgraph
  in place of the real one. The index is treated as usable only when it is complete: if any entity with a
  non-empty normalized name lacks a term row, the corpus reads as unindexed and the full scan runs. This
  covers a corpus ingested before the flag, a document re-ingested unchanged through the idempotency
  no-op path, and an entity whose non-empty name yields no tokens (which keeps that corpus on the full
  scan). The bounded read is therefore realized only once the whole corpus is indexed; until then the
  fallback preserves correctness at full-scan cost. Completing the index today means re-ingesting the
  affected documents, which advances the corpus version and so invalidates the answer cache; a dedicated
  backfill and readiness-inspection surface, and the cache-identity rule for a content-free backfill, are
  tracked as remaining work on the roadmap.

## Multi-query expansion

Before fusion, a deterministic model-free query transform can widen recall. A compound query is decomposed into one
sub-query per content-bearing clause; each variant is retrieved through the same semantic, lexical, and
fusion path, and the per-variant rankings are merged with Reciprocal Rank Fusion and normalized by divide-by-max so the
merged scores stay in `[0, 1]` for the reranker and budgeter. The normalization guards against a max-equal-min list and a
single candidate, never dividing by zero. An ordinary single-clause question produces only the original variant, so its
candidate list and packed context are byte-identical to the single-query path; a question mark alone, or intra-token
punctuation such as an abbreviation, decimal, or version number, never spawns a redundant sub-query. The `expansion` and `rrf` routes appear only when more than one variant list was actually merged.
The transform identity and the `query_expansion` and `max_query_variants` knobs enter the answer-cache scope, so toggling
either recomputes the answer.

### HyDE (Hypothetical Document Embeddings)

HyDE is a second implementation of the same query transform, selected with `REFAAT_QUERY_TRANSFORM_PROVIDER=hyde`. It
pairs the original query with a hypothetical variant. The hypothetical answer moves only the semantic embedding target:
its embedding replaces the semantic search vector for that variant, while the original query text still drives lexical
search. The hypothetical is never lexically searched, never becomes a chunk, is never retrieved, and is never cited, so
every returned hit is a real indexed chunk. The offline default is a deterministic, model-free transform that builds the
hypothetical from the query's content terms, so it has genuine structure and no network dependency. When a real
OpenAI-compatible generator is configured, a model-backed variant writes the hypothetical passage instead, using a
context-free generation call; blank or failed generation degrades to the original-only path. Either variant folds its
identity (`hyde-deterministic-v1`, or `hyde-model:<model-id>`) into the answer-cache scope, exactly like the sub-query
transform. When the transform yields a single original variant, the retrieval path is byte-identical to the single-query
behavior.

### Query rewriting

Query rewriting is another implementation of the same port, selected with `REFAAT_QUERY_TRANSFORM_PROVIDER=query_rewrite`. It retrieves on the original query plus a model restated search query, merged. The offline default is thin by design: it returns one original variant carrying the normalized query, so the single-query retrieval path is byte-identical to today. When a real OpenAI-compatible generator is configured, a model-backed variant asks the generator to restate the query for retrieval and returns the original plus that rewrite, so both are retrieved and RRF-merged and the rewrite actively contributes; a single variant would be short-circuited to the original by the pre-retrieval seam. Blank or failed generation degrades to the normalized original. The rewrite is an ordinary query text that feeds semantic and lexical search over stored chunks. It never becomes a chunk and is never cited. The identity folds as `rewrite-deterministic-v1` or `rewrite-model:<model-id>`.

### Multi-query expansion

Multi-query expansion, selected with `REFAAT_QUERY_TRANSFORM_PROVIDER=multi_query`, reuses the multi-variant seam and Reciprocal Rank Fusion. The offline default returns the single original variant only, so the retrieval path is byte-identical to single-query. When a real OpenAI-compatible generator is configured, a model-backed variant writes several alternative phrasings, one per line, and the runtime retrieves the original query plus each phrasing and merges the per-variant rankings by chunk id. `REFAAT_MAX_QUERY_VARIANTS` bounds the fan-out: the total variant count is capped there. Every phrasing is an ordinary query text over stored chunks; a phrasing never becomes a chunk and is never cited. Blank or failed generation degrades to the original-only path. The identity folds as `multiquery-deterministic-v1` or `multiquery-model:<model-id>`.

### Step-back prompting

Step-back prompting, selected with `REFAAT_QUERY_TRANSFORM_PROVIDER=step_back`, pairs the query with a broader concept query and merges both with Reciprocal Rank Fusion. The offline default derives the broader query deterministically by keeping only the query's content terms and dropping specifics and function words, so the abstraction is genuinely exercised without a model. When a real OpenAI-compatible generator is configured, a model-backed variant writes the broader background question instead. The broader query is an ordinary variant text that feeds both semantic and lexical search over stored chunks. It is not a hypothetical, so it carries no separate embedding target, and it never becomes a chunk and is never cited. A blank broadening, or one that collapses back to the original, yields the byte-identical original-only path. The identity folds as `stepback-deterministic-v1` or `stepback-model:<model-id>`.

## Fusion and token budget

Candidates are fused by one of two strategies. The default, Reciprocal Rank Fusion, combines the lexical, semantic, and
graph rankings by position and normalizes the result; the `linear` strategy uses the normalized weighted sum of the raw
scores. An optional autocut stage then trims the candidate list at its largest relative score drop before packing.
Candidates are deduplicated by canonical chunk identity and packed under a hard token budget. Graph facts reserve
bounded space first, then chunks are selected by score with a per-document diversity penalty. A single oversized first
chunk is truncated instead of producing an empty packet.

Between reranking and the budgeter an optional context-compression stage runs. It is deterministic and model-free: it
splits each retrieved chunk into sentences, scores each sentence by query-term overlap (salience), and drops sentences
that are off-topic below `compression_min_salience` or already emitted verbatim by a higher-ranked chunk. A global set of
kept sentences removes cross-chunk redundancy. Every retained chunk keeps at least its single highest-salience
non-duplicate sentence so it stays citable, and a chunk is dropped only when all of its sentences are global duplicates.
The stage rewrites the in-memory chunk text the generator sees and recomputes `token_count` while preserving the chunk
id, so context tokens fall without changing chunk identity. Compression is off by default. It is enabled through
`REFAAT_COMPRESSION_ENABLED` or the harness `retrieval.compression_enabled`, and both `compression_enabled` and
`compression_min_salience` ride `RetrievalPolicy.as_json` into the answer-cache identity. Citation quotes are always
built from the stored, uncompressed chunk text, so a `citation.quote` remains a verbatim contiguous corpus excerpt even
when the generator saw a compressed chunk.

The generator receives one canonical `ContextPacket`; framework adapters do not maintain independent context logic.

## Source trust ranking

`SourceTrust` grades how much a piece of evidence is trusted, independent of its confidentiality, from `UNVERIFIED` up to `AUTHORITATIVE`. It is read from a top-level `source_trust` metadata key, never from the `acl` block, because trust is a separate axis from access control. When `source_trust_weight` is above zero, the retriever subtracts a bounded, deterministic penalty of `weight * (AUTHORITATIVE - source_trust) / AUTHORITATIVE` from each candidate's ordering score after fusion and reranking, so low-trust evidence cannot dominate the context and the blast radius of untrusted or injected content stays bounded. An absent or malformed tag defaults to `UNVERIFIED`, the minimum: trust must be asserted, never assumed, so untagged or malformed content can neither evade the down-rank nor claim maximum trust. The penalty applies only when `source_trust_weight` is above zero; at the default weight of zero the trust value is never read, so ranking, routes, and the answer-cache identity stay byte-identical whether or not evidence is tagged.

The weight defaults to `0.0`, which disables the stage: no reorder happens, no route is recorded, the weight does not enter the answer-cache identity, and the runtime is byte-identical. When it is above zero, the weight folds into the answer-cache identity so a trust-weighted answer is cached separately.

This is a soft re-rank that only reorders candidates and only touches the ordering score, leaving the semantic, lexical, and graph scores intact for reporting; a penalized chunk stays a candidate and remains citable, and a citation still resolves to its stored chunk. The transform itself never removes a candidate, but because it runs before autocut and the token budget, a downstream selection stage may still omit a candidate whose penalized score widened its gap.

## Hierarchical expansion

When the corpus was ingested with `REFAAT_CHUNKING_STRATEGY` set to `parent_child` or `sentence_window`, retrieval
matches on the precise child chunks and then expands them before packing. Parents never enter candidate generation, so
they cannot steal candidate slots from children; they are fetched by id only during expansion. In `parent_child` mode
each selected child is replaced by its real stored parent section and two children of one parent collapse to a single
parent hit, so the cited id is the parent's stored id. In `sentence_window` mode the child keeps its own id and its
displayed text widens to a bounded window of neighboring sentences sliced directly out of the stored parent text, so the
quote stays a contiguous verbatim substring and is never re-joined or synthesized. The window size is
`REFAAT_SENTENCE_WINDOW_RADIUS` (0 to 8 sentences on each side) and a harness may override it per corpus via
`retrieval.window_radius`. Expansion runs between reranking and the budgeter, recomputes token counts, and adds an
`expand` route. Because citation quotes are always read back from the stored chunk rows, a cited id whose row is missing
is dropped rather than quoted from expanded or compressed in-memory text.

## HyPE (Hypothetical Prompt Embeddings)

HyPE moves the question-to-question match to ingestion time. When it is enabled, each chunk is passed to a hypothetical
question generator, the generated questions are embedded in the same vector space as the chunks, and the question
embeddings are stored in a dedicated `chunk_hypotheticals` table alongside their real chunk id. At query time the query
embedding is matched against those question embeddings in addition to the chunk embeddings, so a question-shaped query can
find a chunk whose stored text is declarative. Every hypothetical match maps back to its real chunk id and is folded into
the ranking by the shipped Reciprocal Rank Fusion, so the final candidates are always real stored chunks and a `hype`
route is added when a match contributes. A hypothetical question is never a chunk, never retrieved as content, and never
cited: the citation resolves to the stored chunk row, and the quote is read back from that row.

The offline default is a deterministic, model-free generator that builds templated questions from the chunk content
terms, so the whole HyPE path is reproducible without a model and re-ingesting an unchanged document reproduces identical
rows. A model-backed generator writes real questions and degrades to the deterministic templates on any dependency
failure, so an ingestion run never stalls on the model path. HyPE is off by default. It is enabled through
`REFAAT_HYPE_ENABLED`, and when enabled the generator identity is folded into the answer-cache identity so an
HyPE-indexed corpus and a plain corpus never share a cache entry. With HyPE disabled the `chunk_hypotheticals` table
stays empty, ingestion writes no rows, and retrieval, routes, and cache identity are unchanged.

## Grounding

Generators cite exact IDs in `[chunk:<id>]` form. Refaat rejects any cited ID that was absent from the context packet.
The API returns structured citations with document IDs and evidence quotes. When a model produces an ungrounded
answer, the configured fallback can return deterministic extractive evidence rather than silently accepting it.

Citation validation proves source binding, not universal factual correctness. Representative evaluations and domain
review remain required.

## Abstention on weak evidence

When the retrieved evidence is too weak, the honest answer is no answer. An opt-in abstention gate runs after retrieval
and before generation. It compares the best retrieved evidence score, the maximum per-hit semantic similarity across the
retrieved chunks, against the `REFAAT_ABSTAIN_BELOW_SCORE` floor. When the floor is above zero and the best score is
below it, the query short circuits to a fixed honest no-answer, "I could not find information about that in this
knowledge base.", with no citations and a distinguishable `abstain` route. It never reaches the extractive fallback, so
an out-of-corpus question returns an honest abstention instead of a confident answer stitched from irrelevant chunks.

The floor is a similarity threshold, so it is compared against the semantic vector score, not the fused rank score. The
default is `0.0`, which disables abstention: the gate short circuits before inspecting any hit, so retrieval, routes,
cache identity, artifact IDs, and every stored response stay byte-identical to the pre-abstention pipeline. An
abstention carries zero citations, so the citation contract holds trivially, and it is deterministic for its scope, so it
is cached and streamed exactly like a grounded answer. The response exposes an `abstained` flag on the query result, the
HTTP schema, and the SDK answer model. Because hash-embedding similarity is not comparable to a real cosine threshold,
the offline deterministic stack keeps the floor at `0.0`; a real cosine floor belongs to a real embedding model.

Abstention applies to the standard query pipeline, both the synchronous answer and the streaming answer. The bounded RLM
is a separate recursive method with its own grounding fallback and is not covered by this gate.

## Requiring a citation

Abstention gates on retrieval score; a separate citation policy gates on the produced answer. `REFAAT_CITATION_POLICY`
governs what happens when an answer resolves no citations: `optional` (the default) serves it, `required_when_evidence`
refuses it when evidence was retrieved but nothing was cited, and `required` refuses any zero-citation answer. A refusal
carries no citations and a distinguishable `citation_required` route. Unlike an abstention it is not cached, because a
later generation may cite; the policy folds into the answer cache identity only when it is not `optional`, so a permissive
answer is never served under a stricter policy and the default path stays byte-identical. The policy is enforced on the
same finalization step for the synchronous answer and a degraded answer, and a cache hit that revalidates to zero
citations (its cited chunks were deleted) is refused as a miss and regenerated under an active policy. When a policy other
than `optional` is active the streaming answer is buffered, exactly as it is under an output governor: the whole answer is
produced and evaluated before any token is emitted, so a refused answer never streams an uncited token to the caller.

## Learning

Explicit feedback can update bounded profile observations, such as increased preference for a user-selected document.
Confidence moves gradually and remains within a fixed range. Anonymous traffic does not create personal preferences.
The immutable grounding policy is not changed by feedback.

## RLM for context larger than one call

The bounded recursive service uses the same canonical chunks and citations. It ranks and partitions externalized corpus
context, executes limited recursive subquestions, carries grounded sub-results into synthesis, and validates the final
answer against original evidence IDs. See [Bounded RLM and autonomy](rlm-and-autonomy.md).
