from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

from refaat.config import Settings
from refaat.core.compression import ContextCompressor, _rewrite_hit_text
from refaat.core.math import cosine_similarity
from refaat.core.session import stored_summary_cited_documents
from refaat.core.sparse import BM25Params
from refaat.core.text import (
    estimate_tokens,
    name_substrings,
    normalized_key,
    sentence_spans,
    tokens,
    truncate_to_tokens,
)
from refaat.domain.authz import (
    AccessPolicy,
    Classification,
    Principal,
    SourceTrust,
    authorized_document_ids,
    chunk_allowed,
    max_classification,
    require_principal_present,
    residency_zones,
    session_summary_visible,
    source_trust_from_metadata,
    summary_documents_authorized,
    summary_evidence_from_documents,
)
from refaat.domain.models import (
    Chunk,
    ContextPacket,
    Entity,
    JsonObject,
    MetadataFilter,
    QueryRequest,
    Relation,
    SearchHit,
)
from refaat.domain.search import SearchIndexCapabilities
from refaat.errors import ValidationError
from refaat.infrastructure.database.store import SQLStore
from refaat.ports import Embedder, GraphReadRepository, QueryTransform, Reranker, SearchIndex


@dataclass(frozen=True, slots=True)
class RetrievalPolicy:
    top_k: int
    candidate_limit: int
    token_budget: int
    semantic_weight: float
    lexical_weight: float
    graph_weight: float
    fusion: str
    rrf_k: int
    autocut_enabled: bool
    autocut_threshold: float
    metadata_filter: MetadataFilter | None
    bm25_k1: float
    bm25_b: float
    query_expansion: bool
    max_query_variants: int
    compression_enabled: bool
    compression_min_salience: float
    expansion_mode: str
    window_radius: int

    def as_json(self) -> JsonObject:
        return {
            "top_k": self.top_k,
            "candidate_limit": self.candidate_limit,
            "token_budget": self.token_budget,
            "semantic_weight": self.semantic_weight,
            "lexical_weight": self.lexical_weight,
            "graph_weight": self.graph_weight,
            "fusion": self.fusion,
            "rrf_k": self.rrf_k,
            "autocut_enabled": self.autocut_enabled,
            "autocut_threshold": self.autocut_threshold,
            "metadata_filter": self.metadata_filter.as_json() if self.metadata_filter is not None else None,
            "bm25_k1": self.bm25_k1,
            "bm25_b": self.bm25_b,
            "query_expansion": self.query_expansion,
            "max_query_variants": self.max_query_variants,
            "compression_enabled": self.compression_enabled,
            "compression_min_salience": self.compression_min_salience,
            "expansion_mode": self.expansion_mode,
            "window_radius": self.window_radius,
        }


class ContextBudgeter:
    def select(self, hits: Sequence[SearchHit], token_budget: int) -> tuple[SearchHit, ...]:
        selected: list[SearchHit] = []
        used = 0
        document_counts: defaultdict[str, int] = defaultdict(int)
        for hit in hits:
            cost = hit.chunk.token_count + 12
            diversity_penalty = 0.08 * document_counts[hit.chunk.document_id]
            adjusted = hit.score - diversity_penalty
            if adjusted <= 0.0:
                continue
            if selected and used + cost > token_budget:
                continue
            if not selected and cost > token_budget:
                # Truncate by the token measure, not a whitespace split, so a single space-free or
                # non-Latin chunk that exceeds the whole budget is actually bounded rather than kept whole.
                truncated_tokens = max(16, token_budget - 12)
                truncated_text = truncate_to_tokens(hit.chunk.text, truncated_tokens)
                selected.append(
                    _rewrite_hit_text(
                        hit,
                        truncated_text,
                        token_count=estimate_tokens(truncated_text),
                    )
                )
                break
            selected.append(hit)
            used += cost
            document_counts[hit.chunk.document_id] += 1
        return tuple(selected)


def autocut(hits: Sequence[SearchHit], threshold: float) -> tuple[tuple[SearchHit, ...], bool]:
    """Trim a best-first ranked list at its largest relative score drop.

    Returns the retained hits and whether anything was dropped. Pure and
    deterministic; expects hits ordered by descending score and always keeps at
    least one hit.
    """
    if len(hits) <= 1:
        return tuple(hits), False
    scores = [hit.score for hit in hits]
    span = scores[0] - scores[-1]
    if span <= 0.0:
        return tuple(hits), False
    best_gap = -1.0
    cut_at = len(hits)
    for index in range(1, len(hits)):
        gap = scores[index - 1] - scores[index]
        if gap > best_gap:
            best_gap = gap
            cut_at = index
    if best_gap / span >= threshold:
        return tuple(hits[:cut_at]), True
    return tuple(hits), False


# The number of graph facts packed into the prompt. Classification of graph evidence uses the same
# bound so the ceiling counts exactly the facts sent, never facts truncated away.
GRAPH_FACT_LIMIT = 20


def select_graph(
    repo: GraphReadRepository, settings: Settings, workspace_id: str, corpus_id: str, query: str
) -> tuple[list[Entity], list[Relation]]:
    """The graph read for the query and RLM hot paths: the full scan, or the opt-in term-indexed read.

    With graph_term_index_enabled off (default) this is exactly repo.list_graph, so the path is
    byte-identical. On, it fetches only the query-relevant subgraph, falling back to the full scan when
    the query exceeds the char cap or the corpus has no term index yet, so it never returns an empty
    graph in place of the real one. The query must be the same string graph_context receives, so the
    candidate spec matches graph_context's matched set and never under-returns.
    """
    if not settings.graph_term_index_enabled:
        return repo.list_graph(workspace_id, corpus_id)
    query_key = normalized_key(query)
    if len(query_key) > settings.graph_term_index_max_query_chars or not repo.graph_index_present(
        workspace_id, corpus_id
    ):
        return repo.list_graph(workspace_id, corpus_id)
    max_len = min(repo.max_entity_name_length(workspace_id, corpus_id), settings.graph_term_index_max_query_chars)
    # "" is added directly: an empty normalized_name is a substring of every query but is excluded by
    # name_substrings, so without it a punctuation-only custom-extractor name would be under-returned.
    candidate_names = name_substrings(query_key, max_len) | {""}
    return repo.graph_neighborhood(
        workspace_id, corpus_id, term_candidates=frozenset(tokens(query)), name_substrings=candidate_names
    )


class AdaptiveRetriever:
    def __init__(
        self,
        store: SQLStore,
        index: SearchIndex,
        settings: Settings,
        reranker: Reranker | None = None,
        embedder: Embedder | None = None,
        query_transform: QueryTransform | None = None,
        hype_cache_id: str | None = None,
        raptor_cache_id: str | None = None,
        access_policy: AccessPolicy | None = None,
        source_trust_weight: float = 0.0,
        classify_evidence: bool = False,
        strict_authorization: bool = False,
        chunking_expansion_mode: str | None = None,
        graph_read_repository: GraphReadRepository | None = None,
    ) -> None:
        self.store = store
        self.index = index
        self.settings = settings
        # The resolved graph read seam, defaulting to the composite store. An injected repo may expose a
        # cache_id (graph reads change results); the store exposes none, so the default folds nothing.
        self._graph_repo: GraphReadRepository = graph_read_repository if graph_read_repository is not None else store
        self.graph_cache_id = gid if isinstance(gid := getattr(graph_read_repository, "cache_id", None), str) else None
        # The resolved chunking strategy's retrieval shape. None means the config-selected strategy,
        # so a config or default path is unchanged; an injected strategy declares its own expansion.
        self._chunking_expansion_mode = (
            chunking_expansion_mode if chunking_expansion_mode is not None else settings.chunking_strategy
        )
        self.reranker = reranker
        self.embedder = embedder
        self.query_transform = query_transform
        self.hype_cache_id = hype_cache_id
        self.hype_enabled = hype_cache_id is not None
        self.raptor_cache_id = raptor_cache_id
        self.raptor_enabled = raptor_cache_id is not None
        self.budgeter = ContextBudgeter()
        self.compressor = ContextCompressor()
        self.access_policy = access_policy
        self.source_trust_weight = source_trust_weight
        # Only derive graph and summary evidence classification when a model router or residency gate
        # is installed to consume it; otherwise the ceiling is never read, so the work is skipped and
        # the packet keeps its open defaults.
        self.classify_evidence = classify_evidence
        # Fail closed (on by default) for an in-process call that omits a principal under an active policy;
        # set strict_authorization off to opt out.
        self.strict_authorization = strict_authorization

    def resolve_policy(
        self,
        request: QueryRequest,
        supplemental_harness: JsonObject,
    ) -> RetrievalPolicy:
        retrieval = supplemental_harness.get("retrieval", {})
        if not isinstance(retrieval, Mapping):
            retrieval = {}
        harness_top_k = retrieval.get("top_k")
        configured_top_k = harness_top_k if isinstance(harness_top_k, int) else self.settings.query_top_k
        top_k = max(1, min(request.top_k or configured_top_k, 100))
        candidate_limit = min(max(top_k * 4, 24), 200)
        token_budget = max(128, min(request.token_budget or self.settings.query_token_budget, 100_000))
        semantic, lexical, graph = self._weights(supplemental_harness)
        return RetrievalPolicy(
            top_k=top_k,
            candidate_limit=candidate_limit,
            token_budget=token_budget,
            semantic_weight=semantic,
            lexical_weight=lexical,
            graph_weight=graph,
            fusion=self._resolve_fusion(request.fusion, retrieval.get("fusion")),
            rrf_k=self._resolve_rrf_k(request.rrf_k, retrieval.get("rrf_k")),
            autocut_enabled=self._resolve_autocut_enabled(request.autocut_enabled, retrieval.get("autocut_enabled")),
            autocut_threshold=self._resolve_autocut_threshold(
                request.autocut_threshold, retrieval.get("autocut_threshold")
            ),
            metadata_filter=self._resolve_metadata_filter(retrieval.get("filters")),
            bm25_k1=self._resolve_bm25_k1(retrieval.get("bm25_k1")),
            bm25_b=self._resolve_bm25_b(retrieval.get("bm25_b")),
            query_expansion=self._resolve_query_expansion(request.query_expansion, retrieval.get("query_expansion")),
            max_query_variants=self._resolve_max_query_variants(
                request.max_query_variants, retrieval.get("max_query_variants")
            ),
            compression_enabled=self._resolve_compression_enabled(retrieval.get("compression_enabled")),
            compression_min_salience=self._resolve_compression_min_salience(retrieval.get("compression_min_salience")),
            expansion_mode=self._chunking_expansion_mode,
            window_radius=self._resolve_window_radius(retrieval.get("window_radius")),
        )

    def _resolve_window_radius(self, harness_value: Any) -> int:
        if isinstance(harness_value, int) and not isinstance(harness_value, bool) and 0 <= harness_value <= 8:
            return harness_value
        return self.settings.sentence_window_radius

    def _resolve_compression_enabled(self, harness_value: Any) -> bool:
        if isinstance(harness_value, bool):
            return harness_value
        return self.settings.compression_enabled

    def _resolve_compression_min_salience(self, harness_value: Any) -> float:
        if isinstance(harness_value, (int, float)) and not isinstance(harness_value, bool):
            return max(0.0, min(float(harness_value), 1.0))
        return self.settings.compression_min_salience

    def _resolve_query_expansion(self, request_value: bool | None, harness_value: Any) -> bool:
        for candidate in (request_value, harness_value):
            if isinstance(candidate, bool):
                return candidate
        return self.settings.query_expansion

    def _resolve_max_query_variants(self, request_value: int | None, harness_value: Any) -> int:
        for candidate in (request_value, harness_value):
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                return max(1, min(candidate, 8))
        return self.settings.max_query_variants

    @staticmethod
    def _resolve_metadata_filter(harness_value: Any) -> MetadataFilter | None:
        if isinstance(harness_value, Mapping):
            return MetadataFilter.from_spec(harness_value)
        return None

    def _resolve_bm25_k1(self, harness_value: Any) -> float:
        if isinstance(harness_value, (int, float)) and not isinstance(harness_value, bool):
            return max(0.0, min(float(harness_value), 5.0))
        return self.settings.bm25_k1

    def _resolve_bm25_b(self, harness_value: Any) -> float:
        if isinstance(harness_value, (int, float)) and not isinstance(harness_value, bool):
            return max(0.0, min(float(harness_value), 1.0))
        return self.settings.bm25_b

    def _resolve_fusion(self, request_value: str | None, harness_value: Any) -> str:
        for candidate in (request_value, harness_value):
            if candidate in ("rrf", "linear"):
                return str(candidate)
        return self.settings.fusion_strategy

    def _resolve_rrf_k(self, request_value: int | None, harness_value: Any) -> int:
        for candidate in (request_value, harness_value):
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                return max(1, candidate)
        return self.settings.rrf_k

    def _resolve_autocut_enabled(self, request_value: bool | None, harness_value: Any) -> bool:
        for candidate in (request_value, harness_value):
            if isinstance(candidate, bool):
                return candidate
        return self.settings.autocut_enabled

    def _resolve_autocut_threshold(self, request_value: float | None, harness_value: Any) -> float:
        for candidate in (request_value, harness_value):
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                return max(0.0, min(float(candidate), 1.0))
        return self.settings.autocut_threshold

    def retrieve(
        self,
        request: QueryRequest,
        query_embedding: tuple[float, ...],
        *,
        profile: JsonObject,
        supplemental_harness: JsonObject,
        policy: RetrievalPolicy | None = None,
    ) -> ContextPacket:
        require_principal_present(self.access_policy, request.principal, strict=self.strict_authorization)
        active_policy = policy or self.resolve_policy(request, supplemental_harness)
        entities, relations = select_graph(
            self._graph_repo, self.settings, request.workspace_id, request.corpus_id, request.query
        )
        documents: list[JsonObject] | None = None
        authorized: frozenset[str] | None = None
        if self.access_policy is not None and request.principal is not None:
            documents = self.store.list_documents(request.workspace_id, request.corpus_id)
            authorized = authorized_document_ids(self.access_policy, request.principal, documents)
        graph_boosts, graph_facts, graph_fact_documents = self.graph_context(
            request.query,
            entities,
            relations,
            authorized_documents=authorized,
        )
        graph_evidence_classification = Classification.PUBLIC
        graph_evidence_zones: frozenset[str] = frozenset()
        if self.classify_evidence and graph_facts:
            # A graph fact reaches the model whether or not per-principal enforcement is active (a
            # router or residency gate runs without an access policy), so classify it regardless, like
            # the hit path, but only when such a gate is installed to consume the value. Classify the
            # documents behind exactly the facts sent, never merely boosted entities. Malformed acls
            # fail closed through the shared helpers.
            if documents is None:
                documents = self.store.list_documents(request.workspace_id, request.corpus_id)
            backing_ids = frozenset().union(*graph_fact_documents[:GRAPH_FACT_LIMIT])
            backing = [doc.get("metadata") for doc in documents if str(doc.get("id")) in backing_ids]
            graph_evidence_classification = max_classification(backing)
            graph_evidence_zones = residency_zones(backing)
        weights = (
            active_policy.semantic_weight,
            active_policy.lexical_weight,
            active_policy.graph_weight,
        )
        hierarchical = active_policy.expansion_mode != "flat"
        fetch_limit = min(active_policy.candidate_limit * 3, 600) if hierarchical else active_policy.candidate_limit
        plan = self._transform_plan(request.query, query_embedding, active_policy, request.workspace_id)
        variant_lists = [
            self._candidate_lists(
                request,
                text,
                embedding,
                active_policy,
                graph_boosts,
                profile,
                weights,
                fetch_limit,
                authorized_documents=authorized,
            )
            for text, embedding in plan
        ]
        expanded = len(variant_lists) > 1
        fused = self._rrf_merge(variant_lists, active_policy.rrf_k) if expanded else variant_lists[0]
        hype_hits = (
            self._hypothetical_hits(request, query_embedding, active_policy, authorized_documents=authorized)
            if self.hype_enabled
            else []
        )
        if hype_hits:
            fused = self._rrf_merge([fused, hype_hits], active_policy.rrf_k)
        raptor_hits = (
            self._raptor_hits(request, query_embedding, active_policy, authorized_documents=authorized)
            if self.raptor_enabled
            else []
        )
        if raptor_hits:
            fused = self._rrf_merge([fused, raptor_hits], active_policy.rrf_k)
        # Backstop the metadata hard-filter over the fused pool so a backend that ignores it is not the
        # sole enforcer; the built-in backends already filtered, so this is a no-op for them.
        if active_policy.metadata_filter is not None:
            fused = [hit for hit in fused if active_policy.metadata_filter.matches(hit.chunk.metadata)]
        if self.access_policy is not None and request.principal is not None:
            fused = [hit for hit in fused if chunk_allowed(self.access_policy, request.principal, hit.chunk.metadata)]
        if hierarchical:
            fused = self._children_only(fused)
        candidates: Sequence[SearchHit] = fused[: active_policy.candidate_limit]
        if self.reranker is not None:
            candidates = list(self.reranker.rerank(request.query, candidates))
        source_trust_applied = False
        if self.source_trust_weight > 0.0:
            candidates, source_trust_applied = self._apply_source_trust(candidates)
        autocut_dropped = False
        if active_policy.autocut_enabled:
            candidates, autocut_dropped = autocut(candidates, active_policy.autocut_threshold)
        if active_policy.compression_enabled:
            candidates = self.compressor.compress(
                request.query,
                candidates,
                enabled=True,
                min_salience=active_policy.compression_min_salience,
            )
        context_expanded = False
        if hierarchical:
            candidates = self._expand_context(request, candidates, active_policy)
            context_expanded = True
        reserved = sum(estimate_tokens(value) for value in graph_facts[:GRAPH_FACT_LIMIT])
        selected = self.budgeter.select(
            candidates,
            max(64, active_policy.token_budget - reserved),
        )[: active_policy.top_k]

        session_summary: str | None = None
        do_semantic, do_lexical = self._search_modes()
        routes = [name for name, on in (("semantic", do_semantic), ("lexical", do_lexical)) if on]
        if active_policy.metadata_filter is not None:
            routes.append("filter")
        if graph_facts:
            routes.append("graph")
        routes.append(active_policy.fusion)
        if expanded:
            routes.append("expansion")
            routes.append("rrf")
        if hype_hits:
            routes.append("hype")
        if raptor_hits:
            routes.append("raptor")
        if self.reranker is not None:
            routes.append("rerank")
        if source_trust_applied:
            routes.append("source_trust")
        if autocut_dropped:
            routes.append("autocut")
        if active_policy.compression_enabled:
            routes.append("compress")
        if context_expanded:
            routes.append("expand")
        summary_evidence_classification = Classification.PUBLIC
        summary_evidence_zones: frozenset[str] = frozenset()
        if request.session_id:
            session = self.store.get_session(request.workspace_id, request.session_id)
            if session["corpus_id"] != request.corpus_id:
                raise ValidationError("Session belongs to another corpus")
            candidate_summary = str(session["summary"]) if session.get("summary") else None
            if candidate_summary is not None:
                # The summary is prior evidence bound to its writer principal: the shared gate
                # surfaces it only when the stored writer scope matches this reader, else fail
                # closed. With no policy the gate is a no-op and the path is unchanged; a principal-less
                # request under an active policy is denied earlier by strict authorization.
                stored_scope = cast(JsonObject, session.get("metadata") or {}).get("summary_auth_scope")
                if not session_summary_visible(stored_scope, request.principal, self.access_policy):
                    candidate_summary = None
            if candidate_summary is not None and self.access_policy is not None and request.principal is not None:
                if documents is None:
                    documents = self.store.list_documents(request.workspace_id, request.corpus_id)
                if not summary_documents_authorized(
                    self.access_policy, request.principal, stored_summary_cited_documents(session), documents
                ):
                    candidate_summary = None
            session_summary = candidate_summary
            if session_summary:
                routes.append("session")
                if self.classify_evidence:
                    # The surfaced summary carries its folded evidence into the prompt, so a router or
                    # egress gate counts it: classify the summary's cited documents from CURRENT ACLs,
                    # like the graph path, so a re-tag after compaction is reflected and the derivation
                    # never depends on an access policy being active.
                    cited = stored_summary_cited_documents(session)
                    if cited and documents is None:
                        documents = self.store.list_documents(request.workspace_id, request.corpus_id)
                    summary_evidence_classification, summary_evidence_zones = summary_evidence_from_documents(
                        cited, documents or []
                    )
        if profile:
            routes.append("profile")
        if supplemental_harness:
            routes.append("harness")
        actual_tokens = sum(hit.chunk.token_count for hit in selected) + reserved
        return ContextPacket(
            query=request.query,
            hits=selected,
            graph_facts=tuple(graph_facts[:GRAPH_FACT_LIMIT]),
            session_summary=session_summary,
            profile=profile,
            supplemental_harness=supplemental_harness,
            token_count=actual_tokens,
            routes=tuple(routes),
            graph_evidence_classification=graph_evidence_classification,
            graph_evidence_zones=graph_evidence_zones,
            session_summary_classification=summary_evidence_classification,
            session_summary_zones=summary_evidence_zones,
        )

    def _apply_source_trust(self, hits: Sequence[SearchHit]) -> tuple[list[SearchHit], bool]:
        """Down-rank low-trust evidence by a bounded, deterministic score penalty.

        A candidate's penalty is ``weight * (AUTHORITATIVE - source_trust) / AUTHORITATIVE``,
        subtracted from the ordering score exactly as the reranker adjusts scores: a new hit via
        ``replace`` (leaving semantic, lexical, and graph scores untouched for reporting), then a
        re-sort. A tagged AUTHORITATIVE chunk gets a zero penalty; an untagged or malformed chunk
        defaults to UNVERIFIED and takes the maximum penalty, so trust must be asserted to escape
        the down-rank. The transform itself never removes a candidate, only reorders it; a
        downstream selection stage (autocut, the token budget, top_k) may still omit a candidate
        whose penalized score widened its gap.
        """
        ceiling = int(SourceTrust.AUTHORITATIVE)
        penalized = False
        adjusted: list[SearchHit] = []
        for hit in hits:
            trust = int(source_trust_from_metadata(hit.chunk.metadata))
            penalty = self.source_trust_weight * (ceiling - trust) / ceiling
            if penalty > 0.0:
                penalized = True
                adjusted.append(replace(hit, score=hit.score - penalty))
            else:
                adjusted.append(hit)
        if not penalized:
            return list(hits), False
        return sorted(adjusted, key=lambda hit: (-hit.score, hit.chunk.id)), True

    def _ranked_slice(self, best: dict[str, float], authorized_documents: frozenset[str] | None, limit: int) -> int:
        # Under a policy, consider the whole ranked set (capped) so the ACL filter runs before truncation.
        if authorized_documents is None:
            return limit
        return min(len(best), self.settings.acl_overfetch_hard_cap)

    def _authorize_and_truncate(
        self,
        hits: list[SearchHit],
        request: QueryRequest,
        limit: int,
        authorized_documents: frozenset[str] | None,
    ) -> list[SearchHit]:
        # Apply the authoritative ACL gate before truncating so authorized hits are not starved; unchanged when off.
        access_policy = self.access_policy
        principal = request.principal
        if authorized_documents is None or access_policy is None or principal is None:
            return hits
        allowed = [hit for hit in hits if chunk_allowed(access_policy, principal, hit.chunk.metadata)]
        return allowed[:limit]

    def _hypothetical_hits(
        self,
        request: QueryRequest,
        query_embedding: tuple[float, ...],
        policy: RetrievalPolicy,
        *,
        authorized_documents: frozenset[str] | None = None,
    ) -> list[SearchHit]:
        """Match the query embedding against stored hypothetical-question embeddings.

        Each hypothetical is precomputed at ingestion from a real chunk, so a match
        maps straight back to that chunk's stored row: a returned hit is always a
        real stored chunk and its citation resolves to the corpus, never to the
        question text. The best score per chunk wins, the metadata filter is applied
        to stay consistent with the semantic path, and the list is capped at the
        candidate limit before the shipped RRF merge folds it in by chunk id.
        """
        rows = self.store.list_hypotheticals(request.workspace_id, request.corpus_id)
        if not rows:
            return []
        best: dict[str, float] = {}
        for chunk_id, embedding in rows:
            score = max(0.0, cosine_similarity(query_embedding, embedding))
            if score > 0.0 and score > best.get(chunk_id, 0.0):
                best[chunk_id] = score
        if not best:
            return []
        slice_limit = self._ranked_slice(best, authorized_documents, policy.candidate_limit)
        ordered = sorted(best.items(), key=lambda item: (-item[1], item[0]))[:slice_limit]
        chunk_ids = [chunk_id for chunk_id, _score in ordered]
        chunks = {
            chunk.id: chunk for chunk in self.store.get_chunks(request.workspace_id, request.corpus_id, chunk_ids)
        }
        hits: list[SearchHit] = []
        for chunk_id, score in ordered:
            chunk = chunks.get(chunk_id)
            if chunk is None:
                continue
            if policy.metadata_filter is not None and not policy.metadata_filter.matches(chunk.metadata):
                continue
            hits.append(SearchHit(chunk=chunk, score=score, semantic_score=score, reasons=("hypothetical",)))
        return self._authorize_and_truncate(hits, request, policy.candidate_limit, authorized_documents)

    def _raptor_hits(
        self,
        request: QueryRequest,
        query_embedding: tuple[float, ...],
        policy: RetrievalPolicy,
        *,
        authorized_documents: frozenset[str] | None = None,
    ) -> list[SearchHit]:
        """Match the query against stored RAPTOR summary nodes, mapped to real leaf chunks.

        Each node is a summary of a cluster of real leaf chunks, embedded in the shared
        space. A summary match therefore resolves to the node's stored ``leaf_chunk_ids``:
        the returned hit is always a real stored chunk and its citation resolves to the
        corpus, never to the summary text or the uncitable node id. The best score per
        leaf wins across matching nodes, the metadata filter is applied to stay
        consistent with the semantic path, and the list is capped at the candidate limit
        before the shipped RRF merge folds it in by chunk id.
        """
        rows = self.store.list_raptor_nodes(request.workspace_id, request.corpus_id)
        if not rows:
            return []
        best: dict[str, float] = {}
        for _node_id, embedding, leaf_chunk_ids in rows:
            score = max(0.0, cosine_similarity(query_embedding, embedding))
            if score <= 0.0:
                continue
            for chunk_id in leaf_chunk_ids:
                if score > best.get(chunk_id, 0.0):
                    best[chunk_id] = score
        if not best:
            return []
        slice_limit = self._ranked_slice(best, authorized_documents, policy.candidate_limit)
        ordered = sorted(best.items(), key=lambda item: (-item[1], item[0]))[:slice_limit]
        chunk_ids = [chunk_id for chunk_id, _score in ordered]
        chunks = {
            chunk.id: chunk for chunk in self.store.get_chunks(request.workspace_id, request.corpus_id, chunk_ids)
        }
        hits: list[SearchHit] = []
        for chunk_id, score in ordered:
            chunk = chunks.get(chunk_id)
            if chunk is None:
                continue
            if policy.metadata_filter is not None and not policy.metadata_filter.matches(chunk.metadata):
                continue
            hits.append(SearchHit(chunk=chunk, score=score, semantic_score=score, reasons=("raptor",)))
        return self._authorize_and_truncate(hits, request, policy.candidate_limit, authorized_documents)

    def _transform_plan(
        self,
        query: str,
        query_embedding: tuple[float, ...],
        policy: RetrievalPolicy,
        workspace_id: str,
    ) -> list[tuple[str, tuple[float, ...]]]:
        transform = self.query_transform
        embedder = self.embedder
        if transform is None or embedder is None or not policy.query_expansion or policy.max_query_variants <= 1:
            return [(query, query_embedding)]
        variants = transform.expand(query, policy.max_query_variants, workspace_id=workspace_id)
        if len(variants) <= 1:
            return [(query, query_embedding)]
        plan: list[tuple[str, tuple[float, ...]]] = [(query, query_embedding)]
        for variant in variants[1:]:
            embed_source = variant.embed_text if variant.embed_text is not None else variant.text
            plan.append((variant.text, embedder.embed([embed_source])[0]))
        return plan

    def authorized_documents(self, request: QueryRequest) -> frozenset[str] | None:
        """The authorized document id set for the request principal, or None when inactive.

        None means no policy is active, so every graph and retrieval path stays byte-identical;
        a principal-less request under a policy is denied earlier by strict authorization. When
        active it resolves each document's ACL once so the graph can be scoped to documents the
        principal may actually see.
        """
        if self.access_policy is None or request.principal is None:
            return None
        documents = self.store.list_documents(request.workspace_id, request.corpus_id)
        return authorized_document_ids(self.access_policy, request.principal, documents)

    def _search_modes(self) -> tuple[bool, bool]:
        caps = getattr(self.index, "capabilities", None)
        if isinstance(caps, SearchIndexCapabilities):
            return caps.supports_semantic, caps.supports_lexical
        return True, True

    def _candidate_lists(
        self,
        request: QueryRequest,
        query_text: str,
        query_embedding: tuple[float, ...],
        policy: RetrievalPolicy,
        graph_boosts: Mapping[str, float],
        profile: JsonObject,
        weights: tuple[float, float, float],
        limit: int,
        *,
        authorized_documents: frozenset[str] | None = None,
    ) -> list[SearchHit]:
        access_policy = self.access_policy
        principal = request.principal
        do_semantic, do_lexical = self._search_modes()
        if authorized_documents is None or access_policy is None or principal is None:
            semantic = (
                self.index.semantic_search(
                    workspace_id=request.workspace_id,
                    corpus_id=request.corpus_id,
                    query_embedding=query_embedding,
                    limit=limit,
                    metadata_filter=policy.metadata_filter,
                )
                if do_semantic
                else []
            )
            lexical = (
                self.index.lexical_search(
                    workspace_id=request.workspace_id,
                    corpus_id=request.corpus_id,
                    query=query_text,
                    limit=limit,
                    metadata_filter=policy.metadata_filter,
                    params=BM25Params(k1=policy.bm25_k1, b=policy.bm25_b),
                )
                if do_lexical
                else []
            )
        else:
            semantic = (
                self._authorized_fetch(
                    lambda size: self.index.semantic_search(
                        workspace_id=request.workspace_id,
                        corpus_id=request.corpus_id,
                        query_embedding=query_embedding,
                        limit=size,
                        metadata_filter=policy.metadata_filter,
                        allowed_document_ids=authorized_documents,
                    ),
                    limit,
                    access_policy,
                    principal,
                )
                if do_semantic
                else []
            )
            lexical = (
                self._authorized_fetch(
                    lambda size: self.index.lexical_search(
                        workspace_id=request.workspace_id,
                        corpus_id=request.corpus_id,
                        query=query_text,
                        limit=size,
                        metadata_filter=policy.metadata_filter,
                        params=BM25Params(k1=policy.bm25_k1, b=policy.bm25_b),
                        allowed_document_ids=authorized_documents,
                    ),
                    limit,
                    access_policy,
                    principal,
                )
                if do_lexical
                else []
            )
        return self._fuse(
            semantic,
            lexical,
            graph_boosts,
            profile,
            weights,
            fusion=policy.fusion,
            rrf_k=policy.rrf_k,
        )

    def _authorized_fetch(
        self,
        fetch: Callable[[int], list[SearchHit]],
        target: int,
        access_policy: AccessPolicy,
        principal: Principal,
    ) -> list[SearchHit]:
        # Widen the fetch until `target` hits survive chunk_allowed (recall only; the filter stays authoritative).
        cap = self.settings.acl_overfetch_hard_cap
        size = min(target * self.settings.acl_overfetch_factor, cap)
        while True:
            raw = fetch(size)
            authorized = [hit for hit in raw if chunk_allowed(access_policy, principal, hit.chunk.metadata)]
            if len(authorized) >= target or len(raw) < size or size >= cap:
                return authorized[:target]
            size = min(size * 2, cap)

    @staticmethod
    def _children_only(hits: Sequence[SearchHit]) -> list[SearchHit]:
        """Drop parent-section hits so they never steal candidate slots from children.

        A missing ``granularity`` is treated as flat and kept, so a mixed corpus and
        the byte-identical flat path both survive unchanged; only chunks explicitly
        marked as a parent section are removed from the candidate pool. Parents are
        fetched by id later, during expansion, never scored against the query.
        """
        return [hit for hit in hits if hit.chunk.metadata.get("granularity") != "parent"]

    def _expand_context(
        self,
        request: QueryRequest,
        hits: Sequence[SearchHit],
        policy: RetrievalPolicy,
    ) -> list[SearchHit]:
        parent_ids: list[str] = []
        for hit in hits:
            meta = hit.chunk.metadata
            parent_id = meta.get("parent_id")
            if meta.get("granularity") == "child" and isinstance(parent_id, str):
                parent_ids.append(parent_id)
        parents: dict[str, Chunk] = (
            {chunk.id: chunk for chunk in self.store.get_chunks(request.workspace_id, request.corpus_id, parent_ids)}
            if parent_ids
            else {}
        )
        if self.access_policy is not None and request.principal is not None:
            # The parent text replaces or widens the hit, so it is authorized on its own ACL: a
            # child's stored parent_id is caller-influenced and must not pull in a forbidden parent.
            parents = {
                cid: chunk
                for cid, chunk in parents.items()
                if chunk_allowed(self.access_policy, request.principal, chunk.metadata)
            }
        if policy.expansion_mode == "parent_child":
            return self._expand_to_parents(hits, parents)
        return self._expand_windows(hits, parents, policy.window_radius)

    @staticmethod
    def _expand_to_parents(hits: Sequence[SearchHit], parents: Mapping[str, Chunk]) -> list[SearchHit]:
        """Replace each child hit with its real stored parent section, deduplicated.

        The cited id becomes the parent's stored id, so citations resolve to a real
        row. Two children of one parent collapse to a single parent hit; a child
        whose parent row is missing keeps its own real id rather than inventing one.
        """
        result: list[SearchHit] = []
        seen: set[str] = set()
        for hit in hits:
            meta = hit.chunk.metadata
            parent_id = meta.get("parent_id")
            if meta.get("granularity") == "child" and isinstance(parent_id, str) and parent_id in parents:
                if parent_id in seen:
                    continue
                seen.add(parent_id)
                result.append(replace(hit, chunk=parents[parent_id]))
            else:
                result.append(hit)
        return result

    def _expand_windows(
        self,
        hits: Sequence[SearchHit],
        parents: Mapping[str, Chunk],
        window_radius: int,
    ) -> list[SearchHit]:
        """Widen each child hit to a bounded sentence window of its stored parent text.

        The child keeps its real id; only the displayed text grows, and it is sliced
        directly out of the stored parent text so it stays a contiguous verbatim
        substring, never a re-joined or synthesized string.
        """
        result: list[SearchHit] = []
        for hit in hits:
            meta = hit.chunk.metadata
            parent_id = meta.get("parent_id")
            sent_start = meta.get("sent_start")
            sent_end = meta.get("sent_end")
            parent = parents.get(parent_id) if isinstance(parent_id, str) else None
            if parent is None or not isinstance(sent_start, int) or not isinstance(sent_end, int):
                result.append(hit)
                continue
            window = self._window_text(parent.text, sent_start, sent_end, window_radius)
            chunk = replace(hit.chunk, text=window, token_count=estimate_tokens(window))
            result.append(replace(hit, chunk=chunk))
        return result

    @staticmethod
    def _window_text(parent_text: str, sent_start: int, sent_end: int, window_radius: int) -> str:
        spans = sentence_spans(parent_text)
        if not spans:
            return parent_text[sent_start:sent_end]
        lo = next((index for index, (start, _end) in enumerate(spans) if start >= sent_start), 0)
        hi = next((index for index in range(len(spans) - 1, -1, -1) if spans[index][1] <= sent_end), len(spans) - 1)
        lo = max(0, lo - window_radius)
        hi = min(len(spans) - 1, hi + window_radius)
        if lo > hi:
            return parent_text[sent_start:sent_end]
        return parent_text[spans[lo][0] : spans[hi][1]]

    def _rrf_merge(self, variant_lists: Sequence[Sequence[SearchHit]], rrf_k: int) -> list[SearchHit]:
        rank_maps = [self._rank_map(hits) for hits in variant_lists]
        by_id: dict[str, SearchHit] = {}
        scores: defaultdict[str, float] = defaultdict(float)
        for hits, rank_map in zip(variant_lists, rank_maps, strict=True):
            for hit in hits:
                by_id.setdefault(hit.chunk.id, hit)
                scores[hit.chunk.id] += 1.0 / (rrf_k + rank_map[hit.chunk.id] + 1)
        if not scores:
            return []
        normalized = self._normalize_scores(scores)
        merged = [replace(by_id[chunk_id], score=score) for chunk_id, score in normalized.items()]
        return sorted(merged, key=lambda hit: (-hit.score, hit.chunk.id))

    @staticmethod
    def _normalize_scores(raw: Mapping[str, float]) -> dict[str, float]:
        maximum = max(raw.values(), default=1.0)
        if maximum <= 0.0:
            maximum = 1.0
        return {key: value / maximum for key, value in raw.items()}

    def _weights(self, supplemental: JsonObject) -> tuple[float, float, float]:
        retrieval = supplemental.get("retrieval", {})
        if not isinstance(retrieval, Mapping):
            retrieval = {}
        semantic = self._bounded_float(retrieval.get("semantic_weight"), self.settings.semantic_weight)
        lexical = self._bounded_float(retrieval.get("lexical_weight"), self.settings.lexical_weight)
        graph = self._bounded_float(retrieval.get("graph_weight"), self.settings.graph_weight)
        total = semantic + lexical + graph
        if total <= 0.0:
            return self.settings.semantic_weight, self.settings.lexical_weight, self.settings.graph_weight
        return semantic / total, lexical / total, graph / total

    @staticmethod
    def _bounded_float(value: Any, default: float) -> float:
        if not isinstance(value, (float, int)):
            return default
        return max(0.0, min(float(value), 1.0))

    def _fuse(
        self,
        semantic: Sequence[SearchHit],
        lexical: Sequence[SearchHit],
        graph_boosts: Mapping[str, float],
        profile: JsonObject,
        weights: tuple[float, float, float],
        *,
        fusion: str,
        rrf_k: int,
    ) -> list[SearchHit]:
        values = self._candidate_scores(semantic, lexical)
        preferred = self._preferred_documents(profile)
        if fusion == "linear":
            hits = self._fuse_linear(values, graph_boosts, preferred, weights)
        else:
            hits = self._fuse_rrf(values, semantic, lexical, graph_boosts, preferred, weights, rrf_k)
        return sorted(hits, key=lambda hit: (-hit.score, hit.chunk.id))

    @staticmethod
    def _candidate_scores(semantic: Sequence[SearchHit], lexical: Sequence[SearchHit]) -> dict[str, dict[str, Any]]:
        values: dict[str, dict[str, Any]] = {}
        for hit in semantic:
            values.setdefault(hit.chunk.id, {"chunk": hit.chunk, "semantic": 0.0, "lexical": 0.0})["semantic"] = (
                hit.semantic_score
            )
        for hit in lexical:
            values.setdefault(hit.chunk.id, {"chunk": hit.chunk, "semantic": 0.0, "lexical": 0.0})["lexical"] = (
                hit.lexical_score
            )
        return values

    @staticmethod
    def _reasons(semantic_val: float, lexical_val: float, graph: float, preference: float) -> tuple[str, ...]:
        reasons = []
        if semantic_val > 0:
            reasons.append("semantic")
        if lexical_val > 0:
            reasons.append("lexical")
        if graph > 0:
            reasons.append("graph")
        if preference > 0:
            reasons.append("profile")
        return tuple(reasons)

    @staticmethod
    def _build_hit(
        chunk: Any,
        score: float,
        semantic_val: float,
        lexical_val: float,
        graph: float,
        reasons: tuple[str, ...],
    ) -> SearchHit:
        return SearchHit(
            chunk=chunk,
            score=score,
            semantic_score=semantic_val,
            lexical_score=lexical_val,
            graph_score=graph,
            reasons=reasons,
        )

    def _fuse_linear(
        self,
        values: Mapping[str, dict[str, Any]],
        graph_boosts: Mapping[str, float],
        preferred: Mapping[str, float],
        weights: tuple[float, float, float],
    ) -> list[SearchHit]:
        semantic_weight, lexical_weight, graph_weight = weights
        hits: list[SearchHit] = []
        for value in values.values():
            chunk = value["chunk"]
            graph = graph_boosts.get(chunk.document_id, 0.0)
            preference = preferred.get(chunk.document_id, 0.0)
            semantic_val = float(value["semantic"])
            lexical_val = float(value["lexical"])
            score = (
                semantic_weight * semantic_val + lexical_weight * lexical_val + graph_weight * graph + 0.05 * preference
            )
            reasons = self._reasons(semantic_val, lexical_val, graph, preference)
            hits.append(self._build_hit(chunk, score, semantic_val, lexical_val, graph, reasons))
        return hits

    def _fuse_rrf(
        self,
        values: Mapping[str, dict[str, Any]],
        semantic: Sequence[SearchHit],
        lexical: Sequence[SearchHit],
        graph_boosts: Mapping[str, float],
        preferred: Mapping[str, float],
        weights: tuple[float, float, float],
        rrf_k: int,
    ) -> list[SearchHit]:
        semantic_weight, lexical_weight, graph_weight = weights
        semantic_rank = self._rank_map(semantic)
        lexical_rank = self._rank_map(lexical)
        graph_rank = self._graph_ranking(values, graph_boosts)
        scored: list[tuple[Any, float, float, float, float, float]] = []
        for value in values.values():
            chunk = value["chunk"]
            graph = graph_boosts.get(chunk.document_id, 0.0)
            preference = preferred.get(chunk.document_id, 0.0)
            semantic_val = float(value["semantic"])
            lexical_val = float(value["lexical"])
            score = 0.05 * preference
            if chunk.id in semantic_rank:
                score += semantic_weight / (rrf_k + semantic_rank[chunk.id] + 1)
            if chunk.id in lexical_rank:
                score += lexical_weight / (rrf_k + lexical_rank[chunk.id] + 1)
            if chunk.id in graph_rank:
                score += graph_weight / (rrf_k + graph_rank[chunk.id] + 1)
            scored.append((chunk, score, semantic_val, lexical_val, graph, preference))
        normalized = self._normalize_scores({chunk.id: score for chunk, score, *_ in scored})
        return [
            self._build_hit(
                chunk,
                normalized[chunk.id],
                semantic_val,
                lexical_val,
                graph,
                self._reasons(semantic_val, lexical_val, graph, preference),
            )
            for chunk, score, semantic_val, lexical_val, graph, preference in scored
        ]

    @staticmethod
    def _rank_map(hits: Sequence[SearchHit]) -> dict[str, int]:
        return {hit.chunk.id: index for index, hit in enumerate(hits)}

    @staticmethod
    def _graph_ranking(values: Mapping[str, dict[str, Any]], graph_boosts: Mapping[str, float]) -> dict[str, int]:
        boosted = [
            (chunk_id, graph_boosts.get(value["chunk"].document_id, 0.0))
            for chunk_id, value in values.items()
            if graph_boosts.get(value["chunk"].document_id, 0.0) > 0.0
        ]
        boosted.sort(key=lambda item: (-item[1], item[0]))
        return {chunk_id: index for index, (chunk_id, _score) in enumerate(boosted)}

    @staticmethod
    def _preferred_documents(profile: JsonObject) -> dict[str, float]:
        values: dict[str, float] = {}
        for key, payload in profile.items():
            if not key.startswith("document:") or not isinstance(payload, Mapping):
                continue
            confidence = payload.get("confidence", 0.0)
            if isinstance(confidence, (float, int)) and confidence >= 0.5:
                values[key.removeprefix("document:")] = float(confidence)
        return values

    @staticmethod
    def graph_context(
        query: str,
        entities: Sequence[Entity],
        relations: Sequence[Relation],
        *,
        authorized_documents: frozenset[str] | None = None,
    ) -> tuple[dict[str, float], list[str], list[frozenset[str]]]:
        # Drop entities from unauthorized documents and any relation touching one, so a fact
        # is revealed only when an authorized document supports it. None leaves it unfiltered.
        # The third return value is the documents backing each returned fact, so a caller can
        # classify exactly the evidence a fact carries to the model, never merely boosted entities.
        if authorized_documents is not None:
            entities = [entity for entity in entities if entity.document_id in authorized_documents]
            allowed_ids = {entity.id for entity in entities}
            # A relation asserts a fact from its own document, so require that document to be authorized
            # too, not only its endpoint entities. Byte-identical for the shipped intra-document
            # extractors (a relation shares its endpoints' document), it closes a cross-document
            # relation from a custom extractor whose backing document the principal may not see.
            relations = [
                relation
                for relation in relations
                if relation.source_entity_id in allowed_ids
                and relation.target_entity_id in allowed_ids
                and relation.document_id in authorized_documents
            ]
        query_key = normalized_key(query)
        query_terms = set(tokens(query))
        matched = {
            entity.id
            for entity in entities
            if entity.normalized_name in query_key or bool(set(tokens(entity.normalized_name)) & query_terms)
        }
        if not matched:
            return {}, [], []
        by_id = {entity.id: entity for entity in entities}
        connected = set(matched)
        relevant_relations: list[Relation] = []
        for relation in relations:
            if relation.source_entity_id in matched or relation.target_entity_id in matched:
                connected.add(relation.source_entity_id)
                connected.add(relation.target_entity_id)
                relevant_relations.append(relation)
        document_scores: defaultdict[str, float] = defaultdict(float)
        for entity_id in connected:
            entity = by_id.get(entity_id)
            if entity is not None:
                document_scores[entity.document_id] += 1.0 if entity_id in matched else 0.4
        maximum = max(document_scores.values(), default=1.0)
        boosts = {document_id: min(1.0, score / maximum) for document_id, score in document_scores.items()}
        facts = []
        fact_documents: list[frozenset[str]] = []
        for relation in relevant_relations:
            source = by_id.get(relation.source_entity_id)
            target = by_id.get(relation.target_entity_id)
            if source is not None and target is not None:
                facts.append(f"{source.name} —{relation.kind}→ {target.name}")
                fact_documents.append(frozenset({source.document_id, target.document_id, relation.document_id}))
        return boosts, facts, fact_documents
