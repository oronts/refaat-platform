from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from refaat.core.raptor import ClusterItem, cluster_items, raptor_node_id
from refaat.domain.models import Chunk, JsonObject, RaptorNode
from refaat.infrastructure.database.store import SQLStore
from refaat.ports import Embedder, Summarizer


class RaptorTreeBuilder:
    """Builds and persists a corpus's RAPTOR summary tree at ingestion time.

    The tree is a pure function of the current stored leaf chunks: deterministic
    clustering groups the leaves, each cluster is summarized and embedded in the
    shared vector space, and the process recurses up to ``max_depth`` levels. Every
    rebuild deletes the corpus's prior nodes and writes the freshly derived set, so
    re-ingesting unchanged content reproduces byte-identical nodes and never
    duplicates them. A corpus smaller than ``min_leaves``, or one whose leaves never
    cluster (no reduction), yields an empty node set rather than trivial singleton
    summaries.
    """

    def __init__(
        self,
        *,
        store: SQLStore,
        embedder: Embedder,
        summarizer: Summarizer,
        max_depth: int = 3,
        cluster_threshold: float = 0.0,
        max_cluster_size: int = 5,
        min_leaves: int = 2,
    ) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if min_leaves < 2:
            raise ValueError("min_leaves must be at least 2")
        self.store = store
        self.embedder = embedder
        self.summarizer = summarizer
        self.max_depth = max_depth
        self.cluster_threshold = cluster_threshold
        self.max_cluster_size = max_cluster_size
        self.min_leaves = min_leaves

    @property
    def provider_id(self) -> str:
        return (
            f"raptor-v1:{self.summarizer.provider_id}:d{self.max_depth}"
            f":c{self.max_cluster_size}:t{self.cluster_threshold}"
        )

    def rebuild(self, workspace_id: str, corpus_id: str) -> int:
        leaves = self.store.list_chunks(workspace_id, corpus_id)
        nodes = self._build(workspace_id, corpus_id, leaves) if len(leaves) >= self.min_leaves else []
        self.store.replace_raptor_nodes(
            workspace_id,
            corpus_id,
            [self._row(workspace_id, node) for node in nodes],
        )
        return len(nodes)

    def _build(self, workspace_id: str, corpus_id: str, leaves: Sequence[Chunk]) -> list[RaptorNode]:
        items = [ClusterItem(key=chunk.id, embedding=chunk.embedding) for chunk in leaves]
        texts = {chunk.id: chunk.text for chunk in leaves}
        coverage: dict[str, tuple[str, ...]] = {chunk.id: (chunk.id,) for chunk in leaves}
        nodes: list[RaptorNode] = []
        parents: dict[str, str] = {}
        level = 1
        while level <= self.max_depth and len(items) > 1:
            clusters = cluster_items(
                items,
                threshold=self.cluster_threshold,
                max_cluster_size=self.max_cluster_size,
            )
            if len(clusters) >= len(items):
                break
            next_items: list[ClusterItem] = []
            next_texts: dict[str, str] = {}
            next_coverage: dict[str, tuple[str, ...]] = {}
            for cluster in clusters:
                child_keys = [item.key for item in cluster]
                covered = tuple(sorted({leaf for key in child_keys for leaf in coverage[key]}))
                node_id = raptor_node_id(corpus_id, level, covered)
                summary = self.summarizer.summarize([texts[key] for key in child_keys], workspace_id=workspace_id)
                embedding = self.embedder.embed([summary])[0]
                nodes.append(
                    RaptorNode(
                        id=node_id,
                        corpus_id=corpus_id,
                        level=level,
                        parent_node_id=None,
                        summary=summary,
                        embedding=embedding,
                        child_node_ids=tuple(child_keys),
                        leaf_chunk_ids=covered,
                    )
                )
                for key in child_keys:
                    parents[key] = node_id
                next_items.append(ClusterItem(key=node_id, embedding=embedding))
                next_texts[node_id] = summary
                next_coverage[node_id] = covered
            items, texts, coverage = next_items, next_texts, next_coverage
            level += 1
        return [replace(node, parent_node_id=parents.get(node.id)) for node in nodes]

    def _row(self, workspace_id: str, node: RaptorNode) -> JsonObject:
        return {
            "id": node.id,
            "workspace_id": workspace_id,
            "corpus_id": node.corpus_id,
            "level": node.level,
            "parent_node_id": node.parent_node_id,
            "summary": node.summary,
            "embedding": list(node.embedding),
            "embedding_model": self.embedder.model_id,
            "child_node_ids": list(node.child_node_ids),
            "leaf_chunk_ids": list(node.leaf_chunk_ids),
        }
