from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from refaat.core.math import cosine_similarity
from refaat.core.text import stable_digest

RAPTOR_ID_PREFIX = "raptor_summary_"


@dataclass(frozen=True, slots=True)
class ClusterItem:
    key: str
    embedding: tuple[float, ...]


def raptor_node_id(corpus_id: str, level: int, leaf_chunk_ids: Sequence[str]) -> str:
    """Content-addressed id for a summary node, in the reserved uncitable namespace.

    The id is a pure function of the corpus, the tree level, and the sorted set of
    real leaf chunks the node covers, so deterministic clustering reproduces the same
    id run to run and re-ingestion never duplicates a node. The ``raptor_summary_``
    prefix is disjoint from the ``chunk_`` prefix every real chunk id carries, so a
    node id can never collide with a stored chunk id.
    """
    return f"{RAPTOR_ID_PREFIX}{stable_digest((corpus_id, str(level), *sorted(leaf_chunk_ids)), length=48)}"


def cluster_items(
    items: Sequence[ClusterItem],
    *,
    threshold: float,
    max_cluster_size: int,
) -> list[tuple[ClusterItem, ...]]:
    """Group items into deterministic embedding-similarity clusters.

    Items are processed in content order (embedding first, key as a final tiebreak) so
    the grouping is reproducible run to run and independent both of input order and of
    volatile item keys such as freshly minted chunk ids. Each still-unassigned item in
    that order seeds a cluster and pulls in the remaining unassigned items whose cosine
    similarity to the seed is at least ``threshold``, nearest first and content-ordered
    on ties, up to ``max_cluster_size`` members. Every item lands in exactly one
    cluster, so a below-threshold item forms its own singleton and the leaf set is
    always fully covered.
    """
    if max_cluster_size < 1:
        raise ValueError("max_cluster_size must be at least 1")
    ordered = sorted(items, key=lambda item: (item.embedding, item.key))
    assigned: set[str] = set()
    clusters: list[tuple[ClusterItem, ...]] = []
    for seed in ordered:
        if seed.key in assigned:
            continue
        assigned.add(seed.key)
        members = [seed]
        if max_cluster_size > 1:
            candidates = sorted(
                (
                    (max(0.0, cosine_similarity(seed.embedding, other.embedding)), other)
                    for other in ordered
                    if other.key not in assigned
                ),
                key=lambda pair: (-pair[0], pair[1].embedding, pair[1].key),
            )
            for score, other in candidates:
                if len(members) >= max_cluster_size or score < threshold:
                    break
                members.append(other)
                assigned.add(other.key)
        clusters.append(tuple(sorted(members, key=lambda item: item.key)))
    return clusters
