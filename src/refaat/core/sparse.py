from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BM25Params:
    k1: float
    b: float


def bm25_scores(
    query_terms: Sequence[str],
    documents: Mapping[str, Sequence[str]],
    params: BM25Params,
) -> dict[str, float]:
    """Score tokenized documents against tokenized query terms with BM25.

    Returns a chunk-id keyed mapping of positive scores divide-by-max normalized
    to (0, 1]. An empty query or empty corpus yields an empty mapping. With
    ``BM25Params(1.2, 0.75)`` this reproduces the pre-refactor
    ``LocalSearchIndex.lexical_search`` output bit for bit: the numerator
    coefficient is ``k1 + 1`` (2.2), the average-document-length guard is
    ``max(avgdl, 1.0)``, and document frequency counts distinct tokens per
    document.
    """
    if not query_terms or not documents:
        return {}
    document_frequency: Counter[str] = Counter()
    for terms in documents.values():
        document_frequency.update(set(terms))
    count = len(documents)
    average_length = sum(len(terms) for terms in documents.values()) / count
    raw: dict[str, float] = {}
    for chunk_id, terms in documents.items():
        frequencies = Counter(terms)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if frequency == 0:
                continue
            inverse = math.log(1 + (count - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            denominator = frequency + params.k1 * (1 - params.b + params.b * len(terms) / max(average_length, 1.0))
            score += inverse * (frequency * (params.k1 + 1.0)) / denominator
        if score > 0.0:
            raw[chunk_id] = score
    maximum = max(raw.values(), default=1.0)
    return {chunk_id: score / maximum for chunk_id, score in raw.items()}
