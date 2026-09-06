from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence

from refaat.core.sparse import BM25Params, bm25_scores


def _legacy_scores(query_terms: Sequence[str], documents: Mapping[str, Sequence[str]]) -> dict[str, float]:
    """The pre-refactor LocalSearchIndex.lexical_search math, verbatim."""
    if not query_terms or not documents:
        return {}
    document_frequency: Counter[str] = Counter()
    tokenized = {chunk_id: list(terms) for chunk_id, terms in documents.items()}
    for values in tokenized.values():
        document_frequency.update(set(values))
    count = len(documents)
    average_length = sum(len(value) for value in tokenized.values()) / count
    scores: dict[str, float] = {}
    for chunk_id, terms in tokenized.items():
        frequencies = Counter(terms)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if frequency == 0:
                continue
            inverse = math.log(1 + (count - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            denominator = frequency + 1.2 * (1 - 0.75 + 0.75 * len(terms) / max(average_length, 1.0))
            score += inverse * (frequency * 2.2) / denominator
        if score > 0.0:
            scores[chunk_id] = score
    maximum = max(scores.values(), default=1.0)
    return {chunk_id: value / maximum for chunk_id, value in scores.items()}


DEFAULT = BM25Params(k1=1.2, b=0.75)


def test_parity_with_legacy_formula_bit_for_bit() -> None:
    documents = {
        "a": ["production", "release", "requires", "review"],
        "b": ["release", "security", "approval", "threat", "review"],
        "c": ["the", "canary", "watches", "the", "release"],
    }
    query_terms = ["release", "review", "security"]
    assert bm25_scores(query_terms, documents, DEFAULT) == _legacy_scores(query_terms, documents)


def test_idf_ranks_rare_over_common() -> None:
    documents = {
        "rare": ["quantum", "filler"],
        "common": ["release", "filler"],
        "x": ["release", "text"],
        "y": ["release", "notes"],
        "z": ["release", "plan"],
    }
    scores = bm25_scores(["quantum", "release"], documents, DEFAULT)
    assert scores["rare"] > scores["common"]


def test_term_frequency_saturates() -> None:
    documents = {
        "once": ["alpha", "beta", "gamma", "delta"],
        "twice": ["alpha", "alpha", "gamma", "delta"],
    }
    scores = bm25_scores(["alpha"], documents, DEFAULT)
    assert scores["twice"] == 1.0
    assert 0.5 < scores["once"] < 1.0


def test_length_parameter_b_penalizes_long_documents() -> None:
    documents = {
        "short": ["x"],
        "long": ["x", "y", "y", "y", "y"],
    }
    penalized = bm25_scores(["x"], documents, BM25Params(k1=1.2, b=1.0))
    assert penalized["short"] == 1.0
    assert penalized["long"] < 1.0

    neutral = bm25_scores(["x"], documents, BM25Params(k1=1.2, b=0.0))
    assert neutral["short"] == neutral["long"] == 1.0


def test_max_normalization_bounds_scores() -> None:
    documents = {"a": ["release", "review"], "b": ["review", "plan"]}
    scores = bm25_scores(["release", "review"], documents, DEFAULT)
    assert max(scores.values()) == 1.0
    assert all(0.0 < value <= 1.0 for value in scores.values())


def test_empty_query_or_corpus_returns_empty() -> None:
    assert bm25_scores([], {"a": ["release"]}, DEFAULT) == {}
    assert bm25_scores(["release"], {}, DEFAULT) == {}


def test_determinism_across_runs() -> None:
    documents = {"a": ["release", "review"], "b": ["review", "plan", "release"]}
    first = bm25_scores(["release", "review"], documents, DEFAULT)
    second = bm25_scores(["release", "review"], documents, DEFAULT)
    assert first == second


def test_non_matching_query_yields_empty() -> None:
    documents = {"a": ["release", "review"]}
    assert bm25_scores(["absent"], documents, DEFAULT) == {}
