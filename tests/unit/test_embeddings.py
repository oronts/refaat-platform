from refaat.core.math import cosine_similarity
from refaat.infrastructure.embeddings import DeterministicEmbedder


def test_deterministic_embeddings_are_stable_and_normalized() -> None:
    embedder = DeterministicEmbedder(dimensions=64)
    left, right, unrelated = embedder.embed(
        [
            "production deployment rollback plan",
            "rollback plan for production deployment",
            "fresh oranges and garden soil",
        ]
    )

    assert left == embedder.embed(["production deployment rollback plan"])[0]
    assert len(left) == 64
    assert cosine_similarity(left, right) > cosine_similarity(left, unrelated)
    assert 0.99 <= cosine_similarity(left, left) <= 1.0


def test_empty_embedding_is_zero_vector() -> None:
    vector = DeterministicEmbedder(dimensions=32).embed([""])[0]
    assert set(vector) == {0.0}
