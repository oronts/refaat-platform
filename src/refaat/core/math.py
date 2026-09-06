from __future__ import annotations

import math
from collections.abc import Sequence


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def normalize_vector(values: Sequence[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return tuple(0.0 for _ in values)
    return tuple(value / norm for value in values)
