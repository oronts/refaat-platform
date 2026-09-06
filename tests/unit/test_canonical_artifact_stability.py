from __future__ import annotations

from refaat.canonical import canonical_digest, canonical_json
from refaat.programs.artifact import ProgramArtifact
from refaat.programs.program import GroundedRagProgram


def test_grounded_rag_artifact_id_is_byte_identical_after_canonical_extraction() -> None:
    """Pin a known existing artifact id so the shared canonical helper never drifts.

    These digests predate extracting the canonical-JSON plus sha256 helper into
    refaat.canonical. If either changes, every stored artifact id would silently
    break, so this test fails loudly instead.
    """
    assert (
        GroundedRagProgram().artifact().artifact_id
        == "sha256:b76d73f8d956bcd38ccc9d4c24372b07b387647a189f3a9e300cded186febbb4"
    )

    fixture = ProgramArtifact(
        program="p",
        backend="native",
        definition_hash="dh",
        immutable_policy_hash="ph",
        signatures={"input": "i", "output": "o"},
        tunables={"top_k": None},
        validators=("grounding",),
        dependency_versions={"refaat": "0.1.0"},
    )
    assert fixture.artifact_id == "sha256:0aa531a8e454fc3b2852b38211d96e7678c254b1c2ab2301fea1848d2dd3d4c5"


def test_canonical_helpers_are_sorted_and_compact() -> None:
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert canonical_digest({"a": 2, "b": 1}) == canonical_digest({"b": 1, "a": 2})


def test_canonical_json_rejects_non_finite_floats() -> None:
    import math

    import pytest

    with pytest.raises(ValueError, match="JSON compliant"):
        canonical_json({"x": math.nan})
    with pytest.raises(ValueError, match="JSON compliant"):
        canonical_json({"x": math.inf})
