from __future__ import annotations

from refaat.application.composition import (
    OmittedComponent,
    ReproducedComponent,
    RuntimeComposition,
    ShadowComposition,
)
from refaat.application.runtime import RuntimeBuilder
from refaat.infrastructure.embeddings import DeterministicEmbedder


def test_shadow_composition_to_json_shape() -> None:
    composition = ShadowComposition(
        reproduced=(ReproducedComponent("embedder"), ReproducedComponent("generator")),
        omitted=(OmittedComponent("raptor_builder", "store-bound"),),
    )
    assert composition.to_json() == {
        "reproduced": ["embedder", "generator"],
        "omitted": [{"component": "raptor_builder", "reason": "store-bound"}],
    }
    assert composition.is_empty() is False


def test_empty_shadow_composition_is_empty() -> None:
    empty = ShadowComposition()
    assert empty.is_empty() is True
    assert empty.to_json() == {"reproduced": [], "omitted": []}


def test_capture_and_reproduce_injected_embedder_marks_it_borrowed() -> None:
    embedder = DeterministicEmbedder()
    source = RuntimeBuilder().with_embedder(embedder)
    composition = source._capture_composition()
    assert composition.embedder is embedder

    target = RuntimeBuilder()
    report = target.reproduce_from(composition)
    assert "embedder" in [component.name for component in report.reproduced]
    assert id(embedder) in target._borrowed_component_ids
    assert target._custom_embedder is embedder


def test_reproduce_refuses_and_reports_refused_slots() -> None:
    composition = RuntimeComposition(raptor_builder=object(), audit_sink=object())
    target = RuntimeBuilder()
    report = target.reproduce_from(composition)
    omitted_names = {component.name for component in report.omitted}
    assert "raptor_builder" in omitted_names
    assert "audit_sink" in omitted_names
    assert target._custom_raptor_builder is None
    assert target._audit_sink is None


def test_empty_composition_reproduces_nothing() -> None:
    report = RuntimeBuilder().reproduce_from(RuntimeComposition())
    assert report.is_empty() is True
