from refaat.core.text import STOPWORDS, STOPWORDS_BY_LANGUAGE, stopwords_for
from refaat.infrastructure.graph import CooccurrenceGraphExtractor

EXPECTED_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "also",
        "before",
        "between",
        "from",
        "have",
        "into",
        "more",
        "other",
        "that",
        "their",
        "these",
        "they",
        "this",
        "through",
        "using",
        "with",
        "your",
    }
)


def test_graph_stopwords_are_the_promoted_single_source_unchanged() -> None:
    assert stopwords_for("en") is STOPWORDS
    assert set(STOPWORDS) == set(EXPECTED_STOPWORDS)


def test_graph_extraction_still_filters_stopword_tokens() -> None:
    extractor = CooccurrenceGraphExtractor(max_entities=20, min_token_length=4)
    graph = extractor.extract(
        corpus_id="corpus_1",
        document_id="doc_1",
        text="these tokens appear appear with other other systems systems using using metrics metrics.",
    )
    concept_names = {entity.normalized_name for entity in graph.entities if entity.kind == "concept"}
    assert concept_names.isdisjoint({"these", "with", "other", "using"})
    assert "systems" in concept_names


def test_graph_extraction_has_stable_unique_ids() -> None:
    extractor = CooccurrenceGraphExtractor(max_entities=20, min_token_length=4)
    graph = extractor.extract(
        corpus_id="corpus_1",
        document_id="doc_1",
        text=(
            "Oronts builds Refaat for Production Intelligence. "
            "Refaat connects Production Intelligence with Oronts. "
            "Production Intelligence requires evidence."
        ),
    )
    repeated = extractor.extract(
        corpus_id="corpus_1",
        document_id="doc_1",
        text=(
            "Oronts builds Refaat for Production Intelligence. "
            "Refaat connects Production Intelligence with Oronts. "
            "Production Intelligence requires evidence."
        ),
    )

    assert graph == repeated
    assert len({entity.id for entity in graph.entities}) == len(graph.entities)
    assert len({relation.id for relation in graph.relations}) == len(graph.relations)
    assert all(relation.source_entity_id != relation.target_entity_id for relation in graph.relations)


_LANGUAGE_TEXT = "werden werden systeme systeme zwischen zwischen diensten diensten."


def test_untagged_default_matches_explicit_english_and_none() -> None:
    extractor = CooccurrenceGraphExtractor(max_entities=20, min_token_length=4)
    default = extractor.extract(corpus_id="corpus_1", document_id="doc_1", text=_LANGUAGE_TEXT)
    english = extractor.extract(corpus_id="corpus_1", document_id="doc_1", text=_LANGUAGE_TEXT, language="en")
    explicit_none = extractor.extract(corpus_id="corpus_1", document_id="doc_1", text=_LANGUAGE_TEXT, language=None)
    assert default == english
    assert default == explicit_none


def test_german_tag_drops_german_stopwords_while_english_keeps_them() -> None:
    extractor = CooccurrenceGraphExtractor(max_entities=20, min_token_length=4)
    german = extractor.extract(corpus_id="corpus_1", document_id="doc_1", text=_LANGUAGE_TEXT, language="de")
    english = extractor.extract(corpus_id="corpus_1", document_id="doc_1", text=_LANGUAGE_TEXT, language="en")

    german_concepts = {entity.normalized_name for entity in german.entities if entity.kind == "concept"}
    english_concepts = {entity.normalized_name for entity in english.entities if entity.kind == "concept"}

    assert {"werden", "zwischen"}.isdisjoint(german_concepts)
    assert {"werden", "zwischen"} <= english_concepts
    assert "systeme" in german_concepts
    assert "diensten" in german_concepts


def test_default_language_setting_selects_stopwords_for_untagged_documents() -> None:
    german_default = CooccurrenceGraphExtractor(max_entities=20, min_token_length=4, default_language="de")
    graph = german_default.extract(corpus_id="corpus_1", document_id="doc_1", text=_LANGUAGE_TEXT)
    concepts = {entity.normalized_name for entity in graph.entities if entity.kind == "concept"}
    assert {"werden", "zwischen"}.isdisjoint(concepts)
    assert "systeme" in concepts


def test_unconfigured_extractor_uses_module_default_and_is_byte_identical() -> None:
    default = CooccurrenceGraphExtractor(max_entities=20, min_token_length=4)
    passed_none = CooccurrenceGraphExtractor(max_entities=20, min_token_length=4, stopwords_by_language=None)
    assert default._stopwords_by_language is None
    text = "these tokens appear appear with other other systems systems using using metrics metrics."
    assert default.extract(corpus_id="corpus_1", document_id="doc_1", text=text) == passed_none.extract(
        corpus_id="corpus_1", document_id="doc_1", text=text
    )


def test_project_override_of_english_drops_custom_words() -> None:
    text = "systems systems metrics metrics evidence evidence pipelines pipelines."
    baseline = CooccurrenceGraphExtractor(max_entities=20, min_token_length=4)
    override = CooccurrenceGraphExtractor(
        max_entities=20,
        min_token_length=4,
        stopwords_by_language={"en": frozenset({"metrics", "systems"})},
    )
    base_concepts = {
        entity.normalized_name
        for entity in baseline.extract(corpus_id="corpus_1", document_id="doc_1", text=text).entities
        if entity.kind == "concept"
    }
    over_concepts = {
        entity.normalized_name
        for entity in override.extract(corpus_id="corpus_1", document_id="doc_1", text=text).entities
        if entity.kind == "concept"
    }
    assert {"metrics", "systems"} <= base_concepts
    assert {"metrics", "systems"}.isdisjoint(over_concepts)
    assert "pipelines" in over_concepts


def test_new_language_registry_drops_its_custom_words() -> None:
    text = "aparato aparato datos datos informe informe sistema sistema."
    registry = {**STOPWORDS_BY_LANGUAGE, "it": frozenset({"datos"})}
    extractor = CooccurrenceGraphExtractor(max_entities=20, min_token_length=4, stopwords_by_language=registry)
    concepts = {
        entity.normalized_name
        for entity in extractor.extract(corpus_id="corpus_1", document_id="doc_1", text=text, language="it").entities
        if entity.kind == "concept"
    }
    assert "datos" not in concepts
    assert "aparato" in concepts
