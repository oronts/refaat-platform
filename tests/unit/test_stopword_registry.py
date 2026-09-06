from refaat.core.text import (
    QUERY_STOPWORDS,
    STOPWORDS,
    STOPWORDS_BY_LANGUAGE,
    query_stopwords_for,
    stopwords_for,
)


def test_english_resolves_to_the_identical_frozenset_objects() -> None:
    assert stopwords_for("en") is STOPWORDS
    assert query_stopwords_for("en") is QUERY_STOPWORDS


def test_none_unknown_and_variants_fall_back_to_english() -> None:
    for code in (None, "", "EN", "en-US", "en_GB", "zz", "xx-YY"):
        assert stopwords_for(code) is STOPWORDS
        assert query_stopwords_for(code) is QUERY_STOPWORDS


def test_curated_non_english_sets_are_selected_and_non_empty() -> None:
    for code in ("de", "fr", "es", "ar"):
        graph_set = stopwords_for(code)
        query_set = query_stopwords_for(code)
        assert graph_set
        assert query_set
        assert graph_set is not STOPWORDS
        assert query_set is not QUERY_STOPWORDS


def test_primary_subtag_normalization_selects_the_language() -> None:
    assert stopwords_for("DE-de") is stopwords_for("de")
    assert stopwords_for("fr_FR") is stopwords_for("fr")
    assert stopwords_for("  es  ") is stopwords_for("es")


def test_no_registry_is_identity_to_the_module_default() -> None:
    assert stopwords_for("en", None) is STOPWORDS
    assert stopwords_for("en") is STOPWORDS
    assert query_stopwords_for("en", None) is QUERY_STOPWORDS
    assert stopwords_for("zz", None) is STOPWORDS


def test_registry_override_replaces_an_existing_language() -> None:
    registry = {**STOPWORDS_BY_LANGUAGE, "en": frozenset({"widget", "sprocket"})}
    resolved = stopwords_for("en", registry)
    assert resolved == frozenset({"widget", "sprocket"})
    assert resolved is not STOPWORDS


def test_registry_adds_a_brand_new_language_code() -> None:
    italian = frozenset({"come", "della", "sono", "questo"})
    registry = {**STOPWORDS_BY_LANGUAGE, "it": italian}
    assert stopwords_for("it", registry) is italian
    assert stopwords_for("it-IT", registry) is italian


def test_registry_falls_back_to_its_own_english_then_module() -> None:
    only_en = {"en": frozenset({"foo"})}
    assert stopwords_for("zz", only_en) == frozenset({"foo"})
    without_en = {"de": frozenset({"und"})}
    assert stopwords_for("zz", without_en) is STOPWORDS
    assert query_stopwords_for("zz", without_en) is QUERY_STOPWORDS
