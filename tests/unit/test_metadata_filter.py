from __future__ import annotations

from refaat.domain.models import MetadataFilter, MetadataPredicate


def test_from_spec_none_and_empty_yield_none() -> None:
    assert MetadataFilter.from_spec(None) is None
    assert MetadataFilter.from_spec({}) is None
    assert MetadataFilter.from_spec({"a": []}) is None


def test_from_spec_scalar_becomes_equality() -> None:
    metadata_filter = MetadataFilter.from_spec({"category": "security"})
    assert metadata_filter is not None
    assert metadata_filter.predicates == (MetadataPredicate(field="category", values=("security",)),)


def test_from_spec_list_becomes_membership_sorted_by_field() -> None:
    metadata_filter = MetadataFilter.from_spec({"tier": [2, 1], "category": "security"})
    assert metadata_filter is not None
    fields = [predicate.field for predicate in metadata_filter.predicates]
    assert fields == ["category", "tier"]
    tier = next(predicate for predicate in metadata_filter.predicates if predicate.field == "tier")
    assert tier.values == (2, 1)


def test_matches_present_and_equal() -> None:
    metadata_filter = MetadataFilter.from_spec({"category": "security"})
    assert metadata_filter is not None
    assert metadata_filter.matches({"category": "security"})
    assert not metadata_filter.matches({"category": "ops"})


def test_matches_missing_key_is_false() -> None:
    metadata_filter = MetadataFilter.from_spec({"category": "security"})
    assert metadata_filter is not None
    assert not metadata_filter.matches({})


def test_matches_membership() -> None:
    metadata_filter = MetadataFilter.from_spec({"tier": [1, 2]})
    assert metadata_filter is not None
    assert metadata_filter.matches({"tier": 2})
    assert not metadata_filter.matches({"tier": 3})


def test_matches_is_type_exact_string_not_int() -> None:
    metadata_filter = MetadataFilter.from_spec({"tier": 1})
    assert metadata_filter is not None
    assert not metadata_filter.matches({"tier": "1"})
    assert metadata_filter.matches({"tier": 1})


def test_matches_bool_is_not_int() -> None:
    metadata_filter = MetadataFilter.from_spec({"flag": True})
    assert metadata_filter is not None
    assert not metadata_filter.matches({"flag": 1})
    assert metadata_filter.matches({"flag": True})


def test_matches_is_and_over_predicates() -> None:
    metadata_filter = MetadataFilter.from_spec({"category": "security", "tier": 1})
    assert metadata_filter is not None
    assert metadata_filter.matches({"category": "security", "tier": 1})
    assert not metadata_filter.matches({"category": "security", "tier": 2})


def test_as_json_sorts_predicates_by_field() -> None:
    metadata_filter = MetadataFilter(
        predicates=(
            MetadataPredicate(field="tier", values=(1,)),
            MetadataPredicate(field="category", values=("security",)),
        )
    )
    fields = [predicate["field"] for predicate in metadata_filter.as_json()["predicates"]]
    assert fields == ["category", "tier"]


def test_as_json_is_order_insensitive_for_values() -> None:
    left = MetadataFilter((MetadataPredicate(field="tier", values=(1, 2, 3)),))
    right = MetadataFilter((MetadataPredicate(field="tier", values=(3, 1, 2)),))
    assert left.as_json() == right.as_json()


def test_as_json_handles_heterogeneous_values_without_typeerror() -> None:
    metadata_filter = MetadataFilter((MetadataPredicate(field="mixed", values=(1, "a", True)),))
    rendered = metadata_filter.as_json()
    assert rendered == metadata_filter.as_json()
    assert rendered["predicates"][0]["field"] == "mixed"
