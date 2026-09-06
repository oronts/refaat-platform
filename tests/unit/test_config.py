from __future__ import annotations

from pathlib import Path

import pytest

from refaat.config import Settings


def test_development_configuration_accepts_safe_defaults(tmp_path: Path) -> None:
    settings = Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path / 'app.db'}",
        blob_root=tmp_path / "blobs",
        connector_file_root=tmp_path / "connectors",
        skill_root=tmp_path / "skills",
        api_key_pepper="test-pepper",
    )
    settings.validate_for_startup()
    assert settings.bm25_k1 == 1.2
    assert settings.bm25_b == 0.75
    assert settings.query_expansion is True
    assert settings.max_query_variants == 4
    assert settings.compression_enabled is False
    assert settings.compression_min_salience == 0.0
    assert settings.contextual_retrieval is False
    assert settings.chunking_strategy == "flat"
    assert settings.sentence_window_radius == 1


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"chunk_overlap_tokens": 420}, "overlap"),
        ({"retained_session_turns": 20}, "retained"),
        ({"semantic_cache_threshold": 1.1}, "threshold"),
        (
            {"semantic_weight": 0.0, "lexical_weight": 0.0, "graph_weight": 0.0},
            "weights",
        ),
        ({"query_top_k": 0}, "positive"),
        ({"graph_max_entities": 0}, "positive"),
        ({"rrf_k": 0}, "positive"),
        ({"autocut_threshold": 2.0}, "autocut"),
        ({"bm25_k1": 6.0}, "bm25_k1"),
        ({"bm25_b": 2.0}, "bm25_b"),
        ({"max_query_variants": 0}, "max_query_variants"),
        ({"max_query_variants": 9}, "max_query_variants"),
        ({"compression_min_salience": 1.5}, "compression"),
        ({"compression_min_salience": -0.1}, "compression"),
        ({"sentence_window_radius": 9}, "sentence_window"),
        ({"sentence_window_radius": -1}, "sentence_window"),
        ({"request_timeout_seconds": 0.0}, "timeout"),
        ({"fine_tune_poll_seconds": 0}, "fine_tune_poll"),
        ({"rlm_max_input_chunks": 0}, "RLM_MAX_INPUT_CHUNKS"),
        ({"rlm_max_input_chunks": -1}, "RLM_MAX_INPUT_CHUNKS"),
        ({"blob_backend": "s3", "s3_bucket": None}, "s3_bucket"),
        ({"answer_cache_backend": "redis"}, "REDIS_URL"),
        (
            {"model_call_cache_backend": "redis", "redis_url": "redis://localhost:6379/0"},
            "MODEL_CALL_CACHE_ENABLED",
        ),
        ({"redis_timeout_seconds": float("nan")}, "finite"),
        ({"redis_timeout_seconds": 0.0}, "finite"),
        ({"stopwords_by_language": {"e": ["und"]}}, "stopwords_by_language"),
        ({"stopwords_by_language": {"it": []}}, "stopwords_by_language"),
        ({"stopwords_by_language": {"it": ["   "]}}, "stopwords_by_language"),
        ({"stopwords_by_language": {"it": ["della", ""]}}, "stopwords_by_language"),
    ],
)
def test_invalid_numeric_and_backend_configuration_is_rejected(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=rf"(?i){message}"):
        Settings(**overrides).validate_for_startup()


def test_stopwords_by_language_accepts_a_valid_override_and_defaults_none() -> None:
    assert Settings().stopwords_by_language is None
    settings = Settings(
        environment="test",
        api_key_pepper="test-pepper",
        stopwords_by_language={"it": ["della", "sono"], "en": ["widget"]},
    )
    settings.validate_for_startup()
    assert settings.stopwords_by_language == {"it": ["della", "sono"], "en": ["widget"]}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({}, "pepper"),
        ({"api_key_pepper": "short"}, "pepper"),
        ({"database_url": "sqlite:///unsafe.db", "api_key_pepper": "x" * 24}, "sqlite"),
        (
            {
                "database_url": "postgresql://db/refaat",
                "api_key_pepper": "x" * 24,
                "auto_create_schema": True,
            },
            "schema",
        ),
        (
            {
                "database_url": "postgresql://db/refaat",
                "api_key_pepper": "x" * 24,
                "auto_create_schema": False,
                "cors_origins": ["*"],
            },
            "cors",
        ),
        (
            {
                "database_url": "postgresql://db/refaat",
                "api_key_pepper": "x" * 24,
                "auto_create_schema": False,
                "bootstrap_token": "short",
            },
            "bootstrap",
        ),
    ],
)
def test_production_rejects_unsafe_configuration(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=rf"(?i){message}"):
        Settings(environment="production", **overrides).validate_for_startup()


def test_production_accepts_explicit_hardened_configuration() -> None:
    Settings(
        environment="production",
        database_url="postgresql://db/refaat",
        auto_create_schema=False,
        api_key_pepper="p" * 32,
        bootstrap_token="b" * 32,
        cors_origins=["https://console.example.com"],
    ).validate_for_startup()
