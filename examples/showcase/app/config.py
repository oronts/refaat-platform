"""Settings loader for the Oronts showcase application.

One env-switched code path. With no API key the app is fully offline and
deterministic: SQLite storage, hash embeddings, an extractive generator, and a
co-occurrence graph. When ``REFAAT_OPENAI_API_KEY`` is present the generation and
embedding roles switch to the OpenAI REST API, while storage stays on SQLite and
vector search stays in process. There are no external services in either mode: no
Postgres, no pgvector, no Redis, no search cluster, and no local model server.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from refaat.config import Settings


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class ShowcaseConfig:
    """Resolved showcase configuration plus the refaat Settings it produced."""

    data_dir: Path
    settings: Settings
    online: bool
    generation_model: str
    embedding_model: str

    @property
    def mode(self) -> str:
        return "openai" if self.online else "offline"


def _offline_settings(data_dir: Path) -> dict[str, object]:
    return {
        "embedding_provider": "deterministic",
        "embedding_model": "refaat-hash-v1",
        "embedding_dimensions": 384,
        "generation_provider": "extractive",
        "generation_model": "extractive-v1",
        "graph_provider": "cooccurrence",
        # Hash-embedding similarity scores are not comparable to a real cosine
        # threshold, so abstention is forced off offline (explicitly, so an env var
        # can never enable it). This keeps the offline entrypoint and gate identical.
        "abstain_below_score": 0.0,
    }


def _online_settings(data_dir: Path) -> dict[str, object]:
    generation_model = os.environ.get("REFAAT_GENERATION_MODEL", "gpt-4o-mini")
    embedding_model = os.environ.get("REFAAT_EMBEDDING_MODEL", "text-embedding-3-small")
    dimensions = int(os.environ.get("REFAAT_EMBEDDING_DIMENSIONS", "1536"))
    return {
        "embedding_provider": "openai_compatible",
        "embedding_model": embedding_model,
        "embedding_dimensions": dimensions,
        "generation_provider": "openai_compatible",
        "generation_model": generation_model,
        # The co-occurrence graph stays deterministic and free even online, so a
        # multi-hop graph answer never depends on a paid extraction call.
        "graph_provider": "cooccurrence",
        "openai_api_key": os.environ["REFAAT_OPENAI_API_KEY"],
        "openai_base_url": os.environ.get("REFAAT_OPENAI_BASE_URL", "https://api.openai.com/v1"),
        # A real text-embedding-3-small cosine floor: an out-of-corpus question scores
        # below it and returns an honest abstention instead of a stitched answer.
        "abstain_below_score": float(os.environ.get("REFAAT_ABSTAIN_BELOW_SCORE", "0.3")),
        # Published USD rates per 1000 tokens for the default models, so the
        # observability panel shows a true cost online. gpt-4o-mini: input 0.00015,
        # output 0.0006. text-embedding-3-small: input 0.00002, no completion tokens.
        "model_prices": {
            generation_model: {"input_per_1k": 0.00015, "output_per_1k": 0.0006},
            embedding_model: {"input_per_1k": 0.00002, "output_per_1k": 0.0},
        },
    }


def _default_data_dir() -> str:
    return os.environ.get("REFAAT_SHOWCASE_DATA_DIR") or os.environ.get("DATA_DIR") or "./data/showcase"


def load_config(data_dir: str | Path | None = None) -> ShowcaseConfig:
    resolved = Path(data_dir or _default_data_dir()).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)

    online = bool(os.environ.get("REFAAT_OPENAI_API_KEY"))
    model_overrides = _online_settings(resolved) if online else _offline_settings(resolved)

    settings = Settings(
        environment="development",
        database_url=f"sqlite+pysqlite:///{resolved / 'refaat.db'}",
        auto_create_schema=True,
        blob_backend="local",
        blob_root=resolved / "blobs",
        connector_file_root=resolved / "connectors",
        skill_root=resolved / "skills",
        api_key_pepper=os.environ.get("REFAAT_API_KEY_PEPPER", "showcase-api-key-pepper-with-enough-entropy"),
        bootstrap_token=os.environ.get("REFAAT_BOOTSTRAP_TOKEN", "showcase-bootstrap-token-with-enough-entropy"),
        trusted_hosts=["127.0.0.1", "localhost", "testserver"],
        # The in-memory rate limiter stays the only limiter; no Redis backend.
        redis_url=None,
        rate_limit_requests=int(os.environ.get("REFAAT_RATE_LIMIT_REQUESTS", "10000")),
        # Feature toggles that make the retrieval catalog meaningful. All degrade to
        # the deterministic default when the offline stack is active.
        reranker_provider="lexical",
        hype_enabled=_flag("REFAAT_HYPE_ENABLED", True),
        raptor_enabled=_flag("REFAAT_RAPTOR_ENABLED", True),
        contextual_retrieval=_flag("REFAAT_CONTEXTUAL_RETRIEVAL", True),
        query_transform_provider="sub_query",
        # Offline the deterministic embedder and extractive generator are never wrapped,
        # so the cache is a transparent no-op; online it serves repeated real model calls.
        model_call_cache_enabled=True,
        default_chunk_tokens=int(os.environ.get("REFAAT_DEFAULT_CHUNK_TOKENS", "110")),
        chunk_overlap_tokens=int(os.environ.get("REFAAT_CHUNK_OVERLAP_TOKENS", "24")),
        query_token_budget=int(os.environ.get("REFAAT_QUERY_TOKEN_BUDGET", "1200")),
        query_top_k=int(os.environ.get("REFAAT_QUERY_TOP_K", "10")),
        session_compaction_turns=6,
        retained_session_turns=3,
        log_json=False,
        log_level=os.environ.get("REFAAT_LOG_LEVEL", "WARNING"),
        **model_overrides,  # type: ignore[arg-type]
    )
    settings.validate_for_startup()

    return ShowcaseConfig(
        data_dir=resolved,
        settings=settings,
        online=online,
        generation_model=str(model_overrides["generation_model"]),
        embedding_model=str(model_overrides["embedding_model"]),
    )


__all__ = ["ShowcaseConfig", "load_config"]
