from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError

from refaat.application.runtime import Refaat

pytestmark = pytest.mark.integration


def test_observe_preference_retries_on_write_conflict(runtime: Refaat, monkeypatch: pytest.MonkeyPatch) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    store = runtime.store
    scope = "user:alice"

    store.observe_preference(
        workspace_id=project.workspace_id,
        corpus_id=project.corpus_id,
        user_scope=scope,
        key="tone",
        value="formal",
        positive=True,
    )

    real = store._observe_preference_once
    calls = {"n": 0}

    def flaky(**kwargs: object) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise IntegrityError("INSERT", {}, Exception("duplicate key"))
        real(**kwargs)

    monkeypatch.setattr(store, "_observe_preference_once", flaky)
    store.observe_preference(
        workspace_id=project.workspace_id,
        corpus_id=project.corpus_id,
        user_scope=scope,
        key="tone",
        value="formal",
        positive=True,
    )

    assert calls["n"] == 2
    preferences = store.get_preferences(project.workspace_id, project.corpus_id, scope)
    assert preferences["tone"]["observations"] == 2


def test_observe_preference_reraises_after_persistent_conflict(
    runtime: Refaat, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    store = runtime.store

    def always_conflict(**kwargs: object) -> None:
        raise IntegrityError("INSERT", {}, Exception("duplicate key"))

    monkeypatch.setattr(store, "_observe_preference_once", always_conflict)
    with pytest.raises(IntegrityError):
        store.observe_preference(
            workspace_id=project.workspace_id,
            corpus_id=project.corpus_id,
            user_scope="user:bob",
            key="tone",
            value="formal",
            positive=True,
        )
