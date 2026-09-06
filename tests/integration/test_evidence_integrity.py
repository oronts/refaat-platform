from __future__ import annotations

import pytest

from refaat.application.runtime import RuntimeBuilder
from refaat.config import Settings
from refaat.core.session import SessionCompactor, tag_turn_metadata
from refaat.security import sanitize_source_url

pytestmark = pytest.mark.integration


def test_sanitize_source_url_strips_credentials_and_optional_query() -> None:
    assert (
        sanitize_source_url("https://user:pass@host.test/path?token=secret#frag", drop_query=True)
        == "https://host.test/path"
    )
    assert (
        sanitize_source_url("https://user:pass@host.test/path?token=secret", drop_query=False)
        == "https://host.test/path?token=secret"
    )
    assert sanitize_source_url("file:///local/file.txt", drop_query=True) == "file:///local/file.txt"
    assert sanitize_source_url("plain-source-id", drop_query=True) == "plain-source-id"


def test_session_compaction_removes_only_same_scope_turns(settings: Settings) -> None:
    runtime = RuntimeBuilder(settings).build()
    try:
        ws = runtime.create_workspace("acme", "Acme")
        corpus = runtime.create_corpus(ws, "c")
        session_id = runtime.store.create_session(ws, corpus, user_id="shared", goal=None)
        alice, bob = "alice-scope", "bob-scope"
        for i in range(3):
            runtime.store.append_session_turn(
                ws, session_id, "user", f"alice message {i}", tag_turn_metadata({}, alice)
            )
            runtime.store.append_session_turn(ws, session_id, "user", f"bob message {i}", tag_turn_metadata({}, bob))
        compactor = SessionCompactor(runtime.store, settings)
        assert compactor.compact_if_needed(ws, session_id, auth_scope=alice) is True
        remaining = [str(turn["content"]) for turn in runtime.store.list_session_turns(ws, session_id)]
        assert sum(1 for content in remaining if "bob message" in content) == 3
    finally:
        runtime.close()
