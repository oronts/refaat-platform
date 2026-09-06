from __future__ import annotations

from refaat.domain.authz import AccessPolicy, chunk_allowed, preference_scope
from refaat.domain.models import Feedback
from refaat.infrastructure.database.store import SQLStore


class PreferenceLearner:
    def __init__(self, store: SQLStore, access_policy: AccessPolicy | None = None) -> None:
        self.store = store
        self.access_policy = access_policy

    def learn(self, workspace_id: str, corpus_id: str, feedback: Feedback) -> None:
        if feedback.rating == 0:
            return
        scope = preference_scope(self.access_policy, feedback.principal, feedback.user_id)
        if scope == "anonymous":
            return
        positive = feedback.rating > 0
        for chunk_id in feedback.selected_chunk_ids[:20]:
            chunks = self.store.get_chunks(workspace_id, corpus_id, [chunk_id])
            if not chunks:
                continue
            # A forbidden chunk records nothing, so it is indistinguishable from a nonexistent one.
            if (
                self.access_policy is not None
                and feedback.principal is not None
                and not chunk_allowed(self.access_policy, feedback.principal, chunks[0].metadata)
            ):
                continue
            self.store.observe_preference(
                workspace_id=workspace_id,
                corpus_id=corpus_id,
                user_scope=scope,
                key=f"document:{chunks[0].document_id}",
                value={"source": "explicit_feedback"},
                positive=positive,
            )
