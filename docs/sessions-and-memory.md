# Sessions, memory, compaction, and preferences

Refaat separates conversational session state from durable user preferences. Both can affect context construction, but neither is treated as source evidence.

## Session lifecycle

A session belongs to one workspace and one corpus and may contain:

- an optional user ID;
- a persistent goal;
- application metadata;
- a compacted summary;
- retained recent turns.

Create a session through the project API:

```python
session_id = project.create_session(
    user_id="alice",
    goal="Prepare the production-readiness review",
)
answer = project.ask(
    "What release evidence is required?",
    user_id="alice",
    session_id=session_id,
)
```

When the number of turns reaches `REFAAT_SESSION_COMPACTION_TURNS`, Refaat summarizes older turns and retains the most recent `REFAAT_RETAINED_SESSION_TURNS`. Both thresholds are validated so the retained count remains below the compaction threshold.

## Cache behavior

The current session summary is part of the effective query-context fingerprint. A cache entry created before compaction cannot be reused after the summary changes. The current user turn is recorded after the query result is produced, so it does not leak into its own retrieval context.

## Explicit preferences

Positive or negative feedback may update bounded observations for a user. The built-in learner records selected document preferences with confidence and observation counts. A profile change creates a new context fingerprint and therefore cannot reuse a cache entry produced under the previous profile.

```python
selected = tuple(citation.chunk_id for citation in answer.citations[:1])
project.submit_feedback(
    rating=1,
    user_id="alice",
    selected_chunk_ids=selected,
    note="This is the authoritative release policy.",
)
```

Anonymous feedback does not create a personal profile. Preferences can affect ranking and appear as the `profile` retrieval route, but the citation list still points only to corpus chunks.

Under an active access policy the session summary and the learned profile are identity-scoped: a compacted summary is surfaced only to the principal who wrote it, and the profile load and feedback write bind to the authenticated principal rather than a client-supplied `user_id`, with the write authorized so it cannot record or reveal a chunk the principal may not see. See [Authorization](authorization.md) and [Learning and preferences](learning-and-preferences.md).

## Long-running goals

Session goals remain stored even when turns are compacted. They can be used by embedding applications to restore work across terminal sessions or worker restarts. Refaat does not autonomously invent or change a user goal; applications control goal creation and updates.

See `examples/02_sessions_feedback.py` for the complete executable flow.
