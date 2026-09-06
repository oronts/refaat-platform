# Learning and preferences

Refaat learns per-identity retrieval preferences from explicit feedback and uses them to rank future results. The learned profile is identity-scoped state, so under an active policy it is bound to the authenticated principal on both read and write. For the enforcement model, see [Authorization](authorization.md).

## How it works

When a caller submits feedback with a rating and the chunk ids they found useful, Refaat records a preference for each cited chunk's document under the caller's scope. At retrieval, a non-empty profile boosts those documents and adds a `profile` route to the result, so a returning identity sees more of what it preferred.

```python
context = project.retrieve("rollback policy", principal=alice)
useful = [context.hits[0].chunk.id]
project.submit_feedback(rating=1, selected_chunk_ids=useful, principal=alice)
```

Feedback with a rating of zero is ignored. Only explicit feedback is learned; there is no implicit tracking.

## Identity scope

The profile scope comes from a single helper, `preference_scope(policy, principal, user_id)`, used by both read sites and the write:

- **With an active policy and a principal**, the scope binds to the principal id. A client-supplied `user_id` cannot select or overwrite another identity's profile.
- **With no policy**, the scope is the `user_id`, byte-identical to the unauthorized behavior. Under an active policy a principal-less call is denied by default under strict authorization, so it never reaches this scope.

This closes two classes of cross-identity leak that a naive per-`user_id` scope would allow: reading another identity's profile by guessing its id, and writing into another identity's profile.

## The write is authorized

A preference is only recorded for a chunk the principal is authorized to see. Under an active policy, feedback that names a forbidden chunk id records nothing, exactly as feedback on a nonexistent chunk records nothing. The two are indistinguishable, so the presence or absence of the `profile` route can never be used as an oracle to test whether a hidden document exists.

## What the profile contains

The profile holds document-level preference weights keyed by document id, learned from explicit feedback. It never stores evidence text. Because the profile influences ranking, it folds into the answer-cache identity, so a feedback change that shifts ranking correctly invalidates a stale cached answer.

## Related pages

- [Authorization](authorization.md)
- [Sessions and memory](sessions-and-memory.md)
- [Adaptive context](adaptive-context.md)
