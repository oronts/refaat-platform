from __future__ import annotations

from collections.abc import Mapping

from refaat.config import Settings
from refaat.core.text import estimate_tokens
from refaat.domain.models import JsonObject
from refaat.infrastructure.database.store import SQLStore

TURN_AUTH_SCOPE_KEY = "_auth_scope"
SUMMARY_AUTH_SCOPE_KEY = "summary_auth_scope"
SUMMARY_CITED_DOCUMENTS_KEY = "summary_cited_documents"


def tag_turn_metadata(metadata: JsonObject | None, auth_scope: str | None) -> JsonObject | None:
    """Stamp a turn's writer authorization scope into its metadata under an active policy.

    Returns the metadata unchanged when no scope is supplied (the default no-policy path), so a
    turn written without authorization is byte-identical to before this binding existed. The tag
    lets compaction fold only same-scope turns, so a shared session never merges two principals'
    evidence into one summary.
    """
    if auth_scope is None:
        return metadata
    tagged = dict(metadata or {})
    tagged[TURN_AUTH_SCOPE_KEY] = auth_scope
    return tagged


def _turn_scope(turn: JsonObject) -> str | None:
    metadata = turn.get("metadata")
    if isinstance(metadata, Mapping):
        scope = metadata.get(TURN_AUTH_SCOPE_KEY)
        if isinstance(scope, str):
            return scope
    return None


def _summary_scope(session: JsonObject) -> str | None:
    metadata = session.get("metadata")
    if isinstance(metadata, Mapping):
        scope = metadata.get(SUMMARY_AUTH_SCOPE_KEY)
        if isinstance(scope, str):
            return scope
    return None


def stored_summary_cited_documents(session: JsonObject) -> frozenset[str] | None:
    """The document ids cited by the summary's folded turns, recomputed against current ACLs at read.

    Returns a frozenset (possibly empty, meaning the summary folded turns that cited nothing) when the
    provenance was recorded, and None when it was not recorded at all: a summary compacted before a
    router was installed has no stored ids and its folding turns are gone, so its provenance is
    unrecoverable and a read-time ceiling must fail closed rather than read it as public.
    """
    metadata = session.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    raw = metadata.get(SUMMARY_CITED_DOCUMENTS_KEY)
    if not isinstance(raw, list):
        return None
    return frozenset(doc for doc in raw if isinstance(doc, str))


class SessionCompactor:
    def __init__(self, store: SQLStore, settings: Settings, *, classify_evidence: bool = False) -> None:
        self.store = store
        self.settings = settings
        # Record the summary's cited document ids only when a model router or residency gate is
        # installed to consume them at read time; otherwise the store is skipped and the no-consumer
        # compaction path is unchanged.
        self.classify_evidence = classify_evidence

    def compact_if_needed(self, workspace_id: str, session_id: str, auth_scope: str | None = None) -> bool:
        turns = self.store.list_session_turns(workspace_id, session_id)
        threshold = self.settings.session_compaction_turns
        if len(turns) <= threshold:
            return False
        retained_count = self.settings.retained_session_turns
        compacted = turns[:-retained_count]
        session = self.store.get_session(workspace_id, session_id)
        previous_scope = _summary_scope(session)
        # A scope-blind caller (the background session.compact job carries no principal) must never
        # recompact content written under an active policy: folding it would keep a stale writer tag
        # on new content and leak it across the principal boundary. It refuses and leaves the
        # session untouched so a principal-driven compaction handles it with the correct scope.
        if auth_scope is None and (
            previous_scope is not None or any(_turn_scope(turn) is not None for turn in compacted)
        ):
            return False
        previous = str(session.get("summary") or "").strip()
        if previous_scope != auth_scope:
            # A prior summary written under a different (or unknown) scope is never merged into this
            # writer's summary, so one summary row never mixes two principals' evidence under one tag.
            previous = ""
        lines: list[str] = []
        folded_ids: list[str] = []
        for turn in compacted:
            # Fold only content whose writer scope matches this compaction's scope. In the default
            # no-policy path every scope is None, so every turn folds and the result is unchanged.
            if _turn_scope(turn) != auth_scope:
                continue
            folded_ids.append(str(turn["id"]))
            content = str(turn["content"])
            if estimate_tokens(content) > 160:
                content = " ".join(content.split()[:160]) + " …"
            lines.append(f"{turn['role']}: {content}")
        # Only same-scope turns are folded, so only they may be removed: a foreign-scope principal's
        # turns are left for its own compaction. With nothing to fold there is nothing to persist.
        if not folded_ids:
            return False
        summary = "\n".join(value for value in [previous, *lines] if value)
        summary_cited_documents: list[str] | None = None
        # Record cited-document provenance whenever a router consumes it OR a policy is active, so the
        # read-time ACL re-check has provenance even in a policy-without-router deployment.
        if self.classify_evidence or auth_scope is not None:
            cited, complete = self._summary_cited_documents(session, compacted, auth_scope, prior_folded=bool(previous))
            # Record the ids only when the provenance is complete. A prior summary with unrecorded
            # provenance (compacted before a router existed) folds its text but not its cited ids, so
            # the merged summary is left unrecorded (None) and a read-time ceiling fails closed on it.
            summary_cited_documents = sorted(cited) if complete else None
        self.store.compact_session(
            workspace_id,
            session_id,
            summary=summary[-16_000:],
            remove_turn_ids=folded_ids,
            auth_scope=auth_scope,
            summary_cited_documents=summary_cited_documents,
        )
        return True

    def _summary_cited_documents(
        self,
        session: JsonObject,
        compacted: list[JsonObject],
        auth_scope: str | None,
        *,
        prior_folded: bool,
    ) -> tuple[frozenset[str], bool]:
        cited: set[str] = set()
        for turn in compacted:
            if _turn_scope(turn) != auth_scope:
                continue
            metadata = turn.get("metadata")
            if isinstance(metadata, Mapping):
                for document_id in metadata.get("cited_document_ids", []) or []:
                    if isinstance(document_id, str):
                        cited.add(document_id)
        complete = True
        if prior_folded:
            prior = stored_summary_cited_documents(session)
            if prior is None:
                complete = False
            else:
                cited |= prior
        return frozenset(cited), complete
