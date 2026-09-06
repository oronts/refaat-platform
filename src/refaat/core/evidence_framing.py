from __future__ import annotations

from collections.abc import Sequence

from refaat.domain.models import SearchHit

_ENVELOPE_OPEN = "<<UNTRUSTED_EVIDENCE"
_ENVELOPE_CLOSE = "<<END_UNTRUSTED_EVIDENCE>>"


def render_untrusted_evidence(hits: Sequence[SearchHit]) -> str:
    """Render retrieved chunk bodies for a model prompt inside an explicit untrusted-data envelope.

    Each retrieved chunk keeps its ``[chunk:<id>]`` citation marker so a model still cites the
    exact stored chunk id, and the body is wrapped so the boundary between developer instructions
    and untrusted retrieved text is legible to the model. This is legibility hardening only, not a
    filter: it makes no claim to detect or neutralize injected instructions in the text. Coverage is
    retrieved chunk bodies alone; graph facts, session summary, profile, and harness stay plain.
    """
    return "\n\n".join(
        f"{_ENVELOPE_OPEN} chunk={hit.chunk.id} document={hit.chunk.document_id} score={hit.score:.4f}>>\n"
        f"[chunk:{hit.chunk.id}]\n{hit.chunk.text}\n"
        f"{_ENVELOPE_CLOSE}"
        for hit in hits
    )
