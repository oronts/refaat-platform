from __future__ import annotations

import re
from collections.abc import Sequence

from refaat.domain.models import Chunk, Citation
from refaat.errors import ValidationError

CITATION_PATTERN = re.compile(r"\[chunk:([\w-]+)\]")
_MARKER_WITH_LEAD = re.compile(r"[ \t]?\[chunk:([\w-]+)\]")


def extract_cited_ids(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(CITATION_PATTERN.findall(text)))


def strip_uncited_markers(text: str, cited_chunk_ids: Sequence[str]) -> str:
    """Drop any ``[chunk:id]`` marker from answer text whose id is not a resolved citation.

    The structured citation list is reconciled against stored chunks, but a model can also leave a
    marker in the prose that resolves to nothing (a hallucinated, unknown, or since-deleted id).
    Such a marker is a citation to no evidence, so it is removed to keep the served answer body
    consistent with the citation contract. A marker whose id is cited is preserved byte-for-byte, so
    a text whose every marker is cited (the deterministic extractive path always is) is unchanged.
    """
    allowed = set(cited_chunk_ids)
    return _MARKER_WITH_LEAD.sub(lambda match: match.group(0) if match.group(1) in allowed else "", text)


def validate_cited_chunks(cited_chunk_ids: Sequence[str], allowed_chunks: Sequence[Chunk]) -> tuple[str, ...]:
    allowed = {chunk.id for chunk in allowed_chunks}
    seen: set[str] = set()
    valid: list[str] = []
    for chunk_id in cited_chunk_ids:
        if chunk_id not in allowed:
            raise ValidationError(f"Generator cited an unknown chunk: {chunk_id}")
        if chunk_id not in seen:
            seen.add(chunk_id)
            valid.append(chunk_id)
    return tuple(valid)


def filter_cited_chunks(cited_chunk_ids: Sequence[str], allowed_chunks: Sequence[Chunk]) -> tuple[str, ...]:
    """Keep only citations that resolve to retrieved chunks, preserving order.

    Unlike :func:`validate_cited_chunks`, an id the model invented is dropped
    rather than raising, so a single hallucinated citation cannot fail an
    otherwise grounded answer.
    """
    allowed = {chunk.id for chunk in allowed_chunks}
    seen: set[str] = set()
    valid: list[str] = []
    for chunk_id in cited_chunk_ids:
        if chunk_id in allowed and chunk_id not in seen:
            seen.add(chunk_id)
            valid.append(chunk_id)
    return tuple(valid)


def build_citations(cited_chunk_ids: Sequence[str], chunks: Sequence[Chunk]) -> tuple[Citation, ...]:
    by_id = {chunk.id: chunk for chunk in chunks}
    citations: list[Citation] = []
    for index, chunk_id in enumerate(cited_chunk_ids, start=1):
        chunk = by_id[chunk_id]
        quote = chunk.text[:500].strip()
        citations.append(
            Citation(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                label=f"[{index}]",
                quote=quote,
            )
        )
    return tuple(citations)
