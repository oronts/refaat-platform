from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from refaat.core.citations import CITATION_PATTERN
from refaat.domain.models import Demonstration

RESERVED_DEMO_CITATION_ID = "example"
_EXAMPLE_CITATION = f"[chunk:{RESERVED_DEMO_CITATION_ID}]"


def sanitize_demonstration_answer(text: str) -> str:
    """Rewrite every citation-shaped token to a reserved example sentinel.

    The citation contract requires every returned citation to resolve to a real
    corpus chunk id. A demonstration must therefore never carry a real chunk id
    into a provider prompt, or a model could echo it and forge a citation to a
    chunk the answer was not grounded in. Rewriting every ``[chunk:...]`` match
    to ``[chunk:example]`` neutralizes any planted id in any demonstration field.
    """
    return CITATION_PATTERN.sub(_EXAMPLE_CITATION, text)


def guidance_digest(instructions: str | None, demonstrations: Sequence[Demonstration]) -> str | None:
    """Content digest of the active guidance, or ``None`` when there is none.

    Returning ``None`` for empty guidance lets the caller keep the unguided
    answer-cache identity byte-identical to the pre-guidance value. The digest
    is order-sensitive over the demonstrations so reordering them is treated as
    an answer-affecting change.
    """
    if not instructions and not demonstrations:
        return None
    payload = {
        "instructions": instructions or "",
        "demonstrations": [demonstration.as_json() for demonstration in demonstrations],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def render_guidance(instructions: str | None, demonstrations: Sequence[Demonstration]) -> str:
    """Render a format-only, never-cite guidance block with a sanitized body.

    The whole rendered string is passed through :func:`sanitize_demonstration_answer`,
    so a citation-shaped token in either a demonstration question or answer can
    never reach a provider verbatim.
    """
    lines = [
        "The guidance below is ILLUSTRATIVE FORMATTING ONLY and is not evidence.",
        (
            "Never cite any chunk id shown in this guidance; cite only real evidence "
            f"chunk ids from the Evidence section. Example citations render as {_EXAMPLE_CITATION}."
        ),
    ]
    if instructions:
        lines.append("")
        lines.append("Instructions:")
        lines.append(instructions.strip())
    if demonstrations:
        lines.append("")
        lines.append("Format examples:")
        for index, demonstration in enumerate(demonstrations, start=1):
            lines.append(f"Q{index}: {demonstration.question}")
            lines.append(f"A{index}: {demonstration.answer}")
    return sanitize_demonstration_answer("\n".join(lines))
