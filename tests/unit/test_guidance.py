from __future__ import annotations

from refaat.core.guidance import (
    RESERVED_DEMO_CITATION_ID,
    guidance_digest,
    render_guidance,
    sanitize_demonstration_answer,
)
from refaat.domain.models import Demonstration

_SENTINEL = f"[chunk:{RESERVED_DEMO_CITATION_ID}]"


def test_sanitizer_rewrites_planted_chunk_ids() -> None:
    text = "Grounded in [chunk:real_abc123] and [chunk:doc-42] evidence."
    sanitized = sanitize_demonstration_answer(text)
    assert "real_abc123" not in sanitized
    assert "doc-42" not in sanitized
    assert sanitized.count(_SENTINEL) == 2


def test_sanitizer_leaves_plain_text_untouched() -> None:
    text = "A plain sentence with no citation markers at all."
    assert sanitize_demonstration_answer(text) == text


def test_guidance_digest_is_none_when_empty() -> None:
    assert guidance_digest(None, ()) is None
    assert guidance_digest("", ()) is None


def test_guidance_digest_stable_under_canonical_json() -> None:
    demos = (Demonstration("What is X?", "X is a thing."),)
    assert guidance_digest("be brief", demos) == guidance_digest("be brief", demos)


def test_guidance_digest_differs_on_any_change() -> None:
    base = guidance_digest("be brief", (Demonstration("q", "a"),))
    assert base != guidance_digest("be verbose", (Demonstration("q", "a"),))
    assert base != guidance_digest("be brief", (Demonstration("q", "b"),))
    assert base != guidance_digest("be brief", ())


def test_guidance_digest_differs_on_demonstration_reordering() -> None:
    first = (Demonstration("a", "1"), Demonstration("b", "2"))
    reordered = (Demonstration("b", "2"), Demonstration("a", "1"))
    assert guidance_digest(None, first) != guidance_digest(None, reordered)


def test_render_guidance_emits_never_cite_banner_and_no_unsanitized_chunk_id() -> None:
    demos = (Demonstration("Where is [chunk:real_q]?", "Here [chunk:real_a]."),)
    rendered = render_guidance("keep it short", demos)
    assert "Never cite" in rendered
    assert "real_q" not in rendered
    assert "real_a" not in rendered
    assert _SENTINEL in rendered
    assert "keep it short" in rendered
