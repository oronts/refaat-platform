from __future__ import annotations

import re

from refaat.core.text import content_terms
from refaat.domain.models import ContextPacket, GenerationRequest
from refaat.errors import DependencyUnavailableError
from refaat.ports import Generator

_HYPE_SYSTEM_PROMPT = (
    "You are a retrieval assistant. Write short questions that the passage directly answers, one per "
    "line. Do not add numbering, sources, citation markers, or commentary. Reply with the questions only."
)
_LIST_PREFIX = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*")

_TEMPLATES = (
    "What is {subject}?",
    "How does {subject} work?",
    "Why is {subject} important?",
)


def _subject(chunk_text: str) -> str:
    terms = content_terms(chunk_text)
    if not terms:
        return ""
    return " ".join(terms[:6])


def _templated_questions(chunk_text: str, max_questions: int) -> list[str]:
    subject = _subject(chunk_text)
    if not subject:
        return []
    return [template.format(subject=subject) for template in _TEMPLATES[:max_questions]]


class DeterministicHypotheticalGenerator:
    """Model-free HyPE question generator: templated questions from the chunk content terms.

    Reproducible and network-free, so the offline index is fully deterministic:
    the same chunk always yields the same questions, which keeps re-ingestion
    idempotent. The questions are built from the chunk's content terms so their
    embeddings land near a real user question about the same subject, giving the
    question-to-question match genuine signal without a model.
    """

    def __init__(self, max_questions: int = 3) -> None:
        if not 1 <= max_questions <= len(_TEMPLATES):
            raise ValueError(f"max_questions must be between 1 and {len(_TEMPLATES)}")
        self._max_questions = max_questions

    @property
    def provider_id(self) -> str:
        return f"hype-deterministic-v1:{self._max_questions}"

    def questions(self, chunk_text: str, *, workspace_id: str = "") -> list[str]:
        del workspace_id
        return _templated_questions(chunk_text, self._max_questions)


class ModelHypotheticalGenerator:
    """Model-backed HyPE question generator with a deterministic fallback.

    A real generator writes the questions the passage answers; the call sees only
    the chunk text (an empty retrieval context), so no stored chunk id reaches the
    model and no question can carry or forge a citation. A failed dependency or a
    blank generation degrades to the deterministic templated questions, so an
    ingestion run never fails or stalls on the model path.
    """

    def __init__(self, generator: Generator, fallback: DeterministicHypotheticalGenerator) -> None:
        self._generator = generator
        self._fallback = fallback

    @property
    def provider_id(self) -> str:
        return f"hype-model:{self._generator.model_id}"

    def questions(self, chunk_text: str, *, workspace_id: str = "") -> list[str]:
        request = GenerationRequest(
            query=chunk_text,
            context=ContextPacket(
                query=chunk_text,
                hits=(),
                graph_facts=(),
                session_summary=None,
                profile={},
                supplemental_harness={},
                token_count=0,
                routes=(),
            ),
            system_prompt=_HYPE_SYSTEM_PROMPT,
            workspace_id=workspace_id,
        )
        try:
            generated = self._generator.generate(request).answer
        except DependencyUnavailableError:
            return self._fallback.questions(chunk_text)
        lines: list[str] = []
        seen: set[str] = set()
        for raw in generated.splitlines():
            cleaned = _LIST_PREFIX.sub("", raw).strip()
            key = cleaned.casefold()
            if cleaned and key not in seen:
                seen.add(key)
                lines.append(cleaned)
        return lines or self._fallback.questions(chunk_text)
