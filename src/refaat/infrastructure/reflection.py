from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from refaat.core.text import content_terms
from refaat.domain.models import ContextPacket, GenerationRequest
from refaat.errors import DependencyUnavailableError

if TYPE_CHECKING:
    from refaat.ports import Generator

RETRIEVE = "retrieve"

_NECESSITY_SYSTEM_PROMPT = (
    "You decide whether a question needs corpus retrieval to answer. Reply with a single word: "
    "'retrieve' if external evidence is needed, 'skip' if it can be answered without retrieval. "
    "No commentary."
)


@dataclass(frozen=True, slots=True)
class ReflectionDecision:
    """The three Self-RAG reflections over one answered query.

    ``retrieve`` is the necessity gate's decision, ``is_supported`` the shipped
    grounding verdict over the final answer, ``is_useful`` the deterministic
    usefulness heuristic. ``rationale`` is a program output only: it never reaches
    the generator and is never citeable.
    """

    retrieve: bool
    is_supported: bool
    is_useful: bool
    rationale: str


class NecessityGate(Protocol):
    @property
    def cache_id(self) -> str: ...

    def decide(self, query: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class DeterministicNecessityGate:
    """Model-free retrieval-necessity gate over the deterministic content-term signal.

    Retrieval is deemed necessary when the query carries at least
    ``min_content_terms`` content terms (function words and short tokens dropped
    by :func:`content_terms`), so a content-rich query retrieves and a trivial one
    skips. Pure, so the decision is reproducible offline and a live discriminator,
    not a constant.
    """

    min_content_terms: int = 2

    @property
    def cache_id(self) -> str:
        return f"selfrag-deterministic-v1:{self.min_content_terms}"

    def decide(self, query: str) -> bool:
        return len(content_terms(query)) >= self.min_content_terms


class ModelNecessityGate:
    """Model-backed necessity gate that degrades to the deterministic decision.

    A generator judges whether retrieval is needed for the query alone. A missing
    dependency degrades to the deterministic gate's decision rather than guessing,
    so the offline path stays honest and no decision is fabricated.
    """

    def __init__(
        self,
        generator: Generator,
        *,
        deterministic: DeterministicNecessityGate | None = None,
    ) -> None:
        self._generator = generator
        self._deterministic = deterministic or DeterministicNecessityGate()

    @property
    def cache_id(self) -> str:
        return f"selfrag-model:{self._generator.model_id}"

    def decide(self, query: str) -> bool:
        baseline = self._deterministic.decide(query)
        try:
            result = self._generator.generate(
                GenerationRequest(query=query, context=_empty_context(query), system_prompt=_NECESSITY_SYSTEM_PROMPT)
            )
        except DependencyUnavailableError:
            return baseline
        return result.answer.strip().casefold().startswith(RETRIEVE)


def is_useful_answer(answer: str, *, grounded: bool, min_length: int) -> bool:
    """Deterministic usefulness heuristic: a grounded, non-trivial answer.

    An answer is useful only when it is grounded and its trimmed length reaches
    ``min_length``, so an empty or ungrounded answer is never useful. Pure, so
    usefulness is offline-checkable; a model usefulness judge activates quality
    on top of this floor.
    """
    return grounded and len(answer.strip()) >= min_length


def _empty_context(query: str) -> ContextPacket:
    return ContextPacket(
        query=query,
        hits=(),
        graph_facts=(),
        session_summary=None,
        profile={},
        supplemental_harness={},
        token_count=0,
        routes=(),
    )
